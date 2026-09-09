#!/usr/bin/env python3
"""Train reconstructed time-indexed PA3FF/PADP V4 on TRAIN data only."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


HOME = Path("/home/feng")
CODE = HOME / "robot_baselines/repos/pa3ff_official/reproduction"
OFFICIAL = HOME / "robot_baselines/repos/pa3ff_official"
RESULTS = HOME / "robot_baselines/results/pa3ff"
DATA = RESULTS / "padp_data_v4_timeindexed_baseframe_fourtask"
REPRESENTATION = RESULTS / "representation_native5_balanced_v1/instance_net_snapshots/instance_net_step10000.pth"
TEXT = RESULTS / "reproduction_v1/frozen_text_embeddings.npz"
MODEL_SOURCE = CODE / "padp_model_fourtask.py"
VARIANT = os.environ.get("PA3FF_PADP_VARIANT", "v4").strip()
if VARIANT not in {"v4", "v5_positional"}:
    raise RuntimeError(f"unknown PA3FF_PADP_VARIANT={VARIANT}")
POSITIONAL = VARIANT == "v5_positional"
if POSITIONAL:
    MODEL_SOURCE = CODE / "padp_model_fourtask_v5.py"
CHECKPOINT_STAGE = (
    "PA3FF_PADP_V5_TIMEINDEXED_BASEFRAME_POSITIONAL_BALANCED"
    if POSITIONAL else "PA3FF_PADP_V4_TIMEINDEXED_BASEFRAME"
)
TASKS = {
    "door_open": ("door", "open door"),
    "door_close": ("door", "close door"),
    "drawer_open": ("drawer", "open drawer"),
    "drawer_close": ("drawer", "close drawer"),
}
TASK_ORDER = tuple(TASKS)
HORIZON = 16
ACTION_DIM = 10
DDPM_STEPS = 100
SEED = 20260907


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def append_json(path: Path, value) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")


def guard(name: str, condition, detail=None) -> None:
    state = "PASS" if bool(condition) else "FAIL"
    print(f"{name:<92s} {state}", flush=True)
    if detail is not None:
        print("   ", detail, flush=True)
    if not condition:
        raise RuntimeError(f"guard failed: {name}: {detail}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class V4Index:
    def __init__(self, task: str, split: str = "train"):
        self.task = task
        self.path = DATA / f"{task}_{split}_v4.npz"
        with np.load(self.path, allow_pickle=False) as z:
            keys = (
                "schema", "task_instruction", "part_cls", "source_shape",
                "source_action_xyz_base_grasptarget",
                "source_action_rotmat_base_grasptarget", "source_action_finger",
                "sample_source_index", "sample_anchor_keyframe",
                "sample_initial_robot_proprioception",
            )
            missing = [key for key in keys if key not in z.files]
            if missing:
                raise RuntimeError(f"{self.path}: missing {missing}")
            for key in keys:
                setattr(self, key, np.array(z[key], copy=True))
        self.point = np.load(DATA / f"{task}_{split}_pointcloud_base_f32.npy", mmap_mode="r")
        self.normal = np.load(DATA / f"{task}_{split}_normal_base_f32.npy", mmap_mode="r")
        self.field = np.load(DATA / f"{task}_{split}_pa3ff_field_f16.npy", mmap_mode="r")
        if self.point.shape != self.normal.shape or self.point.shape[1:] != (1024, 3):
            raise RuntimeError(f"{task}: geometry cache shape mismatch")
        if self.field.shape != (len(self.point), 1024, 768):
            raise RuntimeError(f"{task}: PA3FF field cache shape mismatch {self.field.shape}")
        if len(self.source_shape) != len(self.point):
            raise RuntimeError(f"{task}: source/cache row mismatch")
        self.count = int(len(self.sample_source_index))

    def action(self, sample: int) -> np.ndarray:
        source = int(self.sample_source_index[sample])
        anchor = max(0, int(self.sample_anchor_keyframe[sample]))
        future = np.minimum(anchor + np.arange(HORIZON), HORIZON - 1)
        xyz = self.source_action_xyz_base_grasptarget[source, future]
        rotation = self.source_action_rotmat_base_grasptarget[source, future]
        rot6d = np.concatenate([rotation[..., :, 0], rotation[..., :, 1]], axis=-1)
        finger = self.source_action_finger[source, future, None]
        return np.concatenate([xyz, rot6d, finger], axis=-1).astype(np.float32)


class UniformRowStream:
    def __init__(self, counts: dict[str, int], batch_size: int, seed: int):
        self.counts = {task: int(counts[task]) for task in TASK_ORDER}
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)
        self.offsets = {}
        offset = 0
        for task in TASK_ORDER:
            self.offsets[task] = (offset, offset + self.counts[task])
            offset += self.counts[task]
        self.total = offset
        self.order = self.rng.permutation(self.total)
        self.position = 0
        self.epoch = 0

    def next(self) -> list[tuple[str, int]]:
        ids = []
        while len(ids) < self.batch_size:
            take = min(self.batch_size - len(ids), self.total - self.position)
            ids.extend(self.order[self.position:self.position + take].tolist())
            self.position += take
            if self.position == self.total:
                self.order = self.rng.permutation(self.total)
                self.position = 0
                self.epoch += 1
        result = []
        for gid in ids:
            for task in TASK_ORDER:
                lower, upper = self.offsets[task]
                if lower <= gid < upper:
                    result.append((task, int(gid - lower)))
                    break
        return result

    def state_dict(self) -> dict:
        return {
            "counts": self.counts, "batch_size": self.batch_size,
            "order": self.order, "position": self.position, "epoch": self.epoch,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict) -> None:
        if state["counts"] != self.counts or int(state["batch_size"]) != self.batch_size:
            raise RuntimeError("row stream contract mismatch")
        self.order = np.asarray(state["order"], dtype=np.int64)
        self.position = int(state["position"])
        self.epoch = int(state["epoch"])
        self.rng.bit_generator.state = state["rng_state"]


class BalancedTaskRowStream:
    """Uniform task sampling; each row within a task remains uniform."""
    def __init__(self, counts: dict[str, int], batch_size: int, seed: int):
        if batch_size % len(TASK_ORDER):
            raise ValueError("balanced batch size must be divisible by four")
        self.counts = {task: int(counts[task]) for task in TASK_ORDER}
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)
        self.draws = 0
        self.epoch = 0

    def next(self) -> list[tuple[str, int]]:
        per_task = self.batch_size // len(TASK_ORDER)
        rows = [
            (task, int(row))
            for task in TASK_ORDER
            for row in self.rng.integers(0, self.counts[task], size=per_task)
        ]
        self.rng.shuffle(rows)
        self.draws += self.batch_size
        self.epoch = int(self.draws // sum(self.counts.values()))
        return rows

    def state_dict(self) -> dict:
        return {"counts": self.counts, "batch_size": self.batch_size,
                "draws": self.draws, "epoch": self.epoch,
                "rng_state": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        if state["counts"] != self.counts or int(state["batch_size"]) != self.batch_size:
            raise RuntimeError("balanced row stream contract mismatch")
        self.draws = int(state["draws"])
        self.epoch = int(state["epoch"])
        self.rng.bit_generator.state = state["rng_state"]


def rng_state(torch) -> dict:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(torch, state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("sanity", "formal"), required=True)
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resume", choices=("auto", "fresh"), default="auto")
    args = parser.parse_args()
    updates = int(args.updates or (1000 if args.phase == "sanity" else 30000))
    if updates <= 0 or args.batch_size <= 0:
        raise ValueError("positive updates and batch size required")
    output = RESULTS / ("reproduction_v5" if POSITIONAL else "reproduction_v4") / (
        f"sanity_{updates}" if args.phase == "sanity" else f"training_{updates}"
    )
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    log_path = output / "TRAIN_LOG.jsonl"

    guard("conda environment pa3ff", os.environ.get("CONDA_DEFAULT_ENV") == "pa3ff",
          os.environ.get("CONDA_DEFAULT_ENV"))
    guard("CUDA visible", os.environ.get("CUDA_VISIBLE_DEVICES", "0") in ("0", ""),
          os.environ.get("CUDA_VISIBLE_DEVICES"))
    for path in (DATA / "BUILD_MANIFEST.json", DATA / "GEOMETRY_CACHE_MANIFEST.json",
                 DATA / "PA3FF_FIELD_CACHE_MANIFEST.json",
                 DATA / "ACTION_NORMALIZATION_V4.npz",
                 DATA / "NORMALIZED_ACTION_SUPPORT_AUDIT.json",
                 DATA / "ACTION_FK_ALIGNMENT_AUDIT.json",
                 REPRESENTATION, TEXT, MODEL_SOURCE):
        guard(f"required artifact {path.name}", path.is_file(), str(path))
    build_manifest = json.loads((DATA / "BUILD_MANIFEST.json").read_text(encoding="utf-8"))
    guard("V4 data complete", build_manifest["status"] == "complete")
    guard("no forbidden data in builder", all(value == 0 for value in build_manifest["forbidden_data"].values()),
          build_manifest["forbidden_data"])

    train = {task: V4Index(task) for task in TASK_ORDER}
    counts = {task: train[task].count for task in TASK_ORDER}
    guard("expanded TRAIN rows exact", sum(counts.values()) == 286520, counts)
    all_train_objects = set()
    for task, index in train.items():
        all_train_objects.update(np.asarray(index.source_shape).astype(str).tolist())
        guard(f"{task} static feature cache finite probe",
              np.isfinite(index.point[::max(1, len(index.point)//32)]).all()
              and np.isfinite(index.normal[::max(1, len(index.normal)//32)]).all()
              and np.isfinite(index.field[::max(1, len(index.field)//32)]).all())

    with np.load(DATA / "ACTION_NORMALIZATION_V4.npz", allow_pickle=False) as z:
        action_mean = np.asarray(z["mean"], dtype=np.float32)
        action_std = np.asarray(z["std"], dtype=np.float32)
        normalization_train_only = bool(np.asarray(z["train_only"]).item())
    guard("normalization TRAIN only", normalization_train_only)
    guard("normalization finite positive", action_mean.shape == action_std.shape == (ACTION_DIM,)
          and np.isfinite(action_mean).all() and np.isfinite(action_std).all()
          and np.all(action_std > 0))

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(CODE))
    sys.path.insert(0, str(OFFICIAL))
    sys.path.insert(0, str(OFFICIAL / "libs/sonata"))
    os.chdir(OFFICIAL)
    if POSITIONAL:
        from padp_model_fourtask_v5 import FourTaskPADPPolicyV5 as PolicyClass
        from padp_model_fourtask_v5 import FourTaskX0Diffusion
    else:
        from padp_model_fourtask import FourTaskPADPPolicy as PolicyClass
        from padp_model_fourtask import FourTaskX0Diffusion

    guard("torch CUDA available", torch.cuda.is_available(), torch.__version__)
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    set_seed(SEED)
    field_manifest = json.loads((DATA / "PA3FF_FIELD_CACHE_MANIFEST.json").read_text(encoding="utf-8"))
    guard("official frozen PA3FF field cache complete", field_manifest["status"] == "complete")
    guard("feature cache uses no formal/test data", not field_manifest["formal_or_test_data_used"])

    with np.load(TEXT, allow_pickle=False) as z:
        texts = np.asarray(z["texts"]).astype(str).tolist()
        embeddings = np.asarray(z["embeddings"], dtype=np.float32)
        normalized = bool(np.asarray(z["normalized"]).item())
    guard("frozen text embeddings", normalized and embeddings.shape == (6, 768))
    text_map = {text: embeddings[row] for row, text in enumerate(texts)}

    def extract(index: V4Index, sample: int) -> torch.Tensor:
        source = int(index.sample_source_index[sample])
        # Keep the cache in fp16 through disk/host/device transfer.  Autocast
        # restores the official field to the policy's compute dtype.
        feature = np.array(index.field[source], dtype=np.float16, copy=True)
        if feature.shape != (1024, 768) or not np.isfinite(feature).all():
            raise RuntimeError(f"bad cached PA3FF feature {feature.shape}")
        return torch.from_numpy(feature)

    model = PolicyClass(action_dim=10, d_model=256, scene_layers=4,
                        scene_heads=8, scene_ff=1024,
                        unet_down_dims=(256, 512, 1024)).to(device)
    diffusion = FourTaskX0Diffusion(DDPM_STEPS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4,
                                  betas=(0.9, 0.999), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    stream = (BalancedTaskRowStream(counts, args.batch_size, SEED)
              if POSITIONAL else UniformRowStream(counts, args.batch_size, SEED))
    latest = max(checkpoints.glob("step*.pt"), default=None,
                 key=lambda p: int(p.stem.replace("step", "")))
    start = 1
    losses: list[float] = []
    task_seen: Counter = Counter()
    elapsed_before = 0.0
    if args.resume == "fresh" and latest is not None:
        raise RuntimeError(f"fresh requested but checkpoint exists: {latest}")
    if args.resume == "auto" and latest is not None:
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        guard("resume checkpoint stage", payload["stage"] == CHECKPOINT_STAGE)
        guard("resume variant", payload.get("variant", "v4") == VARIANT)
        guard("resume target updates", int(payload["target_updates"]) == updates)
        guard("resume batch size", int(payload["batch_size"]) == args.batch_size)
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scaler.load_state_dict(payload["scaler"])
        stream.load_state_dict(payload["stream"])
        restore_rng(torch, payload["rng"])
        start = int(payload["step"]) + 1
        losses = [float(x) for x in payload.get("recent_losses", [])]
        task_seen.update(payload.get("task_seen", {}))
        elapsed_before = float(payload.get("elapsed_seconds", 0.0))
        print(f"RESUME {latest} at {start}", flush=True)
    elif log_path.exists():
        raise RuntimeError(f"fresh run found stale log: {log_path}")

    protocol = {
        "stage": CHECKPOINT_STAGE, "variant": VARIANT, "phase": args.phase,
        "seed": SEED, "target_updates": updates, "batch_size": args.batch_size,
        "optimizer": "AdamW lr=3e-4 weight_decay=1e-4 no scheduler",
        "model": (
            "FourTaskPADPPolicyV5 PA3FF+base-xyz positional tokens H16 A10 DDPM100 x0-MSE"
            if POSITIONAL else "FourTaskPADPPolicy H16 A10 DDPM100 x0-MSE"
        ),
        "task_sampling": "uniform_four_task" if POSITIONAL else "uniform_over_all_rows",
        "data_manifest_sha256": sha256(DATA / "BUILD_MANIFEST.json"),
        "geometry_manifest_sha256": sha256(DATA / "GEOMETRY_CACHE_MANIFEST.json"),
        "pa3ff_field_cache_manifest_sha256": sha256(DATA / "PA3FF_FIELD_CACHE_MANIFEST.json"),
        "action_normalization_sha256": sha256(DATA / "ACTION_NORMALIZATION_V4.npz"),
        "action_support_audit_sha256": sha256(DATA / "NORMALIZED_ACTION_SUPPORT_AUDIT.json"),
        "action_fk_alignment_audit_sha256": sha256(DATA / "ACTION_FK_ALIGNMENT_AUDIT.json"),
        "model_source_sha256": sha256(MODEL_SOURCE),
        "representation_sha256": sha256(REPRESENTATION),
        "train_rows": counts, "train_objects_global": len(all_train_objects),
        "checkpoint_selection": "DEV x0-MSE only; no rollout or formal success",
        "formal_or_test_data_used": False, "success_threshold_used": False,
    }
    write_json(output / "TRAINING_PROTOCOL.json", protocol)

    checkpoint_every = 1000 if args.phase == "sanity" else 5000
    model.train()
    run_started = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for step in range(start, updates + 1):
        specs = stream.next()
        features, coordinates, proprio, cls, instruction, actions = [], [], [], [], [], []
        for task, sample in specs:
            index = train[task]
            features.append(extract(index, sample))
            source = int(index.sample_source_index[sample])
            coordinates.append(np.asarray(index.point[source], dtype=np.float32))
            proprio.append(index.sample_initial_robot_proprioception[sample])
            part_text, instruction_text = TASKS[task]
            cls.append(text_map[part_text])
            instruction.append(text_map[instruction_text])
            action = (index.action(sample) - action_mean[None]) / action_std[None]
            if action.shape != (HORIZON, ACTION_DIM) or not np.isfinite(action).all():
                raise RuntimeError(f"bad action {task}/{sample}")
            actions.append(action)
            task_seen[task] += 1
        pf = torch.stack(features).to(device, non_blocking=True)
        coord = torch.from_numpy(np.stack(coordinates)).to(device)
        prop = torch.from_numpy(np.stack(proprio).astype(np.float32)).to(device)
        cls_t = torch.from_numpy(np.stack(cls).astype(np.float32)).to(device)
        instruction_t = torch.from_numpy(np.stack(instruction).astype(np.float32)).to(device)
        x0 = torch.from_numpy(np.stack(actions).astype(np.float32)).to(device)
        timestep = torch.randint(0, DDPM_STEPS, (args.batch_size,), device=device)
        xt = diffusion.q_sample(x0, timestep, torch.randn_like(x0))
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=True):
            if POSITIONAL:
                prediction = model(pf, coord, cls_t, instruction_t, prop, xt, timestep)
            else:
                prediction = model(pf, cls_t, instruction_t, prop, xt, timestep)
            loss = F.mse_loss(prediction.float(), x0.float())
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite loss at {step}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if step == 1 or step % 1000 == 0:
            bad = [name for name, parameter in model.named_parameters()
                   if parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
            guard(f"gradient finite step {step}", not bad, bad[:10])
            guard(f"cached PA3FF field remains input-only step {step}", True)
        scaler.step(optimizer)
        scaler.update()
        value = float(loss.detach().cpu())
        losses.append(value)
        losses = losses[-1000:]
        elapsed = elapsed_before + time.time() - run_started
        if step % 100 == 0 or step == 1 or step == updates:
            record = {
                "step": step, "loss": value,
                "recent100_mean_loss": float(np.mean(losses[-100:])),
                "recent1000_mean_loss": float(np.mean(losses)),
                "examples_seen": step * args.batch_size,
                "stream_epoch": stream.epoch, "task_seen": dict(task_seen),
                "elapsed_seconds": elapsed,
                "updates_per_second": (step - start + 1) / max(time.time() - run_started, 1e-9),
                "gpu_peak_mib": torch.cuda.max_memory_allocated() / (1024**2),
            }
            append_json(log_path, record)
            print(json.dumps(record), flush=True)
        if step % checkpoint_every == 0 or step == updates:
            payload = {
                "stage": CHECKPOINT_STAGE, "phase": args.phase,
                "variant": VARIANT,
                "step": step, "target_updates": updates, "batch_size": args.batch_size,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "stream": stream.state_dict(),
                "rng": rng_state(torch), "recent_losses": losses,
                "task_seen": dict(task_seen), "elapsed_seconds": elapsed,
                "protocol": protocol, "dev_metric_used_during_training": False,
                "formal_success_used": False,
            }
            target = checkpoints / f"step{step:06d}.pt"
            temporary = target.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            os.replace(temporary, target)
            print(f"CHECKPOINT {target} {sha256(target)}", flush=True)
    summary = {
        "status": "complete", "variant": VARIANT, "stage": CHECKPOINT_STAGE,
        "phase": args.phase, "updates": updates,
        "batch_size": args.batch_size, "examples_seen": updates * args.batch_size,
        "recent100_mean_loss": float(np.mean(losses[-100:])),
        "recent1000_mean_loss": float(np.mean(losses)),
        "elapsed_seconds": elapsed_before + time.time() - run_started,
        "gpu_peak_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "last_checkpoint": str(checkpoints / f"step{updates:06d}.pt"),
        "last_checkpoint_sha256": sha256(checkpoints / f"step{updates:06d}.pt"),
    }
    write_json(output / "TRAINING_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    del model, train
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
