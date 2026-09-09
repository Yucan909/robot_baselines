#!/usr/bin/env python3
"""Select a V4 checkpoint by deterministic object-level DEV x0-MSE only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


HOME = Path("/home/feng")
CODE = HOME / "robot_baselines/repos/pa3ff_official/reproduction"
OFFICIAL = HOME / "robot_baselines/repos/pa3ff_official"
DATA = HOME / "robot_baselines/results/pa3ff/padp_data_v4_timeindexed_baseframe_fourtask"
TEXT = HOME / "robot_baselines/results/pa3ff/reproduction_v1/frozen_text_embeddings.npz"
TASKS = {
    "door_open": ("door", "open door"),
    "door_close": ("door", "close door"),
    "drawer_open": ("drawer", "open drawer"),
    "drawer_close": ("drawer", "close drawer"),
}
ANCHORS = (-1, 0, 4, 8, 12)
TIMESTEPS = (25, 50, 75, 90, 99)
SEED = 2026090701


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class DevIndex:
    def __init__(self, task: str):
        with np.load(DATA / f"{task}_dev_v4.npz", allow_pickle=False) as z:
            for key in (
                "source_action_xyz_base_grasptarget",
                "source_action_rotmat_base_grasptarget", "source_action_finger",
                "sample_source_index", "sample_anchor_keyframe",
                "sample_initial_robot_proprioception",
            ):
                setattr(self, key, np.array(z[key], copy=True))
        self.field = np.load(DATA / f"{task}_dev_pa3ff_field_f16.npy", mmap_mode="r")
        self.point = np.load(DATA / f"{task}_dev_pointcloud_base_f32.npy", mmap_mode="r")

    def action(self, sample: int) -> np.ndarray:
        source = int(self.sample_source_index[sample])
        anchor = max(0, int(self.sample_anchor_keyframe[sample]))
        future = np.minimum(anchor + np.arange(16), 15)
        xyz = self.source_action_xyz_base_grasptarget[source, future]
        rotation = self.source_action_rotmat_base_grasptarget[source, future]
        rot6d = np.concatenate([rotation[..., :, 0], rotation[..., :, 1]], axis=-1)
        finger = self.source_action_finger[source, future, None]
        return np.concatenate([xyz, rot6d, finger], axis=-1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples-per-task-anchor", type=int, default=30)
    args = parser.parse_args()
    training = args.training_dir.resolve()
    checkpoints = sorted((training / "checkpoints").glob("step*.pt"))
    if not checkpoints:
        raise RuntimeError(f"no checkpoints in {training}")
    sys.path.insert(0, str(CODE))
    sys.path.insert(0, str(OFFICIAL))
    sys.path.insert(0, str(OFFICIAL / "libs/sonata"))
    os.chdir(OFFICIAL)
    first_payload = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    positional = first_payload.get("stage") == "PA3FF_PADP_V5_TIMEINDEXED_BASEFRAME_POSITIONAL_BALANCED"
    expected_stage = (
        "PA3FF_PADP_V5_TIMEINDEXED_BASEFRAME_POSITIONAL_BALANCED"
        if positional else "PA3FF_PADP_V4_TIMEINDEXED_BASEFRAME"
    )
    if positional:
        from padp_model_fourtask_v5 import FourTaskPADPPolicyV5 as PolicyClass
        from padp_model_fourtask_v5 import FourTaskX0Diffusion
    else:
        from padp_model_fourtask import FourTaskPADPPolicy as PolicyClass
        from padp_model_fourtask import FourTaskX0Diffusion
    del first_payload

    with np.load(DATA / "ACTION_NORMALIZATION_V4.npz", allow_pickle=False) as z:
        mean = np.asarray(z["mean"], dtype=np.float32)
        std = np.asarray(z["std"], dtype=np.float32)
    with np.load(TEXT, allow_pickle=False) as z:
        texts = np.asarray(z["texts"]).astype(str).tolist()
        embeddings = np.asarray(z["embeddings"], dtype=np.float32)
    text_map = {text: embeddings[row] for row, text in enumerate(texts)}
    dev = {task: DevIndex(task) for task in TASKS}
    rng = np.random.default_rng(SEED)
    samples = []
    for task, index in dev.items():
        anchors = np.asarray(index.sample_anchor_keyframe, dtype=np.int64)
        for anchor in ANCHORS:
            eligible = np.flatnonzero(anchors == anchor)
            count = min(args.samples_per_task_anchor, len(eligible))
            chosen = rng.choice(eligible, count, replace=False)
            samples.extend((task, int(row), anchor) for row in chosen)
    # Noise is generated once and reused byte-identically for every checkpoint.
    fixed_noise = rng.standard_normal(
        (len(samples), len(TIMESTEPS), 16, 10), dtype=np.float32
    )
    device = torch.device("cuda")
    diffusion = FourTaskX0Diffusion(100).to(device)
    reports = []
    for checkpoint in checkpoints:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("stage") != expected_stage:
            raise RuntimeError(f"invalid checkpoint {checkpoint}")
        model = PolicyClass(
            action_dim=10, d_model=256, scene_layers=4, scene_heads=8,
            scene_ff=1024, unet_down_dims=(256, 512, 1024),
        ).to(device).eval()
        model.load_state_dict(payload["model"], strict=True)
        sums = defaultdict(float)
        counts = defaultdict(int)
        with torch.inference_mode():
            for start in range(0, len(samples), args.batch_size):
                block = samples[start:start + args.batch_size]
                feature, point, prop, cls, instruction, x0 = [], [], [], [], [], []
                for task, sample, anchor in block:
                    index = dev[task]
                    source = int(index.sample_source_index[sample])
                    feature.append(np.asarray(index.field[source], dtype=np.float32))
                    point.append(np.asarray(index.point[source], dtype=np.float32))
                    prop.append(index.sample_initial_robot_proprioception[sample])
                    part, command = TASKS[task]
                    cls.append(text_map[part])
                    instruction.append(text_map[command])
                    x0.append((index.action(sample) - mean[None]) / std[None])
                feature_t = torch.from_numpy(np.stack(feature)).to(device)
                point_t = torch.from_numpy(np.stack(point)).to(device)
                prop_t = torch.from_numpy(np.stack(prop).astype(np.float32)).to(device)
                cls_t = torch.from_numpy(np.stack(cls).astype(np.float32)).to(device)
                instruction_t = torch.from_numpy(np.stack(instruction).astype(np.float32)).to(device)
                x0_t = torch.from_numpy(np.stack(x0).astype(np.float32)).to(device)
                condition = (
                    model.condition(feature_t, point_t, cls_t, instruction_t, prop_t)
                    if positional else
                    model.condition(feature_t, cls_t, instruction_t, prop_t)
                )
                b = len(block)
                condition = condition[:, None].expand(b, len(TIMESTEPS), -1).reshape(
                    b * len(TIMESTEPS), -1
                )
                x0_many = x0_t[:, None].expand(b, len(TIMESTEPS), 16, 10).reshape(
                    b * len(TIMESTEPS), 16, 10
                )
                timestep = torch.tensor(TIMESTEPS, device=device).repeat(b)
                noise = torch.from_numpy(fixed_noise[start:start + b]).to(device).reshape(
                    b * len(TIMESTEPS), 16, 10
                )
                xt = diffusion.q_sample(x0_many, timestep, noise)
                prediction = model.action_head(xt, timestep, condition)
                mse = torch.mean((prediction.float() - x0_many.float()) ** 2,
                                 dim=(1, 2)).reshape(b, len(TIMESTEPS)).cpu().numpy()
                for row, (task, sample, anchor) in enumerate(block):
                    for column, timestep_value in enumerate(TIMESTEPS):
                        value = float(mse[row, column])
                        for key in (
                            "overall", f"task:{task}", f"anchor:{anchor}",
                            f"task_anchor:{task}:{anchor}", f"timestep:{timestep_value}",
                        ):
                            sums[key] += value
                            counts[key] += 1
        means = {key: sums[key] / counts[key] for key in sorted(sums)}
        macro_task = float(np.mean([means[f"task:{task}"] for task in TASKS]))
        report = {
            "checkpoint": str(checkpoint), "sha256": sha256(checkpoint),
            "step": int(payload["step"]), "macro_task_x0_mse": macro_task,
            "means": means, "counts": dict(counts),
        }
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        del model, payload
        torch.cuda.empty_cache()
    selected = min(reports, key=lambda row: (row["macro_task_x0_mse"], row["step"]))
    result = {
        "schema": "pa3ff_padp_dev_checkpoint_selection_v2",
        "checkpoint_stage": expected_stage, "positional_point_tokens": positional,
        "status": "complete", "selection_metric": "macro over four task DEV x0-MSE",
        "timesteps": list(TIMESTEPS), "seed": SEED,
        "samples_per_task_anchor_requested": args.samples_per_task_anchor,
        "sample_count": len(samples), "formal_rollout_used": False,
        "success_metric_used": False, "reports": reports, "selected": selected,
    }
    target = training / "DEV_CHECKPOINT_SELECTION.json"
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
