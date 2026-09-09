#!/usr/bin/env python3
"""Select the V4 x0-DDIM inference configuration on object-level DEV only."""
from __future__ import annotations

import hashlib
import json
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from pa3ff_policy_runtime_v4_baseframe import rotation_6d_to_matrix


HOME = Path("/home/feng")
DATA = HOME / "robot_baselines/results/pa3ff/padp_data_v4_timeindexed_baseframe_fourtask"
CHECKPOINT = HOME / "robot_baselines/results/pa3ff/reproduction_v4/training_30000/checkpoints/step030000.pt"
TEXT = HOME / "robot_baselines/results/pa3ff/reproduction_v1/frozen_text_embeddings.npz"
OUT = HOME / "robot_baselines/results/pa3ff/reproduction_v4/INFERENCE_CONFIG_DEV_SELECTION.json"
TASKS = {
    "door_open": ("door", "open door"),
    "door_close": ("door", "close door"),
    "drawer_open": ("drawer", "open drawer"),
    "drawer_close": ("drawer", "close drawer"),
}
ROWS_PER_TASK = 20
SEED = 2026090801
CONFIGS = [
    (start, noise, clip, steps)
    for start in (99, 90)
    for noise in (0.0, 0.10, 0.25, 0.50, 1.0)
    for clip in (1.5, 3.0)
    for steps in (10, 20)
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stats(values) -> dict:
    x = np.asarray(values, dtype=np.float64)
    return {"n": len(x), "mean": float(x.mean()), "median": float(np.median(x)),
            "p90": float(np.percentile(x, 90))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    device = torch.device("cuda")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    positional = payload.get("stage") == "PA3FF_PADP_V5_TIMEINDEXED_BASEFRAME_POSITIONAL_BALANCED"
    if positional:
        from padp_model_fourtask_v5 import FourTaskPADPPolicyV5 as PolicyClass
        from padp_model_fourtask_v5 import FourTaskX0Diffusion
    else:
        from padp_model_fourtask import FourTaskPADPPolicy as PolicyClass
        from padp_model_fourtask import FourTaskX0Diffusion
    model = PolicyClass(
        action_dim=10, d_model=256, scene_layers=4, scene_heads=8,
        scene_ff=1024, unet_down_dims=(256, 512, 1024),
    ).to(device).eval()
    model.load_state_dict(payload["model"], strict=True)
    diffusion = FourTaskX0Diffusion(100).to(device)
    with np.load(DATA / "ACTION_NORMALIZATION_V4.npz", allow_pickle=False) as z:
        mean = np.asarray(z["mean"], dtype=np.float32)
        std = np.asarray(z["std"], dtype=np.float32)
    with np.load(TEXT, allow_pickle=False) as z:
        texts = np.asarray(z["texts"]).astype(str).tolist()
        embeddings = np.asarray(z["embeddings"], dtype=np.float32)
    text_map = dict(zip(texts, embeddings))

    specifications, feature, point, prop, cls, instruction, ground_truth = [], [], [], [], [], [], []
    for task_index, (task, (part, command)) in enumerate(TASKS.items()):
        with np.load(DATA / f"{task}_dev_v4.npz", allow_pickle=False) as z:
            sources = len(z["source_shape"])
            rows = np.sort(np.random.default_rng(SEED + task_index).choice(
                sources, min(ROWS_PER_TASK, sources), replace=False
            ))
            xyz = np.asarray(z["source_action_xyz_base_grasptarget"])
            rot = np.asarray(z["source_action_rotmat_base_grasptarget"])
            finger = np.asarray(z["source_action_finger"])
            proprio = np.asarray(z["sample_initial_robot_proprioception"])[::5]
            field = np.load(DATA / f"{task}_dev_pa3ff_field_f16.npy", mmap_mode="r")
            point_cache = np.load(DATA / f"{task}_dev_pointcloud_base_f32.npy", mmap_mode="r")
            for row in rows.tolist():
                rot6 = np.concatenate([rot[row, :, :, 0], rot[row, :, :, 1]], axis=-1)
                ground_truth.append(np.concatenate([
                    xyz[row], rot6, finger[row, :, None]
                ], axis=-1).astype(np.float32))
                feature.append(np.asarray(field[row], dtype=np.float16))
                point.append(np.asarray(point_cache[row], dtype=np.float32))
                prop.append(proprio[row])
                cls.append(text_map[part])
                instruction.append(text_map[command])
                specifications.append({"task": task, "dev_source_row": row})
    prop_t = torch.from_numpy(np.stack(prop).astype(np.float32)).to(device)
    cls_t = torch.from_numpy(np.stack(cls).astype(np.float32)).to(device)
    instruction_t = torch.from_numpy(np.stack(instruction).astype(np.float32)).to(device)
    conditions = []
    with torch.inference_mode(), torch.amp.autocast("cuda"):
        for start in range(0, len(feature), 4):
            field_t = torch.from_numpy(np.stack(feature[start:start + 4])).to(device)
            if positional:
                point_t = torch.from_numpy(np.stack(point[start:start + 4])).to(device)
                value = model.condition(
                    field_t, point_t, cls_t[start:start + 4],
                    instruction_t[start:start + 4], prop_t[start:start + 4],
                )
            else:
                value = model.condition(
                    field_t, cls_t[start:start + 4], instruction_t[start:start + 4],
                    prop_t[start:start + 4],
                )
            conditions.append(value.float())
    condition = torch.cat(conditions)
    gt = np.stack(ground_truth)
    fixed_noise = torch.randn(
        (len(gt), 16, 10), generator=torch.Generator().manual_seed(SEED),
        dtype=torch.float32,
    ).to(device)

    candidates = {}
    for start_t, noise, clip, steps in CONFIGS:
        x = fixed_noise * noise
        timesteps = torch.linspace(start_t, 0, steps, device=device).round().long()
        timesteps = torch.unique_consecutive(timesteps)
        with torch.inference_mode():
            for index, timestep_value in enumerate(timesteps.tolist()):
                timestep = torch.full((len(gt),), timestep_value, device=device, dtype=torch.long)
                pred_x0 = model.action_head(x, timestep, condition).clamp(-clip, clip)
                previous = int(timesteps[index + 1]) if index + 1 < len(timesteps) else -1
                x = diffusion.ddim_step(x, pred_x0, timestep_value, previous)
        pred = x.cpu().numpy() * std[None, None] + mean[None, None]
        values = defaultdict(list)
        task_values = {task: defaultdict(list) for task in TASKS}
        for row, spec in enumerate(specifications):
            pos = float(np.linalg.norm(pred[row, 0, :3] - gt[row, 0, :3]))
            pred_r = rotation_6d_to_matrix(pred[row, 0, 3:9])
            gt_r = rotation_6d_to_matrix(gt[row, 0, 3:9])
            rot = float(np.linalg.norm(Rotation.from_matrix(pred_r @ gt_r.T).as_rotvec()))
            traj = float(np.sqrt(np.mean((pred[row, :, :3] - gt[row, :, :3]) ** 2)))
            gt_disp = gt[row, -1, :3] - gt[row, 0, :3]
            pr_disp = pred[row, -1, :3] - pred[row, 0, :3]
            cosine = float(np.dot(gt_disp, pr_disp) / max(
                np.linalg.norm(gt_disp) * np.linalg.norm(pr_disp), 1e-12
            ))
            measurements = {
                "first_position_error_m": pos,
                "first_rotation_error_rad": rot,
                "first_se3_error": pos + 0.10 * rot / np.pi,
                "trajectory_position_rmse_m": traj,
                "displacement_cosine": cosine,
            }
            for name, value in measurements.items():
                values[name].append(value)
                task_values[spec["task"]][name].append(value)
        key = f"start={start_t}_noise={noise:g}_clip={clip:g}_steps={steps}"
        candidates[key] = {
            "start_timestep": start_t, "noise_scale": noise,
            "clip_pred_x0": clip, "ddim_steps": steps,
            "overall": {name: stats(vals) for name, vals in values.items()},
            "per_task": {task: {name: stats(vals) for name, vals in task_values[task].items()}
                         for task in TASKS},
        }
        print(key, candidates[key]["overall"]["first_se3_error"]["mean"],
              candidates[key]["overall"]["first_position_error_m"]["mean"], flush=True)
    selected_key = min(candidates, key=lambda key: (
        candidates[key]["overall"]["first_se3_error"]["mean"],
        candidates[key]["overall"]["trajectory_position_rmse_m"]["mean"],
        -candidates[key]["overall"]["displacement_cosine"]["mean"],
    ))
    report = {
        "status": "PASS", "scope": "object-level DEV trajectories only",
        "formal_catalog_or_results_used": False, "success_metric_used": False,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
        "positional_point_tokens": positional,
        "rows_per_task_requested": ROWS_PER_TASK, "sample_count": len(gt),
        "selection_metric": "task-balanced mean first SE3 error (m + 0.10*rad/pi)",
        "subset": specifications, "candidates": candidates,
        "selected_key": selected_key, "selected": candidates[selected_key],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("SELECTED", selected_key, json.dumps(report["selected"]["overall"]), flush=True)


if __name__ == "__main__":
    main()
