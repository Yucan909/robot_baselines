#!/usr/bin/env python3
"""Audit PADP V4 candidate recall against Open DEV actions, never formal data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from candidate_rank_padp_v4_trainonly import rank_candidates
from pa3ff_policy_runtime_v4_baseframe import (
    PA3FFPADPRuntimeV4BaseFrame, base_to_world,
)


HOME = Path("/home/feng")
DATA = HOME / "robot_baselines/results/pa3ff/padp_data_v4_timeindexed_baseframe_fourtask"
CATALOG = HOME / "robot_baselines/results/pa3ff/reproduction_v3_dev_rollout_selection/open_dev_catalog_v19.jsonl"
CHECKPOINT = HOME / "robot_baselines/results/pa3ff/reproduction_v4/training_30000/checkpoints/step030000.pt"
OUT = HOME / "robot_baselines/results/pa3ff/reproduction_v4/OPEN_DEV_CANDIDATE_RECALL_AUDIT.json"


def rotation_error(candidate: np.ndarray, target: np.ndarray) -> np.ndarray:
    relative = candidate @ target.T
    return np.linalg.norm(Rotation.from_matrix(relative).as_rotvec(), axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-task", type=int, default=8)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    import torch
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    positional = payload.get("stage") == "PA3FF_PADP_V5_TIMEINDEXED_BASEFRAME_POSITIONAL_BALANCED"
    del payload
    if positional:
        from pa3ff_policy_runtime_v5_positional import PA3FFPADPRuntimeV5Positional
        runtime = PA3FFPADPRuntimeV5Positional(args.checkpoint)
    else:
        runtime = PA3FFPADPRuntimeV4BaseFrame(args.checkpoint)
    runtime.N_CANDIDATES = 128
    runtime.CLIP_PRED_X0 = 3.0
    catalog = [json.loads(line) for line in CATALOG.read_text().splitlines() if line.strip()]
    reports = []
    for task in ("door_open", "drawer_open"):
        cases = [case for case in catalog if case["task"] == task][:args.per_task]
        with np.load(DATA / f"{task}_dev_v4.npz", allow_pickle=False) as z:
            gt_xyz = np.asarray(z["source_action_xyz_base_grasptarget"])
            gt_rot = np.asarray(z["source_action_rotmat_base_grasptarget"])
            points = np.load(DATA / f"{task}_dev_pointcloud_base_f32.npy", mmap_mode="r")
            for case in cases:
                row = int(case["dev_row"])
                world_from_base = base_to_world(case["base_pose"])
                pc_base = np.asarray(points[row], dtype=np.float64)
                pc_world = pc_base @ world_from_base[:3, :3].T + world_from_base[:3, 3]
                prediction = runtime.predict(
                    point_cloud_world=pc_world.astype(np.float32),
                    camera_pose_world=np.asarray(case["camera_pose_world"]),
                    robot_qpos=np.asarray(case["robot_initial_qpos"]),
                    task=task, seed=int(case["seed"]), base_pose=case["base_pose"],
                )
                poses_world = np.asarray(prediction["candidate_poses_world_grasptarget"])
                poses_base = np.einsum("ij,nhjk->nhik", np.linalg.inv(world_from_base), poses_world)
                order, rank = rank_candidates(
                    pc_world, np.asarray(case["camera_pose_world"]), poses_world,
                    task, mode="long_motion",
                    cached_diagnostics=prediction["candidate_diagnostics"],
                )
                pos = np.linalg.norm(poses_base[:, 0, :3, 3] - gt_xyz[row, 0], axis=1)
                rot = rotation_error(poses_base[:, 0, :3, :3], gt_rot[row, 0])
                joint = pos / 0.03 + rot / np.deg2rad(30.0)
                selected = int(order[0])
                oracle = int(np.argmin(joint))
                reports.append({
                    "task": task, "dev_row": row, "shape_id": case["shape_id"],
                    "target_link": case["target_link"], "seed": case["seed"],
                    "selected": selected, "oracle": oracle,
                    "selected_position_error_m": float(pos[selected]),
                    "selected_rotation_error_deg": float(np.degrees(rot[selected])),
                    "oracle_position_error_m": float(pos[oracle]),
                    "oracle_rotation_error_deg": float(np.degrees(rot[oracle])),
                    "candidate_position_error_min_m": float(pos.min()),
                    "candidate_position_error_median_m": float(np.median(pos)),
                    "candidate_rotation_error_min_deg": float(np.degrees(rot.min())),
                    "rank_selected_surface_m": rank["selected_surface_distance_m"],
                })
                print(json.dumps(reports[-1]), flush=True)
    summary = {"status": "PASS", "scope": "object-level Open DEV actions only",
               "checkpoint": str(args.checkpoint.resolve()),
               "positional_point_tokens": positional,
               "formal_or_test_actions_used": False, "reports": reports, "tasks": {}}
    for task in ("door_open", "drawer_open"):
        rows = [row for row in reports if row["task"] == task]
        summary["tasks"][task] = {
            key: float(np.median([row[key] for row in rows])) for key in (
                "selected_position_error_m", "selected_rotation_error_deg",
                "oracle_position_error_m", "oracle_rotation_error_deg",
                "candidate_position_error_min_m", "candidate_rotation_error_min_deg",
            )
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["tasks"], indent=2), flush=True)


if __name__ == "__main__":
    main()
