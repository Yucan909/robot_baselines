#!/usr/bin/env python3
"""Select TRAIN-prior candidate-rank weights on object-level Open DEV.

The online score contains only observation geometry plus statistics frozen from
TRAIN.  DEV demonstrations are used here solely to select two scalar weights;
formal targets/results are never loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from candidate_rank_padp_v4_trainonly import (
    DISPLACEMENT_P75_M, FIRST_POSITION_MEDIAN_BASE,
    FIRST_POSITION_SCALE_BASE, SURFACE_MEDIAN_M,
)
from candidate_rank_train_geometry_v11 import TRAIN_MEDIAN, TRAIN_SCALE, _features
from pa3ff_policy_runtime_v4_baseframe import base_to_world
from pa3ff_policy_runtime_v5_positional import PA3FFPADPRuntimeV5Positional


HOME = Path("/home/feng")
DATA = HOME / "robot_baselines/results/pa3ff/padp_data_v4_timeindexed_baseframe_fourtask"
CATALOG = HOME / "robot_baselines/results/pa3ff/reproduction_v3_dev_rollout_selection/open_dev_catalog_v19.jsonl"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rotation_error(candidate: np.ndarray, target: np.ndarray) -> np.ndarray:
    # The Panda parallel jaws are physically equivalent after a 180 degree
    # rotation around the local approach (Z) axis.  Use the quotient-space
    # error for DEV configuration selection; the policy and executed pose are
    # left untouched.
    symmetry = Rotation.from_rotvec([0.0, 0.0, np.pi]).as_matrix()
    direct = np.linalg.norm(Rotation.from_matrix(candidate @ target.T).as_rotvec(), axis=-1)
    flipped_target = target @ symmetry
    flipped = np.linalg.norm(
        Rotation.from_matrix(candidate @ flipped_target.T).as_rotvec(), axis=-1
    )
    return np.minimum(direct, flipped)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-task", type=int, default=32)
    args = parser.parse_args()

    runtime = PA3FFPADPRuntimeV5Positional(args.checkpoint)
    runtime.N_CANDIDATES = 128
    runtime.START_TIMESTEP = 90
    runtime.NOISE_SCALE = 0.1
    runtime.CLIP_PRED_X0 = 1.5
    runtime.DDIM_STEPS = 10

    catalog = [json.loads(line) for line in CATALOG.read_text().splitlines() if line.strip()]
    records = []
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
            pred = runtime.predict(
                point_cloud_world=pc_world.astype(np.float32),
                camera_pose_world=np.asarray(case["camera_pose_world"]),
                robot_qpos=np.asarray(case["robot_initial_qpos"]),
                task=task, seed=int(case["seed"]), base_pose=case["base_pose"],
            )
            poses_world = np.asarray(pred["candidate_poses_world_grasptarget"], dtype=np.float64)
            poses_base = np.einsum("ij,nhjk->nhik", np.linalg.inv(world_from_base), poses_world)
            xyz_world = poses_world[:, :, :3, 3]
            full_surface = np.min(
                np.linalg.norm(pc_world[None, :, :] - xyz_world[:, :1, :], axis=2), axis=1
            )
            semantic_surface = np.asarray([
                float(diag["semantic_surface_distance_m"]["256"])
                for diag in pred["candidate_diagnostics"]
            ])
            displacement = np.linalg.norm(xyz_world[:, -1] - xyz_world[:, 0], axis=1)
            first_base = poses_base[:, 0, :3, 3]
            position_z = ((first_base - FIRST_POSITION_MEDIAN_BASE[task])
                          / FIRST_POSITION_SCALE_BASE[task])
            position_support = np.sqrt(np.mean(np.clip(position_z, -8, 8) ** 2, axis=1))
            vectors = np.stack([_features(pc_world, trajectory) for trajectory in poses_world])
            zscore = np.clip(
                (vectors - np.asarray(TRAIN_MEDIAN[task])) / np.asarray(TRAIN_SCALE[task]),
                -8, 8,
            )
            geometry = np.sqrt(np.mean(zscore ** 2, axis=1))
            grasp_geometry = np.sqrt(np.mean(zscore[:, :7] ** 2, axis=1))
            base_score = (
                .75 * np.abs(semantic_surface - SURFACE_MEDIAN_M[task])
                + .25 * np.abs(full_surface - SURFACE_MEDIAN_M[task])
                + .25 * np.abs(displacement - DISPLACEMENT_P75_M[task])
            )
            pos_error = np.linalg.norm(first_base - gt_xyz[row, 0], axis=1)
            rot_error = rotation_error(poses_base[:, 0, :3, :3], gt_rot[row, 0])
            records.append({
                "task": task, "base": base_score, "position": position_support,
                "geometry": geometry, "grasp_geometry": grasp_geometry,
                "pos_error": pos_error, "rot_error": rot_error,
            })
            print(f"PREDICT {task} {len([r for r in records if r['task'] == task])}/{len(cases)}", flush=True)

    reports = []
    for position_weight in (0., .02, .1, .4):
        for geometry_weight in (0., .02, .1):
            for grasp_geometry_weight in (0., .02, .05, .1, .2, .4, .8):
                task_metrics = {}
                for task in ("door_open", "drawer_open"):
                    chosen_pos, chosen_rot, joint = [], [], []
                    for record in records:
                        if record["task"] != task:
                            continue
                        score = (record["base"] + position_weight * record["position"]
                                 + geometry_weight * record["geometry"]
                                 + grasp_geometry_weight * record["grasp_geometry"])
                        selected = int(np.argmin(score))
                        pos = float(record["pos_error"][selected])
                        rot = float(record["rot_error"][selected])
                        chosen_pos.append(pos)
                        chosen_rot.append(rot)
                        joint.append(pos / .03 + rot / np.deg2rad(30.))
                    task_metrics[task] = {
                        "mean_position_error_m": float(np.mean(chosen_pos)),
                        "median_position_error_m": float(np.median(chosen_pos)),
                        "mean_rotation_error_deg": float(np.degrees(np.mean(chosen_rot))),
                        "median_rotation_error_deg": float(np.degrees(np.median(chosen_rot))),
                        "mean_joint_error": float(np.mean(joint)),
                    }
                macro_joint = float(np.mean(
                    [m["mean_joint_error"] for m in task_metrics.values()]
                ))
                reports.append({
                    "position_weight": position_weight,
                    "geometry_weight": geometry_weight,
                    "grasp_geometry_weight": grasp_geometry_weight,
                    "macro_mean_joint_error": macro_joint, "tasks": task_metrics,
                })
    reports.sort(key=lambda row: row["macro_mean_joint_error"])
    output = {
        "schema": "PA3FF_PADP_V5_TRAIN_PRIOR_RANK_DEV_SELECTION_V1",
        "scope": "object-level Open DEV demonstrations only",
        "formal_catalog_or_results_used": False,
        "online_rank_inputs": "current observation plus TRAIN-frozen statistics only",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "sampler": {"start_timestep": 90, "noise_scale": .1,
                    "clip_pred_x0": 1.5, "ddim_steps": 10, "candidates": 128},
        "cases_per_task": args.per_task,
        "selection_metric": (
            "macro task mean(first position/0.03 + parallel-jaw-symmetry rotation/30deg)"
        ),
        "selected": reports[0], "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"selected": reports[0], "top5": reports[:5]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
