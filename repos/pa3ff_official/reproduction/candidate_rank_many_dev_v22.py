"""Observation/TRAIN-only ranking for an arbitrary PADP candidate count."""
from __future__ import annotations

import numpy as np

from candidate_rank_devselected_v4 import (
    DISPLACEMENT_WEIGHT, DOOR_NORMAL_WEIGHT_M, DRAWER_PATH_WEIGHT_M,
    SURFACE_TARGET_M, TRAIN_APPROACH_NORMAL_MEDIAN,
    TRAIN_DISPLACEMENT_MEDIAN_M, TRAIN_PATH_RATIO_MEDIAN,
)
from candidate_rank_train_geometry_v11 import TRAIN_MEDIAN, TRAIN_SCALE, _features


TRAIN_DISPLACEMENT_P75_M = {
    "door_open": 0.3710261285305023,
    "drawer_open": 0.21698470786213875,
}


def rank_candidates(
    point_cloud: np.ndarray,
    camera_pose: np.ndarray,
    candidate_poses: np.ndarray,
    task: str,
    *,
    mode: str,
    cached_diagnostics: list[dict],
) -> tuple[list[int], dict]:
    import open3d as o3d

    pc = np.asarray(point_cloud, dtype=np.float64).reshape(1024, 3)
    poses = np.asarray(candidate_poses, dtype=np.float64)
    if poses.ndim != 4 or poses.shape[1:] != (16, 4, 4):
        raise RuntimeError(f"bad candidate poses {poses.shape}")
    count = len(poses)
    diagnostics = list(cached_diagnostics)[:count]
    if len(diagnostics) != count:
        raise RuntimeError("candidate diagnostic length mismatch")
    xyz = poses[:, :, :3, 3]
    distances = np.linalg.norm(pc[None, :, :] - xyz[:, :1, :], axis=2)
    nearest = np.argmin(distances, axis=1)
    surface = distances[np.arange(count), nearest]
    displacement = np.linalg.norm(xyz[:, -1] - xyz[:, 0], axis=1)
    base = np.abs(surface - SURFACE_TARGET_M) + DISPLACEMENT_WEIGHT * np.abs(
        displacement - TRAIN_DISPLACEMENT_MEDIAN_M[task]
    )

    vectors = np.stack([_features(pc, trajectory) for trajectory in poses])
    zscore = np.clip(
        (vectors - np.asarray(TRAIN_MEDIAN[task])) / np.asarray(TRAIN_SCALE[task]),
        -8.0, 8.0,
    )
    grasp_geometry = np.sqrt(np.mean(zscore[:, :7] ** 2, axis=1))
    all_geometry = np.sqrt(np.mean(zscore ** 2, axis=1))

    if mode == "v4":
        score = base.copy()
        if task == "door_open":
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(pc)
            cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=.08, max_nn=30)
            )
            cloud.orient_normals_towards_camera_location(
                np.asarray(camera_pose, dtype=np.float64).reshape(4, 4)[:3, 3]
            )
            normal = np.asarray(cloud.normals)
            cosine = np.einsum("ni,ni->n", poses[:, 0, :3, 2], normal[nearest])
            score += DOOR_NORMAL_WEIGHT_M * np.abs(
                cosine - TRAIN_APPROACH_NORMAL_MEDIAN[task]
            )
        else:
            path = np.linalg.norm(np.diff(xyz, axis=1), axis=2).sum(axis=1)
            ratio = path / np.maximum(displacement, 1e-9)
            score += DRAWER_PATH_WEIGHT_M * np.abs(
                ratio - TRAIN_PATH_RATIO_MEDIAN[task]
            )
        rule = "generalized frozen V4 observation/TRAIN score"
    elif mode == "train_geometry":
        score = grasp_geometry if task == "door_open" else all_geometry
        rule = "TRAIN robust gripper-local demonstration geometry"
    elif mode == "motion_train":
        motion_geometry = np.sqrt(np.mean(zscore[:, 7:] ** 2, axis=1))
        score = motion_geometry
        rule = "TRAIN robust signed local operation-motion geometry"
    elif mode == "all_train":
        score = all_geometry
        rule = "TRAIN robust joint grasp-and-signed-motion geometry"
    elif mode == "semantic_train":
        if task == "door_open":
            # Door DEV reconstruction favored geometry over semantic subsets.
            score = grasp_geometry
            rule = "TRAIN robust door grasp geometry"
        else:
            semantic_surface = np.asarray([
                float(row["semantic_surface_distance_m"]["256"])
                for row in diagnostics
            ])
            score = (
                np.abs(semantic_surface - SURFACE_TARGET_M)
                + DISPLACEMENT_WEIGHT * np.abs(
                    displacement - TRAIN_DISPLACEMENT_MEDIAN_M[task]
                )
                + 0.04 * all_geometry
            )
            rule = "PA3FF drawer semantic top256 surface + TRAIN displacement/geometry"
    elif mode == "long_motion":
        motion_error = np.abs(displacement - TRAIN_DISPLACEMENT_P75_M[task])
        if task == "door_open":
            score = grasp_geometry + 3.0 * motion_error
            rule = "TRAIN robust grasp geometry + TRAIN P75 operation displacement"
        else:
            semantic_surface = np.asarray([
                float(row["semantic_surface_distance_m"]["256"])
                for row in diagnostics
            ])
            score = (
                np.abs(semantic_surface - SURFACE_TARGET_M)
                + 0.25 * motion_error + 0.04 * all_geometry
            )
            rule = "PA3FF drawer semantic surface + TRAIN P75 operation displacement/geometry"
    elif mode == "long_motion_all":
        motion_error = np.abs(displacement - TRAIN_DISPLACEMENT_P75_M[task])
        score = all_geometry + 3.0 * motion_error
        rule = "TRAIN robust full signed geometry + TRAIN P75 operation displacement"
    else:
        raise ValueError(mode)

    order = np.argsort(score, kind="stable").astype(int).tolist()
    selected = order[0]
    return order, {
        "version": "many_candidate_dev_v22", "mode": mode, "rule": rule,
        "candidate_count": count, "formal_success_used": False,
        "selected_candidate_index": selected,
        "selected_score": float(score[selected]),
        "selected_surface_distance_m": float(surface[selected]),
        "selected_displacement_m": float(displacement[selected]),
        "train_displacement_p75_m": TRAIN_DISPLACEMENT_P75_M[task],
        "selected_grasp_geometry_score": float(grasp_geometry[selected]),
        "selected_all_geometry_score": float(all_geometry[selected]),
        "scores": score.tolist(),
    }
