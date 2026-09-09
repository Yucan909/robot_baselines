"""Frozen observation/TRAIN-only ranking of unchanged PA3FF action samples."""
from __future__ import annotations

import numpy as np


SURFACE_TARGET_M = 0.025
DISPLACEMENT_WEIGHT = 0.25
TRAIN_DISPLACEMENT_MEDIAN_M = {
    "door_open": 0.2950354516506195,
    "drawer_open": 0.16538287699222565,
}
TRAIN_APPROACH_NORMAL_MEDIAN = {"door_open": -0.1562314088457686}
TRAIN_PATH_RATIO_MEDIAN = {"drawer_open": 1.0000090333923919}
DOOR_NORMAL_WEIGHT_M = 0.05
DRAWER_PATH_WEIGHT_M = 0.10


def rank_candidates(point_cloud: np.ndarray, camera_pose: np.ndarray,
                    candidate_poses: np.ndarray, task: str) -> tuple[list[int], dict]:
    """Rank only from current observation, predicted chunks, and frozen TRAIN priors."""
    pc = np.asarray(point_cloud, dtype=np.float64).reshape(1024, 3)
    poses = np.asarray(candidate_poses, dtype=np.float64)
    if poses.shape != (32, 16, 4, 4) or not np.isfinite(poses).all():
        raise RuntimeError(f"bad candidate pose batch {poses.shape}")
    if task not in TRAIN_DISPLACEMENT_MEDIAN_M:
        raise ValueError(f"V4 ranking is Open-only, got {task!r}")

    xyz = poses[:, :, :3, 3]
    distances = np.linalg.norm(pc[None, :, :] - xyz[:, :1, :], axis=2)
    nearest = np.argmin(distances, axis=1)
    surface = distances[np.arange(len(poses)), nearest]
    displacement = np.linalg.norm(xyz[:, -1] - xyz[:, 0], axis=1)
    score = (
        np.abs(surface - SURFACE_TARGET_M)
        + DISPLACEMENT_WEIGHT
        * np.abs(displacement - TRAIN_DISPLACEMENT_MEDIAN_M[task])
    )
    component = "none"
    component_value = np.zeros(len(poses), dtype=np.float64)
    component_error = np.zeros(len(poses), dtype=np.float64)
    component_weight = 0.0

    if task == "door_open":
        import open3d as o3d

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pc)
        cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30)
        )
        cloud.orient_normals_towards_camera_location(
            np.asarray(camera_pose, dtype=np.float64).reshape(4, 4)[:3, 3]
        )
        normals = np.asarray(cloud.normals, dtype=np.float64)
        component_value = np.einsum(
            "ni,ni->n", poses[:, 0, :3, 2], normals[nearest]
        )
        component_error = np.abs(
            component_value - TRAIN_APPROACH_NORMAL_MEDIAN[task]
        )
        component_weight = DOOR_NORMAL_WEIGHT_M
        component = "approach_z_nearest_surface_normal_cosine"
    else:
        path_length = np.linalg.norm(np.diff(xyz, axis=1), axis=2).sum(axis=1)
        component_value = path_length / np.maximum(displacement, 1e-9)
        component_error = np.abs(component_value - TRAIN_PATH_RATIO_MEDIAN[task])
        component_weight = DRAWER_PATH_WEIGHT_M
        component = "predicted_path_length_to_chord_ratio"

    score = score + component_weight * component_error
    order = np.argsort(score, kind="stable").astype(np.int64).tolist()
    diagnostic = {
        "version": "open_devselected_v4",
        "formal_success_used": False,
        "surface_target_m": SURFACE_TARGET_M,
        "displacement_weight": DISPLACEMENT_WEIGHT,
        "train_displacement_median_m": TRAIN_DISPLACEMENT_MEDIAN_M[task],
        "task_specific_component": component,
        "task_specific_component_weight_m": component_weight,
        "task_specific_train_median": (
            TRAIN_APPROACH_NORMAL_MEDIAN[task]
            if task == "door_open" else TRAIN_PATH_RATIO_MEDIAN[task]
        ),
        "selected_candidate_index": int(order[0]),
        "selected_score": float(score[order[0]]),
        "selected_surface_distance_m": float(surface[order[0]]),
        "selected_displacement_m": float(displacement[order[0]]),
        "selected_task_specific_component": float(component_value[order[0]]),
        "scores": score.tolist(),
    }
    return order, diagnostic
