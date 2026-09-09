"""Observation-only candidate ranker with priors measured from V4 TRAIN."""
from __future__ import annotations

import os
import numpy as np

from candidate_rank_train_geometry_v11 import TRAIN_MEDIAN, TRAIN_SCALE, _features


SURFACE_MEDIAN_M = {
    "door_open": 0.0233218130,
    "door_close": 0.0254875794,
    "drawer_open": 0.0256380998,
    "drawer_close": 0.0427479707,
}
DISPLACEMENT_MEDIAN_M = {
    "door_open": 0.2975003719,
    "door_close": 0.2979569733,
    "drawer_open": 0.1639862061,
    "drawer_close": 0.1759911925,
}
DISPLACEMENT_P75_M = {
    "door_open": 0.3734320700,
    "door_close": 0.3753196001,
    "drawer_open": 0.2169338167,
    "drawer_close": 0.2134462371,
}

# Robust first-action position support measured once from V4 TRAIN actions.
# This resolves an otherwise underdetermined failure mode on repeated drawers:
# several semantic drawer surfaces are equally close, but only action poses in
# the demonstrated Panda workspace support are valid policy samples.  No DEV,
# formal target, articulation progress, or success statistic enters this prior.
FIRST_POSITION_MEDIAN_BASE = {
    "door_open": np.asarray([0.5227882862, -0.0004688138, 0.6220953465]),
    "door_close": np.asarray([0.4049424231, 0.0385737158, 0.5807345510]),
    "drawer_open": np.asarray([0.5941073895, 0.0026916667, 0.6782575846]),
    "drawer_close": np.asarray([0.4487987012, 0.0351353139, 0.5652216077]),
}
FIRST_POSITION_SCALE_BASE = {
    "door_open": np.asarray([0.0511098494, 0.1090504657, 0.2342590071]),
    "door_close": np.asarray([0.1003428559, 0.2772508655, 0.2227774664]),
    "drawer_open": np.asarray([0.0956077786, 0.1293693072, 0.1428592680]),
    "drawer_close": np.asarray([0.1125759600, 0.1640638888, 0.0894030224]),
}
POSITION_SUPPORT_WEIGHT = float(os.environ.get("PA3FF_DEV_POSITION_PRIOR_WEIGHT", "0.02"))
GEOMETRY_WEIGHT = float(os.environ.get("PA3FF_DEV_GEOMETRY_PRIOR_WEIGHT", "0.02"))
GRASP_GEOMETRY_WEIGHT = float(
    os.environ.get("PA3FF_DEV_GRASP_GEOMETRY_PRIOR_WEIGHT", "0.0")
)


def _task_weight(task: str, name: str, fallback: float) -> float:
    key = f"PA3FF_{task.upper()}_{name}"
    return float(os.environ.get(key, str(fallback)))


def rank_candidates(point_cloud, camera_pose, candidate_poses, task, *, mode,
                    cached_diagnostics):
    del camera_pose
    pc = np.asarray(point_cloud, dtype=np.float64).reshape(1024, 3)
    poses = np.asarray(candidate_poses, dtype=np.float64)
    if poses.ndim != 4 or poses.shape[1:] != (16, 4, 4):
        raise RuntimeError(f"bad candidate poses {poses.shape}")
    if task not in SURFACE_MEDIAN_M:
        raise ValueError(task)
    count = len(poses)
    diagnostics = list(cached_diagnostics)[:count]
    if len(diagnostics) != count:
        raise RuntimeError("candidate diagnostic length mismatch")
    xyz = poses[:, :, :3, 3]
    surface = np.min(
        np.linalg.norm(pc[None, :, :] - xyz[:, :1, :], axis=2), axis=1
    )
    semantic_surface = np.asarray([
        float(row["semantic_surface_distance_m"]["256"]) for row in diagnostics
    ])
    displacement = np.linalg.norm(xyz[:, -1] - xyz[:, 0], axis=1)
    first_position_base = np.asarray([
        row["first_position_base"] for row in diagnostics
    ], dtype=np.float64)
    position_z = (
        (first_position_base - FIRST_POSITION_MEDIAN_BASE[task])
        / FIRST_POSITION_SCALE_BASE[task]
    )
    position_support = np.sqrt(np.mean(np.clip(position_z, -8.0, 8.0) ** 2, axis=1))
    displacement_target = (
        DISPLACEMENT_P75_M[task] if mode in {"long_motion", "long_motion_all"}
        else DISPLACEMENT_MEDIAN_M[task]
    )
    position_weight = _task_weight(
        task, "POSITION_PRIOR_WEIGHT", POSITION_SUPPORT_WEIGHT
    )
    geometry_weight = _task_weight(
        task, "GEOMETRY_PRIOR_WEIGHT", GEOMETRY_WEIGHT
    )
    grasp_geometry_weight = _task_weight(
        task, "GRASP_GEOMETRY_PRIOR_WEIGHT", GRASP_GEOMETRY_WEIGHT
    )
    # The semantic subset is useful for repeated doors/drawers; the full cloud
    # term prevents a noisy semantic outlier from winning on its own.
    score = (
        0.75 * np.abs(semantic_surface - SURFACE_MEDIAN_M[task])
        + 0.25 * np.abs(surface - SURFACE_MEDIAN_M[task])
        + 0.25 * np.abs(displacement - displacement_target)
        + position_weight * position_support
    )
    geometry = np.zeros(count, dtype=np.float64)
    grasp_geometry = np.zeros(count, dtype=np.float64)
    if task in TRAIN_MEDIAN:
        vectors = np.stack([_features(pc, trajectory) for trajectory in poses])
        zscore = np.clip(
            (vectors - np.asarray(TRAIN_MEDIAN[task])) / np.asarray(TRAIN_SCALE[task]),
            -8.0, 8.0,
        )
        geometry = np.sqrt(np.mean(zscore ** 2, axis=1))
        grasp_geometry = np.sqrt(np.mean(zscore[:, :7] ** 2, axis=1))
        score += geometry_weight * geometry
        score += grasp_geometry_weight * grasp_geometry
    order = np.argsort(score, kind="stable").astype(int).tolist()
    selected = order[0]
    return order, {
        "version": "padp_v4_trainonly_v3", "mode": mode,
        "rule": (
            "PA3FF semantic/full surface + TRAIN displacement/base-workspace/"
            "full-geometry/first-grasp-geometry"
        ),
        "formal_success_used": False, "object_progress_used": False,
        "selected_candidate_index": selected,
        "selected_score": float(score[selected]),
        "selected_surface_distance_m": float(surface[selected]),
        "selected_semantic_surface_distance_m": float(semantic_surface[selected]),
        "selected_displacement_m": float(displacement[selected]),
        "train_surface_median_m": SURFACE_MEDIAN_M[task],
        "train_displacement_target_m": displacement_target,
        "selected_geometry_score": float(geometry[selected]),
        "selected_grasp_geometry_score": float(grasp_geometry[selected]),
        "selected_position_support_score": float(position_support[selected]),
        "position_support_weight": position_weight,
        "geometry_weight": geometry_weight,
        "grasp_geometry_weight": grasp_geometry_weight,
        "scores": score.tolist(),
    }
