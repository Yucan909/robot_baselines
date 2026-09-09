"""Canonical keys and validation for PA3FF V3 prediction artifacts."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def input_digest(task: str, seed: int, point_cloud: np.ndarray,
                 camera_pose: np.ndarray, robot_qpos: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(task).encode("utf-8") + b"\0")
    h.update(np.asarray([int(seed)], dtype="<i8").tobytes())
    for array in (point_cloud, camera_pose, robot_qpos):
        canonical = np.round(np.asarray(array, dtype=np.float64), 6).astype("<f8", copy=False)
        h.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
        h.update(canonical.tobytes(order="C"))
    return h.hexdigest()


def relative_case_path(task: str, target_index: int, shape_id: str,
                       target_link: str, trial_index: int, seed: int) -> Path:
    return (
        Path(task)
        / f"{int(target_index):03d}_{shape_id}_{target_link}"
        / f"trial_{int(trial_index):02d}_seed_{int(seed)}"
        / "prediction.npz"
    )
