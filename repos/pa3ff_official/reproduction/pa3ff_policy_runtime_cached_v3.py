"""CPU-side loader for inputs precomputed by the frozen V3 PA3FF runtime."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from policy_cache_v3_common import input_digest


EXPECTED_CHECKPOINT_SHA = "2daab7af74d92f09dd0dea1c6f01d86a1663b0db01efc1b45ce8085055689b51"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class PA3FFPADPRuntimeCachedV3:
    def __init__(self, checkpoint: Path):
        self.checkpoint = Path(checkpoint).resolve()
        self.training_step = 50000
        if sha256(self.checkpoint) != EXPECTED_CHECKPOINT_SHA:
            raise RuntimeError("cached runtime checkpoint SHA mismatch")
        self.cache_root = Path(os.environ["PA3FF_PREDICTION_CACHE_ROOT"]).resolve()

    def predict(self, *, point_cloud_world, camera_pose_world, robot_qpos, task, seed, **kwargs):
        if kwargs:
            raise ValueError(f"cached frozen prediction got unexpected options {sorted(kwargs)}")
        import formal_worker as legacy

        episode_dir = Path(legacy.PA3FF_CURRENT_EPISODE_DIR)
        rel = Path(*episode_dir.parts[-3:]) / "prediction.npz"
        path = self.cache_root / rel
        if not path.is_file():
            raise RuntimeError(f"missing frozen prediction cache {path}")
        digest = input_digest(task, seed, point_cloud_world, camera_pose_world, robot_qpos)
        with np.load(path, allow_pickle=False) as z:
            if str(np.asarray(z["schema"]).item()) != "pa3ff_v3_frozen_prediction_cache_v1":
                raise RuntimeError(f"cache schema mismatch {path}")
            if str(np.asarray(z["checkpoint_sha256"]).item()) != EXPECTED_CHECKPOINT_SHA:
                raise RuntimeError(f"cache checkpoint mismatch {path}")
            if str(np.asarray(z["input_digest"]).item()) != digest:
                raise RuntimeError(f"cache input digest mismatch {path}")
            diagnostics = json.loads(str(np.asarray(z["diagnostics_json"]).item()))
            diagnostics.update({
                "poses_world_grasptarget": np.asarray(z["poses_world_grasptarget"], dtype=np.float64),
                "gripper": np.asarray(z["gripper"], dtype=np.float64),
                "candidate_poses_world_grasptarget": np.asarray(z["candidate_poses_world_grasptarget"], dtype=np.float64),
                "candidate_gripper": np.asarray(z["candidate_gripper"], dtype=np.float64),
                "candidate_rank_order": np.asarray(z["candidate_rank_order"], dtype=np.int64).tolist(),
                "prediction_cache": str(path), "prediction_cache_input_digest": digest,
            })
            return diagnostics
