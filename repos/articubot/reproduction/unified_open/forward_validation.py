"""Checkpoint/config/CUDA and real forward-pass validation for both official models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from articubot_policy_adapter import ArticuBotPolicyAdapter


OUTPUT = Path("/home/feng/robot_baselines/results/articubot/unified_open/forward_validation.json")


def run_forward_validation(device: str = "cuda:0"):
    policy = ArticuBotPolicyAdapter(device)
    rng = np.random.default_rng(20260905)
    t, n = policy.n_obs_steps, policy.num_points
    point_cloud = rng.normal(size=(t, n, 3)).astype(np.float32) * 0.18
    point_cloud[..., 0] += 0.6
    gripper_pcd = rng.normal(size=(t, 4, 3)).astype(np.float32) * 0.02
    gripper_pcd[..., 0] += 0.45
    agent_pos = np.zeros((t, 10), dtype=np.float32)
    agent_pos[:, :3] = [0.45, 0.0, 0.45]
    agent_pos[:, 3:9] = [1, 0, 0, 0, 1, 0]
    agent_pos[:, 9] = 0.04
    diff = np.zeros((t, 4, 3), dtype=np.float32)
    history = {
        "point_cloud": point_cloud,
        "agent_pos": agent_pos,
        "gripper_pcd": gripper_pcd,
        "displacement_gripper_to_object": diff,
    }
    goal, action = policy.infer(history)
    report = {
        "status": "PASS",
        "torch_load": "PASS",
        "config_load": "PASS",
        "cuda": "PASS" if str(device).startswith("cuda") else "CPU",
        "high_level_forward": "PASS",
        "low_level_forward": "PASS",
        "high_level_output_shape": list(goal.shape),
        "low_level_output_shape": list(action.shape),
        "high_level_all_finite": bool(np.all(np.isfinite(goal))),
        "low_level_all_finite": bool(np.all(np.isfinite(action))),
        "low_level_nonconstant": bool(float(np.std(action)) > 0.0),
        "policy_config": policy.audit(),
    }
    if not all((report["high_level_all_finite"], report["low_level_all_finite"], report["low_level_nonconstant"])):
        raise RuntimeError(f"forward validation failed: {report}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report
