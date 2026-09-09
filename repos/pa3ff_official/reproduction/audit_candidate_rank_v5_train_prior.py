#!/usr/bin/env python3
"""Verify and freeze candidate workspace priors from TRAIN actions only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from candidate_rank_padp_v4_trainonly import (
    FIRST_POSITION_MEDIAN_BASE, FIRST_POSITION_SCALE_BASE,
)


HOME = Path("/home/feng")
DATA = HOME / "robot_baselines/results/pa3ff/padp_data_v4_timeindexed_baseframe_fourtask"
OUT = HOME / "robot_baselines/results/pa3ff/reproduction_v5/TRAIN_CANDIDATE_PRIOR_AUDIT.json"
TASKS = ("door_open", "door_close", "drawer_open", "drawer_close")


def main() -> None:
    rows = {}
    for task in TASKS:
        with np.load(DATA / f"{task}_train_v4.npz", allow_pickle=False) as z:
            xyz = np.asarray(z["source_action_xyz_base_grasptarget"], dtype=np.float64)[:, 0]
        median = np.median(xyz, axis=0)
        scale = np.maximum(1.4826 * np.median(np.abs(xyz - median), axis=0), 0.04)
        if not np.allclose(median, FIRST_POSITION_MEDIAN_BASE[task], atol=1e-8):
            raise RuntimeError(f"{task} median constant mismatch")
        if not np.allclose(scale, FIRST_POSITION_SCALE_BASE[task], atol=1e-8):
            raise RuntimeError(f"{task} scale constant mismatch")
        rows[task] = {"train_source_actions": len(xyz), "median_base_m": median.tolist(),
                      "robust_scale_base_m": scale.tolist()}
    report = {
        "status": "PASS", "scope": "V4 object-level TRAIN actions only",
        "formal_or_dev_actions_used": False,
        "definition": "first panda_grasptarget position median and max(1.4826*MAD, 0.04m)",
        "tasks": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
