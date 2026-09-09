#!/usr/bin/env python3
"""Deterministic, numerically robust PA3FF wrapper around ArticuBot OMPL.

Fixes two execution bugs without changing the collision model or goal:
* snap tiny SAPIEN round-off around the frozen 0.10/0.15/0.20 ratios;
* seed Python, NumPy, and OMPL, and bypass the upstream B-spline smoother
  whose returned path can fail the worker's own collision validation.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import traceback
from pathlib import Path

import numpy as np
from ompl import util as ou

sys.path.insert(
    0,
    "/home/feng/robot_baselines/common_env/where2act_four_task_noaff_v7_floating_progress_v2",
)
import motion_planner_collision_worker as implementation


RATIO_SNAP_TOLERANCE = 1e-6
FROZEN_OPEN_RATIOS = (0.10, 0.15, 0.20)


def snap_request_ratio(path: Path) -> dict:
    request = json.loads(path.read_text(encoding="utf-8"))
    raw = float(request["initial_ratio"])
    snapped = raw
    if request.get("initial_object_qpos") is None:
        for nominal in FROZEN_OPEN_RATIOS:
            if abs(raw - nominal) <= RATIO_SNAP_TOLERANCE:
                snapped = nominal
                break
    request["initial_ratio_raw_sapien"] = raw
    request["initial_ratio"] = snapped
    request["initial_ratio_numeric_snap"] = bool(snapped != raw)
    request["initial_ratio_snap_tolerance"] = RATIO_SNAP_TOLERANCE
    path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
    return request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--pose-catalog", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--planner", default="RRTConnect")
    parser.add_argument("--planning-time", type=float, default=5.0)
    parser.add_argument("--ik-attempts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    snap_request_ratio(Path(args.request).resolve())
    seed = int(args.seed) % (2**32)
    upstream = implementation.PbOMPL

    class DeterministicCollisionCheckedPbOMPL(upstream):
        def __init__(self, *ctor_args, **ctor_kwargs):
            # OMPL must be seeded before SimpleSetup/planner construction.
            random.seed(seed)
            np.random.seed(seed)
            ou.RNG.setSeed(seed)
            super().__init__(*ctor_args, **ctor_kwargs)
            # Upstream PbOMPL.__init__ overwrites Python/NumPy with time_ns.
            random.seed(seed)
            np.random.seed(seed)

        def smooth_path(self, path):
            # The upstream B-spline smoother sometimes creates states rejected
            # by the exact same state-validity checker. Keep the original
            # collision-free OMPL solution, which is still interpolated to the
            # configured 100 states and post-validated by the worker.
            return path

    implementation.PbOMPL = DeterministicCollisionCheckedPbOMPL
    try:
        implementation.run(args)
    except Exception:
        print("\n" + "=" * 100)
        print("COLLISION PLANNER ERROR")
        print("=" * 100)
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
