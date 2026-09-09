#!/usr/bin/env python3
"""Save one frozen-camera target-link cloud for geometry diagnostics."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

for path in (
    Path("/home/feng/robot_baselines/repos/articubot/reproduction/perception_reach"),
    Path("/home/feng/robot_baselines/repos/articubot/reproduction/unified_open"),
):
    sys.path.insert(0, str(path))

from frozen_open_episode import DRAWER_POSE_CATALOG, POSE_CATALOG, _create_scene, load_catalog
from articubot_perception_reach_adapter import capture_link_conditioned_observations


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=("door_open", "drawer_open"), required=True)
    p.add_argument("--case-index", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    catalog = POSE_CATALOG if a.task == "door_open" else DRAWER_POSE_CATALOG
    case = load_catalog(catalog, task_mode=a.task)[a.case_index]
    resources = _create_scene(case, 0.15 if case.get("initial_object_qpos") is None else None)
    resources["panda"].wait(100)
    capture = capture_link_conditioned_observations(
        scene=resources["scene"], camera=resources["camera"], obj=resources["obj"],
        target_link=resources["target_link"], panda=resources["panda"], num_points=4500,
        rng=np.random.default_rng(123),
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.output,
        target_points_world=capture["target_points_world"],
        full_points_world=capture["full_points_world"],
        rest_points_world=capture["rest_points_world"],
        camera_position_world=capture["camera_position_world"],
        current_gripper_world=np.asarray([
            resources["panda"].hand_link.get_pose().p,
            resources["panda"].right_finger_link.get_pose().p,
            resources["panda"].left_finger_link.get_pose().p,
            resources["panda"].grasp_link.get_pose().p,
        ]),
    )
    print(json.dumps({"case": case, "output": str(a.output),
                      "target_points": len(capture["target_points_world"])}, indent=2))


if __name__ == "__main__":
    main()

