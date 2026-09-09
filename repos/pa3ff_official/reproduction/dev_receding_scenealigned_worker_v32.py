#!/usr/bin/env python3
"""DEV-only receding PA3FF worker with the repaired scene observation contract."""
from __future__ import annotations

import dev_receding_horizon_worker_v28 as receding
import scene_state_fix_v30
from scene_pointcloud_adapter_v31 import capture_scene_observation


_LATEST_OBJECT_CLOUD = None
_ORIGINAL_BASE_RANK = receding._base_rank
CREATE_SCENE = scene_state_fix_v30.create_scene


def _capture_scene_and_rank_cloud(**kwargs):
    global _LATEST_OBJECT_CLOUD
    observation = capture_scene_observation(**kwargs)
    _LATEST_OBJECT_CLOUD = observation["rank_point_cloud"]
    return observation


def _rank_on_object_only(_scene_cloud, camera_pose, poses, task, diagnostics):
    if _LATEST_OBJECT_CLOUD is None:
        raise RuntimeError("object-only rank cloud was not captured")
    return _ORIGINAL_BASE_RANK(
        _LATEST_OBJECT_CLOUD, camera_pose, poses, task, diagnostics
    )


def main() -> None:
    # Both names are module aliases of formal_worker and are patched before the
    # receding worker snapshots its scene constructor.
    receding.legacy.create_scene = CREATE_SCENE
    receding.legacy.capture_articubot_observation = _capture_scene_and_rank_cloud
    receding._base_rank = _rank_on_object_only
    receding.main()


if __name__ == "__main__":
    main()
