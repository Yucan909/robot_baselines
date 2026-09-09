#!/usr/bin/env python3
"""Bottom-faithful receding worker for positional PADP V5."""
from __future__ import annotations

import numpy as np

import dev_receding_bottomfaithful_worker_v34 as bottom
from candidate_rank_padp_v4_trainonly import rank_candidates as rank_candidates_v5
from pa3ff_policy_runtime_v5_positional import PA3FFPADPRuntimeV5Positional


receding = bottom.receding
_ORIGINAL_SCENE = bottom._REPAIRED_CREATE_SCENE
_ORIGINAL_RUN_EPISODE = receding.run_episode


def _scene_with_base_binding(case: dict) -> dict:
    PA3FFPADPRuntimeV5Positional.set_active_base_pose(
        np.asarray(case["base_pose"], dtype=np.float64)
    )
    return _ORIGINAL_SCENE(case)


def _run_v5(*args, **kwargs):
    result = _ORIGINAL_RUN_EPISODE(*args, **kwargs)
    result["method"] = "PA3FF_REPRODUCTION_PADP_V5_POSITIONAL_BALANCED"
    result["padp_reproduction"] = {
        "official_status": "paper-equivalent reconstructed PADP; private PADP was not released",
        "time_indexed_training": True,
        "action_coordinate_frame": "panda_base",
        "point_token": "frozen_official_PA3FF_feature_plus_base_xyz_position_encoding",
        "task_sampling": "uniform_four_task",
        "execution": "supplied_bottom_grasp_plus_four_action_receding_horizon",
        "formal_or_test_action_labels_used": False,
    }
    return result


def main() -> None:
    bottom._REPAIRED_CREATE_SCENE = _scene_with_base_binding
    receding.PA3FFPADPRuntimeManyCandidatesV17 = PA3FFPADPRuntimeV5Positional
    receding.rank_candidates = rank_candidates_v5
    receding.TRAIN_STEP_P75_M.update({
        "door_close": 0.0256652352, "drawer_close": 0.0142683468,
    })
    receding.TRAIN_ROTATION_STEP_P75_RAD.update({
        "door_close": 0.0479947553, "drawer_close": 0.0050262661,
    })
    receding.run_episode = _run_v5
    bottom.main()


if __name__ == "__main__":
    main()
