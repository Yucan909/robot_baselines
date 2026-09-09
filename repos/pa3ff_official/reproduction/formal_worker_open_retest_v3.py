#!/usr/bin/env python3
"""Open re-test with corrected Jacobian and action-link pose contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import formal_worker_open_retest_v2 as worker


DATA_ROOT = Path("/home/feng/robot_baselines/results/pa3ff/padp_data_v3_initial_state_aligned_fourtask")
BASE_RUNTIME = worker.legacy.PA3FFPADPRuntime


def load_hand_to_grasptarget() -> np.ndarray:
    transforms = []
    for task in ("door_open", "door_close", "drawer_open", "drawer_close"):
        with np.load(DATA_ROOT / f"{task}_train_v3.npz", allow_pickle=False) as data:
            link = str(np.asarray(data["fk_chosen_link"]).item())
            if link != "panda_hand":
                raise RuntimeError(f"unexpected action-pose link for {task}: {link}")
            transforms.append(np.asarray(data["hand_to_grasptarget_T"], dtype=np.float64))
    reference = transforms[0]
    if reference.shape != (4, 4) or not np.isfinite(reference).all():
        raise RuntimeError("invalid hand_to_grasptarget_T")
    if any(not np.allclose(reference, other, atol=1e-9) for other in transforms[1:]):
        raise RuntimeError("task-specific hand_to_grasptarget_T mismatch")
    return reference


HAND_TO_GRASPTARGET = load_hand_to_grasptarget()


class ActionLinkContractRuntime:
    """Decode trained panda_hand poses into panda_grasptarget controller poses."""

    def __init__(self, checkpoint: Path):
        self.base = BASE_RUNTIME(checkpoint)
        self.checkpoint = self.base.checkpoint
        self.training_step = self.base.training_step

    def predict(self, **kwargs):
        prediction = self.base.predict(**kwargs)
        hand_poses = np.asarray(prediction["poses_world_grasptarget"], dtype=np.float64)
        grasp_poses = hand_poses @ HAND_TO_GRASPTARGET[None, :, :]
        prediction["poses_world_grasptarget"] = grasp_poses
        prediction["action_pose_link_in_training"] = "panda_hand"
        prediction["controller_pose_link"] = "panda_grasptarget"
        prediction["hand_to_grasptarget_T"] = HAND_TO_GRASPTARGET.tolist()
        return prediction


def main() -> None:
    worker.METHOD = "PA3FF_reproduction_v1_open_retest_v3"
    worker.EXECUTION_PROTOCOL = "world_cartesian_grasptarget_plus_action_link_transform_v3"
    worker.legacy.PA3FFPADPRuntime = ActionLinkContractRuntime
    worker.main()


if __name__ == "__main__":
    main()
