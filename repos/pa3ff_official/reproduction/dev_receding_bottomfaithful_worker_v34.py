#!/usr/bin/env python3
"""DEV-only PA3FF worker restoring the supplied grasp-phase defaults.

In addition to collision-checked teleport replay from v33, this reproduces
the attached evaluator's default phase discretization: OPEN has 16 waypoints
at 8 simulation steps, CLOSE has 80 waypoints at 8 steps, and HOLD has eight
waypoints covering 0.5 s (31 steps each at 500 Hz).  The target joint is
kinematically held only during CLOSE, exactly as ``unlock_from`` in the
supplied code specifies, and is free for OPEN, TO_GRASP, HOLD, and operation.
Grasp success remains a real final bilateral-finger contact gate.
"""
from __future__ import annotations

import numpy as np

import dev_receding_scenealigned_teleport_worker_v33 as teleport


scene_aligned = teleport.scene_aligned
receding = scene_aligned.receding
_REPAIRED_CREATE_SCENE = scene_aligned.CREATE_SCENE


def _create_scene_with_grasp_lock_handles(case: dict) -> dict:
    resources = _REPAIRED_CREATE_SCENE(case)
    panda = resources["panda"]
    panda.pa3ff_target_joint = resources["target_joint"]
    panda.pa3ff_target_link = resources["target_link"]
    panda.pa3ff_object = resources["obj"]
    panda.pa3ff_target_index = int(resources["target_index"])
    return resources


def _lock_target_state(panda, target_q):
    object_qpos = np.asarray(panda.pa3ff_object.get_qpos(), dtype=np.float64)
    object_qpos[panda.pa3ff_target_index] = float(target_q)
    panda.pa3ff_object.set_qpos(object_qpos)
    object_qvel = np.asarray(panda.pa3ff_object.get_qvel(), dtype=np.float64)
    object_qvel[panda.pa3ff_target_index] = 0.0
    panda.pa3ff_object.set_qvel(object_qvel)


def _step_and_measure(
    panda, target_link, robot_links, target_value, frames, *, lock_q=None
):
    receding.legacy.set_gripper_target(panda, float(target_value))
    if lock_q is not None:
        _lock_target_state(panda, lock_q)
    panda.step()
    if lock_q is not None:
        _lock_target_state(panda, lock_q)
    frames.append(
        receding.legacy.effective_contact(
            panda.scene, robot_links, target_link
        )
    )


def _supplied_grasp_sequence(panda, target_link, primitive: str, gripper_target: float) -> dict:
    if primitive != "pull":
        return receding.legacy_monitor_engagement_original(
            panda, target_link, primitive, gripper_target
        )

    if target_link is not panda.pa3ff_target_link:
        raise RuntimeError("target-link grasp-lock handle mismatch")
    target_joint = panda.pa3ff_target_joint
    object_qpos = np.asarray(panda.pa3ff_object.get_qpos(), dtype=np.float64)
    target_q = float(object_qpos[panda.pa3ff_target_index])
    robot_links = [panda.left_finger_link, panda.right_finger_link]
    frames = []

    # compose_full_grasp_qpath defaults from the supplied evaluator:
    # open_steps=16, close_steps=80, hold_steps=8.  Non-HOLD waypoints are
    # replayed for steps_per_waypoint=8; HOLD distributes 0.5 s over 8
    # waypoints, i.e. round(.5 / (8 * .002)) = 31 simulation steps.
    current_width = float(
        np.mean(np.asarray(panda.robot.get_qpos(), dtype=np.float64)[7:9])
    )
    for width in np.linspace(current_width, 0.04, 16):
        for _ in range(8):
            _step_and_measure(
                panda, target_link, robot_links, width, frames
            )

    # The supplied step callback pins target qpos/qvel on every CLOSE physics
    # step; a stiff drive alone is not equivalent under finger collision.
    for width in np.linspace(0.04, float(gripper_target), 80):
        for _ in range(8):
            _step_and_measure(
                panda, target_link, robot_links, width, frames,
                lock_q=target_q,
            )

    # unlock_from is the first HOLD waypoint, so release before HOLD.
    target_joint.set_drive_property(stiffness=0.0, damping=0.05, force_limit=0.0)
    target_joint.set_drive_target(target_q)
    target_joint.set_drive_velocity_target(0.0)
    for _ in range(8 * 31):
        _step_and_measure(
            panda, target_link, robot_links, gripper_target, frames
        )

    final_frame = frames[-1]
    names = set(final_frame["robot_links"])
    success = bool(
        "panda_leftfinger" in names
        and "panda_rightfinger" in names
        and int(final_frame["points"]) >= 2
    )
    bilateral_tail = [
        "panda_leftfinger" in set(frame["robot_links"])
        and "panda_rightfinger" in set(frame["robot_links"])
        for frame in frames[-100:]
    ]
    return {
        "success": success,
        "definition": "supplied_bottom_final_strict_bilateral_finger_target_contact",
        "open_waypoints": 16,
        "open_steps_per_waypoint": 8,
        "hold_open_steps": 128,
        "close_waypoints": 80,
        "close_steps_per_waypoint": 8,
        "gripper_close_steps": 640,
        "hold_waypoints": 8,
        "hold_steps_per_waypoint": 31,
        "grasp_post_close_settle_steps": 248,
        "target_locked_during_close": True,
        "target_locked_during_hold": False,
        "target_released_before_hold_and_operation": True,
        "tail_steps": 100,
        "valid_contact_frames": int(sum(bilateral_tail)),
        "valid_contact_fraction": float(np.mean(bilateral_tail)),
        "last_frame": final_frame,
        "max_impulse": max(
            (frame["max_impulse"] for frame in frames), default=0.0
        ),
    }


def main() -> None:
    scene_aligned.CREATE_SCENE = _create_scene_with_grasp_lock_handles
    receding.legacy_monitor_engagement_original = receding.legacy.monitor_engagement
    receding.legacy.monitor_engagement = _supplied_grasp_sequence
    teleport.main()


if __name__ == "__main__":
    main()
