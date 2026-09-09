#!/usr/bin/env python3
"""DEV-only PA3FF worker matching the supplied INIT->PREGRASP replay.

The attached canonical bottom evaluator teleports along its already
collision-checked pregrasp joint path (``teleport_init_to_pregrasp=True``),
with zero settle steps.  Earlier PA3FF adapters instead simulated a stiff PD
controller at every OMPL waypoint.  That additional controller rejected valid
collision-free paths because tracking lag accumulated at intermediate
waypoints.  This adapter restores the supplied behavior while retaining the
same OMPL collision check, policy candidates, physical GRASP approach,
bilateral-contact gate, soft weld, and operation controller.
"""
from __future__ import annotations

import numpy as np

import dev_receding_scenealigned_worker_v32 as scene_aligned


def _teleport_collision_checked_waypoint(self, target, *, strict=False):
    target = np.asarray(target, dtype=np.float64).reshape(7)
    qpos = np.asarray(self.robot.get_qpos(), dtype=np.float64).reshape(9)
    qpos[:7] = target
    # Keep the fingers open without calling ``open_gripper`` here.  That helper
    # is a physical controller: it advances the simulation, so the arm PD
    # drives immediately pull the just-teleported arm away from this waypoint
    # before the replay check.  The supplied evaluator's teleport path performs
    # no settle/control steps between collision-checked waypoints.
    qpos[7:9] = 0.04
    self.robot.set_qpos(qpos)
    self.robot.set_qvel(np.zeros_like(np.asarray(self.robot.get_qvel(), dtype=np.float64)))
    after_set = np.asarray(self.robot.get_qpos(), dtype=np.float64).copy()
    for index, joint in enumerate(self.arm_joints):
        joint.set_drive_velocity_target(0.0)
        joint.set_drive_target(float(target[index]))
    for joint in self.finger_joints:
        joint.set_drive_velocity_target(0.0)
        joint.set_drive_target(0.04)
    actual = np.asarray(self.robot.get_qpos(), dtype=np.float64)[:7]
    error = actual - target
    diagnostic = {
        # SAPIEN stores articulation qpos as float32.  The round-trip error is
        # typically 1e-7 even though set_qpos reached the exact representable
        # waypoint, so 1e-8 incorrectly rejects every non-trivial waypoint.
        "converged": bool(
            np.linalg.norm(error) <= 1e-6
            and np.max(np.abs(error)) <= 1e-6
        ),
        "steps": 0,
        "error_norm": float(np.linalg.norm(error)),
        "error_maxabs": float(np.max(np.abs(error))),
        "target_qpos7": target.tolist(),
        "actual_qpos7": actual.tolist(),
        "qpos9_after_set": after_set.tolist(),
        "replay_mode": "supplied_bottom_collision_checked_path_teleport",
        "settle_steps": 0,
    }
    self._last_track_joint_diagnostic = diagnostic
    return diagnostic


def main() -> None:
    # The path itself is still produced and post-validated by the unchanged
    # collision-aware OMPL worker.  Only its replay matches the supplied bottom
    # evaluator's canonical default.
    scene_aligned.receding.JointPDFullBottomController._track_joint_target = (
        _teleport_collision_checked_waypoint
    )
    scene_aligned.main()


if __name__ == "__main__":
    main()
