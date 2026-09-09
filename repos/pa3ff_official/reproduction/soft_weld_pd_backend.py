"""Physical adapter for the supplied senior ``soft_weld_pd`` backend.

The supplied archive documents the implementation but omits its imported
``force_admittance_collect`` package.  This module reconstructs the two pieces
used by that evaluator: contact-gated finite-stiffness SAPIEN drives and the
post-grasp finger lock.  It deliberately contains no policy or task oracle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sapien.core as sapien
from scipy.spatial.transform import Rotation


LINEAR_STIFFNESS = 6000.0
LINEAR_DAMPING = 700.0
LINEAR_FORCE_LIMIT = 1200.0
ANGULAR_STIFFNESS = 1800.0
ANGULAR_DAMPING = 220.0
ANGULAR_FORCE_LIMIT = 500.0
FINGER_STIFFNESS = 20000.0
FINGER_DAMPING = 4000.0
FINGER_FORCE_LIMIT = 800.0
FINGER_GRIP_FORCE = 50.0


def _actor_id(actor) -> int:
    return int(actor.get_id()) if hasattr(actor, "get_id") else int(actor.id)


def _contact_anchor(scene, actor_a, actor_b, impulse_epsilon: float = 1e-8):
    """Return impulse-weighted contact anchor and diagnostics for two actors."""
    aid, bid = _actor_id(actor_a), _actor_id(actor_b)
    positions, weights = [], []
    for contact in scene.get_contacts():
        a0, a1 = _actor_id(contact.actor0), _actor_id(contact.actor1)
        if not ((a0 == aid and a1 == bid) or (a0 == bid and a1 == aid)):
            continue
        for point in contact.points:
            impulse = float(np.linalg.norm(np.asarray(point.impulse, dtype=np.float64)))
            if not np.isfinite(impulse) or impulse <= float(impulse_epsilon):
                continue
            position = np.asarray(point.position, dtype=np.float64).reshape(3)
            if np.isfinite(position).all():
                positions.append(position)
                weights.append(impulse)
    if not positions:
        return None, {"effective_points": 0, "total_impulse": 0.0}
    anchor = np.average(np.stack(positions), axis=0, weights=np.asarray(weights))
    return anchor, {
        "effective_points": len(positions),
        "total_impulse": float(np.sum(weights)),
        "anchor_world": anchor.tolist(),
    }


def _pose_error(actor_a, local_a: sapien.Pose, actor_b, local_b: sapien.Pose):
    world_a = actor_a.get_pose() * local_a
    world_b = actor_b.get_pose() * local_b
    position = float(np.linalg.norm(np.asarray(world_a.p) - np.asarray(world_b.p)))
    ra = Rotation.from_quat(np.roll(np.asarray(world_a.q, dtype=np.float64), -1))
    rb = Rotation.from_quat(np.roll(np.asarray(world_b.q, dtype=np.float64), -1))
    angular = float(np.linalg.norm((ra * rb.inv()).as_rotvec()))
    return position, angular


@dataclass
class SoftContactWeld:
    scene: object
    drives: list
    frames: list

    @classmethod
    def try_create(cls, panda, target_link):
        left_anchor, left_diag = _contact_anchor(
            panda.scene, panda.left_finger_link, target_link
        )
        right_anchor, right_diag = _contact_anchor(
            panda.scene, panda.right_finger_link, target_link
        )
        diag = {
            "enabled": True,
            "created": False,
            "hard": False,
            "require_two_finger": True,
            "hand_weld": True,
            "finger_links": [],
            "raw_contact_points": int(left_diag["effective_points"] + right_diag["effective_points"]),
            "left_contact": left_anchor is not None,
            "right_contact": right_anchor is not None,
            "left": left_diag,
            "right": right_diag,
            "linear_stiffness": LINEAR_STIFFNESS,
            "linear_damping": LINEAR_DAMPING,
            "linear_force_limit": LINEAR_FORCE_LIMIT,
            "angular_stiffness": ANGULAR_STIFFNESS,
            "angular_damping": ANGULAR_DAMPING,
            "angular_force_limit": ANGULAR_FORCE_LIMIT,
            "drive_mode": "force",
            "kinematic_snap": False,
        }
        if left_anchor is None and right_anchor is None:
            diag["fail_reason"] = "no_finger_contact"
            return None, diag
        if left_anchor is None or right_anchor is None:
            diag["fail_reason"] = "not_two_finger_contact"
            return None, diag

        drives, frames = [], []

        def add_drive(other, anchor):
            # Both joint frames coincide at creation.  Their orientation is the
            # current target-link frame, so all six relative DOFs have zero rest
            # error and the finite K/D values act as a soft weld.
            world_anchor = sapien.Pose(anchor.tolist(), target_link.get_pose().q)
            target_local = target_link.get_pose().inv() * world_anchor
            other_local = other.get_pose().inv() * world_anchor
            drive = panda.scene.create_drive(target_link, target_local, other, other_local)
            is_acceleration = False  # archive default: drive_mode="force"
            for setter in (drive.set_x_properties, drive.set_y_properties, drive.set_z_properties):
                setter(LINEAR_STIFFNESS, LINEAR_DAMPING, LINEAR_FORCE_LIMIT, is_acceleration)
            drive.set_slerp_properties(
                ANGULAR_STIFFNESS, ANGULAR_DAMPING, ANGULAR_FORCE_LIMIT, is_acceleration
            )
            drives.append(drive)
            frames.append((target_link, target_local, other, other_local))

        add_drive(panda.left_finger_link, left_anchor)
        add_drive(panda.right_finger_link, right_anchor)
        add_drive(panda.hand_link, 0.5 * (left_anchor + right_anchor))
        diag.update({
            "created": True,
            "n_welds": len(drives),
            "finger_links": ["panda_leftfinger", "panda_rightfinger"],
            "fail_reason": None,
        })
        return cls(panda.scene, drives, frames), diag

    def errors(self):
        errors = [_pose_error(*frame) for frame in self.frames]
        return {
            "weld_position_error_m": max((x[0] for x in errors), default=0.0),
            "weld_angular_error_rad": max((x[1] for x in errors), default=0.0),
        }


def enable_finger_lock(panda, target: float) -> dict:
    target = float(np.clip(target, 0.0, 0.04))
    for joint in panda.finger_joints:
        joint.set_drive_property(FINGER_STIFFNESS, FINGER_DAMPING, FINGER_FORCE_LIMIT)
        joint.set_drive_velocity_target(0.0)
        joint.set_drive_target(target)
    panda.soft_weld_finger_lock_target = target
    panda.soft_weld_finger_grip_force = FINGER_GRIP_FORCE
    return {
        "enabled": True,
        "finger_target": target,
        "finger_stiffness": FINGER_STIFFNESS,
        "finger_damping": FINGER_DAMPING,
        "finger_force_limit": FINGER_FORCE_LIMIT,
        "grip_torque": FINGER_GRIP_FORCE,
        "zero_arm_pd": False,
    }


def apply_finger_lock_force(panda) -> None:
    """Add the archive's sustained closing generalized force before a step."""
    if not hasattr(panda, "soft_weld_finger_lock_target"):
        return
    qf = np.asarray(panda.robot.compute_passive_force(), dtype=np.float64)
    if qf.shape[0] >= 9:
        qf[7:9] -= float(panda.soft_weld_finger_grip_force)
    panda.robot.set_qf(qf)

