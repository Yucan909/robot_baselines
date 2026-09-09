"""Controller bridge for the senior-supplied physical bottom logic.

Reach and pre-grasp use the finite-difference-verified panda_grasptarget
Jacobian.  Post-grasp ArticuBot SE(3) actions are representation-converted by
current-seeded IK and tracked by the senior joint PD drives.  The palm stop and
1 mm / 25 mm retreat are the defaults documented in the supplied archive.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


SHARED = Path("/home/feng/robot_baselines/repos/pa3ff_official/reproduction")
FROZEN = Path("/home/feng/robot_baselines/common_env/flowbot3d_conditionA_physical_v2")
if str(FROZEN) not in sys.path:
    sys.path.insert(0, str(FROZEN))
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

from panda_controller_corrected_v2 import CorrectedPandaTwoFingerController  # noqa: E402
from panda_controller_joint_pd_v8 import JointIKPDOperationMixin  # noqa: E402
from contact_monitor import read_twofinger_contact  # noqa: E402


PALM_RETREAT_STEP_M = 0.001
PALM_RETREAT_MAX_M = 0.025


def _actor_id(actor) -> int:
    return int(actor.get_id()) if hasattr(actor, "get_id") else int(actor.id)


class PerceptionReachPhysicalController(
    JointIKPDOperationMixin, CorrectedPandaTwoFingerController
):
    """Corrected Cartesian reach, palm-stop contact, then joint-PD operation."""

    def palm_object_contact(self, object_links) -> dict:
        hand_id = _actor_id(self.hand_link)
        object_by_id = {_actor_id(link): link for link in object_links}
        object_ids = set(object_by_id)
        points = 0
        maximum = 0.0
        hit_names = set()
        for contact in self.scene.get_contacts():
            ids = {_actor_id(contact.actor0), _actor_id(contact.actor1)}
            if hand_id not in ids or not (ids & object_ids):
                continue
            for actor_id in ids & object_ids:
                actor = object_by_id[actor_id]
                hit_names.add(str(actor.get_name() if hasattr(actor, "get_name") else actor_id))
            for point in contact.points:
                impulse = float(np.linalg.norm(np.asarray(point.impulse, dtype=np.float64)))
                if np.isfinite(impulse) and impulse > 1e-8:
                    points += 1
                    maximum = max(maximum, impulse)
        return {
            "contact": points > 0,
            "effective_points": points,
            "max_impulse": maximum,
            "object_link_names": sorted(hit_names),
        }

    def approach_with_palm_stop(self, target_world_grasp, num_steps, object_links) -> dict:
        target = np.asarray(target_world_grasp, dtype=np.float64).reshape(4, 4)
        target_r, target_p = target[:3, :3], target[:3, 3]
        start = self.get_grasp_pose_matrix()
        approach = target_p - start[:3, 3]
        retreat_axis = approach / max(float(np.linalg.norm(approach)), 1e-12)
        ik = self.solve_operation_pose_ik(target)
        qstar = np.asarray(ik.pop("solution"), dtype=np.float64)
        best_pos = np.inf
        best_rot = np.inf
        hit = None
        used = 0
        reached = False
        for step in range(int(num_steps)):
            current = self.get_grasp_pose_matrix()
            pos_error = target_p - current[:3, 3]
            rotvec = Rotation.from_matrix(target_r @ current[:3, :3].T).as_rotvec()
            pos_norm = float(np.linalg.norm(pos_error))
            rot_norm = float(np.linalg.norm(rotvec))
            best_pos = min(best_pos, pos_norm)
            best_rot = min(best_rot, rot_norm)
            if pos_norm <= 0.005 and rot_norm <= 0.03:
                reached = True
                break
            for index, joint in enumerate(self.arm_joints):
                joint.set_drive_velocity_target(0.0)
                joint.set_drive_target(float(qstar[index]))
            self.step()
            used = step + 1
            contact = self.palm_object_contact(object_links)
            if contact["contact"]:
                hit = {"physics_step": used, **contact}
                break
        self.clear_arm_velocity()

        retreat_steps = 0
        cleared = hit is None
        if hit is not None:
            for retreat_steps in range(
                1, int(round(PALM_RETREAT_MAX_M / PALM_RETREAT_STEP_M)) + 1
            ):
                hold = self.get_grasp_pose_matrix()
                hold[:3, 3] -= PALM_RETREAT_STEP_M * retreat_axis
                super().move_grasp_pose_to(
                    hold, 20, position_tolerance=0.0006, rotation_tolerance=0.03
                )
                if not self.palm_object_contact(object_links)["contact"]:
                    cleared = True
                    break

        final = self.get_grasp_pose_matrix()
        return {
            "reached_control_tolerance": bool(reached),
            "used_steps": int(used),
            "final_position_error": float(np.linalg.norm(target_p - final[:3, 3])),
            "final_rotation_error": float(np.linalg.norm(
                Rotation.from_matrix(target_r @ final[:3, :3].T).as_rotvec()
            )),
            "best_position_error": float(best_pos),
            "best_rotation_error": float(best_rot),
            "controller_version": "current_seed_IK_joint_PD_v8_palm_stop",
            "fixed_joint_target_pd": True,
            "ik_success": bool(ik["success"]),
            "ik_residual": float(ik["residual"]),
            "ik_solution_arm_delta_l2": float(ik["arm_delta_l2"]),
            "ik_min_joint_limit_margin": float(ik["min_joint_limit_margin"]),
            "palm_stop_enabled": True,
            "palm_hit": hit is not None,
            "palm_hit_diagnostic": hit,
            "palm_retreat_step_mm": 1.0,
            "palm_retreat_max_mm": 25.0,
            "palm_retreat_used_mm": float(retreat_steps),
            "palm_cleared": bool(cleared),
        }

    def approach_with_bilateral_preclose(
        self, target_world_grasp, num_steps, object_links, target_link
    ) -> dict:
        """Joint-PD final approach with perception-targeted two-finger pre-close.

        Fingers remain fully open through the distant corridor and close only
        inside the last 90 mm.  Five consecutive bilateral physical-contact
        frames stop the arm before the palm can push the articulated part away.
        """
        target = np.asarray(target_world_grasp, dtype=np.float64).reshape(4, 4)
        target_r, target_p = target[:3, :3], target[:3, 3]
        start = self.get_grasp_pose_matrix()
        start_distance = float(np.linalg.norm(target_p - start[:3, 3]))
        approach = target_p - start[:3, 3]
        retreat_axis = approach / max(start_distance, 1e-12)
        ik = self.solve_operation_pose_ik(target)
        qstar = np.asarray(ik.pop("solution"), dtype=np.float64)
        best_pos = np.inf
        best_rot = np.inf
        hit = None
        bilateral_hit = None
        bilateral_consecutive = 0
        max_bilateral_consecutive = 0
        used = 0
        reached = False
        preclose_start_m = 0.09
        for step in range(int(num_steps)):
            current = self.get_grasp_pose_matrix()
            pos_error = target_p - current[:3, 3]
            rotvec = Rotation.from_matrix(target_r @ current[:3, :3].T).as_rotvec()
            pos_norm = float(np.linalg.norm(pos_error))
            rot_norm = float(np.linalg.norm(rotvec))
            best_pos = min(best_pos, pos_norm)
            best_rot = min(best_rot, rot_norm)
            if pos_norm <= 0.005 and rot_norm <= 0.03:
                reached = True
            for index, joint in enumerate(self.arm_joints):
                joint.set_drive_velocity_target(0.0)
                joint.set_drive_target(float(qstar[index]))
            if pos_norm < preclose_start_m:
                fraction = np.clip(
                    (preclose_start_m - pos_norm) / preclose_start_m, 0.0, 1.0
                )
                finger_target = 0.04 * (1.0 - fraction)
                for joint in self.finger_joints:
                    joint.set_drive_velocity_target(0.0)
                    joint.set_drive_target(float(finger_target))
            self.step()
            used = step + 1
            contact = read_twofinger_contact(
                self.scene, self.left_finger_link, self.right_finger_link, target_link
            )
            if contact["bilateral_contact"]:
                bilateral_consecutive += 1
                max_bilateral_consecutive = max(
                    max_bilateral_consecutive, bilateral_consecutive
                )
            else:
                bilateral_consecutive = 0
            if bilateral_consecutive >= 5:
                bilateral_hit = {"physics_step": used, **contact}
                break
            palm = self.palm_object_contact(object_links)
            if palm["contact"]:
                hit = {"physics_step": used, **palm}
                break
            if reached:
                break
        self.clear_arm_velocity()

        retreat_steps = 0
        cleared = hit is None
        if hit is not None and bilateral_hit is None:
            for retreat_steps in range(
                1, int(round(PALM_RETREAT_MAX_M / PALM_RETREAT_STEP_M)) + 1
            ):
                hold = self.get_grasp_pose_matrix()
                hold[:3, 3] -= PALM_RETREAT_STEP_M * retreat_axis
                super().move_grasp_pose_to(
                    hold, 20, position_tolerance=0.0006, rotation_tolerance=0.03
                )
                if not self.palm_object_contact(object_links)["contact"]:
                    cleared = True
                    break
        final = self.get_grasp_pose_matrix()
        return {
            "reached_control_tolerance": bool(reached),
            "used_steps": int(used),
            "final_position_error": float(np.linalg.norm(target_p - final[:3, 3])),
            "final_rotation_error": float(np.linalg.norm(
                Rotation.from_matrix(target_r @ final[:3, :3].T).as_rotvec()
            )),
            "best_position_error": float(best_pos),
            "best_rotation_error": float(best_rot),
            "controller_version": "current_seed_IK_joint_PD_v8_collision_preclose",
            "fixed_joint_target_pd": True,
            "ik_success": bool(ik["success"]),
            "ik_residual": float(ik["residual"]),
            "ik_solution_arm_delta_l2": float(ik["arm_delta_l2"]),
            "ik_min_joint_limit_margin": float(ik["min_joint_limit_margin"]),
            "bilateral_preclose_enabled": True,
            "bilateral_required_consecutive_frames": 5,
            "max_bilateral_consecutive_frames": int(max_bilateral_consecutive),
            "bilateral_hit": bilateral_hit is not None,
            "bilateral_hit_diagnostic": bilateral_hit,
            "preclose_start_distance_m": preclose_start_m,
            "palm_stop_enabled": True,
            "palm_hit": hit is not None,
            "palm_hit_diagnostic": hit,
            "palm_retreat_step_mm": 1.0,
            "palm_retreat_max_mm": 25.0,
            "palm_retreat_used_mm": float(retreat_steps),
            "palm_cleared": bool(cleared),
        }


def controller_audit() -> dict:
    return {
        "reach_controller": str(SHARED / "panda_controller_corrected_v2.py"),
        "operate_controller": str(SHARED / "panda_controller_joint_pd_v8.py"),
        "palm_stop_source": str(
            SHARED / "formal_worker_soft_weld_pd_devselected_v3.py"
        ),
        "operate_control": "official_SE3_to_current_seed_IK_then_fixed_duration_joint_PD",
        "palm_retreat_step_mm": 1.0,
        "palm_retreat_max_mm": 25.0,
        "bilateral_preclose": "implemented_but_disabled_after_dev_regression",
        "formal_contact_approach": "fingers_open_then_palm_stop_then_PHYSICAL_V2_close",
    }
