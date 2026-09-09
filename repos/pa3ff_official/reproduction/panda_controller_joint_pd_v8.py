"""Joint-PD adapter for PA3FF absolute SE(3) operation actions.

The supplied bottom executor tracks policy-produced joint targets with the
Panda joint drives.  PA3FF instead produces an absolute panda_grasptarget pose,
so this adapter performs only the representation conversion: current-seeded
Pinocchio IK followed by the same physical joint-drive tracking.  It contains
no object state, target joint, contact point, task direction, or success input.
"""
from __future__ import annotations

import numpy as np
import sapien.core as sapien
from scipy.spatial.transform import Rotation


class JointIKPDOperationMixin:
    """Mixin for controllers that already control ``panda_grasptarget``."""

    def initialize_operation_ik(self) -> None:
        if not hasattr(self, "operation_pinocchio"):
            self.operation_pinocchio = self.robot.create_pinocchio_model()

    def _operation_target_in_root(self, target_world: np.ndarray) -> np.ndarray:
        root_world = np.asarray(
            self.robot.get_root_pose().to_transformation_matrix(), dtype=np.float64
        )
        return np.linalg.inv(root_world) @ target_world

    def solve_operation_pose_ik(self, target_world: np.ndarray) -> dict:
        self.initialize_operation_ik()
        current = np.asarray(self.robot.get_qpos(), dtype=np.float64).reshape(9)
        target_root = self._operation_target_in_root(
            np.asarray(target_world, dtype=np.float64).reshape(4, 4)
        )
        solution, success, residual = self.operation_pinocchio.compute_inverse_kinematics(
            self.grasp_link_index,
            sapien.Pose.from_transformation_matrix(target_root),
            current,
            eps=1e-4,
            max_iterations=5000,
            dt=0.1,
            damp=1e-3,
        )
        solution = np.asarray(solution, dtype=np.float64).reshape(9)
        if not np.isfinite(solution).all():
            raise RuntimeError("operation IK returned NaN/Inf")
        solution[7:9] = current[7:9]
        margins = []
        for i, joint in enumerate(self.arm_joints):
            limits = np.asarray(joint.get_limits()[0], dtype=np.float64)
            lo, hi = float(limits[0]), float(limits[1])
            if np.isfinite(lo):
                solution[i] = max(solution[i], lo + 1e-4)
            if np.isfinite(hi):
                solution[i] = min(solution[i], hi - 1e-4)
            finite_margins = []
            if np.isfinite(lo):
                finite_margins.append(solution[i] - lo)
            if np.isfinite(hi):
                finite_margins.append(hi - solution[i])
            margins.append(min(finite_margins) if finite_margins else np.inf)
        return {
            "solution": solution,
            "success": bool(success),
            "residual": float(np.linalg.norm(np.asarray(residual, dtype=np.float64))),
            "arm_delta_l2": float(np.linalg.norm(solution[:7] - current[:7])),
            "min_joint_limit_margin": float(min(margins)),
        }

    def move_operation_pose_to(
        self,
        target_world_grasp,
        num_steps,
        *,
        position_tolerance=0.007,
        rotation_tolerance=0.05,
    ):
        """Convert one PA3FF SE(3) action to q* and hold q* for fixed steps."""
        target = np.asarray(target_world_grasp, dtype=np.float64).reshape(4, 4)
        ik = self.solve_operation_pose_ik(target)
        qstar = ik.pop("solution")
        best_position = np.inf
        best_rotation = np.inf
        first_reached_step = None
        max_arm_qvel = 0.0
        # Fixed-duration tracking matches the supplied executor.  Reaching a
        # tolerance early does not shorten the physical action duration.
        for step in range(int(num_steps)):
            for i, joint in enumerate(self.arm_joints):
                joint.set_drive_velocity_target(0.0)
                joint.set_drive_target(float(qstar[i]))
            self.step()
            pose = self.get_grasp_pose_matrix()
            position_error = float(np.linalg.norm(target[:3, 3] - pose[:3, 3]))
            rotation_error = float(np.linalg.norm(Rotation.from_matrix(
                target[:3, :3] @ pose[:3, :3].T
            ).as_rotvec()))
            best_position = min(best_position, position_error)
            best_rotation = min(best_rotation, rotation_error)
            max_arm_qvel = max(
                max_arm_qvel,
                float(np.max(np.abs(np.asarray(self.robot.get_qvel(), dtype=np.float64)[:7]))),
            )
            if (
                first_reached_step is None
                and position_error <= float(position_tolerance)
                and rotation_error <= float(rotation_tolerance)
            ):
                first_reached_step = step + 1
        self.clear_arm_velocity()
        final = self.get_grasp_pose_matrix()
        final_position = float(np.linalg.norm(target[:3, 3] - final[:3, 3]))
        final_rotation = float(np.linalg.norm(Rotation.from_matrix(
            target[:3, :3] @ final[:3, :3].T
        ).as_rotvec()))
        return {
            "reached_control_tolerance": first_reached_step is not None,
            "used_steps": int(num_steps),
            "first_reached_step": first_reached_step,
            "final_position_error": final_position,
            "final_rotation_error": final_rotation,
            "best_position_error": float(best_position),
            "best_rotation_error": float(best_rotation),
            "controller_version": "pa3ff_se3_to_current_seed_ik_joint_pd_v8",
            "ik_success": ik["success"],
            "ik_residual": ik["residual"],
            "ik_solution_arm_delta_l2": ik["arm_delta_l2"],
            "ik_min_joint_limit_margin": ik["min_joint_limit_margin"],
            "max_abs_arm_qvel": max_arm_qvel,
            "fixed_duration_joint_pd": True,
        }
