"""PA3FF-local correction of the shared Panda Cartesian controller.

The legacy shared controller interpreted SAPIEN's spatial-twist translational
rows as the Cartesian velocity of the controlled point and selected a fixed
hand-link row by ``get_links()`` index.  Both assumptions are invalid for this
URDF.  This implementation controls the actual ``panda_grasptarget`` link with
SAPIEN's world Cartesian Jacobian, whose rows are [linear, angular].
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from panda_controller import PandaTwoFingerController


class CorrectedPandaTwoFingerController(PandaTwoFingerController):
    """Use the finite-difference-verified world Jacobian of panda_grasptarget."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grasp_link_index = self.links.index(self.grasp_link)

    def _get_grasp_world_cartesian_jacobian(self) -> np.ndarray:
        dense = np.asarray(
            self.robot.compute_world_cartesian_jacobian(), dtype=np.float64
        )
        row_start = (self.grasp_link_index - 1) * 6
        raw = dense[row_start : row_start + 6, :7]
        if raw.shape != (6, 7):
            raise RuntimeError(f"grasp Cartesian Jacobian shape invalid: {raw.shape}")
        if not np.all(np.isfinite(raw)):
            raise RuntimeError("grasp Cartesian Jacobian contains NaN/Inf")
        return raw

    def _get_grasp_twist_jacobian(self) -> np.ndarray:
        # SAPIEN world Cartesian Jacobian ordering is [linear xyz, angular xyz].
        raw = self._get_grasp_world_cartesian_jacobian()
        return np.concatenate([raw[3:6], raw[0:3]], axis=0)

    def _get_grasp_point_jacobian(self) -> np.ndarray:
        return self._get_grasp_world_cartesian_jacobian()[:3]

    def move_grasp_pose_to(
        self,
        target_world_grasp,
        num_steps,
        *,
        position_tolerance=0.005,
        rotation_tolerance=0.03,
    ):
        target = np.asarray(target_world_grasp, dtype=np.float64)
        if target.shape != (4, 4):
            raise ValueError("target_world_grasp must be 4x4")
        target_R = target[:3, :3]
        target_p = target[:3, 3]

        best_pos_error = np.inf
        best_rot_error = np.inf
        reached = False
        used_steps = 0

        for step in range(int(num_steps)):
            current = self.get_grasp_pose_matrix()
            current_R = current[:3, :3]
            current_p = current[:3, 3]
            pos_error = target_p - current_p
            pos_norm = float(np.linalg.norm(pos_error))
            rotvec = Rotation.from_matrix(target_R @ current_R.T).as_rotvec()
            rot_norm = float(np.linalg.norm(rotvec))
            best_pos_error = min(best_pos_error, pos_norm)
            best_rot_error = min(best_rot_error, rot_norm)
            if not np.isfinite(pos_norm + rot_norm):
                raise RuntimeError("Panda pose error contains NaN/Inf")
            if pos_norm <= float(position_tolerance) and rot_norm <= float(rotation_tolerance):
                reached = True
                used_steps = step
                break

            linear_velocity = self.position_gain * pos_error
            linear_norm = float(np.linalg.norm(linear_velocity))
            if linear_norm > self.max_cartesian_speed:
                linear_velocity *= self.max_cartesian_speed / linear_norm
            angular_velocity = self.rotation_gain * rotvec
            angular_norm = float(np.linalg.norm(angular_velocity))
            if angular_norm > self.max_angular_speed:
                angular_velocity *= self.max_angular_speed / angular_norm

            twist = np.concatenate([angular_velocity, linear_velocity])
            qvel = np.linalg.pinv(
                self._get_grasp_twist_jacobian(), rcond=self.pinv_rcond
            ) @ twist
            self._apply_arm_velocity(qvel)
            self.step()
            used_steps = step + 1

        self.clear_arm_velocity()
        final = self.get_grasp_pose_matrix()
        final_pos_error = float(np.linalg.norm(target_p - final[:3, 3]))
        final_rot_error = float(
            np.linalg.norm(Rotation.from_matrix(target_R @ final[:3, :3].T).as_rotvec())
        )
        print(
            "Corrected Panda grasp-pose move: "
            f"steps={used_steps}/{int(num_steps)}, "
            f"pos_error={final_pos_error:.4f} m, rot_error={final_rot_error:.4f} rad"
        )
        return {
            "reached_control_tolerance": bool(reached),
            "used_steps": int(used_steps),
            "final_position_error": final_pos_error,
            "final_rotation_error": final_rot_error,
            "best_position_error": float(best_pos_error),
            "best_rotation_error": float(best_rot_error),
            "controller_version": "world_cartesian_grasptarget_v2",
        }
