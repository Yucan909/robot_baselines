"""Original-style 6-DoF floating Panda gripper for the paired formal ablation.

The URDF is the Where2Act ``panda_gripper.urdf``: six virtual Cartesian/
rotational joints followed by the physical Panda hand and two prismatic
fingers.  There is deliberately no 7-DoF arm, IK solver, or path planner.
"""

from pathlib import Path
import sys

import numpy as np
import sapien.core as sapien


WHERE2ACT_CODE = Path("/home/feng/robot_baselines/repos/where2act/code")
if str(WHERE2ACT_CODE) not in sys.path:
    sys.path.insert(0, str(WHERE2ACT_CODE))

from utils import adjoint_matrix, pose2exp_coordinate  # noqa: E402


PANDA_GRIPPER_URDF = WHERE2ACT_CODE / "robots" / "panda_gripper.urdf"


class FloatingPandaTwoFingerController:
    """Physical Panda hand on the official six virtual pose joints."""

    def __init__(
        self,
        scene,
        urdf_path=PANDA_GRIPPER_URDF,
        *,
        pose_stiffness=1000.0,
        pose_damping=400.0,
        finger_stiffness=8000.0,
        finger_damping=1600.0,
        finger_static_friction=2.0,
        finger_dynamic_friction=2.0,
        finger_restitution=0.0,
    ):
        self.scene = scene
        self.timestep = float(scene.get_timestep())

        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        self.robot = loader.load(str(urdf_path))
        if self.robot is None:
            raise RuntimeError(f"floating Panda gripper URDF load failed: {urdf_path}")
        self.robot.set_name("where2act_floating_gripper")

        self.links = list(self.robot.get_links())
        links = {link.get_name(): link for link in self.links}
        for name in ("panda_hand", "panda_leftfinger", "panda_rightfinger"):
            if name not in links:
                raise RuntimeError(f"floating gripper missing link: {name}")
        self.hand_link = links["panda_hand"]
        self.grasp_link = self.hand_link
        self.left_finger_link = links["panda_leftfinger"]
        self.right_finger_link = links["panda_rightfinger"]
        self.hand_link_index = self.links.index(self.hand_link)

        active = list(self.robot.get_active_joints())
        self.finger_joints = [
            joint for joint in active
            if joint.get_name().startswith("panda_finger_joint")
        ]
        self.pose_joints = [
            joint for joint in active
            if not joint.get_name().startswith("panda_finger_joint")
        ]
        if len(self.pose_joints) != 6 or len(self.finger_joints) != 2:
            raise RuntimeError(
                "official floating gripper must expose 6 pose + 2 finger DoF; "
                f"got {len(self.pose_joints)} + {len(self.finger_joints)}"
            )

        for joint in self.pose_joints:
            joint.set_drive_property(
                float(pose_stiffness), float(pose_damping), force_limit=1.0e6
            )
            joint.set_drive_velocity_target(0.0)
        for joint in self.finger_joints:
            joint.set_drive_property(
                float(finger_stiffness), float(finger_damping), force_limit=1.0e4
            )
            joint.set_drive_velocity_target(0.0)

        self.finger_material = scene.create_physical_material(
            float(finger_static_friction),
            float(finger_dynamic_friction),
            float(finger_restitution),
        )
        for link in (self.left_finger_link, self.right_finger_link):
            for shape in link.get_collision_shapes():
                shape.set_physical_material(self.finger_material)

    def set_initial_pose(self, pose_world, *, open_gripper):
        pose_world = np.asarray(pose_world, dtype=np.float64).reshape(4, 4)
        self.root_pose_world = pose_world.copy()
        self.robot.set_root_pose(sapien.Pose().from_transformation_matrix(pose_world))
        qpos = np.zeros(8, dtype=np.float64)
        qpos[-2:] = 0.04 if open_gripper else 0.0
        self.robot.set_qpos(qpos)
        for index, joint in enumerate(self.robot.get_active_joints()):
            joint.set_drive_target(float(qpos[index]))
            joint.set_drive_velocity_target(0.0)

    def get_grasp_center(self):
        return np.asarray(self.hand_link.get_pose().p, dtype=np.float64).copy()

    def get_grasp_pose_matrix(self):
        return np.asarray(
            self.hand_link.get_pose().to_transformation_matrix(), dtype=np.float64
        )

    def get_finger_qpos(self):
        return np.asarray(self.robot.get_qpos(), dtype=np.float64)[-2:].copy()

    def _joint_velocity_from_twist(self, twist):
        twist = np.asarray(twist, dtype=np.float64).reshape(6)
        dense = np.asarray(self.robot.compute_spatial_twist_jacobian(), dtype=np.float64)
        jacobian = np.zeros((6, 6), dtype=np.float64)
        row_end = self.hand_link_index * 6
        jacobian[:3, :] = dense[row_end - 3:row_end, :6]
        jacobian[3:6, :] = dense[row_end - 6:row_end - 3, :6]
        return np.linalg.pinv(jacobian, rcond=1e-2) @ twist

    def _calculate_twist(self, time_to_target, target_pose):
        current = self.hand_link.get_pose().to_transformation_matrix()
        relative = np.linalg.inv(current) @ np.asarray(target_pose, dtype=np.float64)
        unit_twist, theta = pose2exp_coordinate(relative)
        body_twist = unit_twist * (theta / max(float(time_to_target), self.timestep))
        return adjoint_matrix(current) @ body_twist

    def _command_pose_velocity(self, qvel):
        qvel = np.asarray(qvel, dtype=np.float64).reshape(6)
        targets = np.asarray(self.robot.get_drive_target(), dtype=np.float64).copy()
        targets[:6] += qvel * self.timestep
        for index, joint in enumerate(self.pose_joints):
            joint.set_drive_velocity_target(float(qvel[index]))
            joint.set_drive_target(float(targets[index]))

    def step(self):
        self.robot.set_qf(self.robot.compute_passive_force())
        self.scene.step()

    def wait(self, steps):
        self.clear_arm_velocity()
        for _ in range(int(steps)):
            self.step()

    def clear_arm_velocity(self):
        for joint in self.pose_joints:
            joint.set_drive_velocity_target(0.0)

    def open_gripper(self):
        for joint in self.finger_joints:
            joint.set_drive_target(0.04)

    def close_gripper(self):
        for joint in self.finger_joints:
            joint.set_drive_target(0.0)

    def keep_gripper_closed(self):
        self.close_gripper()

    def move_grasp_pose_to(
        self,
        target_pose,
        steps,
        *,
        position_tolerance=0.006,
        rotation_tolerance=0.05,
    ):
        target_pose = np.asarray(target_pose, dtype=np.float64).reshape(4, 4)
        steps = int(steps)
        initial_pose = self.get_grasp_pose_matrix()
        initial_position_error = float(
            np.linalg.norm(initial_pose[:3, 3] - target_pose[:3, 3])
        )
        root = np.asarray(self.root_pose_world, dtype=np.float64)
        local_translation = root[:3, :3].T @ (target_pose[:3, 3] - root[:3, 3])
        relative_rotation = root[:3, :3].T @ target_pose[:3, :3]
        rotation_residual = float(
            np.arccos(np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0))
        )
        if rotation_residual > 1e-5:
            raise RuntimeError(
                "floating formal controller expects the fixed network orientation; "
                f"rotation residual={rotation_residual}"
            )
        joint_target = np.zeros(6, dtype=np.float64)
        joint_target[:3] = local_translation
        for index, joint in enumerate(self.pose_joints):
            joint.set_drive_target(float(joint_target[index]))
            joint.set_drive_velocity_target(0.0)
        best_position = float("inf")
        best_rotation = float("inf")
        for index in range(steps):
            self.step()
            actual = self.get_grasp_pose_matrix()
            position_error = float(np.linalg.norm(actual[:3, 3] - target_pose[:3, 3]))
            trace = float(np.trace(target_pose[:3, :3] @ actual[:3, :3].T))
            rotation_error = float(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)))
            best_position = min(best_position, position_error)
            best_rotation = min(best_rotation, rotation_error)
        actual = self.get_grasp_pose_matrix()
        final_position = float(np.linalg.norm(actual[:3, 3] - target_pose[:3, 3]))
        trace = float(np.trace(target_pose[:3, :3] @ actual[:3, :3].T))
        final_rotation = float(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)))
        return {
            "reached_control_tolerance": bool(
                final_position <= float(position_tolerance)
                and final_rotation <= float(rotation_tolerance)
            ),
            "used_steps": steps,
            "final_position_error": final_position,
            "final_rotation_error": final_rotation,
            "best_position_error": best_position,
            "best_rotation_error": best_rotation,
            "initial_position_error": initial_position_error,
            "joint_target": joint_target.tolist(),
            "final_qpos": np.asarray(self.robot.get_qpos(), dtype=np.float64).tolist(),
        }

    def move_grasp_point_by(self, delta, steps, *, control_tolerance=0.005):
        target = self.get_grasp_pose_matrix()
        target[:3, 3] += np.asarray(delta, dtype=np.float64).reshape(3)
        # The operation primitive keeps the network-selected orientation; do
        # not turn tiny contact-induced angular drift into a runtime failure.
        target[:3, :3] = self.root_pose_world[:3, :3]
        diagnostics = self.move_grasp_pose_to(
            target,
            steps,
            position_tolerance=float(control_tolerance),
            rotation_tolerance=0.05,
        )
        diagnostics.update(
            {
                "target": target[:3, 3].tolist(),
                "final": self.get_grasp_center().tolist(),
                "final_error": diagnostics["final_position_error"],
                "best_error": diagnostics["best_position_error"],
            }
        )
        return diagnostics

    def move_grasp_point_to(self, target_world, steps, *, control_tolerance=0.005):
        target = self.get_grasp_pose_matrix()
        target[:3, 3] = np.asarray(target_world, dtype=np.float64).reshape(3)
        target[:3, :3] = self.root_pose_world[:3, :3]
        diagnostics = self.move_grasp_pose_to(
            target,
            steps,
            position_tolerance=float(control_tolerance),
            rotation_tolerance=0.05,
        )
        diagnostics.update(
            {
                "target": target[:3, 3].tolist(),
                "final": self.get_grasp_center().tolist(),
                "final_error": diagnostics["final_position_error"],
                "best_error": diagnostics["best_position_error"],
            }
        )
        return diagnostics
