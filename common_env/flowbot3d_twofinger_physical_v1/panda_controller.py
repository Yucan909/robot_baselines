import numpy as np
import sapien.core as sapien
from scipy.spatial.transform import Rotation


class PandaTwoFingerController:
    """
    统一 benchmark 的 Franka Panda 二指夹爪控制器。

    设计原则：
    1. 使用学长提供的 ArticuBot Panda URDF；
    2. 机械臂 7 DoF 使用 Jacobian + PD drive；
    3. 两根手指通过真实 prismatic joint 闭合；
    4. 不提供任何 hand-object / finger-object 人工约束；
    5. 抓取是否成立完全由 SAPIEN collision/contact 决定。

    该控制器同时提供：
    - move_grasp_pose_to(): 抓取阶段需要完整 6D 姿态；
    - move_grasp_point_by(): FlowBot 拉动阶段只控制工具中心 XYZ，
      不额外指定后续夹爪旋转轨迹。
    """

    def __init__(
        self,
        scene,
        urdf_path,
        *,
        arm_stiffness=1000.0,
        arm_damping=400.0,
        finger_stiffness=200.0,
        finger_damping=60.0,
        finger_static_friction=1.0,
        finger_dynamic_friction=1.0,
        finger_restitution=0.0,
        max_joint_speed=1.5,
        max_cartesian_speed=0.20,
        max_angular_speed=0.80,
        position_gain=3.0,
        rotation_gain=2.0,
        pinv_rcond=1e-2,
    ):
        self.scene = scene
        self.timestep = float(scene.get_timestep())

        self.max_joint_speed = float(max_joint_speed)
        self.max_cartesian_speed = float(max_cartesian_speed)
        self.max_angular_speed = float(max_angular_speed)
        self.position_gain = float(position_gain)
        self.rotation_gain = float(rotation_gain)
        self.pinv_rcond = float(pinv_rcond)

        loader = scene.create_urdf_loader()
        loader.fix_root_link = True

        self.robot = loader.load(str(urdf_path))
        if self.robot is None:
            raise RuntimeError(f"Panda URDF 加载失败: {urdf_path}")

        if int(self.robot.dof) != 9:
            raise RuntimeError(
                f"Panda 应为 9 自由度，实际为 {self.robot.dof}"
            )

        self.links = list(self.robot.get_links())
        self.link_dict = {
            link.get_name(): link
            for link in self.links
        }

        required_links = [
            "panda_hand",
            "panda_leftfinger",
            "panda_rightfinger",
        ]
        for name in required_links:
            if name not in self.link_dict:
                raise RuntimeError(f"Panda URDF 中找不到 {name}")

        self.hand_link = self.link_dict["panda_hand"]
        self.grasp_link = self.link_dict.get(
            "panda_grasptarget",
            self.hand_link,
        )
        self.left_finger_link = self.link_dict["panda_leftfinger"]
        self.right_finger_link = self.link_dict["panda_rightfinger"]

        self.hand_link_index = self.links.index(self.hand_link)

        active_joints = list(self.robot.get_active_joints())

        self.arm_joints = [
            j
            for j in active_joints
            if not j.get_name().startswith("panda_finger_joint")
        ]
        self.finger_joints = [
            j
            for j in active_joints
            if j.get_name().startswith("panda_finger_joint")
        ]

        if len(self.arm_joints) != 7:
            raise RuntimeError(
                f"Panda arm joint 数量应为 7，实际为 {len(self.arm_joints)}"
            )

        if len(self.finger_joints) != 2:
            raise RuntimeError(
                f"Panda finger joint 数量应为 2，实际为 {len(self.finger_joints)}"
            )

        for joint in self.arm_joints:
            joint.set_drive_property(
                stiffness=float(arm_stiffness),
                damping=float(arm_damping),
            )
            joint.set_drive_velocity_target(0.0)

        for joint in self.finger_joints:
            joint.set_drive_property(
                stiffness=float(finger_stiffness),
                damping=float(finger_damping),
            )
            joint.set_drive_velocity_target(0.0)

        # ArticuBot Panda URDF 的 finger contact 中 lateral_friction=1.0。
        # SAPIEN URDF loader 对 <contact> 标签的支持并不统一，
        # 因此这里显式设置两根 finger collision shape 的物理材料。
        self.finger_material = scene.create_physical_material(
            float(finger_static_friction),
            float(finger_dynamic_friction),
            float(finger_restitution),
        )

        for link in [self.left_finger_link, self.right_finger_link]:
            for shape in link.get_collision_shapes():
                shape.set_physical_material(self.finger_material)

        self._jacobian_printed = False
        self._video_step_callback = None
        self._video_every_n_steps = 16
        self._video_physics_counter = 0

    # --------------------------------------------------------
    # 初始化
    # --------------------------------------------------------

    def set_initial_state(self, base_xyz_yaw, qpos):
        """
        base_xyz_yaw: [x, y, yaw, z]
        qpos: 9D = 7 arm + 2 fingers
        """
        base = np.asarray(base_xyz_yaw, dtype=np.float64).reshape(4)
        qpos = np.asarray(qpos, dtype=np.float64).reshape(9)

        x, y, yaw, z = map(float, base)
        quat = [
            np.cos(yaw / 2.0),
            0.0,
            0.0,
            np.sin(yaw / 2.0),
        ]

        self.robot.set_root_pose(
            sapien.Pose([x, y, z], quat)
        )
        self.robot.set_qpos(qpos.copy())

        for i, joint in enumerate(self.robot.get_active_joints()):
            joint.set_drive_target(float(qpos[i]))
            joint.set_drive_velocity_target(0.0)

    # --------------------------------------------------------
    # 基础状态
    # --------------------------------------------------------

    def get_grasp_center(self):
        return np.asarray(
            self.grasp_link.get_pose().p,
            dtype=np.float64,
        ).copy()

    def get_grasp_pose_matrix(self):
        return np.asarray(
            self.grasp_link.get_pose().to_transformation_matrix(),
            dtype=np.float64,
        )

    def get_hand_pose_matrix(self):
        return np.asarray(
            self.hand_link.get_pose().to_transformation_matrix(),
            dtype=np.float64,
        )

    # --------------------------------------------------------
    # Jacobian
    # --------------------------------------------------------

    def _get_hand_twist_jacobian(self):
        """
        返回 panda_hand 的 6x7 Jacobian。

        输出顺序：
            [angular xyz
             linear  xyz]
        """
        dense = np.asarray(
            self.robot.compute_spatial_twist_jacobian(),
            dtype=np.float64,
        )

        if dense.ndim != 2:
            raise RuntimeError(f"Jacobian 维度异常: {dense.shape}")

        row_start = (self.hand_link_index - 1) * 6
        row_end = self.hand_link_index * 6

        if row_start < 0 or row_end > dense.shape[0]:
            raise RuntimeError("panda_hand 对应 Jacobian 行范围异常")

        raw = dense[row_start:row_end, :7]
        if raw.shape != (6, 7):
            raise RuntimeError(
                f"panda_hand Jacobian shape 异常: {raw.shape}"
            )

        # SAPIEN 1.1 raw: [linear, angular]
        J = np.zeros((6, 7), dtype=np.float64)
        J[:3, :] = raw[3:6, :]
        J[3:6, :] = raw[0:3, :]

        if not self._jacobian_printed:
            print("Panda Jacobian:", dense.shape, "->", J.shape)
            self._jacobian_printed = True

        return J

    @staticmethod
    def _skew(v):
        x, y, z = map(float, np.asarray(v).reshape(3))
        return np.array(
            [
                [0.0, -z, y],
                [z, 0.0, -x],
                [-y, x, 0.0],
            ],
            dtype=np.float64,
        )

    def _get_grasp_point_jacobian(self):
        J = self._get_hand_twist_jacobian()
        Jw = J[:3, :]
        Jv = J[3:6, :]

        hand_p = np.asarray(
            self.hand_link.get_pose().p,
            dtype=np.float64,
        )
        grasp_p = np.asarray(
            self.grasp_link.get_pose().p,
            dtype=np.float64,
        )

        r = grasp_p - hand_p
        Jp = Jv - self._skew(r) @ Jw

        if Jp.shape != (3, 7):
            raise RuntimeError(
                f"grasp point Jacobian shape 异常: {Jp.shape}"
            )

        if not np.all(np.isfinite(Jp)):
            raise RuntimeError("grasp point Jacobian 存在 NaN/Inf")

        return Jp

    # --------------------------------------------------------
    # drive / simulation step
    # --------------------------------------------------------

    def _apply_passive_force(self):
        passive_force = self.robot.compute_passive_force()
        self.robot.set_qf(passive_force)

    def _apply_arm_velocity(self, qvel):
        qvel = np.asarray(qvel, dtype=np.float64).reshape(7)

        if not np.all(np.isfinite(qvel)):
            raise RuntimeError("关节速度存在 NaN/Inf")

        qvel = np.clip(
            qvel,
            -self.max_joint_speed,
            self.max_joint_speed,
        )

        current_qpos = np.asarray(
            self.robot.get_qpos(),
            dtype=np.float64,
        )

        target_qpos = current_qpos[:7] + qvel * self.timestep

        for i, joint in enumerate(self.arm_joints):
            limits = np.asarray(joint.get_limits()[0], dtype=np.float64)
            lo = float(limits[0])
            hi = float(limits[1])

            if np.isfinite(lo):
                target_qpos[i] = max(target_qpos[i], lo + 1e-4)
            if np.isfinite(hi):
                target_qpos[i] = min(target_qpos[i], hi - 1e-4)

            joint.set_drive_velocity_target(float(qvel[i]))
            joint.set_drive_target(float(target_qpos[i]))

        self._apply_passive_force()

    def clear_arm_velocity(self):
        for joint in self.arm_joints:
            joint.set_drive_velocity_target(0.0)

    def set_video_step_callback(self, callback, *, every_n_steps=16):
        self._video_step_callback = callback
        self._video_every_n_steps = max(1, int(every_n_steps))
        self._video_physics_counter = 0

    def clear_video_step_callback(self):
        self._video_step_callback = None

    def step(self):
        self._apply_passive_force()
        self.scene.step()

        self._video_physics_counter += 1
        if (
            self._video_step_callback is not None
            and self._video_physics_counter % self._video_every_n_steps == 0
        ):
            self._video_step_callback()

    def wait(self, n):
        self.clear_arm_velocity()
        for _ in range(int(n)):
            self.step()

    # --------------------------------------------------------
    # 6D grasp-pose controller
    # --------------------------------------------------------

    def move_grasp_pose_to(
        self,
        target_world_grasp,
        num_steps,
        *,
        position_tolerance=0.005,
        rotation_tolerance=0.03,
    ):
        """
        控制 panda_grasptarget 到完整 6D 世界位姿。

        抓取姿态需要明确 finger closing axis，
        所以抓取阶段必须控制 orientation。
        """
        target_grasp = np.asarray(
            target_world_grasp,
            dtype=np.float64,
        )
        if target_grasp.shape != (4, 4):
            raise ValueError("target_world_grasp 必须为 4x4")

        # hand -> grasp 固定变换
        T_world_hand_now = self.get_hand_pose_matrix()
        T_world_grasp_now = self.get_grasp_pose_matrix()
        T_hand_grasp = np.linalg.inv(T_world_hand_now) @ T_world_grasp_now

        target_hand = target_grasp @ np.linalg.inv(T_hand_grasp)
        target_R = target_hand[:3, :3]
        target_p = target_hand[:3, 3]

        best_pos_error = np.inf
        best_rot_error = np.inf
        reached = False
        used_steps = 0

        for step in range(int(num_steps)):
            current = self.get_hand_pose_matrix()
            current_R = current[:3, :3]
            current_p = current[:3, 3]

            pos_error = target_p - current_p
            pos_norm = float(np.linalg.norm(pos_error))

            R_error = target_R @ current_R.T
            rotvec = Rotation.from_matrix(R_error).as_rotvec()
            rot_norm = float(np.linalg.norm(rotvec))

            best_pos_error = min(best_pos_error, pos_norm)
            best_rot_error = min(best_rot_error, rot_norm)

            if not np.isfinite(pos_norm) or not np.isfinite(rot_norm):
                raise RuntimeError("Panda pose error 出现 NaN/Inf")

            if (
                pos_norm <= float(position_tolerance)
                and rot_norm <= float(rotation_tolerance)
            ):
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
            J = self._get_hand_twist_jacobian()
            qvel = np.linalg.pinv(J, rcond=self.pinv_rcond) @ twist

            self._apply_arm_velocity(qvel)
            self.step()
            used_steps = step + 1

        self.clear_arm_velocity()

        final_hand = self.get_hand_pose_matrix()
        final_pos_error = float(
            np.linalg.norm(target_p - final_hand[:3, 3])
        )
        final_rot_error = float(
            np.linalg.norm(
                Rotation.from_matrix(
                    target_R @ final_hand[:3, :3].T
                ).as_rotvec()
            )
        )

        print(
            "Panda grasp-pose move: "
            f"steps={used_steps}/{int(num_steps)}, "
            f"pos_error={final_pos_error:.4f} m, "
            f"rot_error={final_rot_error:.4f} rad"
        )

        return {
            "reached_control_tolerance": bool(reached),
            "used_steps": int(used_steps),
            "final_position_error": final_pos_error,
            "final_rotation_error": final_rot_error,
            "best_position_error": float(best_pos_error),
            "best_rotation_error": float(best_rot_error),
        }

    # --------------------------------------------------------
    # 3D point-only pull controller
    # --------------------------------------------------------

    def move_grasp_point_to(
        self,
        target_world,
        num_steps,
        *,
        control_tolerance=0.005,
    ):
        target = np.asarray(target_world, dtype=np.float64).reshape(3)

        best_error = np.inf
        reached = False
        used_steps = 0

        for step in range(int(num_steps)):
            current = self.get_grasp_center()
            error = target - current
            error_norm = float(np.linalg.norm(error))
            best_error = min(best_error, error_norm)

            if not np.isfinite(error_norm):
                raise RuntimeError("Panda 末端位置误差出现 NaN/Inf")

            if error_norm <= float(control_tolerance):
                reached = True
                used_steps = step
                break

            velocity = self.position_gain * error
            velocity_norm = float(np.linalg.norm(velocity))
            if velocity_norm > self.max_cartesian_speed:
                velocity *= self.max_cartesian_speed / velocity_norm

            Jp = self._get_grasp_point_jacobian()
            qvel = np.linalg.pinv(Jp, rcond=self.pinv_rcond) @ velocity

            self._apply_arm_velocity(qvel)
            self.step()
            used_steps = step + 1

        self.clear_arm_velocity()
        final = self.get_grasp_center()
        final_error = float(np.linalg.norm(target - final))

        print(
            "Panda position move: "
            f"steps={used_steps}/{int(num_steps)}, "
            f"final_error={final_error:.4f} m, "
            f"best_error={best_error:.4f} m"
        )

        return {
            "reached_control_tolerance": bool(reached),
            "used_steps": int(used_steps),
            "target": target.copy(),
            "final": final.copy(),
            "final_error": final_error,
            "best_error": float(best_error),
        }

    def move_grasp_point_by(
        self,
        delta_world,
        num_steps,
        *,
        control_tolerance=0.005,
    ):
        delta = np.asarray(delta_world, dtype=np.float64).reshape(3)
        target = self.get_grasp_center() + delta
        return self.move_grasp_point_to(
            target,
            num_steps,
            control_tolerance=control_tolerance,
        )

    # --------------------------------------------------------
    # Gripper
    # --------------------------------------------------------

    def open_gripper(self):
        for joint in self.finger_joints:
            joint.set_drive_velocity_target(0.0)
            joint.set_drive_target(0.04)

    def close_gripper(self):
        for joint in self.finger_joints:
            joint.set_drive_velocity_target(0.0)
            joint.set_drive_target(0.0)

    def keep_gripper_closed(self):
        self.close_gripper()

    def get_finger_qpos(self):
        qpos = np.asarray(self.robot.get_qpos(), dtype=np.float64)
        return qpos[-2:].copy()
