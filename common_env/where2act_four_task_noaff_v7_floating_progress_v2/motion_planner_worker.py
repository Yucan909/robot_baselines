import argparse
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import pybullet as p

from pybullet_ompl.pb_ompl import (
    PbOMPL,
    PbOMPLRobot,
)


HOME = Path.home()

PANDA_URDF = (
    HOME
    / "robot_baselines"
    / "common_env"
    / "assets"
    / "panda_articubot"
    / "panda.urdf"
)

ARM_JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]

FINGER_JOINT_NAMES = [
    "panda_finger_joint1",
    "panda_finger_joint2",
]

EEF_LINK_NAME = "panda_grasptarget"


class PlannerError(RuntimeError):
    pass


class MinimalEnv:
    """
    ArticuBot PbOMPLRobot 所需的最小 env 接口。

    object_id=None 时，PbOMPLRobot 实际只需要：
        env.id
        env.mobile
    """

    def __init__(self, physics_id):
        self.id = int(
            physics_id
        )
        self.mobile = False


def load_catalog_row(
    catalog_path,
    shape_id,
    target_link,
):
    path = (
        Path(catalog_path)
        .expanduser()
        .resolve()
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    with path.open() as f:
        for line in f:

            line = line.strip()

            if not line:
                continue

            row = json.loads(
                line
            )

            link = (
                row.get("link_name")
                or row.get("target_link")
                or row.get("link")
            )

            if (
                str(row.get("shape_id"))
                == str(shape_id)
                and str(link)
                == str(target_link)
            ):
                return row

    raise PlannerError(
        "pose catalog找不到 "
        f"{shape_id}/{target_link}"
    )


def rotation_matrix_to_quaternion_xyzw(
    R,
):
    """
    3x3 rotation matrix
    -> PyBullet quaternion [x,y,z,w]
    """

    R = np.asarray(
        R,
        dtype=np.float64,
    )

    if R.shape != (
        3,
        3,
    ):
        raise PlannerError(
            f"rotation shape错误: "
            f"{R.shape}"
        )

    trace = float(
        np.trace(
            R
        )
    )

    if trace > 0.0:

        s = math.sqrt(
            trace + 1.0
        ) * 2.0

        qw = 0.25 * s

        qx = (
            R[2, 1]
            - R[1, 2]
        ) / s

        qy = (
            R[0, 2]
            - R[2, 0]
        ) / s

        qz = (
            R[1, 0]
            - R[0, 1]
        ) / s

    elif (
        R[0, 0] > R[1, 1]
        and R[0, 0] > R[2, 2]
    ):

        s = math.sqrt(
            1.0
            + R[0, 0]
            - R[1, 1]
            - R[2, 2]
        ) * 2.0

        qw = (
            R[2, 1]
            - R[1, 2]
        ) / s

        qx = 0.25 * s

        qy = (
            R[0, 1]
            + R[1, 0]
        ) / s

        qz = (
            R[0, 2]
            + R[2, 0]
        ) / s

    elif R[1, 1] > R[2, 2]:

        s = math.sqrt(
            1.0
            + R[1, 1]
            - R[0, 0]
            - R[2, 2]
        ) * 2.0

        qw = (
            R[0, 2]
            - R[2, 0]
        ) / s

        qx = (
            R[0, 1]
            + R[1, 0]
        ) / s

        qy = 0.25 * s

        qz = (
            R[1, 2]
            + R[2, 1]
        ) / s

    else:

        s = math.sqrt(
            1.0
            + R[2, 2]
            - R[0, 0]
            - R[1, 1]
        ) * 2.0

        qw = (
            R[1, 0]
            - R[0, 1]
        ) / s

        qx = (
            R[0, 2]
            + R[2, 0]
        ) / s

        qy = (
            R[1, 2]
            + R[2, 1]
        ) / s

        qz = 0.25 * s

    q = np.asarray(
        [
            qx,
            qy,
            qz,
            qw,
        ],
        dtype=np.float64,
    )

    q /= (
        np.linalg.norm(
            q
        )
        + 1e-12
    )

    return q


def pose_to_pybullet(
    pose,
):
    """
    学长统一benchmark中的 Panda base_pose 定义：

        [x, y, yaw, z]

    注意两个仿真器四元数顺序不同：

    SAPIEN:
        [w, x, y, z]

    PyBullet:
        [x, y, z, w]

    因此绕世界Z轴旋转yaw时，
    PyBullet应使用：

        [0, 0, sin(yaw/2), cos(yaw/2)]
    """

    pose = np.asarray(
        pose,
        dtype=np.float64,
    ).reshape(-1)

    if pose.shape != (4,):
        raise PlannerError(
            "学长 base_pose 应为4维 "
            "[x, y, yaw, z]，"
            f"实际shape={pose.shape}"
        )

    x = float(
        pose[0]
    )

    y = float(
        pose[1]
    )

    yaw = float(
        pose[2]
    )

    z = float(
        pose[3]
    )

    if not np.all(
        np.isfinite(
            [
                x,
                y,
                yaw,
                z,
            ]
        )
    ):
        raise PlannerError(
            "base_pose包含NaN/Inf"
        )

    position = np.asarray(
        [
            x,
            y,
            z,
        ],
        dtype=np.float64,
    )

    quaternion_xyzw = np.asarray(
        [
            0.0,
            0.0,
            math.sin(
                yaw / 2.0
            ),
            math.cos(
                yaw / 2.0
            ),
        ],
        dtype=np.float64,
    )

    quaternion_xyzw /= (
        np.linalg.norm(
            quaternion_xyzw
        )
        + 1e-12
    )

    return (
        position,
        quaternion_xyzw,
    )


def inspect_robot(
    robot_id,
    physics_id,
):
    joint_name_to_index = {}
    link_name_to_index = {}
    movable_joint_indices = []

    num_joints = p.getNumJoints(
        robot_id,
        physicsClientId=physics_id,
    )

    print()
    print("-" * 100)
    print("PYBULLET PANDA")
    print("-" * 100)

    for index in range(
        num_joints
    ):

        info = p.getJointInfo(
            robot_id,
            index,
            physicsClientId=physics_id,
        )

        joint_name = (
            info[1]
            .decode("utf-8")
        )

        joint_type = int(
            info[2]
        )

        link_name = (
            info[12]
            .decode("utf-8")
        )

        joint_name_to_index[
            joint_name
        ] = index

        link_name_to_index[
            link_name
        ] = index

        if joint_type != p.JOINT_FIXED:
            movable_joint_indices.append(
                index
            )

        print(
            f"{index:02d} "
            f"joint={joint_name:<22} "
            f"link={link_name:<22} "
            f"type={joint_type}"
        )

    missing_arm = [
        name
        for name in ARM_JOINT_NAMES
        if name not in joint_name_to_index
    ]

    if missing_arm:
        raise PlannerError(
            f"Panda缺少arm joints: "
            f"{missing_arm}"
        )

    missing_fingers = [
        name
        for name in FINGER_JOINT_NAMES
        if name not in joint_name_to_index
    ]

    if missing_fingers:
        raise PlannerError(
            f"Panda缺少finger joints: "
            f"{missing_fingers}"
        )

    if (
        EEF_LINK_NAME
        not in link_name_to_index
    ):
        raise PlannerError(
            f"找不到EE link: "
            f"{EEF_LINK_NAME}"
        )

    arm_indices = [
        joint_name_to_index[
            name
        ]
        for name in ARM_JOINT_NAMES
    ]

    finger_indices = [
        joint_name_to_index[
            name
        ]
        for name in FINGER_JOINT_NAMES
    ]

    eef_link_index = (
        link_name_to_index[
            EEF_LINK_NAME
        ]
    )

    return {
        "arm_indices":
            arm_indices,

        "finger_indices":
            finger_indices,

        "eef_link_index":
            eef_link_index,

        "movable_joint_indices":
            movable_joint_indices,
    }


def set_robot_qpos(
    robot_id,
    arm_indices,
    finger_indices,
    qpos9,
    physics_id,
):
    qpos9 = np.asarray(
        qpos9,
        dtype=np.float64,
    ).reshape(
        -1
    )

    if len(qpos9) != 9:
        raise PlannerError(
            "robot_initial_qpos必须9维，"
            f"实际={len(qpos9)}"
        )

    for index, value in zip(
        arm_indices,
        qpos9[:7],
    ):

        p.resetJointState(
            robot_id,
            index,
            float(value),
            targetVelocity=0.0,
            physicsClientId=physics_id,
        )

    for index, value in zip(
        finger_indices,
        qpos9[7:9],
    ):

        p.resetJointState(
            robot_id,
            index,
            float(value),
            targetVelocity=0.0,
            physicsClientId=physics_id,
        )


def get_arm_qpos(
    robot_id,
    arm_indices,
    physics_id,
):
    return np.asarray(
        [
            p.getJointState(
                robot_id,
                index,
                physicsClientId=physics_id,
            )[0]
            for index
            in arm_indices
        ],
        dtype=np.float64,
    )


def get_link_pose(
    robot_id,
    link_index,
    physics_id,
):
    state = p.getLinkState(
        robot_id,
        link_index,
        computeForwardKinematics=True,
        physicsClientId=physics_id,
    )

    # worldLinkFramePosition /
    # worldLinkFrameOrientation
    position = np.asarray(
        state[4],
        dtype=np.float64,
    )

    quaternion = np.asarray(
        state[5],
        dtype=np.float64,
    )

    return (
        position,
        quaternion,
    )


def get_arm_bounds(
    robot_id,
    arm_indices,
    physics_id,
):
    lower = []
    upper = []

    for index in arm_indices:

        info = p.getJointInfo(
            robot_id,
            index,
            physicsClientId=physics_id,
        )

        lo = float(
            info[8]
        )

        hi = float(
            info[9]
        )

        if (
            not np.isfinite(lo)
            or not np.isfinite(hi)
            or hi <= lo
        ):
            raise PlannerError(
                f"Panda joint range非法: "
                f"index={index}, "
                f"[{lo}, {hi}]"
            )

        # 与ArticuBot PbOMPLRobot保持一致：
        # 非mobile Panda在关节上下界各缩5%。
        margin = (
            0.05
            * (
                hi - lo
            )
        )

        lower.append(
            lo + margin
        )

        upper.append(
            hi - margin
        )

    return (
        np.asarray(
            lower,
            dtype=np.float64,
        ),
        np.asarray(
            upper,
            dtype=np.float64,
        ),
    )


def build_dof_order(
    robot_id,
    physics_id,
):
    """
    PyBullet calculateInverseKinematics 返回：
    所有 non-fixed joints 按joint index顺序的DoF值。
    """

    indices = []

    for index in range(
        p.getNumJoints(
            robot_id,
            physicsClientId=physics_id,
        )
    ):

        info = p.getJointInfo(
            robot_id,
            index,
            physicsClientId=physics_id,
        )

        if int(
            info[2]
        ) != p.JOINT_FIXED:

            indices.append(
                index
            )

    return indices


def solve_collision_free_ik(
    robot_id,
    eef_link_index,
    arm_indices,
    finger_indices,
    planner_interface,
    target_position,
    target_quaternion,
    physics_id,
    *,
    attempts=100,
    seed=0,
):
    rng = np.random.default_rng(
        int(seed)
    )

    lower_arm, upper_arm = (
        get_arm_bounds(
            robot_id,
            arm_indices,
            physics_id,
        )
    )

    dof_order = build_dof_order(
        robot_id,
        physics_id,
    )

    dof_to_pos = {
        joint_index: i
        for i, joint_index
        in enumerate(
            dof_order
        )
    }

    current_arm = get_arm_qpos(
        robot_id,
        arm_indices,
        physics_id,
    )

    # Fingers保持当前状态。
    finger_values = np.asarray(
        [
            p.getJointState(
                robot_id,
                index,
                physicsClientId=physics_id,
            )[0]
            for index
            in finger_indices
        ],
        dtype=np.float64,
    )

    valid_solutions = []

    for attempt in range(
        int(attempts)
    ):

        rest_arm = rng.uniform(
            lower_arm,
            upper_arm,
        )

        # 先把机器人放到随机rest pose，
        # 与ArticuBot原IK多初值做法一致。
        for index, value in zip(
            arm_indices,
            rest_arm,
        ):
            p.resetJointState(
                robot_id,
                index,
                float(value),
                physicsClientId=physics_id,
            )

        for index, value in zip(
            finger_indices,
            finger_values,
        ):
            p.resetJointState(
                robot_id,
                index,
                float(value),
                physicsClientId=physics_id,
            )

        ik = p.calculateInverseKinematics(
            robot_id,
            eef_link_index,
            targetPosition=(
                np.asarray(
                    target_position,
                    dtype=np.float64,
                ).tolist()
            ),
            targetOrientation=(
                np.asarray(
                    target_quaternion,
                    dtype=np.float64,
                ).tolist()
            ),
            maxNumIterations=10000,
            residualThreshold=1e-5,
            physicsClientId=physics_id,
        )

        ik = np.asarray(
            ik,
            dtype=np.float64,
        )

        try:
            arm_solution = np.asarray(
                [
                    ik[
                        dof_to_pos[
                            index
                        ]
                    ]
                    for index
                    in arm_indices
                ],
                dtype=np.float64,
            )

        except Exception as exc:
            raise PlannerError(
                "无法把PyBullet IK返回值映射到7个arm joints"
            ) from exc

        if np.any(
            arm_solution
            < lower_arm
        ):
            continue

        if np.any(
            arm_solution
            > upper_arm
        ):
            continue

        # 使用ArticuBot自己的state validity checker。
        if not planner_interface.is_state_valid(
            arm_solution
        ):
            continue

        # 再做forward kinematics确认误差。
        for index, value in zip(
            arm_indices,
            arm_solution,
        ):
            p.resetJointState(
                robot_id,
                index,
                float(value),
                physicsClientId=physics_id,
            )

        fk_position, fk_quaternion = (
            get_link_pose(
                robot_id,
                eef_link_index,
                physics_id,
            )
        )

        position_error = float(
            np.linalg.norm(
                fk_position
                - np.asarray(
                    target_position,
                    dtype=np.float64,
                )
            )
        )

        quat_dot = float(
            abs(
                np.dot(
                    fk_quaternion,
                    target_quaternion,
                )
            )
        )

        quat_dot = float(
            np.clip(
                quat_dot,
                0.0,
                1.0,
            )
        )

        rotation_error = float(
            2.0
            * math.acos(
                quat_dot
            )
        )

        if position_error > 0.003:
            continue

        if rotation_error > 0.05:
            continue

        distance = float(
            np.linalg.norm(
                arm_solution
                - current_arm
            )
        )

        valid_solutions.append(
            {
                "q":
                    arm_solution.copy(),

                "position_error":
                    position_error,

                "rotation_error":
                    rotation_error,

                "joint_distance":
                    distance,

                "attempt":
                    attempt,
            }
        )

    # 恢复start state
    for index, value in zip(
        arm_indices,
        current_arm,
    ):
        p.resetJointState(
            robot_id,
            index,
            float(value),
            physicsClientId=physics_id,
        )

    for index, value in zip(
        finger_indices,
        finger_values,
    ):
        p.resetJointState(
            robot_id,
            index,
            float(value),
            physicsClientId=physics_id,
        )

    if not valid_solutions:
        raise PlannerError(
            "IK_FAILED: "
            f"{attempts}次采样没有得到collision-free IK"
        )

    valid_solutions.sort(
        key=lambda x:
            (
                x[
                    "joint_distance"
                ],
                x[
                    "position_error"
                ],
                x[
                    "rotation_error"
                ],
            )
    )

    return valid_solutions[0]


def run_self_test(
    args,
):
    row = load_catalog_row(
        args.pose_catalog,
        args.shape_id,
        args.target_link,
    )

    base_pose = np.asarray(
        row[
            "base_pose"
        ],
        dtype=np.float64,
    )

    robot_initial_qpos = np.asarray(
        row[
            "robot_initial_qpos"
        ],
        dtype=np.float64,
    ).reshape(
        -1
    )

    print()
    print("=" * 100)
    print("WHERE2ACT MOTION PLANNER SELF TEST")
    print("=" * 100)

    print(
        "shape:",
        args.shape_id,
    )

    print(
        "target link:",
        args.target_link,
    )

    print(
        "category:",
        row.get(
            "category",
            "unknown",
        ),
    )

    print(
        "Panda URDF:",
        PANDA_URDF,
    )

    print(
        "robot initial qpos:",
        robot_initial_qpos,
    )

    if not PANDA_URDF.exists():
        raise FileNotFoundError(
            PANDA_URDF
        )

    physics_id = p.connect(
        p.DIRECT
    )

    try:

        p.setGravity(
            0,
            0,
            -9.81,
            physicsClientId=physics_id,
        )

        base_position, base_quaternion = (
            pose_to_pybullet(
                base_pose
            )
        )

        robot_id = p.loadURDF(
            str(
                PANDA_URDF
            ),
            basePosition=(
                base_position.tolist()
            ),
            baseOrientation=(
                base_quaternion.tolist()
            ),
            useFixedBase=True,
            flags=(
                p.URDF_USE_INERTIA_FROM_FILE
            ),
            physicsClientId=physics_id,
        )

        robot_info = inspect_robot(
            robot_id,
            physics_id,
        )

        arm_indices = robot_info[
            "arm_indices"
        ]

        finger_indices = robot_info[
            "finger_indices"
        ]

        eef_link_index = robot_info[
            "eef_link_index"
        ]

        print()
        print(
            "arm indices:",
            arm_indices,
        )

        print(
            "finger indices:",
            finger_indices,
        )

        print(
            "EEF link index:",
            eef_link_index,
        )

        set_robot_qpos(
            robot_id,
            arm_indices,
            finger_indices,
            robot_initial_qpos,
            physics_id,
        )

        start_q = get_arm_qpos(
            robot_id,
            arm_indices,
            physics_id,
        )

        start_position, start_quaternion = (
            get_link_pose(
                robot_id,
                eef_link_index,
                physics_id,
            )
        )

        print()
        print("-" * 100)
        print("START")
        print("-" * 100)

        print(
            "start q:",
            start_q,
        )

        print(
            "start EE position:",
            start_position,
        )

        print(
            "start EE quaternion:",
            start_quaternion,
        )

        # ----------------------------------------------------
        # self-test目标：
        # 当前grasptarget沿世界Z方向抬高3cm，
        # orientation完全不变。
        #
        # 不是Where2Act正式目标。
        # ----------------------------------------------------

        goal_position = (
            start_position
            + np.asarray(
                [
                    0.0,
                    0.0,
                    0.03,
                ],
                dtype=np.float64,
            )
        )

        goal_quaternion = (
            start_quaternion.copy()
        )

        print()
        print("-" * 100)
        print("GOAL")
        print("-" * 100)

        print(
            "goal position:",
            goal_position,
        )

        print(
            "goal quaternion:",
            goal_quaternion,
        )

        env = MinimalEnv(
            physics_id
        )

        ompl_robot = PbOMPLRobot(
            robot_id,
            control_joint_idx=(
                arm_indices
            ),
            object_id=None,
            env=env,
        )

        ompl_robot.set_state(
            start_q.tolist()
        )

        planner_interface = PbOMPL(
            ompl_robot,
            obstacles=[],
            allow_collision_links=[],
            allow_collision_robot_link_pairs=[],
            object_id=None,
            interpolation_num=100,
        )

        planner_interface.set_planner(
            args.planner
        )

        ik_result = solve_collision_free_ik(
            robot_id,
            eef_link_index,
            arm_indices,
            finger_indices,
            planner_interface,
            goal_position,
            goal_quaternion,
            physics_id,
            attempts=(
                args.ik_attempts
            ),
            seed=(
                args.seed
            ),
        )

        goal_q = np.asarray(
            ik_result[
                "q"
            ],
            dtype=np.float64,
        )

        print()
        print("-" * 100)
        print("IK")
        print("-" * 100)

        print(
            "IK attempt:",
            ik_result[
                "attempt"
            ],
        )

        print(
            "goal q:",
            goal_q,
        )

        print(
            "IK position error:",
            ik_result[
                "position_error"
            ],
        )

        print(
            "IK rotation error:",
            ik_result[
                "rotation_error"
            ],
        )

        print(
            "joint distance:",
            ik_result[
                "joint_distance"
            ],
        )

        ompl_robot.set_state(
            start_q.tolist()
        )

        solved, path = (
            planner_interface.plan(
                goal_q.tolist(),
                allowed_time=(
                    args.planning_time
                ),
                smooth_path=True,
            )
        )

        if not solved:
            raise PlannerError(
                "OMPL_FAILED"
            )

        trajectory = np.asarray(
            path,
            dtype=np.float64,
        )

        if (
            trajectory.ndim != 2
            or trajectory.shape[1] != 7
        ):
            raise PlannerError(
                "trajectory shape错误: "
                f"{trajectory.shape}"
            )

        if not np.all(
            np.isfinite(
                trajectory
            )
        ):
            raise PlannerError(
                "trajectory包含NaN/Inf"
            )

        # 路径逐点再用ArticuBot collision checker验证。
        invalid_count = 0

        for q in trajectory:

            if not (
                planner_interface
                .is_state_valid(
                    q
                )
            ):
                invalid_count += 1

        if invalid_count != 0:
            raise PlannerError(
                "OMPL输出路径包含非法状态: "
                f"{invalid_count}"
            )

        print()
        print("-" * 100)
        print("OMPL")
        print("-" * 100)

        print(
            "planner:",
            args.planner,
        )

        print(
            "trajectory shape:",
            trajectory.shape,
        )

        print(
            "invalid states:",
            invalid_count,
        )

        print(
            "first q:",
            trajectory[0],
        )

        print(
            "last q:",
            trajectory[-1],
        )

        start_error = float(
            np.linalg.norm(
                trajectory[0]
                - start_q
            )
        )

        goal_error = float(
            np.linalg.norm(
                trajectory[-1]
                - goal_q
            )
        )

        print(
            "trajectory start error:",
            start_error,
        )

        print(
            "trajectory goal error:",
            goal_error,
        )

        output_dir = (
            Path(
                args.output_dir
            )
            .expanduser()
            .resolve()
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            output_dir
            / "trajectory.npy",
            trajectory,
        )

        result = {
            "success":
                True,

            "planner":
                args.planner,

            "shape_id":
                str(
                    args.shape_id
                ),

            "target_link":
                str(
                    args.target_link
                ),

            "start_q":
                start_q.tolist(),

            "goal_q":
                goal_q.tolist(),

            "start_ee_position":
                start_position.tolist(),

            "goal_ee_position":
                goal_position.tolist(),

            "ik_position_error":
                float(
                    ik_result[
                        "position_error"
                    ]
                ),

            "ik_rotation_error":
                float(
                    ik_result[
                        "rotation_error"
                    ]
                ),

            "trajectory_states":
                int(
                    len(
                        trajectory
                    )
                ),

            "trajectory_start_error":
                start_error,

            "trajectory_goal_error":
                goal_error,

            "invalid_states":
                invalid_count,
        }

        with (
            output_dir
            / "result.json"
        ).open(
            "w"
        ) as f:

            json.dump(
                result,
                f,
                indent=2,
            )

        print()
        print(
            "saved:",
            output_dir,
        )

        print()
        print("=" * 100)
        print(
            "WHERE2ACT MOTION PLANNER SELF TEST: PASS"
        )
        print("=" * 100)

    finally:

        p.disconnect(
            physics_id
        )


def build_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pose-catalog",
        required=True,
    )

    parser.add_argument(
        "--shape-id",
        required=True,
    )

    parser.add_argument(
        "--target-link",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--planner",
        default="RRTConnect",
        choices=[
            "RRTConnect",
            "RRTstar",
            "BITstar",
            "ABITstar",
        ],
    )

    parser.add_argument(
        "--planning-time",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--ik-attempts",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    return parser


if __name__ == "__main__":

    args = (
        build_parser()
        .parse_args()
    )

    try:

        run_self_test(
            args
        )

    except Exception:

        print()
        print("=" * 100)
        print(
            "PLANNER IMPLEMENTATION ERROR"
        )
        print("=" * 100)

        traceback.print_exc()

        raise SystemExit(
            1
        )
