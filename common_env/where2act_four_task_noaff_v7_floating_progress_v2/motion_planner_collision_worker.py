import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import pybullet as p

from motion_planner_worker import (
    PANDA_URDF,
    PlannerError,
    MinimalEnv,
    load_catalog_row,
    pose_to_pybullet,
    inspect_robot,
    set_robot_qpos,
    get_arm_qpos,
    get_link_pose,
    rotation_matrix_to_quaternion_xyzw,
    solve_collision_free_ik,
    PbOMPLRobot,
    PbOMPL,
)


PARTNET_ROOT = (
    Path.home()
    / "robot_baselines"
    / "data"
    / "partnet-mobility"
)

OBJECT_SCALE = 0.75


def find_object_urdf(
    shape_id,
):
    root = (
        PARTNET_ROOT
        / str(shape_id)
    )

    candidates = [
        root / "mobility.urdf",
        root / "mobility_vhacd.urdf",
    ]

    for path in candidates:

        if path.exists():
            return path

    raise FileNotFoundError(
        f"找不到PartNet URDF: {root}"
    )


def load_partnet_object(
    physics_id,
    shape_id,
    target_link_name,
    initial_ratio=None,
    initial_object_qpos=None,
):
    exact_qpos = None
    if initial_object_qpos is not None:
        exact_qpos = np.asarray(
            initial_object_qpos,
            dtype=np.float64,
        ).reshape(-1)
        if not np.all(np.isfinite(exact_qpos)):
            raise PlannerError(
                "initial_object_qpos 包含 NaN/Inf"
            )
    urdf = find_object_urdf(
        shape_id
    )

    print()
    print("-" * 100)
    print("PYBULLET PARTNET MIRROR")
    print("-" * 100)

    print(
        "URDF:",
        urdf,
    )

    print(
        "scale:",
        OBJECT_SCALE,
    )

    object_id = p.loadURDF(
        str(
            urdf
        ),
        basePosition=[
            0.0,
            0.0,
            0.0,
        ],
        baseOrientation=[
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        useFixedBase=True,
        globalScaling=(
            OBJECT_SCALE
        ),
        flags=(
            p.URDF_USE_INERTIA_FROM_FILE
        ),
        physicsClientId=physics_id,
    )

    if object_id < 0:
        raise PlannerError(
            "PyBullet加载PartNet失败"
        )

    num_joints = p.getNumJoints(
        object_id,
        physicsClientId=physics_id,
    )

    target_joint_index = None
    target_limits = None
    target_q = None
    active_counter = 0

    for i in range(
        num_joints
    ):

        info = p.getJointInfo(
            object_id,
            i,
            physicsClientId=physics_id,
        )

        joint_name = (
            info[1]
            .decode("utf-8")
        )

        joint_type = int(
            info[2]
        )

        lower = float(
            info[8]
        )

        upper = float(
            info[9]
        )

        child_link_name = (
            info[12]
            .decode("utf-8")
        )

        if (
            joint_type
            == p.JOINT_FIXED
        ):

            print(
                f"[{i:02d}] "
                f"{joint_name:<20} "
                f"child={child_link_name:<20} "
                f"FIXED"
            )

            continue

        # --------------------------------------------
        # benchmark closed state：
        #
        # 有有限lower：
        #     lower
        #
        # 无有限lower：
        #     0
        # --------------------------------------------

        if exact_qpos is not None:
            if active_counter >= len(exact_qpos):
                raise PlannerError(
                    "initial_object_qpos 长度过短"
                )
            q = float(exact_qpos[active_counter])
            has_finite_range = bool(
                np.isfinite(lower)
                and np.isfinite(upper)
                and upper >= lower
            )
            if has_finite_range and q < lower - 1e-5:
                raise PlannerError(
                    "initial_object_qpos 低于 joint limit"
                )
            if has_finite_range and q > upper + 1e-5:
                raise PlannerError(
                    "initial_object_qpos 高于 joint limit"
                )
        elif np.isfinite(lower):
            q = lower
        else:
            q = 0.0

        role = "closed"

        # --------------------------------------------
        # target joint通过child link name确定
        # --------------------------------------------

        if (
            child_link_name
            == str(
                target_link_name
            )
        ):

            if (
                not np.isfinite(lower)
                or not np.isfinite(upper)
                or upper <= lower
            ):
                raise PlannerError(
                    "target joint没有有限range: "
                    f"{joint_name} "
                    f"[{lower}, {upper}]"
                )

            if exact_qpos is None:
                q = (
                    lower
                    + float(
                        initial_ratio
                    )
                    * (
                        upper
                        - lower
                    )
                )

            target_joint_index = i

            target_limits = (
                lower,
                upper,
            )

            target_q = float(
                q
            )

            role = "TARGET"

        p.resetJointState(
            object_id,
            i,
            float(q),
            targetVelocity=0.0,
            physicsClientId=physics_id,
        )

        print(
            f"[{i:02d}] "
            f"{joint_name:<20} "
            f"child={child_link_name:<20} "
            f"limits=[{lower}, {upper}] "
            f"q={q:.6f} "
            f"{role}"
        )

        active_counter += 1

    if (
        exact_qpos is not None
        and active_counter != len(exact_qpos)
    ):
        raise PlannerError(
            "initial_object_qpos 长度与非固定关节数不一致"
        )

    if target_joint_index is None:

        raise PlannerError(
            "PyBullet PartNet里找不到"
            f"target child link: "
            f"{target_link_name}"
        )

    actual_q = float(
        p.getJointState(
            object_id,
            target_joint_index,
            physicsClientId=physics_id,
        )[0]
    )

    lower, upper = (
        target_limits
    )

    actual_ratio = (
        actual_q
        - lower
    ) / (
        upper
        - lower
    )

    print()
    print(
        "target joint index:",
        target_joint_index,
    )

    print(
        "requested initial ratio:",
        initial_ratio,
    )

    print(
        "actual initial ratio:",
        actual_ratio,
    )

    if (
        initial_ratio is not None
        and abs(
            actual_ratio
            - float(initial_ratio)
        ) > 1e-5
    ):

        raise PlannerError(
            "PyBullet object初始开度不一致"
        )

    return {
        "object_id":
            object_id,

        "target_joint_index":
            target_joint_index,

        "target_limits":
            target_limits,

        "target_q":
            target_q,

        "actual_ratio":
            actual_ratio,

        "urdf":
            str(
                urdf
            ),
    }


def run(
    args,
):
    request_path = (
        Path(
            args.request
        )
        .expanduser()
        .resolve()
    )

    request = json.loads(
        request_path.read_text()
    )

    shape_id = str(
        request[
            "shape_id"
        ]
    )

    target_link = str(
        request[
            "target_link"
        ]
    )

    initial_ratio = float(
        request[
            "initial_ratio"
        ]
    )

    initial_object_qpos = request.get(
        "initial_object_qpos"
    )

    if (
        initial_object_qpos is None
        and not (
            0.10
            <= initial_ratio
            <= 0.20
        )
    ):
        raise PlannerError(
            "request initial_ratio不在"
            "[0.10,0.20]"
        )

    T_goal = np.asarray(
        request[
            "pregrasp_pose_world"
        ],
        dtype=np.float64,
    )

    if T_goal.shape != (
        4,
        4,
    ):
        raise PlannerError(
            "pregrasp_pose_world不是4x4"
        )

    row = load_catalog_row(
        args.pose_catalog,
        shape_id,
        target_link,
    )

    base_pose = np.asarray(
        row[
            "base_pose"
        ],
        dtype=np.float64,
    )

    robot_qpos9 = np.asarray(
        row[
            "robot_initial_qpos"
        ],
        dtype=np.float64,
    )

    start_q_source = "pose_catalog"

    # Backend v2.1:
    # 正式 runtime 若提供 SAPIEN 当前 Panda 状态，
    # planner 必须从完全相同的状态开始。
    if "robot_start_qpos9" in request:

        robot_qpos9 = np.asarray(
            request[
                "robot_start_qpos9"
            ],
            dtype=np.float64,
        ).reshape(-1)

        if (
            robot_qpos9.shape != (9,)
            or not np.all(
                np.isfinite(
                    robot_qpos9
                )
            )
        ):
            raise PlannerError(
                "robot_start_qpos9 非法"
            )

        start_q_source = (
            "sapien_runtime"
        )

    print()
    print("=" * 100)
    print(
        "WHERE2ACT COLLISION-AWARE OMPL"
    )
    print("=" * 100)

    print(
        "shape:",
        shape_id,
    )

    print(
        "target:",
        target_link,
    )

    print(
        "initial ratio:",
        initial_ratio,
    )

    print(
        "network trained:",
        request.get(
            "network_trained"
        ),
    )

    cid = p.connect(
        p.DIRECT
    )

    try:

        p.setGravity(
            0,
            0,
            -9.81,
            physicsClientId=cid,
        )

        # ====================================================
        # Panda
        # ====================================================

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
            physicsClientId=cid,
        )

        robot_info = inspect_robot(
            robot_id,
            cid,
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

        set_robot_qpos(
            robot_id,
            arm_indices,
            finger_indices,
            robot_qpos9,
            cid,
        )

        start_q = get_arm_qpos(
            robot_id,
            arm_indices,
            cid,
        )

        start_ee_position, _ = (
            get_link_pose(
                robot_id,
                eef_link_index,
                cid,
            )
        )

        # ====================================================
        # PartNet mirror
        # ====================================================

        object_info = (
            load_partnet_object(
                cid,
                shape_id,
                target_link,
                initial_ratio,
                initial_object_qpos,
            )
        )

        object_id = object_info[
            "object_id"
        ]

        # ====================================================
        # Goal
        # ====================================================

        goal_position = (
            T_goal[
                :3,
                3
            ]
        )

        goal_quaternion = (
            rotation_matrix_to_quaternion_xyzw(
                T_goal[
                    :3,
                    :3
                ]
            )
        )

        print()
        print("-" * 100)
        print("PREGRASP TARGET")
        print("-" * 100)

        print(
            "start EE:",
            start_ee_position,
        )

        print(
            "goal position:",
            goal_position,
        )

        print(
            "Cartesian distance:",
            float(
                np.linalg.norm(
                    goal_position
                    - start_ee_position
                )
            ),
        )

        # ====================================================
        # ArticuBot OMPL
        #
        # 核心区别：
        #
        #     obstacles=[object_id]
        #
        # 从这里开始所有state validity都会检查：
        #
        # Panda self collision
        # +
        # Panda vs PartNet collision
        # ====================================================

        env = MinimalEnv(
            cid
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
            obstacles=[
                object_id
            ],
            allow_collision_links=[],
            allow_collision_robot_link_pairs=[],
            object_id=None,
            interpolation_num=100,
        )

        planner_interface.set_planner(
            args.planner
        )

        print()
        print(
            "start state valid:",
            planner_interface
            .is_state_valid(
                start_q
            ),
        )

        # ====================================================
        # Collision-free IK
        # ====================================================

        ik = solve_collision_free_ik(
            robot_id,
            eef_link_index,
            arm_indices,
            finger_indices,
            planner_interface,
            goal_position,
            goal_quaternion,
            cid,
            attempts=(
                args.ik_attempts
            ),
            seed=(
                args.seed
            ),
        )

        goal_q = np.asarray(
            ik[
                "q"
            ],
            dtype=np.float64,
        )

        print()
        print("-" * 100)
        print("COLLISION-FREE IK")
        print("-" * 100)

        print(
            "goal q:",
            goal_q,
        )

        print(
            "position error:",
            ik[
                "position_error"
            ],
        )

        print(
            "rotation error:",
            ik[
                "rotation_error"
            ],
        )

        print(
            "goal state valid:",
            planner_interface
            .is_state_valid(
                goal_q
            ),
        )

        # ====================================================
        # OMPL
        # ====================================================

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
                "COLLISION_AWARE_OMPL_FAILED"
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
                "trajectory shape异常: "
                f"{trajectory.shape}"
            )

        invalid_states = []

        for i, q in enumerate(
            trajectory
        ):

            if not (
                planner_interface
                .is_state_valid(
                    q
                )
            ):

                invalid_states.append(
                    i
                )

        if invalid_states:

            raise PlannerError(
                "OMPL路径包含碰撞state: "
                f"{invalid_states[:20]}"
            )

        print()
        print("-" * 100)
        print("COLLISION-AWARE OMPL RESULT")
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
            len(
                invalid_states
            ),
        )

        print(
            "trajectory start:",
            trajectory[
                0
            ],
        )

        print(
            "trajectory goal:",
            trajectory[
                -1
            ],
        )

        # ====================================================
        # Save
        # ====================================================

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

            "shape_id":
                shape_id,

            "target_link":
                target_link,

            "initial_ratio":
                initial_ratio,

            "planner":
                args.planner,

            "object_urdf":
                object_info[
                    "urdf"
                ],

            "object_scale":
                OBJECT_SCALE,

            "start_q":
                start_q.tolist(),

            "start_q_source":
                start_q_source,

            "goal_q":
                goal_q.tolist(),

            "trajectory_states":
                int(
                    len(
                        trajectory
                    )
                ),

            "invalid_states":
                int(
                    len(
                        invalid_states
                    )
                ),

            "ik_position_error":
                float(
                    ik[
                        "position_error"
                    ]
                ),

            "ik_rotation_error":
                float(
                    ik[
                        "rotation_error"
                    ]
                ),

            "pregrasp_pose_world":
                T_goal.tolist(),

            "contact_pose_world":
                request[
                    "contact_pose_world"
                ],

            "approach_axis_world":
                request[
                    "approach_axis_world"
                ],
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
            "WHERE2ACT COLLISION-AWARE OMPL: PASS"
        )
        print("=" * 100)

    finally:

        p.disconnect(
            cid
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--request",
        required=True,
    )

    parser.add_argument(
        "--pose-catalog",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--planner",
        default="RRTConnect",
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

    args = parser.parse_args()

    try:

        run(
            args
        )

    except Exception:

        print()
        print("=" * 100)
        print(
            "COLLISION PLANNER ERROR"
        )
        print("=" * 100)

        traceback.print_exc()

        raise SystemExit(
            1
        )
