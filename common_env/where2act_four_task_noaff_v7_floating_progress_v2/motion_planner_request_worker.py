import argparse
import json
import math
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


def quaternion_error(
    q1,
    q2,
):
    q1 = np.asarray(
        q1,
        dtype=np.float64,
    )

    q2 = np.asarray(
        q2,
        dtype=np.float64,
    )

    dot = float(
        abs(
            np.dot(
                q1,
                q2,
            )
        )
    )

    dot = float(
        np.clip(
            dot,
            0.0,
            1.0,
        )
    )

    return float(
        2.0
        * math.acos(
            dot
        )
    )


def run(args):

    request_path = (
        Path(
            args.request
        )
        .expanduser()
        .resolve()
    )

    with request_path.open() as f:
        request = json.load(f)

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

    if (
        request.get(
            "goal_frame"
        )
        != "panda_grasptarget"
    ):
        raise PlannerError(
            "request goal_frame必须是 "
            "panda_grasptarget"
        )

    T_goal = np.asarray(
        request[
            "pregrasp_pose_world"
        ],
        dtype=np.float64,
    )

    if T_goal.shape != (4, 4):
        raise PlannerError(
            "pregrasp_pose_world "
            "不是4x4"
        )

    R_goal = T_goal[
        :3,
        :3
    ]

    if abs(
        np.linalg.det(
            R_goal
        )
        - 1.0
    ) > 1e-4:
        raise PlannerError(
            "goal rotation det异常"
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

    qpos9 = np.asarray(
        row[
            "robot_initial_qpos"
        ],
        dtype=np.float64,
    )

    print()
    print("=" * 100)
    print("WHERE2ACT -> PANDA -> OMPL")
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
        "network trained:",
        request.get(
            "network_trained"
        ),
    )

    print(
        "pregrasp position:",
        T_goal[
            :3,
            3
        ],
    )

    print(
        "approach axis:",
        request[
            "approach_axis_world"
        ],
    )

    cid = p.connect(
        p.DIRECT
    )

    try:

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

        info = inspect_robot(
            robot_id,
            cid,
        )

        arm_indices = info[
            "arm_indices"
        ]

        finger_indices = info[
            "finger_indices"
        ]

        eef_link_index = info[
            "eef_link_index"
        ]

        set_robot_qpos(
            robot_id,
            arm_indices,
            finger_indices,
            qpos9,
            cid,
        )

        start_q = get_arm_qpos(
            robot_id,
            arm_indices,
            cid,
        )

        start_position, start_quaternion = (
            get_link_pose(
                robot_id,
                eef_link_index,
                cid,
            )
        )

        goal_position = T_goal[
            :3,
            3
        ].copy()

        goal_quaternion = (
            rotation_matrix_to_quaternion_xyzw(
                R_goal
            )
        )

        print()
        print("-" * 100)
        print("START")
        print("-" * 100)

        print(
            "q:",
            start_q,
        )

        print(
            "EE position:",
            start_position,
        )

        print()
        print("-" * 100)
        print("TARGET PREGRASP")
        print("-" * 100)

        print(
            "position:",
            goal_position,
        )

        print(
            "quaternion xyzw:",
            goal_quaternion,
        )

        print(
            "Cartesian distance:",
            float(
                np.linalg.norm(
                    goal_position
                    - start_position
                )
            ),
        )

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
            obstacles=[],
            allow_collision_links=[],
            allow_collision_robot_link_pairs=[],
            object_id=None,
            interpolation_num=100,
        )

        planner_interface.set_planner(
            args.planner
        )

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
        print("IK")
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

        if trajectory.shape != (
            100,
            7,
        ):
            if (
                trajectory.ndim != 2
                or trajectory.shape[1] != 7
            ):
                raise PlannerError(
                    "trajectory shape异常: "
                    f"{trajectory.shape}"
                )

        invalid = 0

        for q in trajectory:

            if not (
                planner_interface
                .is_state_valid(
                    q
                )
            ):
                invalid += 1

        if invalid:
            raise PlannerError(
                "trajectory包含collision状态: "
                f"{invalid}"
            )

        # 最终FK验证
        ompl_robot.set_state(
            goal_q.tolist()
        )

        final_position, final_quaternion = (
            get_link_pose(
                robot_id,
                eef_link_index,
                cid,
            )
        )

        final_pos_error = float(
            np.linalg.norm(
                final_position
                - goal_position
            )
        )

        final_rot_error = (
            quaternion_error(
                final_quaternion,
                goal_quaternion,
            )
        )

        print()
        print("-" * 100)
        print("OMPL RESULT")
        print("-" * 100)

        print(
            "planner:",
            args.planner,
        )

        print(
            "trajectory:",
            trajectory.shape,
        )

        print(
            "invalid states:",
            invalid,
        )

        print(
            "final position error:",
            final_pos_error,
        )

        print(
            "final rotation error:",
            final_rot_error,
        )

        out = (
            Path(
                args.output_dir
            )
            .expanduser()
            .resolve()
        )

        out.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            out
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

            "planner":
                args.planner,

            "start_q":
                start_q.tolist(),

            "goal_q":
                goal_q.tolist(),

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

            "trajectory_states":
                int(
                    len(
                        trajectory
                    )
                ),

            "invalid_states":
                int(
                    invalid
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

            "final_position_error":
                final_pos_error,

            "final_rotation_error":
                final_rot_error,
        }

        with (
            out
            / "result.json"
        ).open(
            "w"
        ) as f:

            json.dump(
                result,
                f,
                indent=2,
            )

        print(
            "saved:",
            out,
        )

        print()
        print("=" * 100)
        print(
            "WHERE2ACT -> OMPL REQUEST: PASS"
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
            "PLANNER REQUEST ERROR"
        )
        print("=" * 100)

        traceback.print_exc()

        raise SystemExit(
            1
        )
