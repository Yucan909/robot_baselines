import argparse
import json
import traceback
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from panda_controller import (
    PandaTwoFingerController,
)

from contact_monitor import (
    monitor_grasp_establishment,
    monitor_post_pull_contact,
)

from smoke_execute_planned_trajectory import (
    PANDA_URDF,
    create_scene,
    initialize_object,
    load_catalog_row,
    execute_joint_trajectory,
)


SUCCESS_RATIO = 0.40

GRASP_SETTLE_STEPS = 300
GRASP_TAIL_STEPS = 100
GRASP_MIN_BILATERAL_FRACTION = 0.50

POST_PULL_CONTACT_STEPS = 50

# Where2Act官方pulling primitive：
# final -> start，距离5cm。
PULL_DISTANCE = 0.05

# 官方floating gripper用2000 steps。
# 完整Panda这里保持相同数量级。
APPROACH_SIM_STEPS = 2000
PULL_SIM_STEPS = 2000


class AtomicPullError(RuntimeError):
    pass


def jsonify(
    value,
):
    if isinstance(
        value,
        dict,
    ):
        return {
            str(k): jsonify(v)
            for k, v
            in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            jsonify(v)
            for v in value
        ]

    if isinstance(
        value,
        np.ndarray,
    ):
        return (
            value
            .tolist()
        )

    if isinstance(
        value,
        np.floating,
    ):
        return float(
            value
        )

    if isinstance(
        value,
        np.integer,
    ):
        return int(
            value
        )

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(
            value
        )

    return value


def rotation_error(
    R_target,
    R_actual,
):
    R_target = np.asarray(
        R_target,
        dtype=np.float64,
    )

    R_actual = np.asarray(
        R_actual,
        dtype=np.float64,
    )

    return float(
        np.linalg.norm(
            Rotation
            .from_matrix(
                R_target
                @ R_actual.T
            )
            .as_rotvec()
        )
    )


def controller_step_n(
    panda,
    steps,
):
    for _ in range(
        int(steps)
    ):
        panda.step()


def save_result(
    output_dir,
    result,
):
    output_dir = (
        Path(
            output_dir
        )
        .expanduser()
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        output_dir
        / "result.json"
    )

    with path.open(
        "w"
    ) as f:

        json.dump(
            jsonify(
                result
            ),
            f,
            indent=2,
        )

    print(
        "result:",
        path,
    )

    return path


def run(
    args,
):
    # ========================================================
    # Inputs
    # ========================================================

    request_path = (
        Path(
            args.request
        )
        .expanduser()
        .resolve()
    )

    trajectory_path = (
        Path(
            args.trajectory
        )
        .expanduser()
        .resolve()
    )

    if not request_path.exists():
        raise FileNotFoundError(
            request_path
        )

    if not trajectory_path.exists():
        raise FileNotFoundError(
            trajectory_path
        )

    request = json.loads(
        request_path.read_text()
    )

    trajectory = np.load(
        trajectory_path
    ).astype(
        np.float64
    )

    if (
        trajectory.ndim != 2
        or trajectory.shape[1] != 7
    ):
        raise AtomicPullError(
            "trajectory必须为N×7，"
            f"实际={trajectory.shape}"
        )

    shape_id = str(
        request[
            "shape_id"
        ]
    )

    target_link_name = str(
        request[
            "target_link"
        ]
    )

    initial_ratio = float(
        request[
            "initial_ratio"
        ]
    )

    approach_axis = np.asarray(
        request[
            "approach_axis_world"
        ],
        dtype=np.float64,
    ).reshape(3)

    approach_norm = float(
        np.linalg.norm(
            approach_axis
        )
    )

    if approach_norm <= 1e-8:
        raise AtomicPullError(
            "approach_axis为零"
        )

    approach_axis /= (
        approach_norm
    )

    # Where2Act pulling：
    # final pose -> start pose
    # 即沿预测up轴反方向退5cm。
    pull_direction = (
        -approach_axis
    )

    pull_delta = (
        PULL_DISTANCE
        * pull_direction
    )

    T_pregrasp = np.asarray(
        request[
            "pregrasp_pose_world"
        ],
        dtype=np.float64,
    )

    T_contact = np.asarray(
        request[
            "contact_pose_world"
        ],
        dtype=np.float64,
    )

    if (
        T_pregrasp.shape != (4, 4)
        or T_contact.shape != (4, 4)
    ):
        raise AtomicPullError(
            "request中的pose不是4×4"
        )

    row = load_catalog_row(
        args.pose_catalog,
        shape_id,
        target_link_name,
    )

    base_pose = np.asarray(
        row[
            "base_pose"
        ],
        dtype=np.float64,
    )

    robot_qpos = np.asarray(
        row[
            "robot_initial_qpos"
        ],
        dtype=np.float64,
    )

    print()
    print("=" * 100)
    print(
        "WHERE2ACT ATOMIC PULL SMOKE"
    )
    print("=" * 100)

    print(
        "shape:",
        shape_id,
    )

    print(
        "target:",
        target_link_name,
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

    print(
        "approach axis:",
        approach_axis,
    )

    print(
        "pull direction:",
        pull_direction,
    )

    print(
        "pull delta:",
        pull_delta,
    )

    print(
        "pull distance:",
        np.linalg.norm(
            pull_delta
        ),
    )

    # ========================================================
    # Result skeleton
    # ========================================================

    result = {
        "method":
            "Where2Act",

        "shape_id":
            shape_id,

        "target_link":
            target_link_name,

        "trial_seed":
            int(
                request[
                    "trial_seed"
                ]
            ),

        "network_trained":
            bool(
                request.get(
                    "network_trained",
                    False,
                )
            ),

        "initial_progress_requested":
            initial_ratio,

        "success_threshold":
            SUCCESS_RATIO,

        "approach_axis_world":
            approach_axis,

        "pull_direction_world":
            pull_direction,

        "pull_distance":
            PULL_DISTANCE,

        "grasp_success":
            False,

        "opening_success":
            False,

        "final_success":
            False,

        "operation_success_given_grasp":
            False,

        "failure_reason":
            None,
    }

    # ========================================================
    # Scene + object
    # ========================================================

    (
        engine,
        renderer,
        scene,
    ) = create_scene()

    (
        obj,
        target_joint,
        get_progress,
    ) = initialize_object(
        scene,
        shape_id,
        target_link_name,
        initial_ratio,
    )

    target_link = (
        target_joint
        .get_child_link()
    )

    actual_initial_progress = float(
        get_progress()
    )

    result[
        "initial_progress_actual"
    ] = actual_initial_progress

    print()
    print(
        "actual initial progress:",
        actual_initial_progress,
    )

    if abs(
        actual_initial_progress
        - initial_ratio
    ) > 1e-5:

        raise AtomicPullError(
            "SAPIEN object初始状态不一致"
        )

    # ========================================================
    # Panda
    # ========================================================

    panda = (
        PandaTwoFingerController(
            scene,
            str(
                PANDA_URDF
            ),
        )
    )

    panda.set_initial_state(
        base_pose,
        robot_qpos,
    )

    # 明确保持open。
    panda.open_gripper()

    controller_step_n(
        panda,
        100,
    )

    actual_start_q = np.asarray(
        panda.robot.get_qpos(),
        dtype=np.float64,
    )[:7]

    start_sync_error = float(
        np.linalg.norm(
            actual_start_q
            - trajectory[
                0
            ]
        )
    )

    result[
        "planner_sapien_start_error"
    ] = start_sync_error

    print()
    print(
        "planner/SAPIEN start error:",
        start_sync_error,
    )

    if start_sync_error > 1e-3:

        raise AtomicPullError(
            "Panda起点和OMPL轨迹不一致"
        )

    # ========================================================
    # Stage 1:
    # OMPL trajectory -> pre-grasp
    # ========================================================

    print()
    print("-" * 100)
    print(
        "STAGE 1: OMPL -> PRE-GRASP"
    )
    print("-" * 100)

    max_waypoint_error = (
        execute_joint_trajectory(
            panda,
            scene,
            trajectory,
            15,
            max_steps_per_waypoint=500,
            waypoint_tolerance=0.02,
            waypoint_maxabs_tolerance=0.015,
        )
    )

    panda.clear_arm_velocity()

    T_pregrasp_actual = (
        panda
        .get_grasp_pose_matrix()
    )

    pregrasp_position_error = float(
        np.linalg.norm(
            T_pregrasp_actual[
                :3,
                3
            ]
            - T_pregrasp[
                :3,
                3
            ]
        )
    )

    pregrasp_rotation_error = (
        rotation_error(
            T_pregrasp[
                :3,
                :3
            ],
            T_pregrasp_actual[
                :3,
                :3
            ],
        )
    )

    progress_after_pregrasp = float(
        get_progress()
    )

    result.update(
        {
            "max_waypoint_error":
                max_waypoint_error,

            "pregrasp_position_error":
                pregrasp_position_error,

            "pregrasp_rotation_error":
                pregrasp_rotation_error,

            "progress_after_pregrasp":
                progress_after_pregrasp,
        }
    )

    print(
        "pregrasp position error:",
        pregrasp_position_error,
    )

    print(
        "pregrasp rotation error:",
        pregrasp_rotation_error,
    )

    print(
        "progress after pregrasp:",
        progress_after_pregrasp,
    )

    if (
        pregrasp_position_error
        > 0.02
    ):

        result[
            "failure_reason"
        ] = (
            "pregrasp_position_error"
        )

        save_result(
            args.output_dir,
            result,
        )

        print(
            "ATOMIC PIPELINE COMPLETE: "
            "TASK FAILURE"
        )

        return result

    # ========================================================
    # Stage 2:
    # straight approach
    # ========================================================

    print()
    print("-" * 100)
    print(
        "STAGE 2: STRAIGHT APPROACH"
    )
    print("-" * 100)

    panda.open_gripper()

    approach_diag = (
        panda.move_grasp_pose_to(
            T_contact,
            APPROACH_SIM_STEPS,
            position_tolerance=0.006,
            rotation_tolerance=0.05,
        )
    )

    panda.clear_arm_velocity()

    T_contact_actual = (
        panda
        .get_grasp_pose_matrix()
    )

    contact_position_error = float(
        np.linalg.norm(
            T_contact_actual[
                :3,
                3
            ]
            - T_contact[
                :3,
                3
            ]
        )
    )

    contact_rotation_error = (
        rotation_error(
            T_contact[
                :3,
                :3
            ],
            T_contact_actual[
                :3,
                :3
            ],
        )
    )

    progress_after_approach = float(
        get_progress()
    )

    approach_progress_drift = float(
        abs(
            progress_after_approach
            - initial_ratio
        )
    )

    result.update(
        {
            "approach_diagnostics":
                approach_diag,

            "contact_position_error":
                contact_position_error,

            "contact_rotation_error":
                contact_rotation_error,

            "progress_after_approach":
                progress_after_approach,

            "approach_progress_drift":
                approach_progress_drift,
        }
    )

    print(
        "target contact position:",
        T_contact[
            :3,
            3
        ],
    )

    print(
        "actual grasp center:",
        panda.get_grasp_center(),
    )

    print(
        "contact position error:",
        contact_position_error,
    )

    print(
        "contact rotation error:",
        contact_rotation_error,
    )

    print(
        "progress after approach:",
        progress_after_approach,
    )

    if contact_position_error > 0.02:

        result[
            "failure_reason"
        ] = (
            "final_approach_failed"
        )

        save_result(
            args.output_dir,
            result,
        )

        print(
            "ATOMIC PIPELINE COMPLETE: "
            "TASK FAILURE"
        )

        return result

    # 在正式闭合前，目标部件不应该被接近阶段撞得明显运动。
    if approach_progress_drift > 0.03:

        result[
            "failure_reason"
        ] = (
            "object_moved_before_grasp"
        )

        save_result(
            args.output_dir,
            result,
        )

        print(
            "ATOMIC PIPELINE COMPLETE: "
            "TASK FAILURE"
        )

        return result

    # ========================================================
    # Stage 3:
    # real two-finger grasp
    # ========================================================

    print()
    print("-" * 100)
    print(
        "STAGE 3: REAL TWO-FINGER GRASP"
    )
    print("-" * 100)

    panda.close_gripper()

    grasp_diag = (
        monitor_grasp_establishment(
            panda,
            target_link,
            settle_steps=(
                GRASP_SETTLE_STEPS
            ),
            tail_steps=(
                GRASP_TAIL_STEPS
            ),
            min_bilateral_fraction=(
                GRASP_MIN_BILATERAL_FRACTION
            ),
        )
    )

    finger_qpos = (
        panda.get_finger_qpos()
    )

    progress_after_grasp = float(
        get_progress()
    )

    result.update(
        {
            "grasp_diagnostics":
                grasp_diag,

            "finger_qpos_after_close":
                finger_qpos,

            "progress_after_grasp":
                progress_after_grasp,
        }
    )

    print(
        "firm grasp:",
        grasp_diag[
            "firm_grasp"
        ],
    )

    print(
        "bilateral fraction:",
        grasp_diag[
            "bilateral_fraction"
        ],
    )

    print(
        "left contact fraction:",
        grasp_diag[
            "left_contact_fraction"
        ],
    )

    print(
        "right contact fraction:",
        grasp_diag[
            "right_contact_fraction"
        ],
    )

    print(
        "finger qpos:",
        finger_qpos,
    )

    print(
        "progress after grasp:",
        progress_after_grasp,
    )

    grasp_success = bool(
        grasp_diag[
            "firm_grasp"
        ]
    )

    result[
        "grasp_success"
    ] = grasp_success

    if not grasp_success:

        result[
            "failure_reason"
        ] = (
            "firm_grasp_not_established"
        )

        if not args.engineering_force_pull:

            result[
                "final_progress"
            ] = float(
                get_progress()
            )

            save_result(
                args.output_dir,
                result,
            )

            print()
            print("=" * 100)
            print(
                "WHERE2ACT ATOMIC PIPELINE: "
                "IMPLEMENTATION PASS / TASK GRASP FAILURE"
            )
            print("=" * 100)

            return result

        # ----------------------------------------------------
        # 工程测试专用：
        #
        # 即使随机网络没有建立firm grasp，也继续执行Stage 4/5，
        # 仅用于证明pull branch能完整运行。
        #
        # 注意：
        # grasp_success仍然保持False；
        # final_success绝不会因此变成True。
        # ----------------------------------------------------

        result[
            "engineering_force_pull"
        ] = True

        print()
        print(
            "[ENGINEERING ONLY] "
            "firm grasp失败，但强制继续测试pull branch"
        )

    # ========================================================
    # Stage 4:
    # release target joint
    # ========================================================

    print()
    print("-" * 100)
    print(
        "STAGE 4: RELEASE TARGET JOINT"
    )
    print("-" * 100)

    pre_pull_progress = float(
        get_progress()
    )

    result[
        "pre_pull_progress"
    ] = pre_pull_progress

    # 到这里目标仍应该大致保持原初始状态。
    if abs(
        pre_pull_progress
        - initial_ratio
    ) > 0.05:

        result[
            "failure_reason"
        ] = (
            "initial_state_lost_before_pull"
        )

        result[
            "final_progress"
        ] = pre_pull_progress

        save_result(
            args.output_dir,
            result,
        )

        print(
            "ATOMIC PIPELINE COMPLETE: "
            "TASK FAILURE"
        )

        return result

    target_joint.set_drive_property(
        stiffness=0,
        damping=10,
    )

    target_joint.set_drive_velocity_target(
        0.0
    )

    print(
        "target joint released"
    )

    # ========================================================
    # Stage 5:
    # Where2Act official pulling primitive
    #
    # final pose -> start pose
    # 也就是沿 -predicted_up 后退5cm。
    # ========================================================

    print()
    print("-" * 100)
    print(
        "STAGE 5: WHERE2ACT PULL 5CM"
    )
    print("-" * 100)

    ee_before_pull = (
        panda.get_grasp_center()
    )

    panda.keep_gripper_closed()

    pull_diag = (
        panda.move_grasp_point_by(
            pull_delta,
            PULL_SIM_STEPS,
            control_tolerance=0.005,
        )
    )

    panda.clear_arm_velocity()

    ee_after_pull = (
        panda.get_grasp_center()
    )

    actual_pull_delta = (
        ee_after_pull
        - ee_before_pull
    )

    actual_pull_distance = float(
        np.linalg.norm(
            actual_pull_delta
        )
    )

    contact_after_pull = (
        monitor_post_pull_contact(
            panda,
            target_link,
            steps=(
                POST_PULL_CONTACT_STEPS
            ),
        )
    )

    final_progress = float(
        get_progress()
    )

    opening_success = bool(
        final_progress
        >= SUCCESS_RATIO
    )

    # 为了保证最终成功是“真的先抓住再操作”，
    # final_success 是 grasp_success 的子集。
    final_success = bool(
        grasp_success
        and opening_success
    )

    result.update(
        {
            "pull_diagnostics":
                pull_diag,

            "post_pull_contact":
                contact_after_pull,

            "ee_before_pull":
                ee_before_pull,

            "ee_after_pull":
                ee_after_pull,

            "commanded_pull_delta":
                pull_delta,

            "actual_pull_delta":
                actual_pull_delta,

            "actual_pull_distance":
                actual_pull_distance,

            "final_progress":
                final_progress,

            "opening_success":
                opening_success,

            "operation_success_given_grasp":
                bool(
                    grasp_success
                    and opening_success
                ),

            "final_success":
                final_success,
        }
    )

    if not final_success:

        if not grasp_success:

            result[
                "failure_reason"
            ] = (
                "firm_grasp_not_established"
            )

        elif contact_after_pull[
            "grasp_lost"
        ]:

            result[
                "failure_reason"
            ] = (
                "grasp_lost_during_pull"
            )

        else:

            result[
                "failure_reason"
            ] = (
                "final_progress_below_40pct"
            )

    print()
    print("-" * 100)
    print(
        "FINAL RESULT"
    )
    print("-" * 100)

    print(
        "pre-pull progress:",
        pre_pull_progress,
    )

    print(
        "final progress:",
        final_progress,
    )

    print(
        "opening success:",
        opening_success,
    )

    print(
        "grasp lost:",
        contact_after_pull[
            "grasp_lost"
        ],
    )

    print(
        "actual pull distance:",
        actual_pull_distance,
    )

    print(
        "FINAL SUCCESS:",
        final_success,
    )

    save_result(
        args.output_dir,
        result,
    )

    print()
    print("=" * 100)

    if final_success:

        print(
            "WHERE2ACT ATOMIC PULL: TASK SUCCESS"
        )

    else:

        print(
            "WHERE2ACT ATOMIC PULL: "
            "IMPLEMENTATION PASS / TASK FAILURE"
        )

    print("=" * 100)

    return result


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--request",
        required=True,
    )

    parser.add_argument(
        "--trajectory",
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
        "--engineering-force-pull",
        action="store_true",
        help=(
            "仅用于工程branch smoke。"
            "即使没有firm grasp也继续执行pull；"
            "不得用于正式benchmark。"
        ),
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
            "WHERE2ACT ATOMIC PULL "
            "IMPLEMENTATION ERROR"
        )
        print("=" * 100)

        traceback.print_exc()

        raise SystemExit(
            1
        )
