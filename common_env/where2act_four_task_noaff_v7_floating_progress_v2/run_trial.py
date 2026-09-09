import argparse
import fcntl
import gc
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from where2act_policy import (
    Where2ActPolicy,
)

from where2act_observation_adapter import (
    capture_where2act_observation,
)

from floating_gripper_controller import (
    FloatingPandaTwoFingerController,
    PANDA_GRIPPER_URDF,
)

from contact_monitor import (
    monitor_grasp_establishment,
    monitor_target_engagement,
)

from progress_operation import (
    OPEN_COMMAND_PROGRESS,
    OPEN_SUCCESS_PROGRESS,
    CLOSE_COMMAND_PROGRESS,
    CLOSE_SUCCESS_PROGRESS,
    OPERATION_SEGMENT_DISTANCE_M,
    OPERATION_MAX_TRAVEL_M,
    command_progress_for_goal,
    success_progress_for_goal,
    command_target_reached,
    final_state_success,
    execute_progress_targeted_operation,
)

from backend_v2_physics import (
    FINGER_STIFFNESS,
    FINGER_DAMPING,
    CONTACT_STATIC_FRICTION,
    CONTACT_DYNAMIC_FRICTION,
    CONTACT_RESTITUTION,
    TARGET_LOCK_STIFFNESS,
    TARGET_LOCK_DAMPING,
    TARGET_FREE_STIFFNESS,
    TARGET_FREE_DAMPING,
    set_target_free,
    lock_target_to_q,
    apply_target_contact_material,
)

from where2act_runtime import (
    APPROACH_DISTANCE,
    create_scene,
    initialize_object,
    get_progress,
    create_fixed_camera,
    load_catalog_row,
    sample_initial_ratio,
    rotation_error,
    save_json,
    Where2ActRuntimeError,
)


GRASP_SETTLE_STEPS = 300
GRASP_TAIL_STEPS = 100
GRASP_MIN_BILATERAL_FRACTION = 0.50

PUSH_ENGAGEMENT_STEPS = 50

APPROACH_SIM_STEPS = 2000

TASK_SPECS = {
    "door_open": {"primitive": "pull", "goal": "open"},
    "door_close": {"primitive": "push", "goal": "close"},
    "drawer_open": {"primitive": "pull", "goal": "open"},
    "drawer_close": {"primitive": "push", "goal": "close"},
}
FLOATING_BACKEND_VERSION = "where2act_floating_gripper_progress_targeted_20260905_v2"
FLOATING_POSE_STIFFNESS = 1000.0
FLOATING_POSE_DAMPING = 400.0
FLOATING_FINGER_STIFFNESS = 200.0
FLOATING_FINGER_DAMPING = 60.0
FLOATING_STATIC_FRICTION = 4.0
FLOATING_DYNAMIC_FRICTION = 4.0
FLOATING_RESTITUTION = 0.01


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _failure_category(detail, primitive):
    if detail is None:
        return None
    if detail in {
        "model_no_valid_action", "planning_failed", "approach_failed",
        "no_target_contact", "grasp_failed", "lost_contact",
        "operation_motion_failed", "insufficient_directional_progress",
        "runtime_error",
    }:
        return detail
    if detail in {"target_not_visible", "target_not_sampled", "policy_invalid_output"}:
        return "model_no_valid_action"
    if detail in {"pregrasp_execution_failed"}:
        return "approach_failed"
    if detail == "firm_grasp_not_established":
        return "grasp_failed" if primitive == "pull" else "no_target_contact"
    if detail in {"grasp_lost_during_pull", "target_contact_lost_during_push"}:
        return "lost_contact"
    if detail in {
        "final_progress_below_open_35pct",
        "final_progress_above_close_15pct",
        "target_progress_not_reached",
    }:
        return "insufficient_directional_progress"
    if "implementation" in str(detail) or "runtime" in str(detail):
        return "runtime_error"
    return "operation_motion_failed"

# 学长新底层：
#
# OPEN 阶段 target 是 free。
OPEN_FREE_WAIT_STEPS = 150

# CLOSE 阶段 target 重新稳定。
#
# 随后再释放 target，单独运行原来的 300-step
# bilateral physical grasp validation。
CLOSE_LOCK_STEPS = 300


def _expected_failure_reason(exc):
    name = type(exc).__name__
    text = f"{name}: {exc}"

    if "target_not_visible" in text:
        return "target_not_visible"

    if "target_not_sampled" in text:
        return "target_not_sampled"

    if name == "Where2ActPolicyError":
        return "policy_invalid_output"

    if (
        "SAPIEN_WAYPOINT_TRACKING_FAILED" in text
        or "SAPIEN_FINAL_TRACKING_FAILED" in text
    ):
        return "pregrasp_execution_failed"

    return None


def run_trial(args):

    start_time = time.time()

    if args.task not in TASK_SPECS:
        raise ValueError(f"unsupported task: {args.task}")
    primitive = TASK_SPECS[args.task]["primitive"]
    goal = TASK_SPECS[args.task]["goal"]

    # ========================================================
    # Strict checkpoint policy
    # ========================================================

    checkpoint = None

    if args.checkpoint is not None:

        checkpoint = (
            Path(args.checkpoint)
            .expanduser()
            .resolve()
        )

        if not checkpoint.exists():

            raise FileNotFoundError(
                checkpoint
            )

    if (
        checkpoint is None
        and not args.engineering_allow_untrained
    ):

        raise Where2ActRuntimeError(
            "CHECKPOINT_REQUIRED: "
            "正式run_trial禁止随机权重。"
        )

    if (
        args.engineering_force_pull
        and not args.engineering_allow_untrained
    ):

        raise Where2ActRuntimeError(
            "ENGINEERING_FORCE_PULL_REQUIRES_UNTRAINED_SMOKE_MODE"
        )

    # ========================================================
    # Seed
    # ========================================================

    seed = int(
        args.trial_seed
    )

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    # ========================================================
    # Benchmark row
    # ========================================================

    row = load_catalog_row(
        args.formal_catalog,
        args.shape_id,
        args.target_link,
    )

    if str(row.get("task")) != args.task:
        raise Where2ActRuntimeError("FORMAL_CATALOG_TASK_MISMATCH")
    if str(row.get("primitive")) != primitive:
        raise Where2ActRuntimeError("FORMAL_CATALOG_PRIMITIVE_MISMATCH")

    initial_object_qpos = row.get("initial_object_qpos")

    if initial_object_qpos is not None:
        if args.initial_ratio is not None:
            raise ValueError("exact initial_object_qpos forbids --initial-ratio override")
        initial_ratio = None
    elif args.initial_ratio is None:

        initial_ratio = (
            sample_initial_ratio(
                seed
            )
        )

    else:

        initial_ratio = float(
            args.initial_ratio
        )

    if initial_ratio is not None and not (
        0.10 <= initial_ratio <= 0.20
    ):

        raise ValueError(
            "initial_ratio必须位于"
            "[0.10,0.20]"
        )

    trial_dir = (
        Path(args.output_root)
        .expanduser()
        .resolve()
        / args.task
        / (
            f"{args.shape_id}_"
            f"{args.target_link}"
        )
        / (
            f"seed_{seed:03d}"
        )
    )

    trial_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path = (
        trial_dir
        / "result.json"
    )

    result = {
        "method":
            "Where2Act",

        "protocol_version":
            "where2act_four_task_noaff_v7_floating_progress_v2",

        "backend_version":
            FLOATING_BACKEND_VERSION,

        "shape_id":
            str(
                args.shape_id
            ),

        "task":
            args.task,

        "primitive":
            primitive,

        "goal":
            goal,

        "trial_id":
            f"{args.shape_id}_{args.target_link}_{seed}",

        "seed":
            seed,

        "target_link":
            str(
                args.target_link
            ),

        "category":
            row.get(
                "category",
                "unknown",
            ),

        "trial_seed":
            seed,

        "initial_ratio_requested":
            initial_ratio,

        "success_progress_threshold":
            success_progress_for_goal(goal),

        "command_progress_target":
            command_progress_for_goal(goal),

        "success_definition":
            (
                "operation_executed AND final_progress >= 0.35"
                if goal == "open"
                else "operation_executed AND final_progress <= 0.15"
            ),

        "checkpoint":
            (
                str(checkpoint)
                if checkpoint
                is not None
                else None
            ),

        "checkpoint_sha256":
            _sha256(checkpoint) if checkpoint is not None else None,

        "scene_source":
            row.get("scene_source"),

        "camera_source":
            row.get("camera_source"),

        "camera_source_path":
            row.get("camera_source_path"),

        "base_pose_source":
            row.get("base_pose_source"),

        "articulation_state_source":
            row.get("articulation_state_source"),

        "engineering_allow_untrained":
            bool(
                args.engineering_allow_untrained
            ),

        "engineering_force_pull_requested":
            bool(
                args.engineering_force_pull
            ),

        "backend_physics":
            {
                "profile":
                    "official_floating_panda_gripper_real_contact_no_weld",

                "floating_pose_pd":
                    [
                        FLOATING_POSE_STIFFNESS,
                        FLOATING_POSE_DAMPING,
                    ],

                "finger_pd":
                    [
                        FLOATING_FINGER_STIFFNESS,
                        FLOATING_FINGER_DAMPING,
                    ],

                "contact_material":
                    {
                        "static_friction":
                            FLOATING_STATIC_FRICTION,

                        "dynamic_friction":
                            FLOATING_DYNAMIC_FRICTION,

                        "restitution":
                            FLOATING_RESTITUTION,
                    },

                "target_lock_drive":
                    [
                        TARGET_LOCK_STIFFNESS,
                        TARGET_LOCK_DAMPING,
                    ],

                "target_free_drive":
                    [
                        TARGET_FREE_STIFFNESS,
                        TARGET_FREE_DAMPING,
                    ],

                "init_to_pregrasp":
                    "direct floating-root placement; no arm IK or path planning",

                "open_to_grasp":
                    "target_free",

                "close":
                    "target_locked",

                "hold":
                    "target_free",

                "operate":
                    "target_free",

                "suction":
                    False,

                "five_cm_suction_tolerance":
                    False,

                "finger_contact_weld":
                    False,

                "gripper_object_weld":
                    False,

                "gt_target_pose_live_tracking":
                    False,

                "full_arm_present":
                    False,

                "ik_or_ompl_used":
                    False,
            },

        "observation_success":
            False,

        "policy_success":
            False,

        "planning_success":
            False,

        "pregrasp_execution_success":
            False,

        "grasp_success":
            False,

        "predicted_contact_point":
            None,

        "predicted_qidx":
            None,

        "predicted_action_orientation":
            None,

        "predicted_action_direction":
            None,

        "operation_attempted_after_grasp":
            False,

        "engineering_operation_executed":
            False,

        "operation_executed":
            False,

        "opening_success":
            False,

        "operation_success":
            False,

        "operation_success_given_grasp":
            False,

        "final_operation_success":
            False,

        "final_state_threshold_met":
            False,

        "command_target_reached":
            False,

        "final_success":
            False,

        "failure_reason":
            None,

        "implementation_error":
            False,
    }

    object_info = None

    def persist_result():
        if object_info is not None:
            qpos = np.asarray(
                object_info["object"].get_qpos(),
                dtype=np.float64,
            ).reshape(-1)
            final_progress = float(get_progress(object_info))
            result["final_articulation_q"] = qpos.tolist()
            result["final_target_q"] = float(qpos[object_info["target_index"]])
            result["final_progress"] = final_progress
            result["final_state_threshold_met"] = bool(
                final_state_success(goal, final_progress)
            )
            result["command_target_reached"] = bool(
                command_target_reached(goal, final_progress)
            )
            if result.get("initial_progress") is not None:
                if goal == "open":
                    directional = final_progress - float(result["initial_progress"])
                else:
                    directional = float(result["initial_progress"]) - final_progress
                result["directional_task_progress"] = float(directional)
        detail = result.get("failure_reason")
        result["failure_detail"] = detail
        result["failure_reason"] = _failure_category(detail, primitive)
        result["runtime_seconds"] = float(time.time() - start_time)
        save_json(result_path, result)

    try:

        print()
        print("=" * 100)
        print(
            "WHERE2ACT FOUR-TASK NO-AFF V7 FORMAL RUN TRIAL"
        )
        print("=" * 100)

        print(
            "shape:",
            args.shape_id,
        )

        print(
            "target:",
            args.target_link,
        )

        print(
            "seed:",
            seed,
        )

        print(
            "initial ratio:",
            initial_ratio,
        )

        print(
            "checkpoint:",
            checkpoint,
        )

        # ====================================================
        # Scene
        # ====================================================

        (
            engine,
            renderer,
            scene,
        ) = create_scene()

        # ====================================================
        # Object
        # ====================================================

        object_info = (
            initialize_object(
                scene,
                args.shape_id,
                args.target_link,
                initial_ratio,
                initial_object_qpos,
            )
        )

        initial_ratio = float(object_info["initial_ratio"])
        result["initial_ratio_actual"] = initial_ratio

        obj = object_info[
            "object"
        ]

        target_joint = object_info[
            "target_joint"
        ]

        target_link = object_info[
            "target_link"
        ]

        target_index = int(
            object_info[
                "target_index"
            ]
        )

        initial_target_q = float(
            np.asarray(
                obj.get_qpos(),
                dtype=np.float64,
            )[
                target_index
            ]
        )

        result[
            "initial_target_q"
        ] = initial_target_q

        initial_qpos_actual = np.asarray(
            obj.get_qpos(),
            dtype=np.float64,
        ).reshape(-1)
        result["initial_articulation_q"] = initial_qpos_actual.tolist()

        initial_progress = float(
            get_progress(
                object_info
            )
        )

        result["initial_progress_actual"] = initial_progress
        result["initial_progress"] = initial_progress

        print(
            "actual initial progress:",
            initial_progress,
        )

        # ====================================================
        # Fixed benchmark camera
        #
        # The paper-style floating gripper is spawned only after inference,
        # matching the original Where2Act observation/execution ordering.
        # ====================================================

        (
            camera_mount,
            camera,
        ) = create_fixed_camera(
            scene,
            row[
                "camera_pose_world"
            ],
        )

        # ====================================================
        # Observation
        # ====================================================

        print()
        print(
            "[1/7] observation"
        )

        observation = (
            capture_where2act_observation(
                scene,
                camera,
                obj,
                target_link,
                object_origin_world=np.zeros(
                    3,
                    dtype=np.float64,
                ),
            )
        )

        diag = observation[
            "diagnostics"
        ]

        result[
            "observation"
        ] = diag

        result[
            "observation_success"
        ] = True

        print(
            "visible object points:",
            diag[
                "visible_object_points"
            ],
        )

        print(
            "visible target points:",
            diag[
                "visible_target_points"
            ],
        )

        # ====================================================
        # Policy
        # ====================================================

        print()
        print(
            "[2/7] Where2Act inference"
        )

        # The shared GPU is also used by other formal baselines.  Serialize
        # only the mathematically identical inference section and release the
        # model before physics execution.  This prevents several episode
        # processes from retaining duplicate model allocations throughout
        # their much longer robot motions.
        inference_lock = Path("/tmp/where2act_noaff_v7_gpu_inference.lock")
        policy = None
        with inference_lock.open("a+", encoding="utf-8") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            try:
                policy = Where2ActPolicy(
                    checkpoint=(
                        str(checkpoint)
                        if checkpoint
                        is not None
                        else None
                    ),
                    device=args.device,
                    allow_untrained=(
                        args.engineering_allow_untrained
                    ),
                )

                with torch.inference_mode():
                    policy_result = policy.predict(
                        observation["points_model"],
                        observation["points_world"],
                        observation["camera_to_world_R"],
                        candidate_mask=observation["candidate_mask"],
                        seed=seed,
                    )
            finally:
                if str(args.device).startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.synchronize()
                policy = None
                gc.collect()
                if str(args.device).startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        result["gpu_inference_serialized_and_released_before_physics"] = True

        trained = bool(
            policy_result.get(
                "trained",
                False,
            )
        )

        if (
            not args.engineering_allow_untrained
            and not trained
        ):

            raise Where2ActRuntimeError(
                "CHECKPOINT_NOT_LOADED_AS_TRAINED"
            )

        result[
            "network_trained"
        ] = trained

        result[
            "interaction_point_world"
        ] = policy_result[
            "interaction_point_world"
        ]

        result[
            "interaction_score"
        ] = policy_result[
            "interaction_score"
        ]

        result[
            "critic_score"
        ] = policy_result[
            "critic_score"
        ]

        result["predicted_contact_point"] = policy_result["interaction_point_world"]
        result["predicted_qidx"] = int(policy_result["original_query_index"])
        result["predicted_sampled_qidx"] = int(policy_result["sampled_query_index"])
        result["predicted_action_orientation"] = policy_result["grasp_pose_world"]
        result["predicted_action_direction"] = policy_result["up_world"]

        result[
            "policy_success"
        ] = True

        # ====================================================
        # Official floating-gripper poses
        #
        # The network pose is the panda_hand/root pose 10 cm behind the
        # selected interaction point.  The pre-contact pose is another
        # 5 cm backwards, exactly as in Where2Act collect/checkcollect.
        # There is intentionally no arm, IK, or OMPL stage.
        # ====================================================

        approach_axis = np.asarray(policy_result["up_world"], dtype=np.float64)
        approach_axis /= np.linalg.norm(approach_axis) + 1e-12

        T_contact = np.asarray(policy_result["grasp_pose_world"], dtype=np.float64)
        T_pregrasp = T_contact.copy()
        T_pregrasp[:3, 3] -= APPROACH_DISTANCE * approach_axis

        operation_direction = (
            -approach_axis
            if primitive == "pull"
            else approach_axis
        )

        result[
            "approach_axis_world"
        ] = approach_axis

        result[
            "operation_direction_world"
        ] = operation_direction

        result["operation_controller"] = "progress_targeted_physical_v2"
        result["operation_segment_distance_m"] = OPERATION_SEGMENT_DISTANCE_M
        result["operation_max_travel_m"] = OPERATION_MAX_TRAVEL_M

        print()
        print("[3/6] spawn official floating gripper at pregrasp")
        panda = FloatingPandaTwoFingerController(
            scene,
            PANDA_GRIPPER_URDF,
            pose_stiffness=FLOATING_POSE_STIFFNESS,
            pose_damping=FLOATING_POSE_DAMPING,
            finger_stiffness=FLOATING_FINGER_STIFFNESS,
            finger_damping=FLOATING_FINGER_DAMPING,
            finger_static_friction=FLOATING_STATIC_FRICTION,
            finger_dynamic_friction=FLOATING_DYNAMIC_FRICTION,
            finger_restitution=FLOATING_RESTITUTION,
        )
        panda.set_initial_pose(T_pregrasp, open_gripper=(primitive == "pull"))
        result["backend_contact_material"] = apply_target_contact_material(
            target_link, panda.finger_material
        )

        result["planner"] = {
            "success": True,
            "type": "not_applicable_official_floating_gripper",
            "full_arm_present": False,
            "ik_used": False,
            "ompl_used": False,
        }
        result["planning_success"] = True
        result["trajectory_execution"] = {
            "type": "direct_floating_root_initialization",
            "trajectory_states": 1,
        }

        T_actual_pregrasp = panda.get_grasp_pose_matrix()
        result["floating_qpos_after_spawn"] = np.asarray(
            panda.robot.get_qpos(), dtype=np.float64
        ).tolist()
        result["floating_pregrasp_target_pose"] = T_pregrasp
        result["floating_pregrasp_actual_pose"] = T_actual_pregrasp

        pregrasp_pos_error = float(
            np.linalg.norm(
                T_actual_pregrasp[
                    :3,
                    3
                ]
                - T_pregrasp[
                    :3,
                    3
                ]
            )
        )

        pregrasp_rot_error = (
            rotation_error(
                T_pregrasp[
                    :3,
                    :3
                ],
                T_actual_pregrasp[
                    :3,
                    :3
                ],
            )
        )

        result[
            "pregrasp_position_error"
        ] = pregrasp_pos_error

        result[
            "pregrasp_rotation_error"
        ] = pregrasp_rot_error

        if (
            pregrasp_pos_error > 0.02
            or pregrasp_rot_error > 0.10
        ):

            result[
                "failure_reason"
            ] = (
                "pregrasp_execution_failed"
            )

            persist_result()

            return result

        result[
            "pregrasp_execution_success"
        ] = True

        result["backend_phase_init_to_pregrasp"] = {
            "executed": True,
            "target_locked": True,
            "execution": "direct floating-root placement",
        }

        # Target is free during the network-selected straight approach.
        set_target_free(obj, target_joint, target_index)
        if primitive == "pull":
            panda.open_gripper()
        else:
            panda.close_gripper()
        # Original Where2Act starts the Cartesian approach immediately after
        # placing the floating root; an idle settle lets the six virtual pose
        # joints drift under finger reaction forces and is not part of that
        # execution condition.
        result["progress_before_free_approach"] = float(get_progress(object_info))
        result["backend_phase_approach"] = {
            "executed": True,
            "target_stiffness": TARGET_FREE_STIFFNESS,
            "target_damping": TARGET_FREE_DAMPING,
            "gripper_state": "open" if primitive == "pull" else "closed",
            "fixed_world_contact_pose": True,
            "gt_target_pose_live_tracking": False,
        }

        print()
        print("[4/6] official floating-gripper straight approach")
        if primitive == "pull":
            panda.open_gripper()
        else:
            panda.close_gripper()
        approach_diag = panda.move_grasp_pose_to(
            T_contact,
            APPROACH_SIM_STEPS,
            position_tolerance=0.006,
            rotation_tolerance=0.05,
        )
        panda.clear_arm_velocity()
        result["approach_diagnostics"] = approach_diag
        T_actual_contact = panda.get_grasp_pose_matrix()
        contact_pos_error = float(np.linalg.norm(T_actual_contact[:3, 3] - T_contact[:3, 3]))
        contact_rot_error = rotation_error(T_contact[:3, :3], T_actual_contact[:3, :3])
        result["contact_position_error"] = contact_pos_error
        result["contact_rotation_error"] = contact_rot_error
        progress_after_approach = float(get_progress(object_info))
        result["progress_after_approach"] = progress_after_approach
        result["progress_drift_during_free_approach"] = float(
            progress_after_approach - initial_progress
        )
        print()
        if primitive == "pull":
            if contact_pos_error > 0.02 or contact_rot_error > 0.10:
                result["failure_reason"] = "approach_failed"
                persist_result()
                return result
            print("[6/7] real two-finger grasp")
            lock_target_to_q(obj, target_joint, target_index, initial_target_q)
            panda.close_gripper()
            for _ in range(CLOSE_LOCK_STEPS):
                lock_target_to_q(obj, target_joint, target_index, initial_target_q)
                panda.keep_gripper_closed()
                panda.step()
            result["progress_after_locked_close"] = float(get_progress(object_info))
            set_target_free(obj, target_joint, target_index)
            grasp_diag = monitor_grasp_establishment(
                panda,
                target_link,
                settle_steps=GRASP_SETTLE_STEPS,
                tail_steps=GRASP_TAIL_STEPS,
                min_bilateral_fraction=GRASP_MIN_BILATERAL_FRACTION,
            )
            grasp_success = bool(grasp_diag["firm_grasp"])
            result["grasp_semantics"] = "real stable bilateral two-finger grasp"
            result["grasp_diagnostics"] = grasp_diag
            failure_if_no_grasp = "firm_grasp_not_established"
        else:
            print("[6/7] closed-gripper target engagement")
            set_target_free(obj, target_joint, target_index)
            panda.close_gripper()
            grasp_diag = monitor_target_engagement(
                panda,
                target_link,
                steps=PUSH_ENGAGEMENT_STEPS,
            )
            grasp_success = bool(grasp_diag["engagement_success"])
            result["grasp_semantics"] = (
                "pre-operation target engagement/contact success for Push"
            )
            result["engagement_diagnostics"] = grasp_diag
            # For Push, target contact can physically prevent the floating
            # root from reaching the nominal network pose.  A verified target
            # engagement therefore takes precedence over pose residual.
            if grasp_success:
                result["push_contact_limited_approach"] = bool(
                    contact_pos_error > 0.02 or contact_rot_error > 0.10
                )
            failure_if_no_grasp = (
                "approach_failed"
                if contact_pos_error > 0.02 or contact_rot_error > 0.10
                else "no_target_contact"
            )

        result["finger_qpos_after_close"] = panda.get_finger_qpos()
        result["grasp_success"] = grasp_success
        if not grasp_success:
            result["failure_reason"] = failure_if_no_grasp
            persist_result()
            print("TRIAL METHOD FAILURE:", failure_if_no_grasp)
            return result

        result["operation_attempted_after_grasp"] = True
        result["operation_executed"] = True
        result["engineering_operation_executed"] = True
        result["pre_operation_progress"] = float(get_progress(object_info))
        set_target_free(obj, target_joint, target_index)
        result["backend_phase_operate"] = {
            "executed": True,
            "primitive": primitive,
            "target_stiffness": TARGET_FREE_STIFFNESS,
            "target_damping": TARGET_FREE_DAMPING,
            "controller": "network_direction_progress_targeted_physical_v2",
            "command_progress_target": command_progress_for_goal(goal),
            "success_progress_threshold": success_progress_for_goal(goal),
            "segment_distance_m": OPERATION_SEGMENT_DISTANCE_M,
            "max_travel_m": OPERATION_MAX_TRAVEL_M,
            "direction_semantics": "-predicted_d1" if primitive == "pull" else "+predicted_d1",
        }

        print()
        print(
            f"[7/7] Where2Act {primitive} progress-targeted: "
            f"command={command_progress_for_goal(goal):.2f}, "
            f"success={success_progress_for_goal(goal):.2f}"
        )
        panda.keep_gripper_closed()
        ee_before_operation = panda.get_grasp_center()
        operation_diag = execute_progress_targeted_operation(
            panda,
            object_info,
            primitive=primitive,
            goal=goal,
            direction_world=operation_direction,
            target_link=target_link,
            get_progress=get_progress,
        )
        panda.clear_arm_velocity()
        ee_after_operation = panda.get_grasp_center()

        final_progress = float(get_progress(object_info))
        directional_progress = (
            final_progress - initial_progress
            if goal == "open"
            else initial_progress - final_progress
        )
        task_success = bool(final_state_success(goal, final_progress))
        final_success = bool(result["operation_executed"] and task_success)

        result["operation_diagnostics"] = operation_diag
        result["post_operation_contact"] = operation_diag.get("last_contact")
        result["ee_before_operation"] = ee_before_operation
        result["ee_after_operation"] = ee_after_operation
        result["actual_operation_delta"] = ee_after_operation - ee_before_operation
        result["actual_operation_distance"] = float(
            np.linalg.norm(ee_after_operation - ee_before_operation)
        )
        result["actual_operation_path_length"] = float(
            operation_diag["actual_path_length_m"]
        )
        result["commanded_operation_travel"] = float(
            operation_diag["commanded_travel_m"]
        )
        result["final_progress"] = final_progress
        result["directional_task_progress"] = float(directional_progress)
        result["final_state_threshold_met"] = task_success
        result["command_target_reached"] = bool(
            command_target_reached(goal, final_progress)
        )
        result["task_success"] = task_success
        result["opening_success"] = task_success
        result["operation_success"] = final_success
        result["operation_success_given_grasp"] = final_success
        result["final_operation_success"] = final_success
        result["final_success"] = final_success

        if final_success:
            result["failure_reason"] = None
        elif operation_diag["termination"] == "contact_lost":
            result["failure_reason"] = (
                "grasp_lost_during_pull"
                if primitive == "pull"
                else "target_contact_lost_during_push"
            )
        elif operation_diag["termination"] == "motion_stalled":
            result["failure_reason"] = "operation_motion_failed"
        else:
            result["failure_reason"] = (
                "final_progress_below_open_35pct"
                if goal == "open"
                else "final_progress_above_close_15pct"
            )

        result[
            "runtime_seconds"
        ] = float(
            time.time()
            - start_time
        )

        persist_result()

        print()
        print("=" * 100)

        print(
            "grasp_success:",
            result[
                "grasp_success"
            ],
        )

        print(
            "final_progress:",
            result[
                "final_progress"
            ],
        )

        print(
            "final_success:",
            result[
                "final_success"
            ],
        )

        print(
            "result:",
            result_path,
        )

        print("=" * 100)

        return result

    except Exception as exc:

        expected_reason = (
            _expected_failure_reason(
                exc
            )
        )

        result[
            "implementation_exception"
        ] = (
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        result[
            "runtime_seconds"
        ] = float(
            time.time()
            - start_time
        )

        if expected_reason is not None:

            result[
                "implementation_error"
            ] = False

            result[
                "failure_reason"
            ] = expected_reason

            if (
                expected_reason
                == "pregrasp_execution_failed"
            ):
                result[
                    "pregrasp_execution_success"
                ] = False

            persist_result()

            print(
                "TRIAL METHOD/EXECUTION FAILURE:",
                expected_reason,
            )

            return result

        result[
            "implementation_error"
        ] = True

        result[
            "failure_reason"
        ] = (
            "implementation_error"
        )

        persist_result()

        raise


def build_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        required=True,
        choices=sorted(TASK_SPECS),
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
        "--formal-catalog",
        "--pose-catalog",
        dest="formal_catalog",
        required=True,
    )

    parser.add_argument(
        "--trial-seed",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
    )

    parser.add_argument(
        "--initial-ratio",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--output-root",
        required=True,
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--planner-env",
        default="where2act_planner",
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
        "--engineering-allow-untrained",
        action="store_true",
        help=(
            "仅允许工程smoke使用随机权重。"
            "正式benchmark禁止开启。"
        ),
    )

    parser.add_argument(
        "--engineering-force-pull",
        action="store_true",
        help=(
            "仅用于工程smoke："
            "即使随机权重没有建立firm grasp，"
            "也继续执行Where2Act pull branch。"
            "grasp_success/final_success不会因此被置True。"
        ),
    )

    return parser


if __name__ == "__main__":

    args = (
        build_parser()
        .parse_args()
    )

    try:

        run_trial(
            args
        )

    except Exception:

        print()
        print("=" * 100)
        print(
            "WHERE2ACT RUN_TRIAL "
            "IMPLEMENTATION ERROR"
        )
        print("=" * 100)

        traceback.print_exc()

        raise SystemExit(1)
