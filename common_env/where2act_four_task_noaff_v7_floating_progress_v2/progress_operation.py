"""Progress-targeted physical operation for the corrected four-task protocol.

The policy still chooses the contact pose and motion direction.  This executor
only replaces the old fixed 5 cm endpoint with a closed-loop absolute
articulation target.  It never sets object qpos or drives the object joint.
"""

import numpy as np

from contact_monitor import monitor_post_pull_contact, monitor_target_engagement


OPEN_COMMAND_PROGRESS = 0.40
OPEN_SUCCESS_PROGRESS = 0.35
CLOSE_COMMAND_PROGRESS = 0.10
CLOSE_SUCCESS_PROGRESS = 0.15

OPERATION_SEGMENT_DISTANCE_M = 0.01
OPERATION_SEGMENT_STEPS = 400
OPERATION_MAX_TRAVEL_M = 0.40
OPERATION_CONTACT_CHECK_STEPS = 10
OPERATION_MAX_CONSECUTIVE_CONTACT_LOSS = 3
OPERATION_MAX_CONSECUTIVE_STALL = 4
OPERATION_MIN_PROGRESS_IMPROVEMENT = 0.001
OPERATION_MIN_EE_MOTION_M = 0.001


def command_progress_for_goal(goal):
    if goal == "open":
        return OPEN_COMMAND_PROGRESS
    if goal == "close":
        return CLOSE_COMMAND_PROGRESS
    raise ValueError(f"unknown goal: {goal}")


def success_progress_for_goal(goal):
    if goal == "open":
        return OPEN_SUCCESS_PROGRESS
    if goal == "close":
        return CLOSE_SUCCESS_PROGRESS
    raise ValueError(f"unknown goal: {goal}")


def command_target_reached(goal, progress):
    progress = float(progress)
    if goal == "open":
        return progress >= OPEN_COMMAND_PROGRESS
    if goal == "close":
        return progress <= CLOSE_COMMAND_PROGRESS
    raise ValueError(f"unknown goal: {goal}")


def final_state_success(goal, progress):
    progress = float(progress)
    if goal == "open":
        return progress >= OPEN_SUCCESS_PROGRESS
    if goal == "close":
        return progress <= CLOSE_SUCCESS_PROGRESS
    raise ValueError(f"unknown goal: {goal}")


def goal_error(goal, progress):
    target = command_progress_for_goal(goal)
    progress = float(progress)
    if goal == "open":
        return max(0.0, target - progress)
    if goal == "close":
        return max(0.0, progress - target)
    raise ValueError(f"unknown goal: {goal}")


def _contact_window(controller, target_link, primitive):
    if primitive == "pull":
        diag = monitor_post_pull_contact(
            controller,
            target_link,
            steps=OPERATION_CONTACT_CHECK_STEPS,
        )
        return (not bool(diag["grasp_lost"])), diag

    diag = monitor_target_engagement(
        controller,
        target_link,
        steps=OPERATION_CONTACT_CHECK_STEPS,
    )
    return bool(diag["engagement_success"]), diag


def execute_progress_targeted_operation(
    controller,
    object_info,
    *,
    primitive,
    goal,
    direction_world,
    target_link,
    get_progress,
):
    """Physically move along the network direction until the joint goal or failure."""

    direction = np.asarray(direction_world, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("operation direction is non-finite or degenerate")
    direction /= norm

    max_segments = int(
        np.ceil(OPERATION_MAX_TRAVEL_M / OPERATION_SEGMENT_DISTANCE_M)
    )
    start_progress = float(get_progress(object_info))
    start_ee = np.asarray(controller.get_grasp_center(), dtype=np.float64).copy()
    trace = [
        {
            "segment": 0,
            "progress": start_progress,
            "goal_error": goal_error(goal, start_progress),
            "ee": start_ee.tolist(),
            "command_target_reached": command_target_reached(goal, start_progress),
        }
    ]

    consecutive_contact_loss = 0
    consecutive_stall = 0
    commanded_travel = 0.0
    actual_path_length = 0.0
    termination = "max_travel_reached"
    last_contact = None

    if command_target_reached(goal, start_progress):
        termination = "command_target_already_reached"
    else:
        for segment in range(1, max_segments + 1):
            before_progress = float(get_progress(object_info))
            before_error = goal_error(goal, before_progress)
            before_ee = np.asarray(controller.get_grasp_center(), dtype=np.float64)

            commanded_target = (
                start_ee
                + direction * OPERATION_SEGMENT_DISTANCE_M * segment
            )
            controller.keep_gripper_closed()
            move_diag = controller.move_grasp_point_to(
                commanded_target,
                OPERATION_SEGMENT_STEPS,
                control_tolerance=0.004,
            )
            controller.clear_arm_velocity()

            after_ee = np.asarray(controller.get_grasp_center(), dtype=np.float64)
            actual_motion = float(np.linalg.norm(after_ee - before_ee))
            actual_path_length += actual_motion
            commanded_travel += OPERATION_SEGMENT_DISTANCE_M
            after_progress = float(get_progress(object_info))
            after_error = goal_error(goal, after_progress)
            improvement = before_error - after_error

            contact_ok, contact_diag = _contact_window(
                controller, target_link, primitive
            )
            last_contact = contact_diag
            consecutive_contact_loss = (
                0 if contact_ok else consecutive_contact_loss + 1
            )
            if (
                actual_motion < OPERATION_MIN_EE_MOTION_M
                and improvement < OPERATION_MIN_PROGRESS_IMPROVEMENT
            ):
                consecutive_stall += 1
            else:
                consecutive_stall = 0

            trace.append(
                {
                    "segment": segment,
                    "progress": after_progress,
                    "goal_error": after_error,
                    "progress_improvement": improvement,
                    "commanded_segment_distance_m": OPERATION_SEGMENT_DISTANCE_M,
                    "commanded_target": commanded_target.tolist(),
                    "actual_segment_distance_m": actual_motion,
                    "ee": after_ee.tolist(),
                    "contact_ok": bool(contact_ok),
                    "contact": contact_diag,
                    "move": move_diag,
                    "command_target_reached": command_target_reached(
                        goal, after_progress
                    ),
                }
            )

            if command_target_reached(goal, after_progress):
                termination = "command_target_reached"
                break
            if (
                consecutive_contact_loss
                >= OPERATION_MAX_CONSECUTIVE_CONTACT_LOSS
            ):
                termination = "contact_lost"
                break
            if consecutive_stall >= OPERATION_MAX_CONSECUTIVE_STALL:
                termination = "motion_stalled"
                break

    final_progress = float(get_progress(object_info))
    final_ee = np.asarray(controller.get_grasp_center(), dtype=np.float64)
    return {
        "controller": "network_direction_progress_targeted_physical_v2",
        "goal": goal,
        "primitive": primitive,
        "command_progress_target": command_progress_for_goal(goal),
        "success_progress_threshold": success_progress_for_goal(goal),
        "segment_distance_m": OPERATION_SEGMENT_DISTANCE_M,
        "segment_steps": OPERATION_SEGMENT_STEPS,
        "max_travel_m": OPERATION_MAX_TRAVEL_M,
        "commanded_travel_m": float(commanded_travel),
        "actual_path_length_m": float(actual_path_length),
        "net_ee_delta": (final_ee - start_ee).tolist(),
        "net_ee_distance_m": float(np.linalg.norm(final_ee - start_ee)),
        "start_progress": start_progress,
        "final_progress": final_progress,
        "command_target_reached": command_target_reached(goal, final_progress),
        "final_state_success": final_state_success(goal, final_progress),
        "termination": termination,
        "consecutive_contact_loss": int(consecutive_contact_loss),
        "consecutive_stall": int(consecutive_stall),
        "last_contact": last_contact,
        "trace": trace,
    }
