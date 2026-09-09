"""ArticuBot-PerceptionReach episode in the frozen OPEN benchmark.

This is an explicitly separate method variant.  It keeps the official weights,
frozen scene/camera/robot, PHYSICAL_V2 grasp gate, senior soft-weld-PD backend,
and 35% evaluator.  Its additions are target-link camera-mask conditioning,
observed-point-cloud reach waypoints, and a post-grasp visual gap direction guard.
"""

from __future__ import annotations

import gc
import json
import os
import random
import shutil
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import frozen_open_episode as frozen_episode
from articubot_perception_action_adapter import execute_articubot_joint_pd_action
from articubot_observation_adapter import ObservationHistory
from frozen_open_episode import (
    ACTION_CONTROL_STEPS,
    GRIPPER_WAIT_STEPS,
    MAX_POLICY_CYCLES,
    SOFT_WELD_CONFIG,
    THRESHOLD,
    _create_scene,
    _pin_target_state,
    _set_target_free,
    _set_target_locked,
    case_link_name,
    case_task_mode,
    catalog_path_for_task,
    initial_ratio_for_seed,
    jsonable,
)
from contact_monitor import monitor_grasp_establishment
from senior_soft_weld_pd import SeniorSoftWeldPD, enable_backend_finger_lock, source_defaults
from perception_reach_controller import PerceptionReachPhysicalController, controller_audit

from articubot_perception_reach_adapter import (
    capture_link_conditioned_observations,
    goal_root_to_world,
    masked_high_level_infer,
    masked_high_level_goal_modes,
    plan_collision_aware_reach,
    shift_goal_to_visible_free_edge,
    shift_goal_to_visible_protrusion,
    visible_gap_metric,
)


PREGRASP_STEPS = (900, 1300, 700)
FINAL_APPROACH_STEPS = 1000
PLANNER_WORKER = Path(
    "/home/feng/robot_baselines/repos/pa3ff_official/reproduction/"
    "motion_planner_collision_worker_pa3ff_fixed_v2.py"
)


def _track_joint_target(panda, q_target, *, min_steps=15, max_steps=600,
                        tolerance_norm=0.03, tolerance_maxabs=0.025):
    q_target = np.asarray(q_target, dtype=np.float64).reshape(7)
    for index, joint in enumerate(panda.arm_joints):
        joint.set_drive_velocity_target(0.0)
        joint.set_drive_target(float(q_target[index]))
    panda.open_gripper()
    norm = maxabs = float("inf")
    for step in range(int(max_steps)):
        panda.step()
        error = np.asarray(panda.robot.get_qpos(), dtype=np.float64)[:7] - q_target
        norm = float(np.linalg.norm(error))
        maxabs = float(np.max(np.abs(error)))
        if step + 1 >= int(min_steps) and norm <= tolerance_norm and maxabs <= tolerance_maxabs:
            return {"converged": True, "steps": step + 1, "error_norm": norm,
                    "error_maxabs": maxabs}
    return {"converged": False, "steps": int(max_steps), "error_norm": norm,
            "error_maxabs": maxabs}


def _execute_joint_trajectory(panda, trajectory):
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 7:
        raise RuntimeError(f"planner trajectory shape invalid: {trajectory.shape}")
    total_steps = 0
    max_error = 0.0
    for index, q in enumerate(trajectory):
        diag = _track_joint_target(panda, q)
        total_steps += int(diag["steps"])
        max_error = max(max_error, float(diag["error_norm"]))
        if not diag["converged"]:
            raise RuntimeError(f"SAPIEN waypoint tracking failed: {index}: {diag}")
    final = _track_joint_target(
        panda, trajectory[-1], min_steps=50, max_steps=1500,
        tolerance_norm=0.005, tolerance_maxabs=0.003,
    )
    if not final["converged"]:
        raise RuntimeError(f"SAPIEN final tracking failed: {final}")
    return {"trajectory_states": len(trajectory), "total_sim_steps": total_steps,
            "max_waypoint_error": max_error, "final_tracking": final}


def _plan_ompl(case, resources, pregrasp_pose, outward_normal, seed):
    temp_root = Path(tempfile.mkdtemp(prefix="articubot_perception_reach_"))
    request_path = temp_root / "request.json"
    output_dir = temp_root / "planner"
    request = {
        "shape_id": str(case["shape_id"]),
        "target_link": case_link_name(case),
        "initial_ratio": float(resources["progress"]()),
        "initial_object_qpos": jsonable(case.get("initial_object_qpos")),
        "robot_start_qpos9": jsonable(np.asarray(resources["panda"].robot.get_qpos())),
        "pregrasp_pose_world": jsonable(pregrasp_pose),
        "contact_pose_world": jsonable(pregrasp_pose),
        "approach_axis_world": jsonable(-np.asarray(outward_normal)),
        "network_trained": False,
    }
    request_path.write_text(json.dumps(request, indent=2) + "\n")
    cmd = [
        "/home/feng/miniconda3/bin/conda", "run", "-n", "where2act_planner", "python",
        str(PLANNER_WORKER), "--request", str(request_path),
        "--pose-catalog", str(catalog_path_for_task(case_task_mode(case))),
        "--output-dir", str(output_dir), "--planner", "RRTConnect",
        "--planning-time", "5.0", "--ik-attempts", "100", "--seed", str(int(seed)),
    ]
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    ompl_root = "/home/feng/robot_baselines/repos/articubot"
    env["PYTHONPATH"] = ompl_root + (os.pathsep + pythonpath if pythonpath else "")
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
    trajectory_path = output_dir / "trajectory.npy"
    planner_result = output_dir / "result.json"
    diag = {
        "returncode": int(proc.returncode),
        "success": bool(proc.returncode == 0 and trajectory_path.is_file()),
        "stdout_tail": proc.stdout[-3000:],
        "stderr_tail": proc.stderr[-3000:],
        "command": cmd,
    }
    if planner_result.is_file():
        diag["result"] = json.loads(planner_result.read_text())
    trajectory = np.load(trajectory_path) if trajectory_path.is_file() else None
    shutil.rmtree(temp_root, ignore_errors=True)
    return trajectory, diag


def _pregrasp_orientation_variants(pregrasp_pose, current_pose, *, prefer_surface):
    base = np.asarray(pregrasp_pose, dtype=np.float64)
    surface_variants = []
    for quarter_turns in (0, 2):
        T = base.copy()
        angle = quarter_turns * np.pi / 2.0
        roll = np.array(
            [[np.cos(angle), -np.sin(angle), 0.0],
             [np.sin(angle), np.cos(angle), 0.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        T[:3, :3] = base[:3, :3] @ roll
        surface_variants.append((f"surface_roll_{quarter_turns}", T))
    T = base.copy()
    T[:3, :3] = np.asarray(current_pose, dtype=np.float64)[:3, :3]
    current_variant = ("current_orientation", T)
    return (
        surface_variants + [("current_orientation_fallback", T)]
        if prefer_surface else
        [current_variant] + surface_variants
    )


def _gripper_points_world(panda):
    return np.asarray(
        [
            panda.hand_link.get_pose().p,
            panda.right_finger_link.get_pose().p,
            panda.left_finger_link.get_pose().p,
            panda.grasp_link.get_pose().p,
        ],
        dtype=np.float64,
    ).reshape(4, 3)


def _summary(records):
    if not records:
        return None
    norms = [float(row["xyz_norm"]) for row in records]
    return {
        "num_actions": len(records),
        "xyz_norm_min": min(norms),
        "xyz_norm_max": max(norms),
        "xyz_norm_mean": float(np.mean(norms)),
        "first": records[:4],
        "last": records[-4:],
    }


def run_perception_reach_episode(policy, case, *, case_index, repeat_id, seed):
    started = time.time()
    task_mode = case_task_mode(case)
    link_name = case_link_name(case)
    initial_ratio = None if case.get("initial_object_qpos") is not None else initial_ratio_for_seed(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    result = {
        "method": "ArticuBot-PerceptionReach",
        "protocol_version": "articubot_perception_reach_soft_weld_pd_relabel35_v3",
        "case_id": f"{case['shape_id']}_{link_name}",
        "case_index": int(case_index),
        "object_id": str(case["shape_id"]),
        "link_id": link_name,
        "category": str(case.get("category", "UNKNOWN")),
        "trial_id": int(repeat_id),
        "repeat_id": int(repeat_id),
        "trial_seed": int(seed),
        "task_mode": task_mode,
        "requested_initial_articulation_state": initial_ratio,
        "requested_initial_object_qpos": jsonable(case.get("initial_object_qpos")),
        "initial_articulation_state": None,
        "final_articulation_state": None,
        "articulation_progress": None,
        "articulation_success_threshold": THRESHOLD,
        "reach_35": False,
        "grasp_success": False,
        "operation_success": False,
        "final_success": False,
        "termination_reason": None,
        "timeout": False,
        "ik_failure": False,
        "policy_failure": False,
        "controller_failure": False,
        "exception": None,
        "runtime_sec": None,
        "camera_source": str(catalog_path_for_task(task_mode)),
        "interaction_mode": "perception_reach_then_physical_twofinger_soft_weld_pd",
        "target_conditioning": {
            "source": "frozen_camera_sapien_link_instance_segmentation",
            "requested_link_identity_selects_mask": True,
            "ground_truth_handle": False,
            "ground_truth_joint_axis": False,
            "ground_truth_motion_direction": False,
            "training": False,
            "finetuning": False,
        },
        "collision_aware_reach": None,
        "visual_direction_guard": {
            "metric": "visible_target_to_rest_q75_nearest_distance",
            "uses_articulation_state": False,
            "translation_sign_flipped": False,
            "gap_samples": [],
        },
        "senior_soft_weld_pd": {
            **source_defaults(),
            "enabled": True,
            "created": False,
            "n_welds": 0,
            "fail_reason": "not_reached",
            "source_archive": "/home/feng/robot_baselines/repos/articubot/reproduction/provenance/low_level_execution_logic.zip",
        },
        "finger_grasp_lock": {"enabled": False},
        "grasp_monitor": None,
        "high_level_goal": None,
        "low_level_action_summary": None,
    }
    resources = None
    weld = None
    action_records = []
    try:
        # Reuse the exact frozen scene constructor while replacing only the
        # robot controller implementation with the corrected/senior bridge.
        frozen_episode.PandaTwoFingerController = PerceptionReachPhysicalController
        resources = _create_scene(case, initial_ratio)
        panda = resources["panda"]
        panda.wait(100)
        result["initial_articulation_state"] = float(resources["progress"]())
        policy.reset()
        rng = np.random.default_rng(np.random.SeedSequence([seed, 921731]))
        capture = capture_link_conditioned_observations(
            scene=resources["scene"], camera=resources["camera"], obj=resources["obj"],
            target_link=resources["target_link"], panda=panda,
            num_points=policy.num_points, rng=rng,
        )
        full_history = ObservationHistory(policy.n_obs_steps)
        target_history = ObservationHistory(policy.n_obs_steps)
        full_history.reset(capture["full"])
        target_history.reset(capture["target"])
        goal_modes_root, goal_modes_diag = masked_high_level_goal_modes(
            policy, full_history.stack(), capture["target_mask_on_full_sample"]
        )
        goal_root = goal_modes_root[:1]
        goal_world = goal_root_to_world(goal_root[0, 0], capture)
        goal_modes_world = [
            goal_root_to_world(candidate[0], capture) for candidate in goal_modes_root
        ]
        # A second, still-official forward is conditioned on the selected
        # link cloud itself.  The masked-full forward preserves surrounding
        # object context for the low-level policy, while this target-only
        # forward supplies a geometrically meaningful height when the frozen
        # camera sees a door/drawer nearly edge-on.
        target_goal_root = policy.high_level_infer(target_history.stack())
        target_goal_world = goal_root_to_world(target_goal_root[0, 0], capture)
        target_centered = np.asarray(capture["target_points_world"]) - np.asarray(
            capture["target_points_world"]
        ).mean(axis=0)
        target_eigenvalues = np.sort(np.maximum(
            np.linalg.eigvalsh(target_centered.T @ target_centered / len(target_centered)),
            0.0,
        ))
        edge_like_target = bool(
            target_eigenvalues[-1] / max(target_eigenvalues[-2], 1e-12) > 25.0
        )
        reach_goal_world = target_goal_world if edge_like_target else goal_world
        free_edge_diag = {"applied": False, "reason": "drawer_or_edge_like"}
        if task_mode == "door_open" and not edge_like_target:
            reach_goal_world, free_edge_diag = shift_goal_to_visible_free_edge(
                reach_goal_world,
                capture["target_points_world"],
                capture["rest_points_world"],
            )
        result["high_level_goal"] = {
            "masked_full_goal_world": jsonable(goal_world),
            "target_only_goal_world": jsonable(target_goal_world),
            "operate_conditioning": "official_full_object_pointcloud_and_masked_full_goal",
            "reach_anchor_conditioning": (
                "target_only_goal" if edge_like_target else "masked_full_goal"
            ),
            "target_cloud_eigenvalues": jsonable(target_eigenvalues),
            "edge_like_target": edge_like_target,
            "visible_free_edge_conditioning": free_edge_diag,
            "masked_goal_modes": goal_modes_diag,
        }

        current_pose = np.asarray(panda.get_grasp_pose_matrix(), dtype=np.float64)
        current_gripper = _gripper_points_world(panda)
        # Keep the official masked softmax aggregate first.  Distinct modes are
        # perception-only fallbacks if its collision-free reach is infeasible.
        if edge_like_target:
            reach_goal_candidates = [(target_goal_world, "target_only_edge")]
        else:
            reach_goal_candidates = [(goal_modes_world[0], "masked_aggregate")]
            protrusion_goal, protrusion_diag = shift_goal_to_visible_protrusion(
                goal_modes_world[0], capture["target_points_world"],
                capture["camera_position_world"],
            )
            if protrusion_diag.get("applied"):
                reach_goal_candidates.append((protrusion_goal, "visible_protrusion"))
            reach_goal_candidates.extend(
                (candidate, f"masked_mode_{index}")
                for index, candidate in enumerate(goal_modes_world[1:4], start=1)
            )
        result["high_level_goal"]["visible_protrusion_conditioning"] = (
            protrusion_diag if not edge_like_target else
            {"applied": False, "reason": "edge_like_target"}
        )
        conditioned_goal_candidates = []
        for candidate_index, (candidate, candidate_source) in enumerate(reach_goal_candidates):
            conditioned = candidate
            candidate_diag = {"applied": False, "reason": "drawer_or_edge_like"}
            # Moving a broad door prediction to its visually exposed free edge
            # is useful as a fallback, but applying it to the aggregate broke a
            # previously valid official-policy trajectory during development.
            if task_mode == "door_open" and not edge_like_target and candidate_index > 0:
                conditioned, candidate_diag = shift_goal_to_visible_free_edge(
                    candidate,
                    capture["target_points_world"],
                    capture["rest_points_world"],
                )
            candidate_diag["source"] = candidate_source
            conditioned_goal_candidates.append((conditioned, candidate_diag))
        plans = []
        for goal_mode_rank, (candidate, candidate_diag) in enumerate(
            conditioned_goal_candidates
        ):
            # A point-cloud-clear corridor is not necessarily collision-free
            # for the full Panda.  Preserve the two strongest perceptual
            # corridors so the authoritative PyBullet collision/IK stage can
            # reject an unreachable side without discarding the same learned
            # goal.  This is reach-interface redundancy, not an object oracle.
            for corridor_rank in (0, 1):
                plan_candidate = plan_collision_aware_reach(
                    current_pose=current_pose,
                    current_gripper_points=current_gripper,
                    goal_points=candidate,
                    target_points=capture["target_points_world"],
                    obstacle_points=capture["full_points_world"],
                    camera_position=capture["camera_position_world"],
                    approach_offset=0.35,
                    candidate_rank=corridor_rank,
                    visible_free_edge_grasp=bool(candidate_diag.get("applied")),
                )
                plans.append((goal_mode_rank, candidate_diag, plan_candidate))
        # Door panels require a lateral free-edge pinch; face-normal aggregate
        # approaches produced palm-first contacts across the pre-freeze pilot.
        # The official aggregate remains a fallback.  Drawers retain the
        # aggregate when it already lies inside the Panda pinch envelope.
        aggregate_plan = plans[0]
        geometry_key = lambda item: (
            item[2].predicted_fingertip_mean_distance_m,
            -item[2].predicted_palm_clearance_m,
        )
        if task_mode == "door_open" and not edge_like_target:
            edge_plans = [item for item in plans if item[1].get("applied")]
            edge_plans = sorted(edge_plans, key=geometry_key)
            # Vertical panels use the lateral free edge.  For strongly tilted
            # visible surfaces the released aggregate/current-wrist path is
            # retained: forcing a lateral edge there preserved grasp but broke
            # the official post-grasp motion in regression.
            if abs(float(aggregate_plan[2].outward_normal[2])) > 0.50:
                plans = [aggregate_plan] + edge_plans
            else:
                plans = edge_plans + [aggregate_plan]
        elif task_mode == "drawer_open":
            # A drawer front and its handle normally belong to the same active
            # link.  The aggregate can therefore sit on the broad panel and
            # still have a deceptively small point distance.  When the frozen
            # camera itself exposes a >12 mm protrusion, try that perception-
            # derived handle hypothesis before the panel aggregate.  Keep both
            # aggregate corridors as fallbacks for handle-less geometry.
            protrusion_plans = [
                item for item in plans
                if item[1].get("source") == "visible_protrusion"
            ]
            aggregate_plans = [
                item for item in plans
                if item[1].get("source") == "masked_aggregate"
            ]
            other_plans = [
                item for item in plans
                if item[1].get("source") not in {
                    "visible_protrusion", "masked_aggregate"
                }
            ]
            plans = (
                sorted(protrusion_plans, key=geometry_key)
                + sorted(aggregate_plans, key=geometry_key)
                + sorted(other_plans, key=geometry_key)
            )
        elif aggregate_plan[2].predicted_fingertip_mean_distance_m <= 0.065:
            aggregate_plans = [
                item for item in plans
                if item[1].get("source") == "masked_aggregate"
            ]
            other_plans = [
                item for item in plans
                if item[1].get("source") != "masked_aggregate"
            ]
            plans = sorted(aggregate_plans, key=geometry_key) + sorted(
                other_plans, key=geometry_key,
            )
        else:
            plans = sorted(plans, key=geometry_key)
        # Bound the formal runtime while retaining the aggregate/protrusion and
        # strongest distinct masked hypotheses/corridors.  Four geometry
        # choices x three wrist symmetries gives at most twelve independent
        # OMPL attempts per episode.
        plans = plans[:4]
        result["collision_aware_reach"] = {
            "planner": "observed_pointcloud_multi_corridor",
            "target_pixel_count": capture["target_pixel_count"],
            "object_pixel_count": capture["object_pixel_count"],
            "waypoint_results": [],
            "controller": controller_audit(),
            "geometric_candidates": [
                {
                    "rank": rank,
                    "goal_source": conditioning_diag.get("source"),
                    "candidate_index": item.candidate_index,
                    "predicted_clearance_m": item.predicted_clearance_m,
                    "approach_offset_m": item.approach_offset_m,
                    "outward_normal_world": jsonable(item.outward_normal),
                    "pregrasp_position_world": jsonable(item.waypoints[-1][:3, 3]),
                    "predicted_palm_clearance_m": item.predicted_palm_clearance_m,
                    "predicted_fingertip_mean_distance_m": item.predicted_fingertip_mean_distance_m,
                    "grasp_geometry_score": item.grasp_geometry_score,
                    "current_tool_leading_alignment": float(np.dot(
                        (current_gripper[3] - current_gripper[0])
                        / max(np.linalg.norm(current_gripper[3] - current_gripper[0]), 1e-12),
                        -np.asarray(item.outward_normal)
                        / max(np.linalg.norm(item.outward_normal), 1e-12),
                    )),
                }
                for rank, (_, conditioning_diag, item) in enumerate(plans)
            ],
        }

        # The target remains at the frozen start articulation during reaching and
        # physical closing.  The robot itself is never teleported.
        _set_target_locked(resources)
        panda.set_video_step_callback(lambda: _pin_target_state(resources), every_n_steps=1)
        trajectory = None
        selected_pregrasp = None
        plan = None
        planner_attempts = []
        attempt_index = 0
        for geometric_rank, (goal_mode_rank, conditioning_diag, plan_try) in enumerate(plans):
            tool_forward = current_gripper[3] - current_gripper[0]
            tool_forward /= max(np.linalg.norm(tool_forward), 1e-12)
            approach_direction = -np.asarray(plan_try.outward_normal, dtype=np.float64)
            approach_direction /= max(np.linalg.norm(approach_direction), 1e-12)
            tool_leading_alignment = float(np.dot(tool_forward, approach_direction))
            prefer_surface = bool(
                conditioning_diag.get("source") != "masked_aggregate"
            )
            for name, candidate_pose in _pregrasp_orientation_variants(
                plan_try.waypoints[-1], current_pose,
                prefer_surface=prefer_surface,
            ):
                candidate_trajectory, planner_diag = _plan_ompl(
                    case, resources, candidate_pose, plan_try.outward_normal,
                    seed + 1009 * attempt_index,
                )
                planner_diag["geometric_rank"] = geometric_rank
                planner_diag["goal_mode_rank"] = goal_mode_rank
                planner_diag["geometric_candidate_index"] = plan_try.candidate_index
                planner_diag["orientation_variant"] = name
                planner_diag["current_tool_leading_alignment"] = tool_leading_alignment
                planner_diag["prefer_surface_orientation"] = prefer_surface
                planner_attempts.append(planner_diag)
                attempt_index += 1
                if candidate_trajectory is not None:
                    trajectory = candidate_trajectory
                    selected_pregrasp = candidate_pose
                    plan = plan_try
                    break
            if trajectory is not None:
                break
        result["collision_aware_reach"]["ompl_attempts"] = jsonable(planner_attempts)
        if trajectory is None:
            result["termination_reason"] = "planning_failed"
            result["ik_failure"] = any(
                "IK_FAILED" in (item.get("stdout_tail", "") + item.get("stderr_tail", ""))
                for item in planner_attempts
            )
            raise RuntimeError("collision-aware OMPL planning failed")
        result["collision_aware_reach"].update({
            "selected_goal_mode_rank": planner_attempts[-1]["goal_mode_rank"],
            "selected_geometric_candidate_index": plan.candidate_index,
            "predicted_clearance_m": plan.predicted_clearance_m,
            "goal_fit_rmse_m": plan.goal_fit_rmse_m,
            "approach_offset_m": plan.approach_offset_m,
            "outward_normal_world": jsonable(plan.outward_normal),
            "goal_pose_world": jsonable(plan.goal_pose),
        })
        trajectory_diag = _execute_joint_trajectory(panda, trajectory)
        result["collision_aware_reach"]["trajectory_execution"] = jsonable(trajectory_diag)
        result["collision_aware_reach"]["selected_pregrasp_pose_world"] = jsonable(
            selected_pregrasp
        )
        # At the collision-free standoff, release the target exactly as in the
        # supplied OPEN/TO_GRASP bottom logic.  No target pinning is active in
        # the physical contact phase.
        panda.clear_video_step_callback()
        _set_target_free(resources)
        # Align to the surface-consistent learned closing orientation before
        # the deliberate-contact approach.
        # Retain the exact collision-planned wrist roll.  A 180-degree roll is
        # a symmetric parallel-jaw grasp, but forcing it back to the nominal
        # ordering after OMPL can create a needless wrist singularity.
        contact_pose = np.asarray(plan.goal_pose, dtype=np.float64).copy()
        contact_pose[:3, :3] = np.asarray(selected_pregrasp, dtype=np.float64)[:3, :3]
        align_pose = np.asarray(selected_pregrasp, dtype=np.float64).copy()
        align_diag = panda.move_grasp_pose_to(align_pose, 900)
        result["collision_aware_reach"]["pregrasp_alignment"] = jsonable(align_diag)
        # The fingers must stay open until the selected grasp pose is reached.
        # Development showed that closing during free-space approach can pass
        # an already-closed gripper over a thin door edge.  Collision awareness
        # therefore remains in OMPL + palm stop; physical closure happens once
        # at the contact pose under the unchanged PHYSICAL_V2 grasp gate.
        final_diag = panda.approach_with_palm_stop(
            contact_pose, FINAL_APPROACH_STEPS, resources["obj"].get_links()
        )
        result["collision_aware_reach"]["executed_contact_pose_world"] = jsonable(
            contact_pose
        )
        result["collision_aware_reach"]["final_approach"] = jsonable(final_diag)
        if final_diag["final_position_error"] > 0.04 or final_diag["final_rotation_error"] > 0.35:
            result["ik_failure"] = True
        # Fully physical close and the unchanged PHYSICAL_V2 firm bilateral gate.
        for _ in range(GRIPPER_WAIT_STEPS):
            panda.keep_gripper_closed()
            panda.step()
        grasp_monitor = monitor_grasp_establishment(
            panda, resources["target_link"], settle_steps=300, tail_steps=100,
            min_bilateral_fraction=0.50,
        )
        result["grasp_monitor"] = jsonable(grasp_monitor)
        result["grasp_success"] = bool(grasp_monitor["firm_grasp"])
        if not result["grasp_success"]:
            result["termination_reason"] = "grasp_failed"
        else:
            weld, weld_diag = SeniorSoftWeldPD.try_create(
                resources["scene"], panda, resources["target_link"], SOFT_WELD_CONFIG
            )
            result["senior_soft_weld_pd"] = jsonable(weld_diag)
            if weld is not None:
                finger_target = float(weld.finger_target)
                result["finger_grasp_lock"] = jsonable(weld.enable_finger_lock(panda))
            else:
                finger_target = 0.0
                result["finger_grasp_lock"] = jsonable(
                    enable_backend_finger_lock(panda, finger_target)
                )
            resources["target_joint"].set_drive_property(
                SOFT_WELD_CONFIG.operate_target_stiffness,
                SOFT_WELD_CONFIG.operate_target_damping,
                SOFT_WELD_CONFIG.operate_target_force_limit,
            )
            resources["target_joint"].set_drive_target(
                float(resources["obj"].get_qpos()[resources["target_index"]])
            )
            gap_before = visible_gap_metric(
                capture["target_points_world"], capture["rest_points_world"]
            )
            direction_sign = 1.0
            for cycle in range(MAX_POLICY_CYCLES):
                capture = capture_link_conditioned_observations(
                    scene=resources["scene"], camera=resources["camera"], obj=resources["obj"],
                    target_link=resources["target_link"], panda=panda,
                    num_points=policy.num_points, rng=rng,
                )
                full_history.append(capture["full"])
                target_history.append(capture["target"])
                goal_root = masked_high_level_infer(
                    policy, full_history.stack(), capture["target_mask_on_full_sample"]
                )
                # Preserve the released low-level policy's full-object input
                # distribution.  Active-link masking conditions the high-level
                # goal and collision-aware reach only; forcing a target-only
                # low-level cloud was an out-of-distribution regression.
                actions = policy.low_level_infer(full_history.stack(), goal_root)
                for action_index, raw_action in enumerate(actions):
                    action = np.asarray(raw_action, dtype=np.float64).copy()
                    raw_root_xyz = action[:3].copy()
                    # Official actions are robot-root-frame translation deltas.
                    action[:3] = direction_sign * (
                        action[:3] @ capture["R_world_root"].T
                    )
                    pose_before_action = np.asarray(
                        panda.get_grasp_pose_matrix(), dtype=np.float64
                    )[:3, 3].copy()
                    diag = execute_articubot_joint_pd_action(
                        panda, action,
                        num_steps=SOFT_WELD_CONFIG.operate_steps_per_action,
                        finger_target_override=finger_target,
                    )
                    pose_after_action = np.asarray(
                        panda.get_grasp_pose_matrix(), dtype=np.float64
                    )[:3, 3].copy()
                    if weld is not None:
                        weld.maintain_finger_lock(panda)
                    ctrl = diag["control"]
                    if not ctrl["reached_control_tolerance"] and (
                        ctrl["final_position_error"] > 0.04
                        or ctrl["final_rotation_error"] > 0.35
                    ):
                        result["ik_failure"] = True
                    action_records.append({
                        "cycle": cycle,
                        "action_index": action_index,
                        "xyz_norm": float(np.linalg.norm(action[:3])),
                        "raw_root_xyz": jsonable(raw_root_xyz),
                        "command_world_xyz": jsonable(action[:3]),
                        "achieved_world_xyz": jsonable(
                            pose_after_action - pose_before_action
                        ),
                        "control_final_position_error": float(
                            ctrl["final_position_error"]
                        ),
                        "control_final_rotation_error": float(
                            ctrl["final_rotation_error"]
                        ),
                        "translation_sign": float(direction_sign),
                        "target_finger": float(diag["target_finger"]),
                    })
                    current_progress = float(resources["progress"]())
                    if current_progress >= THRESHOLD:
                        result["termination_reason"] = "articulation_threshold_reached"
                        break
                if result["termination_reason"] is not None:
                    break
                # The guard uses only a second frozen-camera observation.  A
                # meaningful reduction in visible link/rest separation flips the
                # future learned translation direction once.
                try:
                    after = capture_link_conditioned_observations(
                        scene=resources["scene"], camera=resources["camera"], obj=resources["obj"],
                        target_link=resources["target_link"], panda=panda,
                        num_points=policy.num_points, rng=rng,
                    )
                    gap_after = visible_gap_metric(
                        after["target_points_world"], after["rest_points_world"]
                    )
                    result["visual_direction_guard"]["gap_samples"].append(
                        {"cycle": cycle, "before": gap_before, "after": gap_after}
                    )
                    if (
                        direction_sign > 0
                        and gap_before is not None and gap_after is not None
                        and gap_after < gap_before - 0.0015
                    ):
                        direction_sign = -1.0
                        result["visual_direction_guard"]["translation_sign_flipped"] = True
                    gap_before = gap_after
                except Exception as exc:
                    result["visual_direction_guard"]["capture_warning"] = str(exc)
            if result["termination_reason"] is None:
                result["termination_reason"] = "policy_horizon"
                result["timeout"] = True

        final_progress = float(resources["progress"]())
        result["final_articulation_state"] = final_progress
        result["articulation_progress"] = final_progress
        result["reach_35"] = bool(final_progress >= THRESHOLD)
        result["operation_success"] = bool(result["grasp_success"] and result["reach_35"])
        result["final_success"] = bool(result["operation_success"])
    except Exception as exc:
        result["exception"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        text = str(exc).lower()
        result["policy_failure"] = any(x in text for x in ("policy", "checkpoint", "cuda", "tensor"))
        result["controller_failure"] = any(x in text for x in ("controller", "jacobian", "rotation", "panda"))
        if result["termination_reason"] is None:
            result["termination_reason"] = "exception"
        if resources is not None:
            try:
                final_progress = float(resources["progress"]())
                result["final_articulation_state"] = final_progress
                result["articulation_progress"] = final_progress
                result["reach_35"] = bool(final_progress >= THRESHOLD)
                result["operation_success"] = bool(
                    result["grasp_success"] and result["reach_35"]
                )
                result["final_success"] = bool(result["operation_success"])
            except Exception:
                pass
    finally:
        if weld is not None:
            try:
                weld.destroy()
            except Exception:
                pass
        result["low_level_action_summary"] = _summary(action_records)
        result["runtime_sec"] = float(time.time() - started)
        resources = None
        gc.collect()
    return jsonable(result)
