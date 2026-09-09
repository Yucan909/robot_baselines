#!/usr/bin/env python3
"""DEV worker that faithfully restores the supplied receding-horizon loop.

The supplied bottom evaluator predicts at most 60 chunks, executes four
actions from each chunk, and caps the trial at 80 actions.  PA3FF/PADP emits
absolute panda_grasptarget SE(3) chunks rather than joint-position chunks, so
after the first grasp this adapter chooses the new chunk whose first pose is
most continuous with the current end-effector pose.  The continuity scales
come only from TRAIN action-step distributions.  Object progress is used only
for the evaluator's 0.40 termination condition, never as policy/rank input.
"""
from __future__ import annotations

import gc
import os
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import formal_worker_soft_weld_pd_devselected_v3 as implementation
import formal_worker_soft_weld_pd as backend
from candidate_rank_many_dev_v22 import rank_candidates
from pa3ff_policy_runtime_many_candidates_v17 import (
    PA3FFPADPRuntimeManyCandidatesV17,
)
from panda_controller_joint_pd_v8 import JointIKPDOperationMixin


legacy = implementation.legacy
worker = implementation.worker
RANK_MODE = os.environ.get("PA3FF_DEV_RANK_MODE", "long_motion")
DOOR_RANK_MODE = os.environ.get("PA3FF_DEV_DOOR_RANK_MODE", RANK_MODE)
DRAWER_RANK_MODE = os.environ.get("PA3FF_DEV_DRAWER_RANK_MODE", RANK_MODE)
CANDIDATE_COUNT = int(os.environ.get("PA3FF_DEV_CANDIDATE_COUNT", "32"))
CLIP_PRED_X0 = float(os.environ.get("PA3FF_DEV_CLIP_PRED_X0", "1.5"))
NOISE_SCALE = float(os.environ.get("PA3FF_DEV_NOISE_SCALE", "1.0"))
DDIM_STEPS = int(os.environ.get("PA3FF_DEV_DDIM_STEPS", "10"))
START_TIMESTEP = int(os.environ.get("PA3FF_DEV_START_TIMESTEP", "99"))
EXECUTE_STEPS = 4
MAX_POLICY_STEPS = 60
MAX_ACTIONS = 80
PUSH_TO_GRASP_MIN_CONSECUTIVE_CONTACT_FRAMES = 5
TRAIN_STEP_P75_M = {"door_open": 0.0277326032, "drawer_open": 0.0143516672}
TRAIN_ROTATION_STEP_P75_RAD = {
    "door_open": 0.0475014068,
    "drawer_open": 0.0014857982,
}


class _PregraspPlanningFailure(RuntimeError):
    pass


class JointPDFullBottomController(
    JointIKPDOperationMixin, implementation.FullBottomLogicController
):
    def step(self):
        """Advance physics and, when armed, sample correct-part Push contact.

        Push contact can be transient while the commanded pose is moving the
        articulated part.  Sampling only after ``move_grasp_pose_to`` returns
        misses precisely that valid engagement interval, so the TO_GRASP
        phase arms this recorder around the motion itself.
        """
        super().step()
        target_link = getattr(self, "_pa3ff_push_contact_target_link", None)
        frames = getattr(self, "_pa3ff_push_contact_frames", None)
        if target_link is not None and frames is not None:
            frames.append(legacy.effective_contact(
                self.scene,
                [self.hand_link, self.left_finger_link, self.right_finger_link],
                target_link,
            ))

    def move_grasp_pose_to(self, target_world_grasp, num_steps, **kwargs):
        if getattr(self, "_pose_move_count", 0) == 0:
            return super().move_grasp_pose_to(target_world_grasp, num_steps, **kwargs)
        self._pose_move_count += 1
        return self.move_operation_pose_to(target_world_grasp, num_steps, **kwargs)


def _push_motion_engagement(frames: list[dict], static_tail: dict) -> dict:
    """Combine in-motion and post-motion correct-target contact evidence."""
    consecutive = 0
    max_consecutive = 0
    valid_frames = 0
    for frame in frames:
        if frame["effective"]:
            valid_frames += 1
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0
    motion_success = (
        max_consecutive >= PUSH_TO_GRASP_MIN_CONSECUTIVE_CONTACT_FRAMES
    )
    success = bool(motion_success or static_tail["success"])
    effective_frames = [frame for frame in frames if frame["effective"]]
    return {
        "success": success,
        "definition": (
            "correct_target_part_effective_robot_contact_"
            f"consecutive_frames>={PUSH_TO_GRASP_MIN_CONSECUTIVE_CONTACT_FRAMES}_"
            "during_TO_GRASP_motion_or_static_tail_protocol"
        ),
        "movement_frames_total": len(frames),
        "movement_valid_contact_frames": valid_frames,
        "movement_valid_contact_fraction": (
            float(valid_frames / len(frames)) if frames else 0.0
        ),
        "movement_max_consecutive_contact_frames": max_consecutive,
        "movement_contact_success": bool(motion_success),
        "movement_last_effective_frame": (
            effective_frames[-1] if effective_frames else None
        ),
        "movement_max_impulse": max(
            (frame["max_impulse"] for frame in frames), default=0.0
        ),
        "static_tail_contact": static_tail,
        "progress_used_for_engagement": False,
    }


def _unpack_prediction(prediction: dict) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    prediction = dict(prediction)
    prediction.pop("poses_world_grasptarget")
    prediction.pop("gripper")
    poses = np.asarray(
        prediction.pop("candidate_poses_world_grasptarget"), dtype=np.float64
    )[:CANDIDATE_COUNT]
    gripper = np.asarray(
        prediction.pop("candidate_gripper"), dtype=np.float64
    )[:CANDIDATE_COUNT]
    prediction.pop("candidate_rank_order")
    diagnostics = list(prediction.pop("candidate_diagnostics"))[:CANDIDATE_COUNT]
    return poses, gripper, diagnostics, prediction


def _base_rank(
    point_cloud: np.ndarray,
    camera_pose: np.ndarray,
    poses: np.ndarray,
    task: str,
    diagnostics: list[dict],
) -> tuple[list[int], dict]:
    # Family-specific overrides apply to both directions.  The old equality
    # check silently routed door_close through the drawer setting.
    task_rank_mode = DOOR_RANK_MODE if task.startswith("door_") else DRAWER_RANK_MODE
    return rank_candidates(
        point_cloud, camera_pose, poses, task, mode=task_rank_mode,
        cached_diagnostics=diagnostics,
    )


def _continuity_rank(
    current_pose: np.ndarray,
    base_order: list[int],
    base_diagnostic: dict,
    poses: np.ndarray,
    task: str,
) -> tuple[list[int], dict]:
    position = np.linalg.norm(poses[:, 0, :3, 3] - current_pose[:3, 3], axis=1)
    rotation = np.asarray([
        np.linalg.norm(Rotation.from_matrix(
            pose[0, :3, :3] @ current_pose[:3, :3].T
        ).as_rotvec())
        for pose in poses
    ])
    base = np.asarray(base_diagnostic["scores"], dtype=np.float64)
    median = float(np.median(base))
    scale = float(np.median(np.abs(base - median)))
    base_robust = np.clip((base - median) / max(scale, 1e-6), -8.0, 8.0)
    score = (
        position / TRAIN_STEP_P75_M[task]
        + rotation / TRAIN_ROTATION_STEP_P75_RAD[task]
        + 0.05 * base_robust
    )
    order = np.argsort(score, kind="stable").astype(int).tolist()
    selected = order[0]
    return order, {
        "rule": (
            "absolute-SE3 receding continuity using TRAIN P75 adjacent-step "
            "position/rotation scales + 0.05 robust frozen base-rank score"
        ),
        "selected_candidate_index": selected,
        "selected_current_to_first_position_m": float(position[selected]),
        "selected_current_to_first_rotation_rad": float(rotation[selected]),
        "selected_score": float(score[selected]),
        "base_rank_position": int(base_order.index(selected)),
        "train_step_p75_m": TRAIN_STEP_P75_M[task],
        "train_rotation_step_p75_rad": TRAIN_ROTATION_STEP_P75_RAD[task],
        "formal_success_used": False,
        "object_progress_or_target_joint_used": False,
    }


def _capture(resources: dict, seed: int, policy_step: int) -> dict:
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), 2718, int(policy_step)])
    )
    return legacy.capture_articubot_observation(
        scene=resources["scene"], camera=resources["camera"], obj=resources["obj"],
        panda=resources["panda"], num_points=1024, rng=rng,
    )


def run_episode(runtime, case: dict, checkpoint_sha: str, catalog_sha: str) -> dict:
    started = time.time()
    seed = int(case["seed"])
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    result = {
        "episode_status": "running",
        "method": "PA3FF_DEV_RECEDING_HORIZON_V28",
        "task": case["task"], "primitive": case["primitive"], "goal": case["goal"],
        "shape_id": case["shape_id"], "target_link": case["target_link"],
        "target_index": case["target_index"], "trial_index": case["trial_index"],
        "seed": seed, "checkpoint": str(runtime.checkpoint),
        "checkpoint_sha256": checkpoint_sha, "training_step": runtime.training_step,
        "catalog_sha256": catalog_sha, "grasp_success": False,
        "final_success": False, "reached_target_40": False,
        "initial_progress": None, "final_progress": None,
        "directional_task_progress": None, "failure_reason": None,
        "exception": None,
        "policy_input_contract": [
            "current_object_point_cloud_xyz_1024", "current_robot_qpos_9d",
            "part_siglip_embedding", "instruction_siglip_embedding",
        ],
        "forbidden_formal_fields_used_by_policy": [],
        "bottom_logic": {
            "source": "attached senior receding-horizon soft_weld_pd defaults",
            "max_policy_steps": MAX_POLICY_STEPS, "execute_steps": EXECUTE_STEPS,
            "max_actions": MAX_ACTIONS, "steps_per_action": backend.ACTION_STEPS,
            "clip_pred_x0": CLIP_PRED_X0,
            "operate_control": "joint_PD_via_current_seed_absolute_SE3_IK",
            "approach_target_joint": "free_K0_D0.05", "palm_stop_approach": True,
            "soft_weld_scope": "pull_only_after_bilateral_real_contact",
        },
    }
    resources = None
    weld = None
    try:
        resources = legacy.create_scene(case)
        panda = resources["panda"]
        panda.wait(100)
        initial_progress = float(resources["progress"]())
        result["initial_progress"] = initial_progress

        observation = _capture(resources, seed, 0)
        prediction = runtime.predict(
            point_cloud_world=observation["point_cloud"],
            camera_pose_world=resources["camera_pose"],
            robot_qpos=np.asarray(panda.robot.get_qpos(), dtype=np.float32),
            task=case["task"], seed=seed,
        )
        poses_all, gripper_all, diagnostics, initial_meta = _unpack_prediction(prediction)
        candidate_order, rank_diag = _base_rank(
            observation["point_cloud"], resources["camera_pose"], poses_all,
            case["task"], diagnostics,
        )
        initial_meta["rank"] = rank_diag
        result["initial_policy_output_diagnostics"] = initial_meta

        planner_attempts = []
        chosen = None
        for rank, candidate_i in enumerate(candidate_order):
            poses_try = poses_all[candidate_i]
            pregrasp = poses_try[0].copy()
            pregrasp[:3, 3] -= legacy.PREGRASP_DISTANCE * poses_try[0, :3, 2]
            panda._pose_move_count = 0
            panda._pregrasp_planning_failed = False
            panda.pa3ff_artifact_dir = (
                Path(legacy.PA3FF_CURRENT_EPISODE_DIR)
                / f"planner_pregrasp_candidate_{candidate_i:03d}"
            )
            diag = panda.move_grasp_pose_to(pregrasp, legacy.PREGRASP_STEPS)
            planner_attempts.append({
                "rank": rank, "candidate_index": candidate_i, **diag,
            })
            if diag.get("planning_success", True):
                chosen = candidate_i
                result["pregrasp_control"] = diag
                break
        result["pregrasp_candidate_attempts"] = planner_attempts
        result["selected_candidate_index"] = chosen
        if chosen is None:
            result["episode_status"] = "complete"
            result["failure_reason"] = "pregrasp_planning_failed"
            result["final_progress"] = initial_progress
            result["directional_task_progress"] = 0.0
            raise _PregraspPlanningFailure("pregrasp_planning_failed")

        poses = poses_all[chosen]
        gripper = gripper_all[chosen]
        result["first_action"] = {
            "position": poses[0, :3, 3].tolist(),
            "rotation": poses[0, :3, :3].tolist(), "gripper": float(gripper[0]),
        }
        target_joint = resources["target_joint"]
        target_joint.set_drive_property(0.0, 0.05, 0.0)
        target_joint.set_drive_target(
            float(resources["obj"].get_qpos()[resources["target_index"]])
        )
        target_joint.set_drive_velocity_target(0.0)
        result["approach_control"] = panda.approach_with_palm_stop(
            poses[0], legacy.APPROACH_STEPS, resources["obj"].get_links()
        )
        result["progress_before_engagement"] = float(resources["progress"]())
        engagement = legacy.monitor_engagement(
            panda, resources["target_link"], case["primitive"], float(gripper[0])
        )
        push_engagement_control = []
        next_initial_action = 1
        if case["primitive"] == "push" and not engagement["success"]:
            # A PADP chunk may place a Push action just outside contact at a_0
            # and establish contact in the immediately following actions.  In
            # the supplied phase semantics these actions are still TO_GRASP;
            # formal OPERATE begins only after correct-part engagement.  Use
            # at most the frozen execution prefix and never inspect progress
            # to decide engagement.
            for action_i in range(1, min(1 + EXECUTE_STEPS, 16)):
                legacy.set_gripper_target(panda, float(gripper[action_i]))
                panda._pa3ff_push_contact_target_link = resources["target_link"]
                panda._pa3ff_push_contact_frames = []
                try:
                    diag = panda.move_grasp_pose_to(
                        poses[action_i], backend.ACTION_STEPS,
                        position_tolerance=0.007, rotation_tolerance=0.05,
                    )
                finally:
                    motion_contact_frames = list(
                        panda._pa3ff_push_contact_frames
                    )
                    panda._pa3ff_push_contact_target_link = None
                    panda._pa3ff_push_contact_frames = None
                static_contact = legacy.monitor_engagement(
                    panda, resources["target_link"], case["primitive"],
                    float(gripper[action_i]),
                )
                contact = _push_motion_engagement(
                    motion_contact_frames, static_contact
                )
                push_engagement_control.append({
                    "chunk_action_index": action_i, **diag,
                    "gripper": float(gripper[action_i]),
                    "contact_monitor": contact,
                })
                next_initial_action = action_i + 1
                engagement = contact
                if contact["success"]:
                    break
        result["push_engagement_control"] = push_engagement_control
        if case["primitive"] == "push":
            engagement = dict(engagement)
            engagement["phase"] = "TO_GRASP prediction prefix before formal operation"
            engagement["prefix_actions_executed"] = len(push_engagement_control)
            engagement["progress_used_for_engagement"] = False
        result["engagement_monitor"] = engagement
        result["grasp_success"] = bool(engagement["success"])

        if case["primitive"] == "pull":
            weld, weld_diag = backend.SoftContactWeld.try_create(
                panda, resources["target_link"]
            )
            result["grasp_finger_contact_weld"] = weld_diag
            if weld is not None:
                width = float(np.mean(np.asarray(panda.robot.get_qpos())[7:9]))
                result["finger_grasp_lock"] = backend.enable_finger_lock(panda, width)
                for _ in range(backend.ATTACHMENT_SETTLE_STEPS):
                    panda.step()
                weld_diag.update(weld.errors())
            else:
                result["finger_grasp_lock"] = backend.enable_finger_lock(panda, 0.0)
        else:
            result["grasp_finger_contact_weld"] = {
                "enabled": False, "created": False, "reason": "push_primitive",
            }
            result["finger_grasp_lock"] = {
                "enabled": False, "reason": "push_primitive",
            }

        controls = []
        replans = []
        n_actions = len(push_engagement_control)
        policy_step = 0
        active_poses, active_gripper = poses, gripper
        action_indexes = list(range(next_initial_action, min(1 + EXECUTE_STEPS, 16)))
        operation_allowed = not (
            case["primitive"] == "push" and not result["grasp_success"]
        )
        while operation_allowed and policy_step < MAX_POLICY_STEPS and n_actions < MAX_ACTIONS:
            for action_i in action_indexes:
                if n_actions >= MAX_ACTIONS:
                    break
                if case["primitive"] == "push":
                    legacy.set_gripper_target(panda, float(active_gripper[action_i]))
                diag = panda.move_grasp_pose_to(
                    active_poses[action_i], backend.ACTION_STEPS,
                    position_tolerance=0.007, rotation_tolerance=0.05,
                )
                n_actions += 1
                row = {
                    "policy_step": policy_step, "chunk_action_index": action_i,
                    "global_action_index": n_actions, **diag,
                    "gripper": float(active_gripper[action_i]),
                }
                if weld is not None:
                    row.update(weld.errors())
                controls.append(row)
                progress = float(resources["progress"]())
                directional_now = (
                    progress - initial_progress if case["goal"] == "open"
                    else initial_progress - progress
                )
                if directional_now >= legacy.COMMAND_TARGET:
                    break
            progress = float(resources["progress"]())
            directional_now = (
                progress - initial_progress if case["goal"] == "open"
                else initial_progress - progress
            )
            policy_step += 1
            if directional_now >= legacy.COMMAND_TARGET or n_actions >= MAX_ACTIONS:
                break

            observation = _capture(resources, seed, policy_step)
            prediction = runtime.predict(
                point_cloud_world=observation["point_cloud"],
                camera_pose_world=resources["camera_pose"],
                robot_qpos=np.asarray(panda.robot.get_qpos(), dtype=np.float32),
                task=case["task"], seed=seed + policy_step * 15485863,
            )
            poses_all, gripper_all, diagnostics, meta = _unpack_prediction(prediction)
            base_order, base_diag = _base_rank(
                observation["point_cloud"], resources["camera_pose"], poses_all,
                case["task"], diagnostics,
            )
            continuity_order, continuity_diag = _continuity_rank(
                panda.get_grasp_pose_matrix(), base_order, base_diag, poses_all,
                case["task"],
            )
            chosen = continuity_order[0]
            active_poses = poses_all[chosen]
            active_gripper = gripper_all[chosen]
            action_indexes = list(range(min(EXECUTE_STEPS, 16)))
            replans.append({
                "policy_step": policy_step, "policy_seed": seed + policy_step * 15485863,
                "selected_candidate_index": chosen,
                "candidate_count": CANDIDATE_COUNT, "base_rank": base_diag,
                "continuity_rank": continuity_diag,
                "sampler": meta,
            })

        panda.wait(100)
        result["operation_control"] = controls
        result["receding_replans"] = replans
        result["n_policy_steps"] = policy_step
        result["n_actions"] = n_actions
        result["operation_timed_out"] = bool(n_actions >= MAX_ACTIONS)
        final_progress = float(resources["progress"]())
        directional = (
            final_progress - initial_progress if case["goal"] == "open"
            else initial_progress - final_progress
        )
        result["final_progress"] = final_progress
        result["directional_task_progress"] = float(directional)
        result["reached_target_40"] = bool(directional >= legacy.COMMAND_TARGET)
        result["final_success"] = bool(
            result["grasp_success"] and directional >= legacy.SUCCESS_THRESHOLD
        )
        if result["final_success"]:
            result["failure_reason"] = None
        elif not result["grasp_success"]:
            result["failure_reason"] = (
                "grasp_failed" if case["primitive"] == "pull"
                else "push_engagement_failed"
            )
        elif directional < 0:
            result["failure_reason"] = "wrong_direction"
        else:
            result["failure_reason"] = "insufficient_directional_progress"
        result["episode_status"] = "complete"
    except _PregraspPlanningFailure:
        pass
    except Exception as exc:
        result["episode_status"] = "crash"
        result["failure_reason"] = "infrastructure_exception"
        result["exception"] = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        if resources is not None:
            try:
                result["final_progress"] = float(resources["progress"]())
            except Exception:
                pass
    finally:
        result["runtime_seconds"] = float(time.time() - started)
        weld = None
        resources = None
        gc.collect()
        torch.cuda.empty_cache()
    return legacy.jsonable(result)


def main() -> None:
    valid_rank_modes = {
        "v4", "train_geometry", "motion_train", "all_train",
        "semantic_train", "long_motion", "long_motion_all",
    }
    if RANK_MODE not in valid_rank_modes:
        raise ValueError(RANK_MODE)
    if DOOR_RANK_MODE not in valid_rank_modes:
        raise ValueError(DOOR_RANK_MODE)
    if DRAWER_RANK_MODE not in valid_rank_modes:
        raise ValueError(DRAWER_RANK_MODE)
    if CANDIDATE_COUNT not in {32, 128}:
        raise ValueError(CANDIDATE_COUNT)
    # V4 uses a TRAIN-distribution-derived +/-3.0 support bound.  Keep the
    # historical values accepted so older frozen protocols remain resumable.
    if CLIP_PRED_X0 not in {1.0, 1.5, 3.0}:
        raise ValueError(CLIP_PRED_X0)
    if NOISE_SCALE not in {0.0, 0.1, 0.25, 0.5, 1.0}:
        raise ValueError(NOISE_SCALE)
    if DDIM_STEPS not in {10, 20}:
        raise ValueError(DDIM_STEPS)
    if START_TIMESTEP not in {90, 99}:
        raise ValueError(START_TIMESTEP)
    PA3FFPADPRuntimeManyCandidatesV17.N_CANDIDATES = CANDIDATE_COUNT
    PA3FFPADPRuntimeManyCandidatesV17.CLIP_PRED_X0 = CLIP_PRED_X0
    PA3FFPADPRuntimeManyCandidatesV17.NOISE_SCALE = NOISE_SCALE
    PA3FFPADPRuntimeManyCandidatesV17.DDIM_STEPS = DDIM_STEPS
    PA3FFPADPRuntimeManyCandidatesV17.START_TIMESTEP = START_TIMESTEP
    implementation.FullBottomLogicController = JointPDFullBottomController
    original_create_scene = legacy.create_scene

    def create_configured_scene(case: dict):
        resources = original_create_scene(case)
        planner_case = dict(case)
        planner_case["initial_target_ratio"] = float(resources["progress"]())
        episode_dir = Path(legacy.PA3FF_CURRENT_EPISODE_DIR)
        resources["panda"].configure_pa3ff_planner(
            planner_case, episode_dir / "planner_pregrasp"
        )
        return resources

    legacy.create_scene = create_configured_scene
    legacy.run_episode = run_episode
    worker.METHOD = "PA3FF_DEV_RECEDING_HORIZON_V28"
    worker.EXECUTION_PROTOCOL = (
        f"attached_receding_n{CANDIDATE_COUNT}_door-{DOOR_RANK_MODE}_"
        f"drawer-{DRAWER_RANK_MODE}_clip{CLIP_PRED_X0:g}_noise{NOISE_SCALE:g}_"
        f"ddim{DDIM_STEPS}_start{START_TIMESTEP}_joint_pd"
    )
    worker.CONTROLLER_CLASS = JointPDFullBottomController
    legacy.PA3FFPADPRuntime = PA3FFPADPRuntimeManyCandidatesV17
    worker.main()


if __name__ == "__main__":
    main()
