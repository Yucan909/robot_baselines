#!/usr/bin/env python3
"""PA3FF Open V4 worker with DEV-frozen sampling/ranking and supplied bottom logic.

Relative to the earlier soft-weld adapter this restores two defaults from the
attached senior implementation: the target joint is free during approach and
the palm stops/retreats on object contact before finger closure.  The OMPL
pregrasp bug fixes remain active.  No target state is exposed to the policy.
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

import formal_worker_soft_weld_pd as backend
import panda_controller_ompl_v5 as ompl_module

ompl_module.PLANNER = Path(
    "/home/feng/robot_baselines/repos/pa3ff_official/reproduction/"
    "motion_planner_collision_worker_pa3ff_fixed_v2.py"
)

from pa3ff_policy_runtime_devselected_v3 import PA3FFPADPRuntimeDEVSelectedV3
from candidate_rank_devselected_v4 import rank_candidates


worker = backend.worker
legacy = backend.legacy
ACTION_STEPS = backend.ACTION_STEPS
ATTACHMENT_SETTLE_STEPS = backend.ATTACHMENT_SETTLE_STEPS
PALM_RETREAT_STEP_M = 0.001
PALM_RETREAT_MAX_M = 0.025


class _PregraspPlanningFailure(RuntimeError):
    pass


def _actor_id(actor) -> int:
    return int(actor.get_id()) if hasattr(actor, "get_id") else int(actor.id)


class FullBottomLogicController(backend.SoftWeldPDController):
    """Soft-weld controller plus the supplied palm-stop approach behavior."""

    def palm_object_contact(self, object_links) -> dict:
        hand_id = _actor_id(self.hand_link)
        object_ids = {_actor_id(link) for link in object_links}
        points = 0
        maximum = 0.0
        for contact in self.scene.get_contacts():
            ids = {_actor_id(contact.actor0), _actor_id(contact.actor1)}
            if hand_id not in ids or not (ids & object_ids):
                continue
            for point in contact.points:
                impulse = float(np.linalg.norm(np.asarray(point.impulse, dtype=np.float64)))
                if np.isfinite(impulse) and impulse > 1e-8:
                    points += 1
                    maximum = max(maximum, impulse)
        return {"contact": points > 0, "effective_points": points, "max_impulse": maximum}

    def approach_with_palm_stop(self, target_world_grasp, num_steps, object_links) -> dict:
        """Resolved-rate approach, stopping on the first effective palm contact."""
        target = np.asarray(target_world_grasp, dtype=np.float64)
        target_r, target_p = target[:3, :3], target[:3, 3]
        start = self.get_grasp_pose_matrix()
        approach = target_p - start[:3, 3]
        approach_norm = float(np.linalg.norm(approach))
        retreat_axis = approach / max(approach_norm, 1e-12)
        best_pos = np.inf
        best_rot = np.inf
        hit = None
        used = 0
        reached = False
        for step in range(int(num_steps)):
            current = self.get_grasp_pose_matrix()
            pos_error = target_p - current[:3, 3]
            rotvec = Rotation.from_matrix(target_r @ current[:3, :3].T).as_rotvec()
            pos_norm = float(np.linalg.norm(pos_error))
            rot_norm = float(np.linalg.norm(rotvec))
            best_pos = min(best_pos, pos_norm)
            best_rot = min(best_rot, rot_norm)
            if pos_norm <= 0.005 and rot_norm <= 0.03:
                reached = True
                break
            linear = self.position_gain * pos_error
            ln = float(np.linalg.norm(linear))
            if ln > self.max_cartesian_speed:
                linear *= self.max_cartesian_speed / ln
            angular = self.rotation_gain * rotvec
            an = float(np.linalg.norm(angular))
            if an > self.max_angular_speed:
                angular *= self.max_angular_speed / an
            qvel = np.linalg.pinv(
                self._get_grasp_twist_jacobian(), rcond=self.pinv_rcond
            ) @ np.concatenate([angular, linear])
            self._apply_arm_velocity(qvel)
            self.step()
            used = step + 1
            contact = self.palm_object_contact(object_links)
            if contact["contact"]:
                hit = {"physics_step": used, **contact}
                break
        self.clear_arm_velocity()

        retreat_steps = 0
        cleared = hit is None
        if hit is not None:
            # The archive retreats in 1 mm increments up to 25 mm and freezes
            # the arm once clear.  Keep the current orientation while retreating.
            for retreat_steps in range(1, int(round(PALM_RETREAT_MAX_M / PALM_RETREAT_STEP_M)) + 1):
                hold = self.get_grasp_pose_matrix()
                hold[:3, 3] -= PALM_RETREAT_STEP_M * retreat_axis
                super().move_grasp_pose_to(
                    hold, 20, position_tolerance=0.0006, rotation_tolerance=0.03
                )
                if not self.palm_object_contact(object_links)["contact"]:
                    cleared = True
                    break

        final = self.get_grasp_pose_matrix()
        return {
            "reached_control_tolerance": bool(reached),
            "used_steps": int(used),
            "final_position_error": float(np.linalg.norm(target_p - final[:3, 3])),
            "final_rotation_error": float(np.linalg.norm(Rotation.from_matrix(target_r @ final[:3, :3].T).as_rotvec())),
            "best_position_error": float(best_pos),
            "best_rotation_error": float(best_rot),
            "controller_version": "world_cartesian_grasptarget_v2_palm_stop",
            "palm_stop_enabled": True,
            "palm_hit": hit is not None,
            "palm_hit_diagnostic": hit,
            "palm_retreat_step_mm": 1.0,
            "palm_retreat_max_mm": 25.0,
            "palm_retreat_used_mm": float(retreat_steps),
            "palm_cleared": bool(cleared),
        }


def run_episode(runtime, case: dict, checkpoint_sha: str, catalog_sha: str) -> dict:
    started = time.time()
    seed = int(case["seed"])
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    result = {
        "episode_status": "running",
        "method": "PA3FF_reproduction_v4_devselected_geometry_rank_full_soft_weld_pd",
        "task": case["task"], "primitive": case["primitive"], "goal": case["goal"],
        "shape_id": case["shape_id"], "target_link": case["target_link"],
        "target_index": case["target_index"], "trial_index": case["trial_index"],
        "seed": seed, "checkpoint": str(runtime.checkpoint),
        "checkpoint_sha256": checkpoint_sha, "training_step": runtime.training_step,
        "catalog_sha256": catalog_sha, "grasp_success": False, "final_success": False,
        "reached_target_40": False, "initial_progress": None, "final_progress": None,
        "directional_task_progress": None, "failure_reason": None, "exception": None,
        "policy_input_contract": ["object_point_cloud_xyz_1024", "robot_qpos_9d", "part_siglip_embedding", "instruction_siglip_embedding"],
        "forbidden_formal_fields_used_by_policy": [],
        "bottom_logic": {
            "source": "attached senior soft_weld_pd defaults",
            "approach_target_joint": "free_K0_D0.05",
            "palm_stop_approach": True,
            "palm_retreat_step_mm": 1.0, "palm_retreat_max_mm": 25.0,
            "soft_weld_scope": "pull_only_after_bilateral_real_contact",
            "steps_per_action": ACTION_STEPS,
        },
    }
    resources = None
    weld = None
    try:
        resources = legacy.create_scene(case)
        panda = resources["panda"]
        panda.wait(100)
        initial_progress = resources["progress"]()
        result["initial_progress"] = initial_progress
        rng = np.random.default_rng(np.random.SeedSequence([seed, 2718]))
        obs = legacy.capture_articubot_observation(
            scene=resources["scene"], camera=resources["camera"], obj=resources["obj"],
            panda=panda, num_points=1024, rng=rng,
        )
        prediction = runtime.predict(
            point_cloud_world=obs["point_cloud"], camera_pose_world=resources["camera_pose"],
            robot_qpos=np.asarray(panda.robot.get_qpos(), dtype=np.float32),
            task=case["task"], seed=seed,
        )
        prediction.pop("poses_world_grasptarget")
        prediction.pop("gripper")
        candidate_poses = np.asarray(prediction.pop("candidate_poses_world_grasptarget"), dtype=np.float64)
        candidate_gripper = np.asarray(prediction.pop("candidate_gripper"), dtype=np.float64)
        cached_v3_order = [int(x) for x in prediction.pop("candidate_rank_order")]
        candidate_order, rank_diagnostic = rank_candidates(
            obs["point_cloud"], resources["camera_pose"], candidate_poses, case["task"]
        )
        prediction["cached_v3_candidate_rank_order_superseded"] = cached_v3_order
        prediction["v4_candidate_rank_diagnostic"] = rank_diagnostic
        result["policy_output_diagnostics"] = prediction

        # The attached bottom pipeline evaluates only collision-free grasp
        # proposals.  PADP is stochastic, so try its 32 unchanged samples in
        # the task-specific Open-DEV-frozen observation/TRAIN-only rank order
        # and retain the first plan that the unchanged collision planner can
        # execute.
        planner_attempts = []
        chosen = None
        for rank, candidate_i in enumerate(candidate_order):
            poses_try = candidate_poses[candidate_i]
            pregrasp_try = poses_try[0].copy()
            pregrasp_try[:3, 3] -= legacy.PREGRASP_DISTANCE * poses_try[0, :3, 2]
            panda._pose_move_count = 0
            panda._pregrasp_planning_failed = False
            panda.pa3ff_artifact_dir = Path(legacy.PA3FF_CURRENT_EPISODE_DIR) / f"planner_pregrasp_candidate_{candidate_i:02d}"
            diag = panda.move_grasp_pose_to(pregrasp_try, legacy.PREGRASP_STEPS)
            planner_attempts.append({"rank": rank, "candidate_index": candidate_i, **diag})
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
        poses = candidate_poses[chosen]
        gripper = candidate_gripper[chosen]
        result["first_action"] = {
            "position": poses[0, :3, 3].tolist(), "rotation": poses[0, :3, :3].tolist(),
            "gripper": float(gripper[0]),
        }

        # Attached logic default: OPEN and TO_GRASP already use the operate
        # target-joint physics, rather than the scene-init lock.
        target_joint = resources["target_joint"]
        target_joint.set_drive_property(0.0, 0.05, 0.0)
        target_joint.set_drive_target(float(resources["obj"].get_qpos()[resources["target_index"]]))
        target_joint.set_drive_velocity_target(0.0)
        result["approach_target_drive"] = {"stiffness": 0.0, "damping": 0.05, "force_limit": 0.0}
        result["approach_control"] = panda.approach_with_palm_stop(
            poses[0], legacy.APPROACH_STEPS, resources["obj"].get_links()
        )
        result["progress_before_engagement"] = resources["progress"]()
        engagement = legacy.monitor_engagement(
            panda, resources["target_link"], case["primitive"], float(gripper[0])
        )
        result["engagement_monitor"] = engagement
        result["grasp_success"] = bool(engagement["success"])

        if case["primitive"] == "pull":
            weld, weld_diag = backend.SoftContactWeld.try_create(panda, resources["target_link"])
            result["grasp_finger_contact_weld"] = weld_diag
            if weld is not None:
                width = float(np.mean(np.asarray(panda.robot.get_qpos())[7:9]))
                result["finger_grasp_lock"] = backend.enable_finger_lock(panda, width)
                for _ in range(ATTACHMENT_SETTLE_STEPS):
                    panda.step()
                weld_diag.update(weld.errors())
            else:
                result["finger_grasp_lock"] = backend.enable_finger_lock(panda, 0.0)
        else:
            result["grasp_finger_contact_weld"] = {"enabled": False, "created": False, "reason": "push_primitive"}
            result["finger_grasp_lock"] = {"enabled": False, "reason": "push_primitive"}

        control = []
        for action_i in range(1, 16):
            if case["primitive"] == "push":
                legacy.set_gripper_target(panda, float(gripper[action_i]))
            diag = panda.move_grasp_pose_to(
                poses[action_i], ACTION_STEPS, position_tolerance=0.007, rotation_tolerance=0.05
            )
            row = {"action_index": action_i, **diag, "gripper": float(gripper[action_i])}
            if weld is not None:
                row.update(weld.errors())
            control.append(row)
        panda.wait(100)
        result["operation_control"] = control
        final_progress = resources["progress"]()
        directional = final_progress - initial_progress if case["goal"] == "open" else initial_progress - final_progress
        result["final_progress"] = final_progress
        result["directional_task_progress"] = float(directional)
        result["reached_target_40"] = bool(directional >= legacy.COMMAND_TARGET)
        result["final_success"] = bool(result["grasp_success"] and directional >= legacy.SUCCESS_THRESHOLD)
        if result["final_success"]:
            result["failure_reason"] = None
        elif not result["grasp_success"]:
            result["failure_reason"] = "grasp_failed" if case["primitive"] == "pull" else "push_engagement_failed"
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
        result["exception"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        if resources is not None:
            try:
                result["final_progress"] = resources["progress"]()
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
    original_create_scene = legacy.create_scene

    def create_configured_scene(case: dict):
        resources = original_create_scene(case)
        planner_case = dict(case)
        planner_case["initial_target_ratio"] = float(resources["progress"]())
        episode_dir = Path(legacy.PA3FF_CURRENT_EPISODE_DIR)
        resources["panda"].configure_pa3ff_planner(planner_case, episode_dir / "planner_pregrasp")
        return resources

    legacy.create_scene = create_configured_scene
    legacy.run_episode = run_episode
    worker.METHOD = "PA3FF_reproduction_v4_devselected_geometry_rank_full_soft_weld_pd"
    worker.EXECUTION_PROTOCOL = "attached_soft_weld_pd_defaults_plus_deterministic_ompl_v4_dev_geometry_rank"
    worker.CONTROLLER_CLASS = FullBottomLogicController
    if os.environ.get("PA3FF_PREDICTION_CACHE_ROOT"):
        from pa3ff_policy_runtime_cached_v3 import PA3FFPADPRuntimeCachedV3
        legacy.PA3FFPADPRuntime = PA3FFPADPRuntimeCachedV3
    else:
        legacy.PA3FFPADPRuntime = PA3FFPADPRuntimeDEVSelectedV3
    worker.main()


if __name__ == "__main__":
    main()
