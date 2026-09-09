#!/usr/bin/env python3
"""PA3FF V2 policy evaluated with the supplied soft-weld/PD bottom logic."""

from __future__ import annotations

import gc
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import formal_worker_open_retest_v3 as runtime_sources
from panda_controller_ompl_v5 import OMPLPregraspPandaController
from soft_weld_pd_backend import (
    SoftContactWeld,
    apply_finger_lock_force,
    enable_finger_lock,
)


worker = runtime_sources.worker
legacy = worker.legacy
ACTION_STEPS = 160
ATTACHMENT_SETTLE_STEPS = 4


class SoftWeldPDController(OMPLPregraspPandaController):
    """Existing OMPL/Cartesian adapter with the supplied post-grasp lock force."""

    def step(self):
        if hasattr(self, "soft_weld_finger_lock_target"):
            for joint in self.finger_joints:
                joint.set_drive_velocity_target(0.0)
                joint.set_drive_target(float(self.soft_weld_finger_lock_target))
            apply_finger_lock_force(self)
            # Bypass PandaTwoFingerController.step's second passive-force write.
            self.scene.step()
            self._video_physics_counter += 1
            if (
                self._video_step_callback is not None
                and self._video_physics_counter % self._video_every_n_steps == 0
            ):
                self._video_step_callback()
            return
        super().step()


def run_episode(runtime, case: dict, checkpoint_sha: str, catalog_sha: str) -> dict:
    started = time.time()
    seed = int(case["seed"])
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    result = {
        "episode_status": "running",
        "method": "PA3FF_reproduction_v2_soft_weld_pd",
        "task": case["task"],
        "primitive": case["primitive"],
        "goal": case["goal"],
        "shape_id": case["shape_id"],
        "target_link": case["target_link"],
        "target_index": case["target_index"],
        "trial_index": case["trial_index"],
        "seed": seed,
        "checkpoint": str(runtime.checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "training_step": runtime.training_step,
        "catalog_sha256": catalog_sha,
        "grasp_success": False,
        "final_success": False,
        "reached_target_40": False,
        "initial_progress": None,
        "final_progress": None,
        "directional_task_progress": None,
        "failure_reason": None,
        "exception": None,
        "policy_input_contract": [
            "object_point_cloud_xyz_1024",
            "robot_qpos_9d",
            "part_siglip_embedding",
            "instruction_siglip_embedding",
        ],
        "forbidden_formal_fields_used_by_policy": [],
        "bottom_logic": {
            "name": "senior_soft_weld_pd_PA3FF_adapter_v1",
            "policy_output": "PA3FF native panda_grasptarget SE(3)+gripper",
            "operate_control": "drive",
            "steps_per_action": ACTION_STEPS,
            "soft_weld_scope": "pull_only_after_bilateral_real_contact",
            "push_scope": "real correct-part engagement; no artificial pull grasp",
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
            scene=resources["scene"],
            camera=resources["camera"],
            obj=resources["obj"],
            panda=panda,
            num_points=1024,
            rng=rng,
        )
        prediction = runtime.predict(
            point_cloud_world=obs["point_cloud"],
            camera_pose_world=resources["camera_pose"],
            robot_qpos=np.asarray(panda.robot.get_qpos(), dtype=np.float32),
            task=case["task"],
            seed=seed,
        )
        poses = prediction.pop("poses_world_grasptarget")
        gripper = prediction.pop("gripper")
        result["policy_output_diagnostics"] = prediction
        result["first_action"] = {
            "position": poses[0, :3, 3].tolist(),
            "rotation": poses[0, :3, :3].tolist(),
            "gripper": float(gripper[0]),
        }

        pregrasp = poses[0].copy()
        pregrasp[:3, 3] -= legacy.PREGRASP_DISTANCE * poses[0, :3, 2]
        result["pregrasp_control"] = panda.move_grasp_pose_to(pregrasp, legacy.PREGRASP_STEPS)
        result["approach_control"] = panda.move_grasp_pose_to(poses[0], legacy.APPROACH_STEPS)
        result["progress_before_engagement"] = resources["progress"]()
        engagement = legacy.monitor_engagement(
            panda, resources["target_link"], case["primitive"], float(gripper[0])
        )
        result["engagement_monitor"] = engagement
        result["grasp_success"] = bool(engagement["success"])

        if case["primitive"] == "pull":
            weld, weld_diag = SoftContactWeld.try_create(panda, resources["target_link"])
            result["grasp_finger_contact_weld"] = weld_diag
            if weld is not None:
                current_width = float(np.mean(np.asarray(panda.robot.get_qpos())[7:9]))
                result["finger_grasp_lock"] = enable_finger_lock(panda, current_width)
                for _ in range(ATTACHMENT_SETTLE_STEPS):
                    panda.step()
                weld_diag.update(weld.errors())
            else:
                # Supplied evaluator still locks closed fingers if contact weld
                # creation fails; strict-grasp/weld failure does not abort operate.
                result["finger_grasp_lock"] = enable_finger_lock(panda, 0.0)
        else:
            result["grasp_finger_contact_weld"] = {
                "enabled": False,
                "created": False,
                "reason": "push_primitive_uses_real_engagement_not_pull_attachment",
            }
            result["finger_grasp_lock"] = {"enabled": False, "reason": "push_primitive"}

        target_joint = resources["target_joint"]
        target_joint.set_drive_property(0.0, 0.05, 0.0)
        target_joint.set_drive_target(float(resources["obj"].get_qpos()[resources["target_index"]]))
        target_joint.set_drive_velocity_target(0.0)
        result["operate_target_drive"] = {
            "stiffness": 0.0,
            "damping": 0.05,
            "force_limit": 0.0,
        }
        result["operate_joint_drives"] = {
            "arm_stiffness": [1800.0] * 7,
            "arm_damping": [360.0] * 7,
            "drive_soft_base_joint": False,
        }

        control = []
        for action_i in range(1, 16):
            if case["primitive"] == "push":
                legacy.set_gripper_target(panda, float(gripper[action_i]))
            diag = panda.move_grasp_pose_to(
                poses[action_i],
                ACTION_STEPS,
                position_tolerance=0.007,
                rotation_tolerance=0.05,
            )
            row = {"action_index": action_i, **diag, "gripper": float(gripper[action_i])}
            if weld is not None:
                row.update(weld.errors())
            control.append(row)
        panda.wait(100)
        result["operation_control"] = control
        if weld is not None:
            end_errors = weld.errors()
            result["grasp_finger_contact_weld"].update(
                {
                    "weld_position_error_m_end": end_errors["weld_position_error_m"],
                    "weld_angular_error_rad_end": end_errors["weld_angular_error_rad"],
                }
            )

        final_progress = resources["progress"]()
        directional = (
            final_progress - initial_progress
            if case["goal"] == "open"
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
                "grasp_failed" if case["primitive"] == "pull" else "push_engagement_failed"
            )
        elif directional < 0:
            result["failure_reason"] = "wrong_direction"
        else:
            result["failure_reason"] = "insufficient_directional_progress"
        result["episode_status"] = "complete"
    except Exception as exc:
        result["episode_status"] = "crash"
        result["failure_reason"] = "infrastructure_exception"
        result["exception"] = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
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
        resources["panda"].configure_pa3ff_planner(
            planner_case, episode_dir / "planner_pregrasp"
        )
        return resources

    legacy.create_scene = create_configured_scene
    legacy.run_episode = run_episode
    worker.METHOD = "PA3FF_reproduction_v2_soft_weld_pd"
    worker.EXECUTION_PROTOCOL = "senior_soft_weld_pd_PA3FF_adapter_v1"
    worker.CONTROLLER_CLASS = SoftWeldPDController
    legacy.PA3FFPADPRuntime = runtime_sources.BASE_RUNTIME
    worker.main()


if __name__ == "__main__":
    main()
