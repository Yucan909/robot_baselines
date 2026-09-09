"""Collision-aware pregrasp planning plus physical PA3FF action tracking."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from panda_controller_corrected_v2 import CorrectedPandaTwoFingerController


HOME = Path("/home/feng")
PLANNER = HOME / "robot_baselines/common_env/where2act_four_task_noaff_v7_floating_progress_v2/motion_planner_collision_worker.py"
PLANNER_PYTHON = HOME / "miniconda3/envs/where2act_planner/bin/python"
ARTICUBOT_REPO = HOME / "robot_baselines/repos/articubot"


class OMPLPregraspPandaController(CorrectedPandaTwoFingerController):
    """OMPL for the first pregrasp move; corrected resolved-rate control thereafter."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pa3ff_case = None
        self.pa3ff_artifact_dir = None
        self._pose_move_count = 0
        self._pregrasp_planning_failed = False

    def configure_pa3ff_planner(self, case: dict, artifact_dir: Path) -> None:
        self.pa3ff_case = dict(case)
        self.pa3ff_artifact_dir = Path(artifact_dir)

    def _track_joint_target(self, target: np.ndarray, *, strict: bool = False) -> dict:
        target = np.asarray(target, dtype=np.float64).reshape(7)
        max_steps = 1500 if strict else 500
        min_steps = 50 if strict else 15
        tol_norm = 0.005 if strict else 0.02
        tol_max = 0.003 if strict else 0.015
        for index, joint in enumerate(self.arm_joints):
            joint.set_drive_velocity_target(0.0)
            joint.set_drive_target(float(target[index]))
        self.open_gripper()
        for step in range(max_steps):
            self.step()
            actual = np.asarray(self.robot.get_qpos(), dtype=np.float64)[:7]
            error = actual - target
            norm = float(np.linalg.norm(error))
            maxabs = float(np.max(np.abs(error)))
            if step + 1 >= min_steps and norm <= tol_norm and maxabs <= tol_max:
                return {"converged": True, "steps": step + 1, "error_norm": norm, "error_maxabs": maxabs}
        return {"converged": False, "steps": max_steps, "error_norm": norm, "error_maxabs": maxabs}

    def _noop_diagnostic(self, target: np.ndarray, reason: str) -> dict:
        current = self.get_grasp_pose_matrix()
        diagnostic = {
            "reached_control_tolerance": False,
            "used_steps": 0,
            "final_position_error": float(np.linalg.norm(target[:3, 3] - current[:3, 3])),
            "final_rotation_error": float(np.linalg.norm(Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec())),
            "best_position_error": float(np.linalg.norm(target[:3, 3] - current[:3, 3])),
            "best_rotation_error": float(np.linalg.norm(Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec())),
            "controller_version": "collision_aware_ompl_pregrasp_v5",
            "planning_success": False,
            "planning_failure": reason,
        }
        if hasattr(self, "_last_track_joint_diagnostic"):
            diagnostic["last_track_joint_diagnostic"] = self._last_track_joint_diagnostic
        return diagnostic

    def _plan_and_execute_pregrasp(self, target: np.ndarray) -> dict:
        if self.pa3ff_case is None or self.pa3ff_artifact_dir is None:
            raise RuntimeError("PA3FF planner was not configured for this episode")
        case = self.pa3ff_case
        output = self.pa3ff_artifact_dir
        output.mkdir(parents=True, exist_ok=True)
        current_qpos = np.asarray(self.robot.get_qpos(), dtype=np.float64).reshape(9)
        exact_object_qpos = case.get("initial_object_qpos")
        initial_ratio = case.get("initial_target_ratio")
        if initial_ratio is None:
            initial_ratio = 0.0
        request = {
            "shape_id": case["shape_id"],
            "target_link": case["target_link"],
            "initial_ratio": float(initial_ratio),
            "initial_object_qpos": exact_object_qpos,
            "pregrasp_pose_world": target.tolist(),
            "contact_pose_world": (
                target
                + np.pad(
                    np.outer(target[:3, 2], np.asarray([0.0, 0.0, 0.0, 0.08])),
                    ((0, 1), (0, 0)),
                    mode="constant",
                )
            ).tolist(),
            "approach_axis_world": target[:3, 2].tolist(),
            "robot_start_qpos9": current_qpos.tolist(),
            "network_trained": True,
            "goal_frame": "panda_grasptarget",
        }
        catalog_row = {
            "shape_id": case["shape_id"],
            "target_link": case["target_link"],
            "base_pose": case["base_pose"],
            "robot_initial_qpos": current_qpos.tolist(),
        }
        request_path = output / "planning_request.json"
        catalog_path = output / "planning_pose_catalog.jsonl"
        request_path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
        catalog_path.write_text(json.dumps(catalog_row) + "\n", encoding="utf-8")
        env = os.environ.copy()
        # formal_env.sh deliberately exposes the ArticuBot Python-3.9 packages to
        # PA3FF.  The planner is Python 3.10, so inheriting that site-packages path
        # breaks NumPy's binary extension import.  The planner only needs the
        # ArticuBot source repository in PYTHONPATH.
        env["PYTHONPATH"] = str(ARTICUBOT_REPO)
        env.pop("PYTHONHOME", None)
        command = [
            str(PLANNER_PYTHON), str(PLANNER),
            "--request", str(request_path),
            "--pose-catalog", str(catalog_path),
            "--output-dir", str(output),
            "--planner", "RRTConnect",
            "--planning-time", "5.0",
            "--ik-attempts", "100",
            "--seed", str(int(case["seed"])),
        ]
        proc = subprocess.run(command, cwd=str(PLANNER.parent), env=env, text=True, capture_output=True, timeout=180)
        (output / "planner_stdout.txt").write_text(proc.stdout, encoding="utf-8")
        (output / "planner_stderr.txt").write_text(proc.stderr, encoding="utf-8")
        trajectory_path = output / "trajectory.npy"
        if proc.returncode != 0 or not trajectory_path.exists():
            self._pregrasp_planning_failed = True
            return self._noop_diagnostic(target, f"planner_exit_{proc.returncode}")

        trajectory = np.asarray(np.load(trajectory_path), dtype=np.float64)
        if trajectory.ndim != 2 or trajectory.shape[1] != 7:
            raise RuntimeError(f"invalid OMPL trajectory shape: {trajectory.shape}")
        start_error = float(np.linalg.norm(np.asarray(self.robot.get_qpos())[:7] - trajectory[0]))
        if start_error > 1e-3:
            raise RuntimeError(f"OMPL/SAPIEN start mismatch: {start_error}")
        total_steps = 0
        max_error = 0.0
        for waypoint_index, waypoint in enumerate(trajectory):
            diag = self._track_joint_target(waypoint)
            total_steps += int(diag["steps"])
            max_error = max(max_error, float(diag["error_norm"]))
            if not diag["converged"]:
                self._pregrasp_planning_failed = True
                return self._noop_diagnostic(target, f"waypoint_{waypoint_index}_tracking_failed")
        final_diag = self._track_joint_target(trajectory[-1], strict=True)
        total_steps += int(final_diag["steps"])
        if not final_diag["converged"]:
            self._pregrasp_planning_failed = True
            return self._noop_diagnostic(target, "final_waypoint_tracking_failed")

        final = self.get_grasp_pose_matrix()
        pos_error = float(np.linalg.norm(target[:3, 3] - final[:3, 3]))
        rot_error = float(np.linalg.norm(Rotation.from_matrix(target[:3, :3] @ final[:3, :3].T).as_rotvec()))
        return {
            "reached_control_tolerance": bool(pos_error <= 0.005 and rot_error <= 0.03),
            "used_steps": total_steps,
            "final_position_error": pos_error,
            "final_rotation_error": rot_error,
            "best_position_error": pos_error,
            "best_rotation_error": rot_error,
            "controller_version": "collision_aware_ompl_pregrasp_v5",
            "planning_success": True,
            "planner": "RRTConnect",
            "planning_time_s": 5.0,
            "ik_attempts": 100,
            "trajectory_states": int(len(trajectory)),
            "trajectory_total_sim_steps": total_steps,
            "trajectory_max_waypoint_error": max_error,
            "planner_sapien_start_error": start_error,
        }

    def move_grasp_pose_to(self, target_world_grasp, num_steps, **kwargs):
        target = np.asarray(target_world_grasp, dtype=np.float64)
        move_index = self._pose_move_count
        self._pose_move_count += 1
        if move_index == 0:
            return self._plan_and_execute_pregrasp(target)
        if self._pregrasp_planning_failed:
            return self._noop_diagnostic(target, "pregrasp_planning_failed")
        return super().move_grasp_pose_to(target, num_steps, **kwargs)
