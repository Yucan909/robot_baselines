#!/usr/bin/env python3
"""Verify V7 provenance and freeze the four-task formal protocol before rollout."""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from formal_protocol import (
    OPEN_COMMAND_PROGRESS,
    OPEN_SUCCESS_PROGRESS,
    CLOSE_COMMAND_PROGRESS,
    CLOSE_SUCCESS_PROGRESS,
    TASK_SPECS,
    TRIAL_SEEDS,
    check_unique_targets,
    expected_trials,
    load_jsonl,
    read_json,
    sha256,
    write_json,
)
from where2act_policy import Where2ActPolicy


BACKEND_FILES = (
    "run_trial.py",
    "floating_gripper_controller.py",
    "progress_operation.py",
    "where2act_runtime.py",
    "where2act_policy.py",
    "where2act_observation_adapter.py",
    "backend_v2_physics.py",
    "backend_v2_protocol.json",
    "contact_monitor.py",
    "formal_protocol.py",
    "prepare_formal_protocol.py",
    "run_four_task_formal.py",
    "aggregate_four_task_formal.py",
    "build_formal_scene_catalogs.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-manifest", required=True)
    parser.add_argument("--scene-manifest", required=True)
    parser.add_argument("--result-root", required=True)
    args = parser.parse_args()

    backend = Path(__file__).resolve().parent
    training_manifest_path = Path(args.training_manifest).resolve()
    scene_manifest_path = Path(args.scene_manifest).resolve()
    result_root = Path(args.result_root).resolve()
    output_path = result_root / "protocol_manifest.json"
    training = read_json(training_manifest_path)
    scene_manifest = read_json(scene_manifest_path)
    if training.get("protocol") != "where2act_four_task_train_v7_noaff_schema_robust":
        raise RuntimeError("not the current no-affordance V7 training manifest")
    if training.get("formal_or_test_rollout_used") is not False:
        raise RuntimeError("training manifest reports formal/test rollout use")
    run_root = Path(training["run_root"]).resolve()
    if run_root.name != "20260905_032913":
        raise RuntimeError(f"unexpected V7 run id: {run_root.name}")
    if scene_manifest.get("schema") != "where2act_four_task_noaff_v7_formal_scene_catalog_v1":
        raise RuntimeError("unexpected scene catalog schema")
    if scene_manifest.get("policy_input_leakage") is not False:
        raise RuntimeError("scene builder reported policy leakage")

    frozen_tasks: Dict[str, Dict[str, Any]] = {}
    total = 0
    for task in TASK_SPECS:
        trained = training["tasks"][task]
        checkpoint = Path(trained["joint"]).resolve()
        expected_checkpoint = run_root / task / "joint" / "best-network.pth"
        if checkpoint != expected_checkpoint:
            raise RuntimeError(f"{task}: checkpoint is not from this V7 run")
        actual_checkpoint_sha = sha256(checkpoint)
        if actual_checkpoint_sha != trained["joint_sha256"]:
            raise RuntimeError(f"{task}: V7 checkpoint SHA mismatch")
        policy = Where2ActPolicy(checkpoint=str(checkpoint), device="cpu")
        parameter_count = sum(parameter.numel() for parameter in policy.network.parameters())
        del policy
        gc.collect()

        scene = scene_manifest["tasks"][task]
        catalog = Path(scene["output_catalog"]).resolve()
        actual_catalog_sha = sha256(catalog)
        if actual_catalog_sha != scene["output_catalog_sha256"]:
            raise RuntimeError(f"{task}: formal scene catalog SHA mismatch")
        targets = check_unique_targets(task, load_jsonl(catalog))
        task_trials = len(targets) * len(TRIAL_SEEDS)
        if task_trials != expected_trials(task):
            raise RuntimeError(f"{task}: trial count mismatch")
        total += task_trials
        frozen_tasks[task] = {
            "primitive": TASK_SPECS[task]["primitive"],
            "goal": TASK_SPECS[task]["goal"],
            "targets": len(targets),
            "trials": task_trials,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": actual_checkpoint_sha,
            "checkpoint_load_verification": {
                "strict": True,
                "missing_keys": [],
                "unexpected_keys": [],
                "parameter_count": parameter_count,
                "device": "cpu",
            },
            "critic_checkpoint": trained["critic"],
            "critic_checkpoint_sha256": trained["critic_sha256"],
            "training_index": trained["index"],
            "training_index_sha256": trained["index_sha256"],
            "formal_catalog": str(catalog),
            "formal_catalog_sha256": actual_catalog_sha,
            "source_target_catalog": scene["source_target_catalog"],
            "source_target_catalog_sha256": scene["source_target_catalog_sha256"],
            "camera_sources": scene["camera_sources"],
        }
    if total != 3640:
        raise RuntimeError(f"expected 3640 total episodes, got {total}")

    backend_hashes = {}
    for relative in BACKEND_FILES:
        path = backend / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        backend_hashes[relative] = sha256(path)

    seed_source = Path(
        "/home/feng/robot_baselines/results/where2act/FAITHFUL_FREEZE_MANIFEST.json"
    )
    old_freeze = read_json(seed_source)
    if old_freeze.get("trial_seeds") != TRIAL_SEEDS:
        raise RuntimeError("established local freeze manifest has a different seed set")

    training_script = Path(
        "/home/feng/robot_baselines/repos/where2act/reproduction/v7/train_v7.sh"
    )
    index_manifest = Path(
        "/home/feng/robot_baselines/results/where2act/"
        "four_task_train_v7_noaff_schema_robust_indices/INDEX_MANIFEST.json"
    )
    builder = Path(
        "/home/feng/robot_baselines/repos/where2act/reproduction/v7/"
        "build_four_task_indices_v7_noaff_schema_robust.py"
    )
    external_artifacts = {
        str(training_manifest_path): sha256(training_manifest_path),
        str(training_script): sha256(training_script),
        str(index_manifest): sha256(index_manifest),
        str(builder): sha256(builder),
        str(scene_manifest_path): sha256(scene_manifest_path),
        str(seed_source): sha256(seed_source),
        "/home/feng/robot_baselines/repos/where2act/code/models/model_3d.py": sha256(
            Path("/home/feng/robot_baselines/repos/where2act/code/models/model_3d.py")
        ),
        "/home/feng/robot_baselines/repos/where2act/code/robots/panda_gripper.urdf": sha256(
            Path("/home/feng/robot_baselines/repos/where2act/code/robots/panda_gripper.urdf")
        ),
    }
    payload = {
        "status": "FROZEN",
        "protocol": "where2act_four_task_noaff_v7_floating_progress_v2",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "formal_result_root": str(result_root),
        "training_manifest": str(training_manifest_path),
        "training_manifest_sha256": sha256(training_manifest_path),
        "v7_run_root": str(run_root),
        "scene_catalog_manifest": str(scene_manifest_path),
        "scene_catalog_manifest_sha256": sha256(scene_manifest_path),
        "trial_seeds": TRIAL_SEEDS,
        "trial_seed_source": str(seed_source),
        "trial_seed_source_sha256": sha256(seed_source),
        "trials_per_target": len(TRIAL_SEEDS),
        "expected_total_trials": total,
        "parallel_workers_max_validated": 8,
        "parallel_workers_runtime_range": [1, 8],
        "parallelism_note": (
            "Concurrency may be reduced without changing any episode, seed, "
            "policy, physics, or success rule when the shared GPU is occupied; "
            "workers use a continuous bounded queue without a per-wave barrier."
        ),
        "shared_gpu_oom_policy": (
            "wait 15 seconds and rerun the identical seed; never treat CUDA OOM "
            "as a policy outcome and never retry a valid policy/physics failure"
        ),
        "task_execution_order": list(TASK_SPECS),
        "tasks": frozen_tasks,
        "backend_root": str(backend),
        "backend_hashes": backend_hashes,
        "external_artifact_hashes": external_artifacts,
        "scientific_contract": {
            "execution_condition": "official_where2act_floating_panda_gripper_progress_targeted_v2",
            "full_arm_present": False,
            "arm_ik_used": False,
            "ompl_used": False,
            "floating_gripper_urdf": "/home/feng/robot_baselines/repos/where2act/code/robots/panda_gripper.urdf",
            "observation_before_floating_gripper_spawn": True,
            "training_affordance_fields_used": False,
            "formal_trajectory_contact_or_action_given_to_policy": False,
            "policy_selects_contact_and_action_from_observation": True,
            "gpu_inference_concurrency": (
                "serialized across episode processes; model released before physics; "
                "no mathematical policy change"
            ),
            "target_link_identity_use": "environment, faithful candidate mask, contact and physics scoring",
            "pull_grasp_success": "real stable bilateral two-finger grasp",
            "push_grasp_success": "pre-operation closed-gripper target engagement/contact",
            "push_does_not_require_pull_style_enclosing_grasp": True,
            "operation_controller": "incremental network direction with absolute progress target; no fixed 5cm endpoint",
            "operation_segment_distance_m": 0.01,
            "operation_max_travel_m": 0.40,
            "pull_direction": "-predicted_d1",
            "push_direction": "+predicted_d1",
            "open_directional_progress": "final_progress - initial_progress",
            "close_directional_progress": "initial_progress - final_progress",
            "open_command_final_progress": OPEN_COMMAND_PROGRESS,
            "open_success_final_progress": OPEN_SUCCESS_PROGRESS,
            "close_command_final_progress": CLOSE_COMMAND_PROGRESS,
            "close_success_final_progress": CLOSE_SUCCESS_PROGRESS,
            "final_operation_success": (
                "operation_executed AND ((open AND final_progress>=0.35) OR "
                "(close AND final_progress<=0.15))"
            ),
            "primary_report_metric": "final_operation_success / total_trials",
            "reached_target_40_metric": False,
            "reach_35_diagnostic_metric": (
                "ungated final-state reach / total_trials: Open final_progress>=0.35; "
                "Close final_progress<=0.15"
            ),
            "reach_35_is_primary_success": False,
            "articulation_state_pairing_allowed": False,
            "camera_pairing_allowed_only_when_native_missing": True,
            "hard_target_skips": False,
            "retry_policy_failure_until_success": False,
            "resume_incomplete_same_seed": True,
        },
        "scene_metadata_allowlist": scene_manifest["loaded_npz_fields_allowlist"],
        "smoke_root": str(result_root / "smoke"),
        "engineering_schema_notes": [
            (
                "Episodes with target_not_visible terminate before policy construction; "
                "they retain the frozen checkpoint path/SHA, network_trained=false, and "
                "count once as model_no_valid_action without retry."
            )
        ],
    }
    write_json(output_path, payload)
    print(f"FROZEN: {output_path}")
    print(f"SHA256: {sha256(output_path)}")
    for task, frozen in frozen_tasks.items():
        print(f"{task}: {frozen['checkpoint_sha256']} | {frozen['targets']}x20={frozen['trials']}")
    print(f"TOTAL: {total}")


if __name__ == "__main__":
    main()
