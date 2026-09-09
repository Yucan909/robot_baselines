#!/usr/bin/env python3
"""Freeze a DEV-selected PADP V4 checkpoint before smoke/formal rollout."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


HOME = Path("/home/feng")
CODE = HOME / "robot_baselines/repos/pa3ff_official/reproduction"
RESULTS = HOME / "robot_baselines/results/pa3ff"
DATA = RESULTS / "padp_data_v4_timeindexed_baseframe_fourtask"
CATALOG = HOME / "robot_baselines/configs/pa3ff/reproduction_v1_formal/formal_episode_catalog.jsonl"
SCENE_MANIFEST = HOME / "robot_baselines/configs/pa3ff/reproduction_v1_formal/SCENE_CATALOG_MANIFEST.json"
SMOKE = HOME / "robot_baselines/configs/pa3ff/reproduction_v2_soft_weld_pd_smoke_cases.json"
ATTACHED_BOTTOM = HOME / "下载/底层执行逻辑 (1).zip"
WORKER = CODE / "dev_receding_bottomfaithful_padp_v4_worker_v38.py"
RUNTIME = CODE / "pa3ff_policy_runtime_v4_baseframe.py"
RANK = CODE / "candidate_rank_padp_v4_trainonly.py"
MODEL = CODE / "padp_model_fourtask.py"
PARENT = RESULTS / "reproduction_v4_padp_timeindexed_formal"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    global WORKER, RUNTIME, MODEL, PARENT
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--variant", choices=("v4", "v5_positional"), default="v4")
    args = parser.parse_args()
    if args.variant == "v5_positional":
        WORKER = CODE / "dev_receding_bottomfaithful_padp_v5_worker_v39.py"
        RUNTIME = CODE / "pa3ff_policy_runtime_v5_positional.py"
        MODEL = CODE / "padp_model_fourtask_v5.py"
        PARENT = RESULTS / "reproduction_v5_padp_positional_balanced_formal"
    training = args.training_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    selection_path = training / "DEV_CHECKPOINT_SELECTION.json"
    inference_selection_path = RESULTS / "reproduction_v5/INFERENCE_CONFIG_DEV_SELECTION.json"
    rank_selection_path = RESULTS / "reproduction_v5/RANK_CONFIG_DEV_SELECTION_GRASP_GEOMETRY.json"
    door_physical_dev = RESULTS / "reproduction_v5/door_dev_cfg_s90_n05_c15_p0_g01_gg08/DEV_METRICS.json"
    drawer_physical_dev = RESULTS / "reproduction_v5/drawer_dev_cfg_s90_n05_c15_p0_g0_gg08/DEV_METRICS.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection["selected"]
    if checkpoint != Path(selected["checkpoint"]).resolve():
        raise RuntimeError("checkpoint path is not the DEV-selected checkpoint")
    if sha256(checkpoint) != selected["sha256"]:
        raise RuntimeError("DEV-selected checkpoint hash mismatch")
    for path in (CATALOG, SCENE_MANIFEST, SMOKE, ATTACHED_BOTTOM, WORKER,
                 RUNTIME, RANK, MODEL, DATA / "BUILD_MANIFEST.json",
                 DATA / "DIRECTION_PAIR_AUDIT.json",
                 DATA / "NORMALIZED_ACTION_SUPPORT_AUDIT.json",
                 DATA / "ACTION_FK_ALIGNMENT_AUDIT.json",
                 DATA / "PA3FF_FIELD_CACHE_ACCURACY_AUDIT.json",
                 inference_selection_path if args.variant == "v5_positional" else RANK,
                 rank_selection_path if args.variant == "v5_positional" else RANK,
                 door_physical_dev if args.variant == "v5_positional" else RANK,
                 drawer_physical_dev if args.variant == "v5_positional" else RANK,
                 RESULTS / "reproduction_v5/TRAIN_CANDIDATE_PRIOR_AUDIT.json"
                 if args.variant == "v5_positional" else RANK,
                 training / "TRAINING_PROTOCOL.json"):
        if not path.is_file():
            raise RuntimeError(f"missing formal prerequisite {path}")

    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    root = PARENT / now.strftime("%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "protocol_name": (
            "PA3FF_REPRODUCTION_V5_POSITIONAL_BALANCED_FORMAL"
            if args.variant == "v5_positional" else
            "PA3FF_REPRODUCTION_V4_TIMEINDEXED_BASEFRAME_FORMAL"
        ),
        "padp_variant": args.variant,
        "status": "FROZEN_BEFORE_SMOKE_AND_FORMAL", "created_at": now.isoformat(),
        "formal_root": str(root), "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_training_step": int(selected["step"]),
        "checkpoint_selected_by_dev_x0_mse_only": True,
        "dev_selection": str(selection_path), "dev_selection_sha256": sha256(selection_path),
        "formal_success_used_for_checkpoint_or_sampler_selection": False,
        "dev_runtime_selection": {
            "inference_config": str(inference_selection_path),
            "inference_config_sha256": sha256(inference_selection_path),
            "rank_config": str(rank_selection_path),
            "rank_config_sha256": sha256(rank_selection_path),
            "door_open_physical_dev": str(door_physical_dev),
            "door_open_physical_dev_sha256": sha256(door_physical_dev),
            "drawer_open_physical_dev": str(drawer_physical_dev),
            "drawer_open_physical_dev_sha256": sha256(drawer_physical_dev),
            "physical_dev_reached_target_35": {
                "door_open": "7/32", "drawer_open": "7/32"
            },
            "formal_catalog_or_results_used": False,
        } if args.variant == "v5_positional" else None,
        "paper_method": {
            "name": "PA3FF reproduction",
            "official_components": ["PA3FF source", "Sonata backbone", "trained PA3FF instance refinement weights"],
            "reconstructed_components": ["PADP dataset adapter", "PADP diffusion policy training", "simulator executor"],
            "padp_official_private_implementation_used": False,
            "observation": "current 1024-point cloud + current Panda qpos + task-critical part/instruction embeddings",
            "action": "H=16 absolute panda_grasptarget SE(3)+gripper chunk",
            "loss": "DDPM x0 MSE", "inference": "DDIM 10 steps",
            "implementation_choice": (
                "time-indexed suffix samples, Panda-base actions, explicit base-xyz point positional tokens, and uniform four-task sampling; static non-future visual reused when per-time visual is absent"
                if args.variant == "v5_positional" else
                "time-indexed suffix samples and Panda-base coordinates; static non-future visual reused when per-time visual is absent"
            ),
        },
        "training": {
            "protocol": str(training / "TRAINING_PROTOCOL.json"),
            "protocol_sha256": sha256(training / "TRAINING_PROTOCOL.json"),
            "actual_updates": int(selected["step"]),
            "paper_training_budget": "exact PartInstruct PADP update count is not published; appendix reports selecting PADP epochs 300-400 for the separate RLBench study",
            "shortened_budget_reason": "reasonable comparison baseline rather than matching the unpublished full compute budget",
        },
        "data": {
            "root": str(DATA), "manifest": str(DATA / "BUILD_MANIFEST.json"),
            "manifest_sha256": sha256(DATA / "BUILD_MANIFEST.json"),
            "direction_audit_sha256": sha256(DATA / "DIRECTION_PAIR_AUDIT.json"),
            "action_support_audit_sha256": sha256(DATA / "NORMALIZED_ACTION_SUPPORT_AUDIT.json"),
            "action_fk_alignment_audit_sha256": sha256(DATA / "ACTION_FK_ALIGNMENT_AUDIT.json"),
            "feature_cache_accuracy_audit_sha256": sha256(DATA / "PA3FF_FIELD_CACHE_ACCURACY_AUDIT.json"),
            "object_level_split": True, "affordance_fields_used": False,
            "formal_or_test_trajectory_actions_used": False,
        },
        "scene_catalog": str(CATALOG), "scene_catalog_sha256": sha256(CATALOG),
        "scene_catalog_manifest": str(SCENE_MANIFEST),
        "scene_catalog_manifest_sha256": sha256(SCENE_MANIFEST),
        "formal_worker": str(WORKER), "formal_worker_sha256": sha256(WORKER),
        "formal_launcher": str(CODE / "run_padp_v5_formal.sh"),
        "formal_launcher_sha256": sha256(CODE / "run_padp_v5_formal.sh"),
        "runtime": str(RUNTIME), "runtime_sha256": sha256(RUNTIME),
        "runtime_base": str(CODE / "pa3ff_policy_runtime_v4_baseframe.py"),
        "runtime_base_sha256": sha256(CODE / "pa3ff_policy_runtime_v4_baseframe.py"),
        "candidate_rank": str(RANK), "candidate_rank_sha256": sha256(RANK),
        "candidate_train_prior_audit": (
            str(RESULTS / "reproduction_v5/TRAIN_CANDIDATE_PRIOR_AUDIT.json")
            if args.variant == "v5_positional" else None
        ),
        "model": str(MODEL), "model_sha256": sha256(MODEL),
        "execution_dependency_hashes": {
            name: sha256(CODE / name) for name in (
                "dev_receding_bottomfaithful_worker_v34.py",
                "dev_receding_scenealigned_teleport_worker_v33.py",
                "dev_receding_horizon_worker_v28.py",
                "formal_worker_soft_weld_pd_devselected_v3.py",
                "formal_worker_soft_weld_pd.py",
                "panda_controller_joint_pd_v8.py",
                "padp_model.py", "padp_model_fourtask.py",
            )
        },
        "supplied_bottom_logic_archive": str(ATTACHED_BOTTOM),
        "supplied_bottom_logic_archive_sha256": sha256(ATTACHED_BOTTOM),
        "smoke_cases": str(SMOKE), "smoke_cases_sha256": sha256(SMOKE),
        "smoke_validator": str(CODE / "validate_padp_v4_smoke.py"),
        "smoke_validator_sha256": sha256(CODE / "validate_padp_v4_smoke.py"),
        "tasks": {
            "door_open": {"primitive": "pull", "goal": "open", "targets": 56, "trials_per_target": 20, "episodes": 1120},
            "door_close": {"primitive": "push", "goal": "close", "targets": 56, "trials_per_target": 20, "episodes": 1120},
            "drawer_open": {"primitive": "pull", "goal": "open", "targets": 36, "trials_per_target": 20, "episodes": 720},
            "drawer_close": {"primitive": "push", "goal": "close", "targets": 34, "trials_per_target": 20, "episodes": 680},
        },
        "total_episodes": 3640,
        "formal_seeds": list(range(2026082900, 2026082920)),
        "success_definition": {
            "open_directional": "final_progress - initial_progress",
            "close_directional": "initial_progress - final_progress",
            "grasp_success_pull": "real final bilateral finger contact on correct target part before operation",
            "grasp_success_push": "effective real robot contact on correct target part before operation",
            "final_success": "grasp_success AND directional_task_progress >= 0.35",
            "reached_target_35": "directional_task_progress >= 0.35",
            "reached_target_40": "directional_task_progress >= 0.40",
        },
        "execution": {
            "bottom_logic": "supplied OPEN/CLOSE/HOLD timing, target pinned only during CLOSE, soft contact weld after valid Pull grasp",
            "receding_horizon": "execute 4 actions then re-observe, max 80 actions",
            "candidates": 128, "rank_inputs": "current observation + predicted chunks + TRAIN-only priors",
            "predicted_x0_clip": 1.5,
            "start_timestep": 90, "noise_scale": 0.5, "ddim_steps": 10,
            "sampler_selection": "fixed on object-level DEV before formal; 0.5 retained candidate diversity needed for physical grasp",
            "rank_weights": {
                "close_default": {"position": 0.02, "full_geometry": 0.02, "grasp_geometry": 0.0},
                "door_open": {"position": 0.0, "full_geometry": 0.1, "grasp_geometry": 0.8},
                "drawer_open": {"position": 0.0, "full_geometry": 0.0, "grasp_geometry": 0.8},
            },
            "rank_weight_basis": "candidate scoring uses current observation and TRAIN-frozen statistics; weights selected on object-level DEV",
            "normal_policy_failure_retry": False,
            "infrastructure_crash_retry_same_seed_max_attempts": 2,
        },
        "policy_forbidden_inputs": [
            "test operation/contact/grasp point", "test successful trajectory/action/direction",
            "test success/affordance/final qpos", "target joint/progress",
        ],
        "official_pa3ff_commit": subprocess.check_output(
            ["git", "-C", str(OFFICIAL := HOME / "robot_baselines/repos/pa3ff_official"), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "host": {"platform": platform.platform(), "python": platform.python_version()},
    }
    manifest_path = root / "protocol_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    PARENT.mkdir(parents=True, exist_ok=True)
    (PARENT / "LATEST_FORMAL_ROOT.txt").write_text(str(root) + "\n", encoding="utf-8")
    print(root)


if __name__ == "__main__":
    main()
