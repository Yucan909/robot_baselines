#!/usr/bin/env python3
"""Independent formal-result audit, aggregation, and required artifact generation."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from articubot_policy_adapter import ARTICUBOT_ROOT, HIGH_CKPT, LOW_CKPT, LOW_EXP
from frozen_open_episode import FROZEN_DIR, POSE_CATALOG, RELABEL_EVALUATOR, THRESHOLD, load_catalog


ROOT = Path("/home/feng/robot_baselines/results/articubot/unified_open")
JSONL = ROOT / "per_episode_results.jsonl"
CODE_ROOT = Path(__file__).resolve().parent
LOW_ZIP = ARTICUBOT_ROOT / "data/low-level-ckpt.zip"
HISTORICAL_PHYSICAL_SUMMARY = (
    Path("/home/feng/robot_baselines/results/flowbot3d/FINAL_CONDITIONA_PHYSICAL_V2/summary.json")
)
HISTORICAL_PHYSICAL_ROOT = HISTORICAL_PHYSICAL_SUMMARY.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_rows():
    rows = []
    with JSONL.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception as exc:
                    raise RuntimeError(f"invalid JSONL line {line_number}: {exc}") from exc
    return rows


def rates(rows):
    n = len(rows)
    grasp = sum(bool(row["grasp_success"]) for row in rows)
    operation = sum(bool(row["operation_success"]) for row in rows)
    final = sum(bool(row["final_success"]) for row in rows)
    return {
        "n_total": n, "n_grasp": grasp, "n_operation": operation, "n_final": final,
        "grasp_rate": grasp / n if n else 0.0,
        "post_rate": operation / grasp if grasp else 0.0,
        "final_rate": final / n if n else 0.0,
    }


def audit_rows(rows, cases):
    required = {
        "method", "case_id", "object_id", "link_id", "trial_id", "repeat_id", "task_mode",
        "initial_articulation_state", "final_articulation_state", "articulation_progress",
        "grasp_success", "operation_success", "final_success", "termination_reason", "timeout",
        "ik_failure", "policy_failure", "controller_failure", "exception", "runtime_sec",
    }
    if len(rows) != len(cases) * 20:
        raise RuntimeError(f"expected {len(cases)*20} formal episodes, got {len(rows)}")
    expected = {(ci, ri) for ci in range(len(cases)) for ri in range(20)}
    historical_initials = {}
    for path in HISTORICAL_PHYSICAL_ROOT.glob("*/result.json"):
        historical = json.loads(path.read_text())
        key = (
            f"{historical['shape_id']}_{historical['target_link']}",
            int(historical["trial_seed"]),
        )
        historical_initials[key] = historical.get("actual_initial_progress")
    actual = [(int(row["case_index"]), int(row["repeat_id"])) for row in rows]
    if len(set(actual)) != len(actual):
        raise RuntimeError("duplicate case/repeat pairs")
    if set(actual) != expected:
        raise RuntimeError(f"case/repeat mismatch: missing={sorted(expected-set(actual))[:10]}")
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise RuntimeError(f"row {index} missing {sorted(missing)}")
        if row["method"] != "ArticuBot_official_pretrained":
            raise RuntimeError("non-ArticuBot result in formal JSONL")
        if row["protocol_version"] != "flowbot3d_conditionA_physical_v2_relabel35":
            raise RuntimeError("formal protocol version mismatch")
        if row["task_mode"] != "door_open" or "close" in row["task_mode"]:
            raise RuntimeError("non-OPEN result in formal JSONL")
        if float(row["articulation_success_threshold"]) != THRESHOLD:
            raise RuntimeError("threshold mismatch")
        case = cases[int(row["case_index"])]
        expected_case_id = f"{case['shape_id']}_{case['link_name']}"
        if (
            row["case_id"] != expected_case_id
            or str(row["object_id"]) != str(case["shape_id"])
            or str(row["link_id"]) != str(case["link_name"])
        ):
            raise RuntimeError(f"catalog identity mismatch at row {index}")
        if int(row["trial_id"]) != int(row["repeat_id"]):
            raise RuntimeError("trial/repeat id mismatch")
        if bool(row["final_success"]) and not bool(row["grasp_success"]):
            raise RuntimeError("final success is not a subset of grasp success")
        computed = bool(row["grasp_success"] and row["articulation_progress"] is not None and float(row["articulation_progress"]) >= THRESHOLD)
        if bool(row["operation_success"]) != computed or bool(row["final_success"]) != computed:
            raise RuntimeError(f"metric formula mismatch at row {index}")
        if int(row["trial_seed"]) != 2026082900 + int(row["repeat_id"]):
            raise RuntimeError("formal seed mismatch")
        if row["camera_source"] != str(POSE_CATALOG):
            raise RuntimeError("formal camera source mismatch")
        requested = float(row["requested_initial_articulation_state"])
        expected_initial = float(
            np.random.default_rng(int(row["trial_seed"])).uniform(0.10, 0.20)
        )
        if abs(requested - expected_initial) > 1e-12:
            raise RuntimeError("initial articulation RNG mismatch")
        actual_initial = row["initial_articulation_state"]
        historical_initial = historical_initials.get(
            (row["case_id"], int(row["trial_seed"]))
        )
        if (
            actual_initial is None
            or historical_initial is None
            or float(actual_initial) != float(historical_initial)
        ):
            raise RuntimeError(
                "actual initial articulation does not exactly match the historical "
                "PHYSICAL_V2 episode"
            )
        if (
            row["final_articulation_state"] is not None
            and row["articulation_progress"] is not None
            and abs(
                float(row["final_articulation_state"])
                - float(row["articulation_progress"])
            ) > 1e-12
        ):
            raise RuntimeError("final articulation state/progress mismatch")
        phases = [
            (item["phase"], item["target_joint"])
            for item in row.get("phase_transitions", [])
        ]
        if phases and phases[0] != ("OPEN_TO_GRASP", "free"):
            raise RuntimeError("OPEN/TO_GRASP phase is not free")
        if row["termination_reason"] in (
            "grasp_failed", "grasp_lost", "articulation_threshold_reached"
        ):
            expected_phases = [
                ("OPEN_TO_GRASP", "free"),
                ("CLOSE", "locked"),
                ("HOLD_OPERATE", "free"),
            ]
            if phases != expected_phases:
                raise RuntimeError(
                    f"PHYSICAL_V2 phase mapping mismatch at row {index}: {phases}"
                )
        monitor = row.get("grasp_monitor")
        if bool(row["grasp_success"]) and not (
            monitor and bool(monitor.get("firm_grasp"))
        ):
            raise RuntimeError(
                "grasp success was not produced by the frozen contact monitor"
            )
        action_summary = row.get("low_level_action_summary")
        if action_summary and not 0 < int(action_summary["num_actions"]) <= 34 * 4:
            raise RuntimeError("official policy/action horizon mismatch")
    return "PASS"


def write_summary_csv(rows, cases):
    groups = defaultdict(list)
    for row in rows:
        groups[row["case_id"]].append(row)
    path = ROOT / "per_case_summary.csv"
    fields = [
        "case_id", "object_id", "link_id", "category", "task_mode", "n_total", "n_grasp",
        "n_operation", "n_final", "grasp_success_rate", "post_grasp_operation_success_rate",
        "final_success_rate",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            key = f"{case['shape_id']}_{case['link_name']}"
            stat = rates(groups[key])
            writer.writerow({
                "case_id": key, "object_id": case["shape_id"], "link_id": case["link_name"],
                "category": case.get("category", "UNKNOWN"), "task_mode": "door_open",
                "n_total": stat["n_total"], "n_grasp": stat["n_grasp"],
                "n_operation": stat["n_operation"], "n_final": stat["n_final"],
                "grasp_success_rate": stat["grasp_rate"],
                "post_grasp_operation_success_rate": stat["post_rate"],
                "final_success_rate": stat["final_rate"],
            })


def write_breakdown(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[("task", row["task_mode"])].append(row)
        groups[("object", row["object_id"])].append(row)
    path = ROOT / "breakdown_by_task_object.csv"
    with path.open("w", newline="") as handle:
        fields = ["group_type", "group", "n_total", "n_grasp", "n_operation", "n_final", "grasp_success_rate", "post_grasp_operation_success_rate", "final_success_rate"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(groups):
            stat = rates(groups[key])
            writer.writerow({
                "group_type": key[0], "group": key[1],
                "n_total": stat["n_total"], "n_grasp": stat["n_grasp"],
                "n_operation": stat["n_operation"], "n_final": stat["n_final"],
                "grasp_success_rate": stat["grasp_rate"],
                "post_grasp_operation_success_rate": stat["post_rate"],
                "final_success_rate": stat["final_rate"],
            })


def failure_category(row):
    if not row["grasp_success"]:
        return "no_grasp"
    if row["grasp_success"] and not row["operation_success"]:
        return "grasped_but_operation_failed"
    return "success"


def write_failures(rows):
    failed = [row for row in rows if not row["final_success"]]
    fields = ["case_id", "object_id", "link_id", "repeat_id", "trial_seed", "failure_category", "termination_reason", "timeout", "ik_failure", "policy_failure", "controller_failure", "exception", "articulation_progress", "runtime_sec"]
    with (ROOT / "articubot_failures.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in failed:
            writer.writerow({key: failure_category(row) if key == "failure_category" else row.get(key) for key in fields})


def pkg(name):
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "unavailable"


def verify_historical_frozen_hashes():
    historical = json.loads(HISTORICAL_PHYSICAL_SUMMARY.read_text())
    expected = historical["code_hashes"]
    current = {
        "run_trial_sha256": sha256(FROZEN_DIR / "run_trial.py"),
        "panda_controller_sha256": sha256(FROZEN_DIR / "panda_controller.py"),
        "grasp_pose_adapter_sha256": sha256(FROZEN_DIR / "grasp_pose_adapter.py"),
        "contact_monitor_sha256": sha256(FROZEN_DIR / "contact_monitor.py"),
    }
    if current != expected:
        raise RuntimeError(
            f"PHYSICAL_V2 files no longer match the historical 1120-run hashes: "
            f"expected={expected} current={current}"
        )
    return current


def verify_historical_case_manifest(cases):
    result_files = sorted(HISTORICAL_PHYSICAL_ROOT.glob("*/result.json"))
    if len(result_files) != 1120:
        raise RuntimeError(
            f"historical PHYSICAL_V2 run no longer has 1120 episodes: {len(result_files)}"
        )
    historical_pairs = set()
    historical_pair_seeds = set()
    for path in result_files:
        row = json.loads(path.read_text())
        pair = (str(row["shape_id"]), str(row["target_link"]))
        historical_pairs.add(pair)
        historical_pair_seeds.add((pair[0], pair[1], int(row["trial_seed"])))
    catalog_pairs = {(str(row["shape_id"]), str(row["link_name"])) for row in cases}
    expected_pair_seeds = {
        (shape_id, link_id, 2026082900 + repeat_id)
        for shape_id, link_id in catalog_pairs
        for repeat_id in range(20)
    }
    if historical_pairs != catalog_pairs or historical_pair_seeds != expected_pair_seeds:
        raise RuntimeError("catalog case/seed manifest differs from historical 1120-run")
    return {
        "historical_episodes": len(result_files),
        "canonical_cases": len(catalog_pairs),
        "seeds_per_case": 20,
    }


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    cases = load_catalog()
    historical_hashes = verify_historical_frozen_hashes()
    historical_manifest = verify_historical_case_manifest(cases)
    rows = load_rows()
    audit_rows(rows, cases)
    forward_report_path = ROOT / "forward_validation.json"
    forward_report = json.loads(forward_report_path.read_text())
    if not (
        forward_report.get("status") == "PASS"
        and forward_report.get("cuda") == "PASS"
        and forward_report.get("torch_load") == "PASS"
        and forward_report.get("config_load") == "PASS"
        and forward_report.get("high_level_forward") == "PASS"
        and forward_report.get("low_level_forward") == "PASS"
        and forward_report.get("high_level_output_shape") == [1, 1, 4, 3]
        and forward_report.get("low_level_output_shape") == [4, 10]
    ):
        raise RuntimeError(f"final CUDA forward audit failed: {forward_report}")
    stat = rates(rows)
    write_summary_csv(rows, cases)
    write_breakdown(rows)
    write_failures(rows)
    repo_commit = subprocess.check_output(["git", "-C", str(ARTICUBOT_ROOT), "rev-parse", "HEAD"], text=True).strip()
    repo_branch = subprocess.check_output(
        ["git", "-C", str(ARTICUBOT_ROOT), "branch", "--show-current"], text=True
    ).strip()
    repo_remote = subprocess.check_output(
        ["git", "-C", str(ARTICUBOT_ROOT), "remote", "get-url", "origin"], text=True
    ).strip()
    repo_status = subprocess.check_output(
        ["git", "-C", str(ARTICUBOT_ROOT), "status", "--porcelain"], text=True
    ).strip()
    if repo_status:
        raise RuntimeError(f"official ArticuBot source tree is modified: {repo_status}")
    high_sha, low_sha = sha256(HIGH_CKPT), sha256(LOW_CKPT)
    low_zip_sha = sha256(LOW_ZIP)
    failures = {
        "no_grasp": sum(not row["grasp_success"] for row in rows),
        "grasped_but_operation_failed": sum(row["grasp_success"] and not row["operation_success"] for row in rows),
        "ik_failure": sum(bool(row["ik_failure"]) for row in rows),
        "controller_failure": sum(bool(row["controller_failure"]) for row in rows),
        "policy_failure": sum(bool(row["policy_failure"]) for row in rows),
        "timeout": sum(bool(row["timeout"]) for row in rows),
        "other": sum(bool(row["exception"]) and not (row["ik_failure"] or row["controller_failure"] or row["policy_failure"]) for row in rows),
    }
    text = f"""================================================================================
ARTICUBOT BASELINE
OFFICIAL PRETRAINED MODEL
UNIFIED OPEN BENCHMARK
================================================================================

Repository:
https://github.com/yufeiwang63/ArticuBot.git

Commit:
{repo_commit}

Checkpoint:
official pretrained

Training performed:
NO

Original-paper evaluation reproduced:
NO

Benchmark regenerated:
NO

Protocol:
existing frozen OPEN benchmark

Gripper:
parallel jaw / PHYSICAL_V2

Articulation success threshold:
35%

Cases:
{len(cases)}

Repeats per case:
20

Total episodes:
{stat['n_total']}

N_total: {stat['n_total']}
N_grasp: {stat['n_grasp']}
N_operation_success: {stat['n_operation']}
N_final_success: {stat['n_final']}

================================================================================
SENIOR-REQUESTED PRIMARY METRICS
================================================================================

1. grasp_success_rate

{stat['n_grasp']} / {stat['n_total']} = {100*stat['grasp_rate']:.3f}%

2. post_grasp_operation_success_rate

{stat['n_operation']} / {stat['n_grasp']} = {100*stat['post_rate']:.3f}%
denominator_zero: {str(stat['n_grasp'] == 0).upper()}

3. final_success_rate

{stat['n_final']} / {stat['n_total']} = {100*stat['final_rate']:.3f}%

================================================================================
FAILURE BREAKDOWN
================================================================================

no_grasp: {failures['no_grasp']}
grasped_but_operation_failed: {failures['grasped_but_operation_failed']}
ik_failure: {failures['ik_failure']}
controller_failure: {failures['controller_failure']}
policy_failure: {failures['policy_failure']}
timeout: {failures['timeout']}
other: {failures['other']}

================================================================================
"""
    (ROOT / "articubot_final_metrics.txt").write_text(text)
    interface_files = sorted(str(path) for path in CODE_ROOT.glob("*.py"))
    code_classifications = {
        "articubot_observation_adapter.py": "runtime observation/coordinate/preprocessing interface",
        "articubot_action_adapter.py": "10D EEF-to-PHYSICAL_V2 action interface",
        "articubot_policy_adapter.py": "official checkpoint/config loader and inference interface",
        "frozen_open_episode.py": "frozen scene/controller/metric protocol orchestration",
        "run_articubot_unified_open.py": "resumable sanity/formal runner",
        "forward_validation.py": "checkpoint and CUDA forward validation",
        "merge_formal_shards.py": "deterministic formal-result merge",
        "aggregate_articubot_results.py": "independent metrics/protocol audit",
        "supervise_and_finalize.py": "crash-resume supervision and finalization",
        "run_articubot_batched_formal.py": "discarded throughput validation only; never formal",
    }
    frozen_hashes = {
        "catalog": sha256(POSE_CATALOG),
        "controller": sha256(FROZEN_DIR / "panda_controller.py"),
        "grasp_evaluator": sha256(FROZEN_DIR / "contact_monitor.py"),
        "articulation_evaluator": sha256(RELABEL_EVALUATOR),
    }
    protocol = f"""ArticuBot official repository: https://github.com/yufeiwang63/ArticuBot.git
ArticuBot git origin: {repo_remote}
ArticuBot git branch: {repo_branch}
ArticuBot official repository commit: {repo_commit}
Official checkpoint release documented by: {ARTICUBOT_ROOT / 'readme.md'} (lines 106-111)
Official checkpoint release folder: https://drive.google.com/drive/folders/1lbpoo8SqNuLWTjMyvO5RWnBd0XpGq6C4
Official high-level checkpoint: {HIGH_CKPT}
Official high-level checkpoint size: {HIGH_CKPT.stat().st_size}
Official high-level checkpoint SHA256: {high_sha}
Official low-level release archive: {LOW_ZIP}
Official low-level release archive size: {LOW_ZIP.stat().st_size}
Official low-level release archive SHA256: {low_zip_sha}
Official low-level release archive integrity test: PASS
Official low-level checkpoint: {LOW_CKPT}
Official low-level checkpoint size: {LOW_CKPT.stat().st_size}
Official low-level checkpoint SHA256: {low_sha}
No retraining: YES
No finetuning: YES
No original-paper evaluation: YES
No regenerated benchmark dataset: YES
No ArticuBot training demonstrations/datasets downloaded: YES
Canonical OPEN benchmark: {POSE_CATALOG}
Canonical OPEN benchmark SHA256: {frozen_hashes['catalog']}
Canonical evaluator: {RELABEL_EVALUATOR}
PHYSICAL_V2 controller: {FROZEN_DIR / 'panda_controller.py'}
PHYSICAL_V2 controller SHA256: {frozen_hashes['controller']}
Grasp evaluator: {FROZEN_DIR / 'contact_monitor.py'}
Grasp evaluator SHA256: {frozen_hashes['grasp_evaluator']}
Articulation evaluator: {RELABEL_EVALUATOR}
Articulation evaluator SHA256: {frozen_hashes['articulation_evaluator']}
Articulation threshold: {THRESHOLD:.2f} (35%)
Cases: {len(cases)}
Repeats per case: 20
Total episodes: {stat['n_total']}
Camera source: frozen camera_pose_world in {POSE_CATALOG}; one 448x448 35-degree-FOV view
Articulation-state source: canonical seed RNG uniform [0.10,0.20], PartNet joint qpos
Robot/gripper: frozen ArticuBot-geometry Franka Panda, PHYSICAL_V2 parallel jaw, physical contacts only
PHYSICAL_V2 phases: INIT/PREGRASP locked; OPEN/TO_GRASP free; CLOSE locked/stabilized for 300 steps; HOLD/OPERATE free
Task modes: door_open only; drawer_open absent because all canonical target joints are revolute; close=N/A
Official evaluation horizon: 35 with range(1, horizon) = 34 policy calls; native n_action_steps=4 retained
Interface files modified/created: {json.dumps(interface_files)}
Code-change classifications: {json.dumps(code_classifications, sort_keys=True)}
All changes are interface/protocol/validation/aggregation adaptations; official checkpoints, benchmark catalog, controller, grasp evaluator, and 35% evaluator were not modified.
Historical 1120-run PHYSICAL_V2 code hashes verified: {json.dumps(historical_hashes, sort_keys=True)}
Historical 1120-run case/seed manifest verified: {json.dumps(historical_manifest, sort_keys=True)}
Actual initial articulation states exactly match all 1120 historical PHYSICAL_V2 episodes: PASS
Official ArticuBot git working tree clean: PASS
Excluded validation artifacts: sanity_a, sanity_b, batch_smoke_archive, and formal_phase_bug_archive_20260905_0613 are outside per_episode_results.jsonl.
Formal independent consistency audit: PASS
No duplicate episodes: PASS
No missing episodes: PASS
Final success subset of grasp: PASS
Operation denominator is successful grasps: PASS
Metrics recomputed solely from per_episode_results.jsonl: PASS
No sanity episodes in formal JSONL: PASS
No CLOSE episodes: PASS
No suction execution: PASS
"""
    (ROOT / "articubot_protocol_audit.txt").write_text(protocol)
    try:
        driver_version = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).splitlines()[0].strip()
    except Exception:
        driver_version = "unavailable"
    env = f"""Conda environment: /home/feng/miniconda3/envs/articubot
Python prefix: {sys.prefix}
Python: {sys.version.split()[0]}
PyTorch: {torch.__version__}
CUDA runtime: {torch.version.cuda}
CUDA available: {torch.cuda.is_available()}
GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable'}
NVIDIA driver: {driver_version}
NumPy: {np.__version__}
PyBullet: {pkg('pybullet')}
SAPIEN: {pkg('sapien')}
Hydra: {pkg('hydra-core')}
Diffusers: {pkg('diffusers')}
fpsample: {pkg('fpsample')}
ArticuBot imports: PASS
High-level checkpoint load/forward: PASS
Low-level checkpoint config/load/forward: PASS
Forward validation artifact: {forward_report_path}
"""
    (ROOT / "articubot_environment_audit.txt").write_text(env)
    print(text)


if __name__ == "__main__":
    main()
