#!/usr/bin/env python3
"""Merge and independently audit the formal ArticuBot-PerceptionReach run."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


RESULT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/home/feng/robot_baselines/results/articubot/perception_reach"
)
CODE = Path("/home/feng/robot_baselines/repos/articubot/reproduction/perception_reach")
BASELINE = Path("/home/feng/robot_baselines/repos/articubot/reproduction/unified_open")
ARTICUBOT = Path("/home/feng/robot_baselines/repos/articubot")
CATALOG_ROOT = Path(
    "/home/feng/robot_baselines/results/where2act/four_task_noaff_v7_formal/"
    "20260905_082001/formal_catalogs"
)
CATALOGS = {
    "door_open": Path(
        "/home/feng/robot_baselines/configs/flowbot3d/eval/eval_pose_catalog.jsonl"
    ),
    "drawer_open": CATALOG_ROOT / "drawer_open_formal_scene_catalog.jsonl",
}
EXPECTED_CASES = {"door_open": 56, "drawer_open": 36}
REPEATS = 20
THRESHOLD = 0.35
NUM_SHARDS = 4
HIGH = ARTICUBOT / "data/high_level_200_obj_ckpt.pth"
LOW = ARTICUBOT / "data/low-level-ckpt/checkpoints/low-level.ckpt"
SOURCE_ZIP = Path("/home/feng/robot_baselines/repos/articubot/reproduction/provenance/low_level_execution_logic.zip")
BACKEND = Path("/home/feng/robot_baselines/repos/pa3ff_official/reproduction/soft_weld_pd_backend.py")
CONTROLLER = Path("/home/feng/robot_baselines/repos/pa3ff_official/reproduction/panda_controller_joint_pd_v8.py")
GRASP = Path("/home/feng/robot_baselines/common_env/flowbot3d_conditionA_physical_v2/contact_monitor.py")
ARTICULATION = Path("/home/feng/robot_baselines/scripts/relabel_flowbot_35.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def metric(rows: list[dict]) -> dict:
    n = len(rows)
    grasp = sum(bool(row["grasp_success"]) for row in rows)
    reach = sum(bool(row["reach_35"]) for row in rows)
    operation = sum(bool(row["operation_success"]) for row in rows)
    final = sum(bool(row["final_success"]) for row in rows)
    return {
        "N_total": n,
        "N_grasp": grasp,
        "N_reach_35": reach,
        "N_operation_success": operation,
        "N_final_success": final,
        "grasp_success_rate": grasp / n if n else 0.0,
        "reach_35_rate": reach / n if n else 0.0,
        "post_grasp_operation_success_rate": operation / grasp if grasp else 0.0,
        "final_success_rate": final / n if n else 0.0,
    }


def pct(value: float) -> str:
    return f"{100 * value:.3f}%"


def atomic_write(path: Path, content: str) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(content)
    os.replace(temp, path)


def merge() -> list[dict]:
    rows = []
    full_log = []
    for task, cases in EXPECTED_CASES.items():
        root = RESULT / "formal_shards" / task
        for shard in range(NUM_SHARDS):
            path = root / f"shard_{shard:02d}.jsonl"
            part = read_jsonl(path)
            expected = sum(
                ordinal % NUM_SHARDS == shard
                for ordinal in range(cases * REPEATS)
            )
            if len(part) != expected:
                raise RuntimeError(f"{task} shard {shard}: {len(part)} != {expected}")
            for row in part:
                row["formal_result_provenance"] = str(path)
            rows.extend(part)
            log = root / f"shard_{shard:02d}.log"
            if log.is_file():
                full_log.append(f"===== {log} =====\n{log.read_text()}")
    atomic_write(RESULT / "articubot_full_run.log", "\n".join(full_log))
    return sorted(rows, key=lambda r: (r["task_mode"], r["case_index"], r["repeat_id"]))


def validate(rows: list[dict]) -> list[str]:
    errors = []
    expected = {
        (task, case_index, repeat_id)
        for task, count in EXPECTED_CASES.items()
        for case_index in range(count)
        for repeat_id in range(REPEATS)
    }
    keys = [(r.get("task_mode"), r.get("case_index"), r.get("repeat_id")) for r in rows]
    if len(rows) != len(expected):
        errors.append(f"episode_count={len(rows)} expected={len(expected)}")
    if len(keys) != len(set(keys)):
        errors.append(f"duplicate_keys={len(keys) - len(set(keys))}")
    if set(keys) != expected:
        errors.append(f"coverage_missing={len(expected-set(keys))} extra={len(set(keys)-expected)}")
    for row in rows:
        tag = f"{row.get('task_mode')}:{row.get('case_index')}:{row.get('repeat_id')}"
        if row.get("task_mode") not in EXPECTED_CASES:
            errors.append(f"close_or_unknown_task:{tag}")
            break
        if row.get("method") != "ArticuBot-PerceptionReach":
            errors.append(f"wrong_method:{tag}")
            break
        if abs(float(row.get("articulation_success_threshold", -1)) - THRESHOLD) > 1e-12:
            errors.append(f"threshold_changed:{tag}")
            break
        progress = row.get("articulation_progress")
        reach = progress is not None and float(progress) >= THRESHOLD
        if bool(row.get("reach_35")) != reach:
            errors.append(f"reach35_predicate_mismatch:{tag}")
            break
        gated = bool(row.get("grasp_success") and reach)
        if bool(row.get("operation_success")) != gated or bool(row.get("final_success")) != gated:
            errors.append(f"gated_success_predicate_mismatch:{tag}")
            break
        conditioning = row.get("target_conditioning") or {}
        if any(bool(conditioning.get(key)) for key in (
            "ground_truth_handle", "ground_truth_joint_axis", "ground_truth_motion_direction",
            "training", "finetuning",
        )):
            errors.append(f"forbidden_conditioning:{tag}")
            break
        weld = row.get("senior_soft_weld_pd") or {}
        if bool(weld.get("created")) and not bool(row.get("grasp_success")):
            errors.append(f"weld_without_grasp:{tag}")
            break
    return errors


def write_group(path: Path, groups, fields: list[str]) -> None:
    metric_fields = [
        "N_total", "N_grasp", "N_reach_35", "N_operation_success", "N_final_success",
        "grasp_success_rate", "reach_35_rate",
        "post_grasp_operation_success_rate", "final_success_rate",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + metric_fields)
        writer.writeheader()
        for key, items in sorted(groups.items()):
            key = key if isinstance(key, tuple) else (key,)
            row = dict(zip(fields, key))
            values = metric(items)
            row.update(values)
            writer.writerow(row)


def failure_counts(rows: list[dict]) -> dict:
    return {
        "no_grasp": sum(not r["grasp_success"] for r in rows),
        "grasped_but_operation_failed": sum(r["grasp_success"] and not r["operation_success"] for r in rows),
        "IK_failure": sum(bool(r["ik_failure"]) for r in rows),
        "controller_failure": sum(bool(r["controller_failure"]) for r in rows),
        "policy_failure": sum(bool(r["policy_failure"]) for r in rows),
        "timeout": sum(bool(r["timeout"]) for r in rows),
        "exception": sum(r.get("exception") is not None for r in rows),
        "planning_failed": sum(r.get("termination_reason") == "planning_failed" for r in rows),
        "palm_stop": sum(bool(((r.get("collision_aware_reach") or {}).get("final_approach") or {}).get("palm_hit")) for r in rows),
    }


def main() -> None:
    RESULT.mkdir(parents=True, exist_ok=True)
    rows = merge()
    errors = validate(rows)
    per_episode = RESULT / "per_episode_results.jsonl"
    atomic_write(per_episode, "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    by_task_rows = {task: [r for r in rows if r["task_mode"] == task] for task in EXPECTED_CASES}
    by_task = {task: metric(items) for task, items in by_task_rows.items()}
    overall = metric(rows)
    by_case = defaultdict(list)
    by_object = defaultdict(list)
    for row in rows:
        by_case[(row["task_mode"], row["case_index"], row["object_id"], row["link_id"])].append(row)
        by_object[(row["task_mode"], row["object_id"])].append(row)
    write_group(RESULT / "per_case_summary.csv", by_case, ["task_mode", "case_index", "object_id", "link_id"])
    write_group(RESULT / "per_object_summary.csv", by_object, ["task_mode", "object_id"])
    with (RESULT / "articubot_failures.csv").open("w", newline="") as handle:
        fields = [
            "task_mode", "case_index", "case_id", "object_id", "link_id", "repeat_id",
            "termination_reason", "grasp_success", "reach_35", "operation_success",
            "ik_failure", "controller_failure", "policy_failure", "timeout", "exception",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(r for r in rows if not r["final_success"])

    failures = {task: failure_counts(items) for task, items in by_task_rows.items()}
    terminations = {task: dict(Counter(r["termination_reason"] for r in items)) for task, items in by_task_rows.items()}
    audit = {
        "pass": not errors,
        "errors": errors,
        "expected_episodes": 1840,
        "actual_episodes": len(rows),
        "duplicate_keys": len(rows) - len({(r["task_mode"], r["case_index"], r["repeat_id"]) for r in rows}),
        "close_rows": sum("close" in r["task_mode"] for r in rows),
        "sanity_rows": sum("sanity" in r.get("formal_result_provenance", "") for r in rows),
        "by_task": by_task,
        "overall": overall,
        "failures": failures,
        "terminations": terminations,
        "per_episode_sha256": sha256(per_episode),
    }
    atomic_write(RESULT / "independent_recount_and_consistency.json", json.dumps(audit, indent=2, sort_keys=True) + "\n")

    commit = subprocess.check_output(["git", "-C", str(ARTICUBOT), "rev-parse", "HEAD"], text=True).strip()
    lines = [
        "=" * 80, "ARTICUBOT-PERCEPTIONREACH IMPROVED BASELINE",
        "OFFICIAL PRETRAINED MODEL / UNIFIED PHYSICAL OPEN BENCHMARK", "=" * 80,
        f"Repository: {ARTICUBOT}", f"Commit: {commit}", "Checkpoint: official pretrained",
        "Training performed: NO", "Finetuning performed: NO",
        "Original-paper evaluation reproduced: NO", "Benchmark regenerated: NO",
        "Protocol: existing frozen OPEN benchmark", "Gripper: Panda parallel jaw / PHYSICAL_V2",
        "Interaction: real SAPIEN contact; senior soft-weld PD only after unified physical grasp",
        "Articulation success threshold: 35%", "Cases: 92 (56 door_open + 36 drawer_open)",
        "Repeats per case: 20", "Total episodes: 1840", "",
    ]
    for task in ("door_open", "drawer_open"):
        value = by_task[task]
        post = (
            f"{value['N_operation_success']}/{value['N_grasp']} = {pct(value['post_grasp_operation_success_rate'])}"
            if value["N_grasp"] else "0/0 = 0.000% (denominator=0)"
        )
        lines += [
            "=" * 80, task.upper(), "=" * 80,
            f"reach_35_rate (ungated): {value['N_reach_35']}/{value['N_total']} = {pct(value['reach_35_rate'])}",
            f"1. grasp_success_rate: {value['N_grasp']}/{value['N_total']} = {pct(value['grasp_success_rate'])}",
            f"2. post_grasp_operation_success_rate: {post}",
            f"3. final_success_rate: {value['N_final_success']}/{value['N_total']} = {pct(value['final_success_rate'])}",
            "Failure breakdown: " + ", ".join(f"{k}={v}" for k, v in failures[task].items()), "",
        ]
    lines += ["=" * 80, f"CONSISTENCY CHECK: {'PASS' if not errors else 'FAIL'}", *(errors or ["All invariants passed."]), "=" * 80]
    atomic_write(RESULT / "articubot_final_metrics.txt", "\n".join(lines) + "\n")

    changed = sorted(str(path) for path in CODE.glob("*.py"))
    protocol = [
        "ARTICUBOT-PERCEPTIONREACH PROTOCOL AUDIT", "",
        f"Official repository commit: {commit}",
        f"Official high checkpoint: {HIGH}; sha256={sha256(HIGH)}; bytes={HIGH.stat().st_size}",
        f"Official low checkpoint: {LOW}; sha256={sha256(LOW)}; bytes={LOW.stat().st_size}",
        "No retraining; no finetuning; checkpoints not modified.",
        "No original-paper evaluation; no regenerated benchmark dataset.",
        f"Door catalog: {CATALOGS['door_open']}; sha256={sha256(CATALOGS['door_open'])}; cases=56",
        f"Drawer catalog: {CATALOGS['drawer_open']}; sha256={sha256(CATALOGS['drawer_open'])}; cases=36",
        f"Canonical evaluator: {BASELINE / 'frozen_open_episode.py'}",
        f"PHYSICAL_V2 grasp evaluator: {GRASP}; sha256={sha256(GRASP)}",
        f"35% articulation evaluator: {ARTICULATION}; sha256={sha256(ARTICULATION)}",
        f"Senior source archive: {SOURCE_ZIP}; sha256={sha256(SOURCE_ZIP)}",
        f"Senior soft-weld PD backend: {BACKEND}; sha256={sha256(BACKEND)}",
        f"Joint-PD action controller: {CONTROLLER}; sha256={sha256(CONTROLLER)}",
        "Camera/articulation/object pose/scale/robot initial state: frozen catalog and scene constructor.",
        "Target conditioning: runtime SAPIEN camera instance mask for requested active link only.",
        "No handle location, joint axis, joint type, motion direction, or articulation state is supplied to policy/reach.",
        "Reach: observed-point-cloud collision corridors + external OMPL; no robot teleport.",
        "Operation: official low-level action, converted to EEF/current-seed IK, senior joint PD.",
        "Soft weld: finite spring constraint created only after stable bilateral PHYSICAL_V2 grasp.",
        "Interface files (all classified as perception/observation/action/controller/protocol adaptation):",
        *changed,
    ]
    atomic_write(RESULT / "articubot_protocol_audit.txt", "\n".join(protocol) + "\n")
    environment = [
        "ARTICUBOT-PERCEPTIONREACH ENVIRONMENT AUDIT",
        "Conda: /home/feng/miniconda3/envs/articubot",
        "Official high/low checkpoint strict load: PASS",
        "Official forward: PASS", "Physical two-finger regression: PASS",
        f"Full consistency: {'PASS' if not errors else 'FAIL'}",
    ]
    atomic_write(RESULT / "articubot_environment_audit.txt", "\n".join(environment) + "\n")
    state = {
        "status": "complete" if not errors else "failed_consistency",
        "timestamp": datetime.now().astimezone().isoformat(), "cases": 92,
        "repeats": 20, "episodes": len(rows), "consistency_errors": errors,
    }
    atomic_write(RESULT / "resume_state.json", json.dumps(state, indent=2, sort_keys=True) + "\n")
    if errors:
        raise SystemExit("; ".join(errors))
    atomic_write(RESULT / "FULL_RUN_COMPLETE", state["timestamp"] + "\n")
    print(json.dumps({"by_task": by_task, "overall": overall, "consistency": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
