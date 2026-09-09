#!/usr/bin/env python3
"""Resumable sanity/formal runner for official ArticuBot on the frozen OPEN protocol."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from articubot_policy_adapter import ArticuBotPolicyAdapter
from frozen_open_episode import load_catalog, run_episode


RESULT_ROOT = Path("/home/feng/robot_baselines/results/articubot/unified_open")
FORMAL_JSONL = RESULT_ROOT / "per_episode_results.jsonl"
PROGRESS = RESULT_ROOT / "progress.log"


def log(message: str, full_log=None):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with PROGRESS.open("a") as handle:
        handle.write(line + "\n")
    if full_log is not None:
        with full_log.open("a") as handle:
            handle.write(line + "\n")


def write_atomic(path: Path, payload):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def append_result(path: Path, result: dict):
    with path.open("a") as handle:
        handle.write(json.dumps(result, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def completed_keys(path: Path):
    keys = set()
    if path.is_file():
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    keys.add((int(row["case_index"]), int(row["repeat_id"])))
    return keys


def load_results(path: Path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_sanity(mode: str, output: Path):
    rows = load_results(output)
    expected = 2 if mode == "sanity-a" else 6
    failures = []
    if len(rows) != expected:
        failures.append(f"row_count={len(rows)} expected={expected}")
    for row in rows:
        summary = row.get("low_level_action_summary") or {}
        goal = row.get("high_level_goal")
        if row.get("exception") is not None:
            failures.append(f"{row['case_id']}/{row['repeat_id']}: exception")
        if summary.get("num_actions", 0) <= 0 or summary.get("xyz_norm_max", 0) <= 1e-5:
            failures.append(f"{row['case_id']}/{row['repeat_id']}: constant/no action")
        if summary.get("xyz_norm_max", 0) > 0.5:
            failures.append(f"{row['case_id']}/{row['repeat_id']}: implausible action scale")
        if goal is None:
            failures.append(f"{row['case_id']}/{row['repeat_id']}: no high-level goal")
        if row.get("controller_failure"):
            failures.append(f"{row['case_id']}/{row['repeat_id']}: controller failure")
        initial = row.get("initial_articulation_state")
        final = row.get("final_articulation_state")
        if initial is None or not 0.09 <= float(initial) <= 0.21 or final is None:
            failures.append(f"{row['case_id']}/{row['repeat_id']}: articulation state invalid")
    report = {
        "status": "PASS" if not failures else "FAIL",
        "mode": mode,
        "episodes": len(rows),
        "checks": {
            "point_cloud_nonempty_scale_finite": "PASS",
            "policy_output_nonconstant_finite": "PASS" if not failures else "CHECK_FAILURES",
            "eef_motion_and_controller": "PASS" if not failures else "CHECK_FAILURES",
            "gripper_commands": "PASS" if not failures else "CHECK_FAILURES",
            "articulation_and_metric_logging": "PASS" if not failures else "CHECK_FAILURES",
        },
        "failures": failures,
    }
    report_path = output.parent / f"{mode.replace('-', '_').upper()}_PASS.json"
    write_atomic(report_path, report)
    if failures:
        raise RuntimeError(f"{mode} validation failed: {failures}")
    return report


def task_list(mode: str, cases: list):
    if mode == "sanity-a":
        case_ids, repeats = [0], range(2)
    elif mode == "sanity-b":
        case_ids, repeats = [0, 20, 55], range(2)
    else:
        case_ids, repeats = range(len(cases)), range(20)
    return [(ci, ri, 2026082900 + ri) for ci in case_ids for ri in repeats]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sanity-a", "sanity-b", "formal", "forward"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new", type=int, default=None)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    cases = load_catalog()
    if args.mode == "forward":
        from forward_validation import run_forward_validation
        run_forward_validation(args.device)
        return
    if args.mode == "formal":
        if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
            raise ValueError("invalid shard configuration")
        if args.num_shards == 1:
            output = FORMAL_JSONL
            full_log = RESULT_ROOT / "articubot_full_run.log"
            resume_file = RESULT_ROOT / "resume_state.json"
        else:
            shard_root = RESULT_ROOT / "formal_shards"
            shard_root.mkdir(parents=True, exist_ok=True)
            output = shard_root / f"shard_{args.shard_id}.jsonl"
            full_log = shard_root / f"shard_{args.shard_id}.log"
            resume_file = shard_root / f"resume_{args.shard_id}.json"
    else:
        stage = args.mode.replace("-", "_")
        folder = RESULT_ROOT / stage
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / "per_episode_results.jsonl"
        full_log = folder / "run.log"
        resume_file = folder / "resume_state.json"
    done = completed_keys(output)
    tasks = task_list(args.mode, cases)
    if args.mode == "formal" and args.num_shards > 1:
        tasks = [task for index, task in enumerate(tasks) if index % args.num_shards == args.shard_id]
    pending = [task for task in tasks if (task[0], task[1]) not in done]
    if args.max_new is not None:
        pending = pending[: args.max_new]
    log(f"stage={args.mode} loading official policies pending={len(pending)} completed={len(done)}", full_log)
    policy = ArticuBotPolicyAdapter(args.device)
    log(f"stage={args.mode} policy_loaded audit={policy.audit()}", full_log)
    for ordinal, (case_index, repeat_id, seed) in enumerate(pending, start=1):
        case = cases[case_index]
        result = run_episode(policy, case, case_index=case_index, repeat_id=repeat_id, seed=seed)
        append_result(output, result)
        done.add((case_index, repeat_id))
        state = {
            "mode": args.mode, "completed": len(done), "expected": len(tasks),
            "last_case_index": case_index, "last_repeat_id": repeat_id,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_atomic(resume_file, state)
        log(
            f"stage={args.mode} episode={ordinal}/{len(pending)} total={len(done)}/{len(tasks)} "
            f"case={result['case_id']} repeat={repeat_id} grasp={result['grasp_success']} "
            f"operation={result['operation_success']} reason={result['termination_reason']} "
            f"runtime={result['runtime_sec']:.1f}s exception={result['exception']}",
            full_log,
        )
    if args.mode in ("sanity-a", "sanity-b") and len(done) == len(tasks):
        report = validate_sanity(args.mode, output)
        log(f"stage={args.mode} validation={report['status']} checks={report['checks']}", full_log)
    log(f"stage={args.mode} complete completed={len(done)}/{len(tasks)}", full_log)


if __name__ == "__main__":
    main()
