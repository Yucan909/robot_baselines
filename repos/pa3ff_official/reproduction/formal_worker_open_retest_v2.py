#!/usr/bin/env python3
"""Frozen-checkpoint Open re-test with the corrected Cartesian controller."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
from pathlib import Path

import formal_worker as legacy
from panda_controller_corrected_v2 import CorrectedPandaTwoFingerController


METHOD = "PA3FF_reproduction_v1_open_retest_v2"
EXECUTION_PROTOCOL = "world_cartesian_grasptarget_v2"
CONTROLLER_CLASS = CorrectedPandaTwoFingerController


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def valid_existing(path: Path, checkpoint_sha: str, catalog_sha: str, protocol_sha: str) -> bool:
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        return (
            row.get("episode_status") == "complete"
            and row.get("checkpoint_sha256") == checkpoint_sha
            and row.get("catalog_sha256") == catalog_sha
            and row.get("execution_protocol_sha256") == protocol_sha
        )
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--task", choices=["door_open", "door_close", "drawer_open", "drawer_close"])
    parser.add_argument("--smoke-cases", type=Path)
    args = parser.parse_args()

    legacy.PandaTwoFingerController = CONTROLLER_CLASS
    checkpoint_sha = sha256(args.checkpoint)
    catalog_sha = sha256(args.catalog)
    protocol_sha = sha256(args.protocol_manifest)
    cases = [
        json.loads(line)
        for line in args.catalog.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    indexed = list(enumerate(cases))
    if args.task:
        indexed = [(i, case) for i, case in indexed if case["task"] == args.task]
    if args.smoke_cases:
        keys = {
            (row["task"], int(row["target_index"]), int(row["trial_index"]))
            for row in json.loads(args.smoke_cases.read_text(encoding="utf-8"))
        }
        indexed = [
            (i, case)
            for i, case in indexed
            if (case["task"], int(case["target_index"]), int(case["trial_index"])) in keys
        ]
    else:
        indexed = [
            (i, case) for i, case in indexed if i % args.num_shards == args.shard_id
        ]

    runtime = legacy.PA3FFPADPRuntime(args.checkpoint)
    for ordinal, (global_index, case) in enumerate(indexed, 1):
        episode_dir = (
            args.formal_root
            / case["task"]
            / f"{int(case['target_index']):03d}_{case['shape_id']}_{case['target_link']}"
            / f"trial_{int(case['trial_index']):02d}_seed_{case['seed']}"
        )
        episode_dir.mkdir(parents=True, exist_ok=True)
        result_path = episode_dir / "result.json"
        if valid_existing(result_path, checkpoint_sha, catalog_sha, protocol_sha):
            print(
                f"SKIP {ordinal}/{len(indexed)} {case['task']} "
                f"{case['shape_id']}/{case['target_link']} seed={case['seed']}",
                flush=True,
            )
            continue

        result = None
        for attempt in range(1, 3):
            legacy.PA3FF_CURRENT_EPISODE_DIR = episode_dir
            with (episode_dir / f"attempt_{attempt}.log").open("w", encoding="utf-8") as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    result = legacy.run_episode(runtime, case, checkpoint_sha, catalog_sha)
            result.update(
                {
                    "method": METHOD,
                    "execution_protocol": EXECUTION_PROTOCOL,
                    "execution_protocol_sha256": protocol_sha,
                    "infrastructure_attempt": attempt,
                }
            )
            planning = result.get("pregrasp_control", {}).get("planning_success")
            result["pregrasp_planning_success"] = planning
            if planning is False and result.get("failure_reason") == "grasp_failed":
                result["failure_reason"] = "pregrasp_planning_failed"
            result_path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            if result["episode_status"] == "complete":
                break
        print(
            f"DONE {ordinal}/{len(indexed)} global={global_index} {case['task']} "
            f"{case['shape_id']}/{case['target_link']} seed={case['seed']} "
            f"status={result['episode_status']} grasp={result['grasp_success']} "
            f"progress={result['directional_task_progress']} final={result['final_success']} "
            f"sec={result['runtime_seconds']:.2f}",
            flush=True,
        )
        gc.collect()


if __name__ == "__main__":
    main()
