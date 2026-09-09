#!/usr/bin/env python3
"""Resumable runner for the ArticuBot-PerceptionReach method variant."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


BASELINE = Path("/home/feng/robot_baselines/repos/articubot/reproduction/unified_open")
VARIANT = Path("/home/feng/robot_baselines/repos/articubot/reproduction/perception_reach")
for path in (VARIANT, BASELINE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from perception_reach_policy_adapter import PerceptionReachPolicyAdapter
from frozen_open_episode import DRAWER_POSE_CATALOG, POSE_CATALOG, THRESHOLD, load_catalog
from perception_reach_episode import run_perception_reach_episode


SEED_BASE = 2026090800


def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_done(path):
    done = set()
    if path.is_file():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row["case_index"]), int(row["repeat_id"]))
            if key in done:
                raise RuntimeError(f"duplicate {key}")
            done.add(key)
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-mode", choices=("door_open", "drawer_open"), required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--case-indices", default="")
    parser.add_argument("--repeat-ids", default="0,1")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--max-new", type=int)
    args = parser.parse_args()
    catalog = POSE_CATALOG if args.task_mode == "door_open" else DRAWER_POSE_CATALOG
    cases = load_catalog(catalog, task_mode=args.task_mode)
    case_indices = list(range(len(cases))) if not args.case_indices else [
        int(v) for v in args.case_indices.split(",") if v.strip()
    ]
    repeat_ids = list(range(20)) if args.formal else [
        int(v) for v in args.repeat_ids.split(",") if v.strip()
    ]
    tasks = [(c, r) for c in case_indices for r in repeat_ids]
    tasks = [task for ordinal, task in enumerate(tasks) if ordinal % args.num_shards == args.shard_id]
    stage = args.result_root / ("formal_shards" if args.formal else "sanity") / args.task_mode
    stage.mkdir(parents=True, exist_ok=True)
    output = stage / f"shard_{args.shard_id:02d}.jsonl"
    log_path = stage / f"shard_{args.shard_id:02d}.log"
    done = load_done(output)
    pending = [task for task in tasks if task not in done]
    if args.max_new is not None:
        pending = pending[:args.max_new]
    args.result_root.mkdir(parents=True, exist_ok=True)

    def log(message):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(line, flush=True)
        with log_path.open("a") as handle:
            handle.write(line + "\n")
        with (args.result_root / "progress.log").open("a") as handle:
            handle.write(line + "\n")

    log(f"task={args.task_mode} shard={args.shard_id}/{args.num_shards} pending={len(pending)} loading")
    policy = PerceptionReachPolicyAdapter(args.device)
    log(f"policy={policy.audit()}")
    for ordinal, (case_index, repeat_id) in enumerate(pending, 1):
        row = run_perception_reach_episode(
            policy, cases[case_index], case_index=case_index, repeat_id=repeat_id,
            seed=SEED_BASE + repeat_id,
        )
        if abs(float(row["articulation_success_threshold"]) - THRESHOLD) > 1e-12:
            raise RuntimeError("threshold changed")
        if bool(row["operation_success"]) != bool(row["grasp_success"] and row["reach_35"]):
            raise RuntimeError("gated metric mismatch")
        append(output, row)
        done.add((case_index, repeat_id))
        log(
            f"episode={ordinal}/{len(pending)} case={row['case_id']} repeat={repeat_id} "
            f"grasp={row['grasp_success']} reach35={row['reach_35']} "
            f"reason={row['termination_reason']} runtime={row['runtime_sec']:.1f}s "
            f"exception={row['exception']}"
        )
    log(f"complete rows={len(done)}/{len(tasks)}")


if __name__ == "__main__":
    main()
