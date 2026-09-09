#!/usr/bin/env python3
"""Run the frozen object-level Open DEV physical rollout for PADP V4.

This is a method/runtime validation set, never the formal target catalog.  It
uses two shards per task so SAPIEN and CUDA stay below the host limits while
still exercising the requested parallel execution.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path


HOME = Path("/home/feng")
CODE = HOME / "robot_baselines/repos/pa3ff_official/reproduction"
VARIANT = os.environ.get("PA3FF_PADP_VARIANT", "v4")
DEFAULT_ROOT = HOME / "robot_baselines/results/pa3ff" / (
    "reproduction_v5/open_dev_rollout_v1" if VARIANT == "v5_positional"
    else "reproduction_v4/open_dev_rollout_v1"
)
ROOT = Path(os.environ.get("PA3FF_DEV_ROOT", str(DEFAULT_ROOT)))
CATALOG = HOME / "robot_baselines/results/pa3ff/reproduction_v3_dev_rollout_selection/open_dev_catalog_v19.jsonl"
CHECKPOINT = Path(os.environ.get(
    "PA3FF_DEV_CHECKPOINT",
    str(HOME / "robot_baselines/results/pa3ff/reproduction_v4/training_30000/checkpoints/step030000.pt"),
))
WORKER = CODE / (
    "dev_receding_bottomfaithful_padp_v5_worker_v39.py"
    if VARIANT == "v5_positional" else
    "dev_receding_bottomfaithful_padp_v4_worker_v38.py"
)
TASKS = tuple(
    task.strip() for task in os.environ.get(
        "PA3FF_DEV_TASKS", "door_open,drawer_open"
    ).split(",") if task.strip()
)
if not TASKS or any(task not in {"door_open", "drawer_open"} for task in TASKS):
    raise ValueError(f"invalid PA3FF_DEV_TASKS={TASKS}")
SHARDS = int(os.environ.get("PA3FF_DEV_SHARDS_PER_TASK", "2"))
NOISE_SCALE = float(os.environ.get("PA3FF_DEV_NOISE_SCALE", "1.0"))
DDIM_STEPS = int(os.environ.get("PA3FF_DEV_DDIM_STEPS", "10"))
START_TIMESTEP = int(os.environ.get("PA3FF_DEV_START_TIMESTEP", "99"))
POSITION_PRIOR_WEIGHT = float(os.environ.get("PA3FF_DEV_POSITION_PRIOR_WEIGHT", "0.02"))
GEOMETRY_PRIOR_WEIGHT = float(os.environ.get("PA3FF_DEV_GEOMETRY_PRIOR_WEIGHT", "0.02"))
GRASP_GEOMETRY_PRIOR_WEIGHT = float(
    os.environ.get("PA3FF_DEV_GRASP_GEOMETRY_PRIOR_WEIGHT", "0.0")
)
CLIP_PRED_X0 = float(os.environ.get("PA3FF_DEV_CLIP_PRED_X0", "3.0"))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize() -> dict:
    rows = []
    for task in TASKS:
        for result in (ROOT / task).glob("**/result.json"):
            if any(part.startswith("planner_") for part in result.parts):
                continue
            try:
                row = json.loads(result.read_text(encoding="utf-8"))
            except Exception:
                continue
            if row.get("episode_status") == "complete":
                rows.append(row)
    expected = 32 * len(TASKS)
    report = {
        "status": "PASS" if len(rows) == expected else "INCOMPLETE",
        "total": len(rows), "expected": expected,
    }
    for task in TASKS:
        task_rows = [row for row in rows if row.get("task") == task]
        grasp = sum(bool(row.get("grasp_success")) for row in task_rows)
        r35 = sum(float(row.get("directional_task_progress", float("-inf"))) >= 0.35 for row in task_rows)
        r40 = sum(float(row.get("directional_task_progress", float("-inf"))) >= 0.40 for row in task_rows)
        report[task] = {
            "complete": len(task_rows), "grasp_success": grasp,
            "reached_target_35": r35, "reached_target_40": r40,
            "reached_target_35_rate": r35 / len(task_rows) if task_rows else None,
        }
    (ROOT / "DEV_METRICS.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / "protocol_manifest.json"
    task_rank_weights = {}
    for task in TASKS:
        prefix = f"PA3FF_{task.upper()}_"
        task_rank_weights[task] = {
            "position_prior_weight": float(os.environ.get(
                prefix + "POSITION_PRIOR_WEIGHT", str(POSITION_PRIOR_WEIGHT)
            )),
            "geometry_prior_weight": float(os.environ.get(
                prefix + "GEOMETRY_PRIOR_WEIGHT", str(GEOMETRY_PRIOR_WEIGHT)
            )),
            "grasp_geometry_prior_weight": float(os.environ.get(
                prefix + "GRASP_GEOMETRY_PRIOR_WEIGHT",
                str(GRASP_GEOMETRY_PRIOR_WEIGHT),
            )),
        }
    manifest.write_text(json.dumps({
        "status": "FROZEN_OPEN_DEV_RUNTIME_VALIDATION",
        "scope": "object-level Open DEV only",
        "episodes": 32 * len(TASKS),
        "tasks": list(TASKS),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha(CHECKPOINT),
        "catalog": str(CATALOG),
        "catalog_sha256": sha(CATALOG),
        "worker": str(WORKER),
        "worker_sha256": sha(WORKER),
        "candidate_count": 128,
        "rank_mode": "long_motion_train_only",
        "normalized_action_clip": CLIP_PRED_X0,
        "noise_scale": NOISE_SCALE, "ddim_steps": DDIM_STEPS,
        "start_timestep": START_TIMESTEP,
        "position_prior_weight": POSITION_PRIOR_WEIGHT,
        "geometry_prior_weight": GEOMETRY_PRIOR_WEIGHT,
        "grasp_geometry_prior_weight": GRASP_GEOMETRY_PRIOR_WEIGHT,
        "task_rank_weights": task_rank_weights,
        "formal_catalog_or_results_used": False,
        "normal_policy_failure_retry": False,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    jobs = []
    for task in TASKS:
        for shard in range(SHARDS):
            log_path = ROOT / f"{task}_shard_{shard}.log"
            log = log_path.open("w", encoding="utf-8")
            env = os.environ.copy()
            env.update({
                "PA3FF_DEV_CANDIDATE_COUNT": "128",
                "PA3FF_DEV_RANK_MODE": "long_motion",
                "PA3FF_DEV_CLIP_PRED_X0": str(CLIP_PRED_X0),
                "PA3FF_DEV_NOISE_SCALE": str(NOISE_SCALE),
                "PA3FF_DEV_DDIM_STEPS": str(DDIM_STEPS),
                "PA3FF_DEV_START_TIMESTEP": str(START_TIMESTEP),
                "PA3FF_DEV_POSITION_PRIOR_WEIGHT": str(POSITION_PRIOR_WEIGHT),
                "PA3FF_DEV_GEOMETRY_PRIOR_WEIGHT": str(GEOMETRY_PRIOR_WEIGHT),
                "PA3FF_DEV_GRASP_GEOMETRY_PRIOR_WEIGHT": str(
                    GRASP_GEOMETRY_PRIOR_WEIGHT
                ),
            })
            command = [
                str(CODE / "formal_env.sh"), str(WORKER),
                "--checkpoint", str(CHECKPOINT),
                "--formal-root", str(ROOT),
                "--catalog", str(CATALOG),
                "--protocol-manifest", str(manifest),
                "--task", task,
                "--num-shards", str(SHARDS),
                "--shard-id", str(shard),
            ]
            proc = subprocess.Popen(
                command, cwd=str(CODE), env=env, stdout=log,
                stderr=subprocess.STDOUT,
            )
            jobs.append((task, shard, proc, log))
            print(f"START {task} shard={shard} pid={proc.pid}", flush=True)
    failed = []
    for task, shard, proc, log in jobs:
        code = proc.wait()
        log.close()
        print(f"DONE {task} shard={shard} exit={code}", flush=True)
        if code:
            failed.append({"task": task, "shard": shard, "exit": code})
    report = summarize()
    report["failed_workers"] = failed
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if failed or report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
