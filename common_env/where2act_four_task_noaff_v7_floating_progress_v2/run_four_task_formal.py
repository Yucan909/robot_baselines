#!/usr/bin/env python3
"""Resume-safe launcher (up to the validated Parallel-8 concurrency)."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from formal_protocol import (
    TASK_SPECS,
    TRIAL_SEEDS,
    check_unique_targets,
    episode_log_path,
    expected_trials,
    load_jsonl,
    read_json,
    read_valid_result,
    result_path,
    sha256,
    write_json,
)


def verify_freeze(manifest_path: Path, backend: Path) -> Dict[str, Any]:
    manifest = read_json(manifest_path)
    if manifest.get("status") != "FROZEN":
        raise RuntimeError("protocol manifest is not FROZEN")
    if manifest.get("trial_seeds") != TRIAL_SEEDS:
        raise RuntimeError("frozen seed set differs from the established formal seeds")
    if int(manifest.get("trials_per_target", -1)) != len(TRIAL_SEEDS):
        raise RuntimeError("invalid frozen trials_per_target")
    if int(manifest.get("expected_total_trials", -1)) != 3640:
        raise RuntimeError("invalid frozen total trial count")
    for relative, expected in manifest["backend_hashes"].items():
        path = backend / relative
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"backend changed after freeze: {path}: {actual} != {expected}")
    for task, spec in TASK_SPECS.items():
        frozen = manifest["tasks"][task]
        checkpoint = Path(frozen["checkpoint"])
        catalog = Path(frozen["formal_catalog"])
        if sha256(checkpoint) != frozen["checkpoint_sha256"]:
            raise RuntimeError(f"{task}: checkpoint changed after freeze")
        if sha256(catalog) != frozen["formal_catalog_sha256"]:
            raise RuntimeError(f"{task}: formal catalog changed after freeze")
        rows = check_unique_targets(task, load_jsonl(catalog))
        if len(rows) * len(TRIAL_SEEDS) != expected_trials(task):
            raise RuntimeError(f"{task}: frozen trial count mismatch")
        if int(frozen["targets"]) != spec["targets"]:
            raise RuntimeError(f"{task}: frozen target count mismatch")
    return manifest


def run_job(
    *,
    index: int,
    total: int,
    task: str,
    target: Dict[str, Any],
    seed: int,
    backend: Path,
    output_root: Path,
    python: Path,
    frozen_task: Dict[str, Any],
) -> Dict[str, Any]:
    shape_id = str(target["shape_id"])
    target_link = str(target["target_link"])
    checkpoint = str(frozen_task["checkpoint"])
    checkpoint_sha = str(frozen_task["checkpoint_sha256"])
    rp = result_path(output_root, task, shape_id, target_link, seed)
    existing, invalid_reason = read_valid_result(
        rp,
        task=task,
        shape_id=shape_id,
        target_link=target_link,
        seed=seed,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha,
    )
    if existing is not None:
        return {
            "index": index,
            "total": total,
            "task": task,
            "shape_id": shape_id,
            "target_link": target_link,
            "seed": seed,
            "status": "resume-skip",
            "returncode": 0,
            "failure_reason": existing.get("failure_reason"),
        }

    log_path = episode_log_path(output_root, task, shape_id, target_link, seed)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(python),
        str(backend / "run_trial.py"),
        "--task",
        task,
        "--shape-id",
        shape_id,
        "--target-link",
        target_link,
        "--formal-catalog",
        str(frozen_task["formal_catalog"]),
        "--trial-seed",
        str(seed),
        "--checkpoint",
        checkpoint,
        "--output-root",
        str(output_root),
        "--device",
        "cuda:0",
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    started = time.time()
    resource_retries = 0
    while True:
        process = subprocess.run(
            command,
            cwd=str(backend),
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(
                f"\n===== launcher attempt {resource_retries + 1} =====\n"
            )
            stream.write(process.stdout)
        result, validation_reason = read_valid_result(
            rp,
            task=task,
            shape_id=shape_id,
            target_link=target_link,
            seed=seed,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha,
        )
        if process.returncode == 0 and result is not None:
            break
        if "CUDA error: out of memory" not in process.stdout:
            break
        resource_retries += 1
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(
                "\nCUDA OOM is a shared-resource failure, not an episode "
                "outcome; waiting 15 seconds and rerunning the same seed.\n"
            )
        time.sleep(15)
    status = "complete" if process.returncode == 0 and result is not None else "implementation-failure"
    return {
        "index": index,
        "total": total,
        "task": task,
        "shape_id": shape_id,
        "target_link": target_link,
        "seed": seed,
        "status": status,
        "returncode": int(process.returncode),
        "failure_reason": None if result is None else result.get("failure_reason"),
        "invalid_existing_reason": invalid_reason,
        "validation_reason": validation_reason,
        "log": str(log_path),
        "resource_retries": resource_retries,
        "elapsed_seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--python",
        default="/home/feng/miniconda3/envs/where2act/bin/python",
    )
    args = parser.parse_args()

    backend = Path(__file__).resolve().parent
    manifest_path = Path(args.protocol_manifest).resolve()
    output_root = Path(args.output_root).resolve()
    python = Path(args.python).resolve()
    if not python.exists():
        raise FileNotFoundError(python)
    if not 1 <= args.workers <= 8:
        raise RuntimeError("workers must be between 1 and the validated maximum of 8")
    manifest = verify_freeze(manifest_path, backend)
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / "formal_progress.json"

    all_failures: List[Dict[str, Any]] = []
    completed_total = 0
    print(f"Frozen formal root: {output_root}", flush=True)
    print(f"Protocol manifest SHA256: {sha256(manifest_path)}", flush=True)

    for task in TASK_SPECS:
        frozen_task = manifest["tasks"][task]
        targets = check_unique_targets(task, load_jsonl(Path(frozen_task["formal_catalog"])))
        jobs: List[Tuple[Dict[str, Any], int]] = [
            (target, seed) for target in targets for seed in TRIAL_SEEDS
        ]
        task_total = len(jobs)
        task_done = 0
        task_skipped = 0
        print(
            f"\n[{task}] starting/resuming {task_total} episodes "
            f"with {args.workers} worker(s)",
            flush=True,
        )

        # Queue several waves at once so a long physical operation does not
        # leave the other workers idle at an artificial batch barrier.
        queue_span = args.workers * 8
        for batch_start in range(0, task_total, queue_span):
            batch = jobs[batch_start : batch_start + queue_span]
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [
                    pool.submit(
                        run_job,
                        index=batch_start + offset + 1,
                        total=task_total,
                        task=task,
                        target=target,
                        seed=seed,
                        backend=backend,
                        output_root=output_root,
                        python=python,
                        frozen_task=frozen_task,
                    )
                    for offset, (target, seed) in enumerate(batch)
                ]
                reports = [future.result() for future in concurrent.futures.as_completed(futures)]
            reports.sort(key=lambda item: item["index"])
            batch_failures = [item for item in reports if item["status"] == "implementation-failure"]
            all_failures.extend(batch_failures)
            for report in reports:
                task_done += int(report["status"] in ("complete", "resume-skip"))
                task_skipped += int(report["status"] == "resume-skip")
            completed_total += sum(
                int(report["status"] in ("complete", "resume-skip")) for report in reports
            )
            write_json(
                progress_path,
                {
                    "protocol_manifest": str(manifest_path),
                    "protocol_manifest_sha256": sha256(manifest_path),
                    "current_task": task,
                    "current_task_completed": task_done,
                    "current_task_expected": task_total,
                    "current_task_resume_skipped": task_skipped,
                    "completed_in_this_launcher_pass": completed_total,
                    "expected_total": 3640,
                    "implementation_failures": all_failures,
                    "last_batch": reports,
                    "updated_unix": time.time(),
                },
            )
            print(
                f"[{task}] {task_done:4d}/{task_total} | resume={task_skipped} | "
                f"last={reports[-1]['status']}",
                flush=True,
            )
            if batch_failures:
                details = json.dumps(batch_failures, indent=2, ensure_ascii=False)
                raise RuntimeError(f"implementation failure(s); fix and resume same seeds:\n{details}")

        if task_done != task_total:
            raise RuntimeError(f"{task}: incomplete launcher count {task_done}/{task_total}")
        print(f"[{task}] COMPLETE {task_done}/{task_total}", flush=True)

    write_json(
        progress_path,
        {
            "status": "COMPLETE",
            "protocol_manifest": str(manifest_path),
            "protocol_manifest_sha256": sha256(manifest_path),
            "completed": 3640,
            "expected_total": 3640,
            "workers": args.workers,
            "updated_unix": time.time(),
        },
    )
    print("\nFOUR-TASK FORMAL COMPLETE: 3640/3640", flush=True)


if __name__ == "__main__":
    main()
