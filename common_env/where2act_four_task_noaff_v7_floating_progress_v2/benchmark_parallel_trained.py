#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


HOME = Path.home()
CODE_DIR = Path(__file__).resolve().parent

RUN_TRIAL = CODE_DIR / "run_trial.py"

POSE_CATALOG = (
    HOME
    / "robot_baselines/configs/flowbot3d/eval"
    / "eval_pose_catalog.jsonl"
)

CKPT = (
    HOME
    / "robot_baselines/repos/where2act/logs"
    / "where2act_full_adapt/best-network.pth"
)

FREEZE_FILE = (
    CODE_DIR
    / "FORMAL_FREEZE_TRAINED.json"
)

OUT_ROOT = (
    HOME
    / "robot_baselines/results/where2act"
    / "FINAL_BACKEND_V2_TRAINED_1120"
)

LOG_ROOT = OUT_ROOT / "logs"

TRIAL_SEEDS = [
    2026082900 + i
    for i in range(20)
]

EXPECTED_TARGETS = 56
TRIALS_PER_TARGET = 20
EXPECTED_TRIALS = 1120


# ============================================================
# Hash / freeze
# ============================================================

def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def verify_freeze():

    if not FREEZE_FILE.is_file():
        raise RuntimeError(
            f"缺少 freeze manifest: {FREEZE_FILE}"
        )

    freeze = json.loads(
        FREEZE_FILE.read_text()
    )

    if freeze.get("status") != "FROZEN":
        raise RuntimeError(
            "FORMAL_FREEZE_TRAINED.json status != FROZEN"
        )

    expected_ckpt = Path(
        freeze["checkpoint"]
    ).resolve()

    if expected_ckpt != CKPT.resolve():
        raise RuntimeError(
            "checkpoint 路径和 freeze 不一致"
        )

    for raw_path, expected_hash in (
        freeze["artifacts_sha256"].items()
    ):
        path = Path(raw_path)

        if not path.is_file():
            raise RuntimeError(
                f"frozen artifact missing: {path}"
            )

        actual = sha256(path)

        if actual != expected_hash:
            raise RuntimeError(
                "FROZEN FILE CHANGED:\n"
                f"{path}\n"
                f"expected={expected_hash}\n"
                f"actual  ={actual}"
            )

    return freeze


# ============================================================
# Targets
# ============================================================

def load_targets():

    targets = []

    with open(POSE_CATALOG) as f:

        for line in f:

            if not line.strip():
                continue

            row = json.loads(line)

            link = (
                row.get("link_name")
                or row.get("target_link")
                or row.get("link")
            )

            if link is None:
                raise RuntimeError(
                    "pose catalog entry missing link"
                )

            targets.append(
                {
                    "shape_id":
                        str(row["shape_id"]),

                    "link_name":
                        str(link),

                    "category":
                        str(
                            row.get(
                                "category",
                                "UNKNOWN"
                            )
                        ),
                }
            )

    keys = [
        (
            x["shape_id"],
            x["link_name"]
        )
        for x in targets
    ]

    if len(targets) != EXPECTED_TARGETS:
        raise RuntimeError(
            f"expected 56 targets, got {len(targets)}"
        )

    if len(set(keys)) != EXPECTED_TARGETS:
        raise RuntimeError(
            "duplicate shape/link targets"
        )

    return targets


def trial_dir(task):

    return (
        OUT_ROOT
        / (
            f"{task['shape_id']}_"
            f"{task['link_name']}"
        )
        / (
            f"seed_"
            f"{int(task['trial_seed']):03d}"
        )
    )


def make_tasks(targets):

    tasks = []

    for target_index, target in enumerate(targets):

        for trial_index, seed in enumerate(
            TRIAL_SEEDS
        ):

            task = {
                **target,

                "target_index":
                    target_index,

                "trial_index":
                    trial_index,

                "trial_seed":
                    int(seed),
            }

            td = trial_dir(task)

            task["trial_dir"] = td
            task["result_file"] = td / "result.json"

            task["log_file"] = (
                LOG_ROOT
                / (
                    f"{target['shape_id']}_"
                    f"{target['link_name']}_"
                    f"seed_{seed}.log"
                )
            )

            tasks.append(task)

    if len(tasks) != EXPECTED_TRIALS:
        raise RuntimeError(
            f"expected 1120 tasks, got {len(tasks)}"
        )

    return tasks


# ============================================================
# Result validation
# ============================================================

def load_result(path, task, freeze):

    with open(path) as f:
        d = json.load(f)

    required = [
        "method",
        "shape_id",
        "target_link",
        "trial_seed",
        "checkpoint",
        "network_trained",
        "engineering_allow_untrained",
        "observation_success",
        "policy_success",
        "planning_success",
        "pregrasp_execution_success",
        "grasp_success",
        "operation_success_given_grasp",
        "final_success",
        "failure_reason",
        "implementation_error",
    ]

    missing = [
        k for k in required
        if k not in d
    ]

    if missing:
        raise RuntimeError(
            f"result missing fields: {missing}"
        )

    if str(d["method"]) != "Where2Act":
        raise RuntimeError(
            "method != Where2Act"
        )

    if str(d["shape_id"]) != str(
        task["shape_id"]
    ):
        raise RuntimeError(
            "shape mismatch"
        )

    if str(d["target_link"]) != str(
        task["link_name"]
    ):
        raise RuntimeError(
            "link mismatch"
        )

    if int(d["trial_seed"]) != int(
        task["trial_seed"]
    ):
        raise RuntimeError(
            "seed mismatch"
        )

    protocol = freeze.get(
        "protocol_version"
    )

    if protocol is not None:

        if d.get(
            "protocol_version"
        ) != protocol:

            raise RuntimeError(
                "protocol_version mismatch"
            )

    backend = freeze.get(
        "backend_version"
    )

    if backend is not None:

        if d.get(
            "backend_version"
        ) != backend:

            raise RuntimeError(
                "backend_version mismatch"
            )

    got_ckpt = Path(
        d["checkpoint"]
    ).resolve()

    if got_ckpt != CKPT.resolve():
        raise RuntimeError(
            "trial checkpoint mismatch"
        )

    if not bool(
        d["network_trained"]
    ):
        raise RuntimeError(
            "network_trained != True"
        )

    if bool(
        d["engineering_allow_untrained"]
    ):
        raise RuntimeError(
            "engineering flag enabled"
        )

    if bool(
        d["implementation_error"]
    ):
        raise RuntimeError(
            "implementation_error=True"
        )

    grasp = bool(
        d["grasp_success"]
    )

    op = bool(
        d["operation_success_given_grasp"]
    )

    final = bool(
        d["final_success"]
    )

    # final success must be a subset of grasp success.
    if final and not grasp:
        raise RuntimeError(
            "final_success=True with grasp_success=False"
        )

    if final and not op:
        raise RuntimeError(
            "final_success=True with operation_success_given_grasp=False"
        )

    # Current executor defines these identically.
    if op != final:
        raise RuntimeError(
            "operation_success_given_grasp != final_success"
        )

    if (
        not final
        and d.get("failure_reason") is None
    ):
        raise RuntimeError(
            "failed trial missing failure_reason"
        )

    return d


# ============================================================
# One formal trial
# ============================================================

def run_one(task, freeze):

    result_file = task["result_file"]
    td = task["trial_dir"]

    # Resume support.
    if result_file.is_file():

        try:

            d = load_result(
                result_file,
                task,
                freeze
            )

            return {
                **task,
                "status": "SKIP",
                "failure_reason":
                    d.get("failure_reason"),
                "grasp_success":
                    bool(d["grasp_success"]),
                "final_success":
                    bool(d["final_success"]),
            }

        except Exception:
            pass

    # Any incomplete/invalid old trial is removed completely.
    if td.exists():
        shutil.rmtree(td)

    task["log_file"].parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cmd = [
        sys.executable,
        str(RUN_TRIAL),

        "--shape-id",
        task["shape_id"],

        "--target-link",
        task["link_name"],

        "--pose-catalog",
        str(POSE_CATALOG),

        "--trial-seed",
        str(task["trial_seed"]),

        "--checkpoint",
        str(CKPT),

        "--output-root",
        str(OUT_ROOT),

        "--device",
        "cuda:0",

        "--planner-env",
        "where2act_planner",

        "--planner",
        "RRTConnect",

        "--planning-time",
        "5",

        "--ik-attempts",
        "100",
    ]

    # CRITICAL:
    # no --engineering-allow-untrained

    env = dict(os.environ)

    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    t0 = time.time()

    try:

        with open(
            task["log_file"],
            "w"
        ) as log:

            proc = subprocess.run(
                cmd,
                cwd=str(CODE_DIR),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )

        elapsed = (
            time.time() - t0
        )

        if result_file.is_file():

            try:

                d = load_result(
                    result_file,
                    task,
                    freeze
                )

                return {
                    **task,
                    "status": "OK",
                    "returncode":
                        int(proc.returncode),
                    "elapsed_s":
                        elapsed,
                    "failure_reason":
                        d.get("failure_reason"),
                    "grasp_success":
                        bool(
                            d["grasp_success"]
                        ),
                    "final_success":
                        bool(
                            d["final_success"]
                        ),
                }

            except Exception as exc:

                return {
                    **task,
                    "status":
                        "IMPLEMENTATION_ERROR",
                    "returncode":
                        int(proc.returncode),
                    "elapsed_s":
                        elapsed,
                    "error":
                        f"invalid result: {exc}",
                }

        return {
            **task,
            "status":
                "IMPLEMENTATION_ERROR",
            "returncode":
                int(proc.returncode),
            "elapsed_s":
                elapsed,
            "error":
                "result.json missing",
        }

    except Exception as exc:

        return {
            **task,
            "status":
                "IMPLEMENTATION_ERROR",
            "elapsed_s":
                time.time() - t0,
            "error":
                repr(exc),
        }


# ============================================================
# Parallel runner
# ============================================================

def run_pool(
    tasks,
    workers,
    freeze,
    *,
    label,
):

    results = []

    completed = 0
    total = len(tasks)

    t0 = time.time()

    with ThreadPoolExecutor(
        max_workers=workers
    ) as pool:

        futures = {
            pool.submit(
                run_one,
                task,
                freeze
            ): task

            for task in tasks
        }

        for future in as_completed(
            futures
        ):

            item = future.result()

            results.append(item)

            completed += 1

            elapsed = max(
                time.time() - t0,
                1e-6
            )

            throughput = (
                completed
                / elapsed
                * 60.0
            )

            remain = (
                total
                - completed
            )

            eta = (
                remain / throughput
                if throughput > 0
                else 0.0
            )

            print(
                f"[{label} "
                f"{completed:04d}/{total:04d}] "
                f"{item['status']:<20s} "
                f"{item['shape_id']}/"
                f"{item['link_name']} "
                f"seed={item['trial_seed']} "
                f"reason="
                f"{item.get('failure_reason')} "
                f"| {throughput:.2f} trial/min "
                f"| ETA≈{eta:.1f} min",
                flush=True
            )

    return results


# ============================================================
# Final summary
# ============================================================

def write_final_summary(
    targets,
    tasks,
    freeze,
    workers,
):

    records = []
    missing = []

    for task in tasks:

        path = task["result_file"]

        if not path.is_file():

            missing.append(
                {
                    "shape_id":
                        task["shape_id"],
                    "link_name":
                        task["link_name"],
                    "trial_seed":
                        task["trial_seed"],
                }
            )

            continue

        try:

            d = load_result(
                path,
                task,
                freeze
            )

        except Exception as exc:

            missing.append(
                {
                    "shape_id":
                        task["shape_id"],
                    "link_name":
                        task["link_name"],
                    "trial_seed":
                        task["trial_seed"],
                    "error":
                        str(exc),
                }
            )

            continue

        records.append(
            {
                "shape_id":
                    task["shape_id"],

                "link_name":
                    task["link_name"],

                "category":
                    task["category"],

                "trial_index":
                    task["trial_index"],

                "trial_seed":
                    task["trial_seed"],

                "initial_ratio_requested":
                    d.get(
                        "initial_ratio_requested"
                    ),

                "initial_progress_actual":
                    d.get(
                        "initial_progress_actual"
                    ),

                "observation_success":
                    bool(
                        d.get(
                            "observation_success",
                            False
                        )
                    ),

                "policy_success":
                    bool(
                        d.get(
                            "policy_success",
                            False
                        )
                    ),

                "planning_success":
                    bool(
                        d.get(
                            "planning_success",
                            False
                        )
                    ),

                "pregrasp_execution_success":
                    bool(
                        d.get(
                            "pregrasp_execution_success",
                            False
                        )
                    ),

                "grasp_success":
                    bool(
                        d[
                            "grasp_success"
                        ]
                    ),

                "operation_success_given_grasp":
                    bool(
                        d[
                            "operation_success_given_grasp"
                        ]
                    ),

                "final_success":
                    bool(
                        d[
                            "final_success"
                        ]
                    ),

                "final_progress":
                    d.get(
                        "final_progress"
                    ),

                "interaction_score":
                    d.get(
                        "interaction_score"
                    ),

                "critic_score":
                    d.get(
                        "critic_score"
                    ),

                "failure_reason":
                    d.get(
                        "failure_reason"
                    ),
            }
        )

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Trial CSV
    # --------------------------------------------------------

    trial_csv = (
        OUT_ROOT
        / "trial_results.csv"
    )

    if records:

        with open(
            trial_csv,
            "w",
            newline=""
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    records[0].keys()
                )
            )

            writer.writeheader()
            writer.writerows(records)

    # --------------------------------------------------------
    # Target-level CSV
    # --------------------------------------------------------

    grouped = defaultdict(list)

    for r in records:

        grouped[
            (
                r["shape_id"],
                r["link_name"],
                r["category"],
            )
        ].append(r)

    target_rows = []

    for (
        shape,
        link,
        category
    ), xs in sorted(
        grouped.items()
    ):

        n = len(xs)

        grasp_n = sum(
            int(x["grasp_success"])
            for x in xs
        )

        final_n = sum(
            int(x["final_success"])
            for x in xs
        )

        target_rows.append(
            {
                "shape_id":
                    shape,

                "link_name":
                    link,

                "category":
                    category,

                "completed_trials":
                    n,

                "grasp_success_trials":
                    grasp_n,

                "grasp_success_rate":
                    (
                        grasp_n / n
                        if n
                        else None
                    ),

                "final_success_trials":
                    final_n,

                "post_grasp_operation_success_rate":
                    (
                        final_n / grasp_n
                        if grasp_n > 0
                        else None
                    ),

                "final_success_rate":
                    (
                        final_n / n
                        if n
                        else None
                    ),
            }
        )

    with open(
        OUT_ROOT
        / "target_results.csv",
        "w",
        newline=""
    ) as f:

        fields = [
            "shape_id",
            "link_name",
            "category",
            "completed_trials",
            "grasp_success_trials",
            "grasp_success_rate",
            "final_success_trials",
            "post_grasp_operation_success_rate",
            "final_success_rate",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(
            target_rows
        )

    # --------------------------------------------------------
    # Senior's three metrics
    # --------------------------------------------------------

    grasp_n = sum(
        int(x["grasp_success"])
        for x in records
    )

    final_n = sum(
        int(x["final_success"])
        for x in records
    )

    op_n = sum(
        int(
            x[
                "operation_success_given_grasp"
            ]
        )
        for x in records
    )

    if op_n != final_n:
        raise RuntimeError(
            "operation/final count mismatch"
        )

    complete = (
        len(missing) == 0
        and len(records)
        == EXPECTED_TRIALS
    )

    grasp_rate = (
        grasp_n
        / EXPECTED_TRIALS
        if complete
        else None
    )

    post_grasp_rate = (
        final_n
        / float(grasp_n)
        if complete
        and grasp_n > 0
        else None
    )

    final_rate = (
        final_n
        / EXPECTED_TRIALS
        if complete
        else None
    )

    failures = Counter(
        x["failure_reason"]
        for x in records
        if not x["final_success"]
    )

    diagnostics = {
        "observation_success_trials":
            sum(
                int(
                    x[
                        "observation_success"
                    ]
                )
                for x in records
            ),

        "policy_success_trials":
            sum(
                int(
                    x[
                        "policy_success"
                    ]
                )
                for x in records
            ),

        "planning_success_trials":
            sum(
                int(
                    x[
                        "planning_success"
                    ]
                )
                for x in records
            ),

        "pregrasp_execution_success_trials":
            sum(
                int(
                    x[
                        "pregrasp_execution_success"
                    ]
                )
                for x in records
            ),
    }

    summary = {
        "method":
            "Where2Act",

        "protocol_version":
            freeze.get(
                "protocol_version"
            ),

        "backend_version":
            freeze.get(
                "backend_version"
            ),

        "interaction":
            "physical_two_finger_no_object_drive",

        "checkpoint":
            str(CKPT),

        "expected_targets":
            EXPECTED_TARGETS,

        "trials_per_target":
            TRIALS_PER_TARGET,

        "expected_trials":
            EXPECTED_TRIALS,

        "completed_trials":
            len(records),

        "missing_trials":
            len(missing),

        "grasp_success_trials":
            grasp_n,

        "grasp_success_rate":
            grasp_rate,

        "final_success_trials":
            final_n,

        "post_grasp_operation_success_rate":
            post_grasp_rate,

        "final_success_rate":
            final_rate,

        "failure_reasons":
            dict(failures),

        "diagnostics":
            diagnostics,

        "trial_seeds":
            TRIAL_SEEDS,

        "workers":
            workers,

        "freeze_manifest":
            str(FREEZE_FILE),

        "artifacts_sha256":
            freeze[
                "artifacts_sha256"
            ],
    }

    with open(
        OUT_ROOT
        / "summary.json",
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    with open(
        OUT_ROOT
        / "missing_trials.json",
        "w"
    ) as f:

        json.dump(
            missing,
            f,
            indent=2
        )

    print()
    print("=" * 110)
    print("WHERE2ACT FINAL 1120 SUMMARY")
    print("=" * 110)

    print(
        "completed:",
        f"{len(records)}/{EXPECTED_TRIALS}"
    )

    print(
        "missing:",
        len(missing)
    )

    print()

    if complete:

        print(
            "1. Grasp Success Rate:"
        )

        print(
            f"   {grasp_n}/1120 "
            f"= {100.0 * grasp_rate:.2f}%"
        )

        print()

        print(
            "2. Post-Grasp Operation Success Rate:"
        )

        if grasp_n > 0:

            print(
                f"   {final_n}/{grasp_n} "
                f"= "
                f"{100.0 * post_grasp_rate:.2f}%"
            )

        else:

            print(
                "   N/A (grasp success = 0)"
            )

        print()

        print(
            "3. Final Success Rate:"
        )

        print(
            f"   {final_n}/1120 "
            f"= {100.0 * final_rate:.2f}%"
        )

    else:

        print(
            "正式三个指标暂不计算："
            "仍有 missing/invalid trials"
        )

    print()
    print("Failure reasons:")

    for reason, n in (
        failures.most_common()
    ):

        print(
            f"  {reason}: {n}"
        )

    print()
    print(
        OUT_ROOT
        / "summary.json"
    )

    print(
        OUT_ROOT
        / "trial_results.csv"
    )

    print(
        OUT_ROOT
        / "target_results.csv"
    )

    print("=" * 110)

    return summary, missing


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        type=int,
        default=6
    )

    parser.add_argument(
        "--preflight-only",
        action="store_true"
    )

    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError(
            "--workers must >= 1"
        )

    freeze = verify_freeze()

    targets = load_targets()
    tasks = make_tasks(targets)

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    LOG_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUT_ROOT
        / "benchmark_manifest.json",
        "w"
    ) as f:

        json.dump(
            {
                "method":
                    "Where2Act",

                "protocol_version":
                    freeze.get(
                        "protocol_version"
                    ),

                "backend_version":
                    freeze.get(
                        "backend_version"
                    ),

                "targets":
                    56,

                "trials_per_target":
                    20,

                "expected_trials":
                    1120,

                "trial_seeds":
                    TRIAL_SEEDS,

                "workers":
                    args.workers,

                "gpu_device":
                    "cuda:0",

                "checkpoint":
                    str(CKPT),

                "freeze_manifest":
                    str(FREEZE_FILE),
            },
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Formal preflight:
    # 8 different targets, same first formal seed.
    #
    # Their task outcomes are already formal results.
    # Only implementation errors may block the benchmark.
    # --------------------------------------------------------

    preflight_target_indices = [
        0, 7, 14, 21,
        28, 35, 42, 49
    ]

    preflight_tasks = [
        tasks[
            i * TRIALS_PER_TARGET
        ]
        for i
        in preflight_target_indices
    ]

    print("=" * 110)
    print("WHERE2ACT FORMAL PREFLIGHT")
    print("=" * 110)
    print(
        "8 targets × 1 formal seed"
    )
    print(
        "Only implementation errors block launch."
    )
    print("=" * 110)

    pre_results = run_pool(
        preflight_tasks,
        min(
            args.workers,
            len(preflight_tasks)
        ),
        freeze,
        label="PREFLIGHT",
    )

    pre_errors = [
        x for x in pre_results
        if x["status"]
        == "IMPLEMENTATION_ERROR"
    ]

    if pre_errors:

        path = (
            OUT_ROOT
            / "PREFLIGHT_IMPLEMENTATION_ERRORS.json"
        )

        with open(
            path,
            "w"
        ) as f:

            json.dump(
                pre_errors,
                f,
                indent=2,
                default=str
            )

        print()
        print(
            "PREFLIGHT IMPLEMENTATION ERROR"
        )

        print(path)

        return 2

    with open(
        OUT_ROOT
        / "PREFLIGHT_PASS.json",
        "w"
    ) as f:

        json.dump(
            {
                "status":
                    "PASS",

                "rule":
                    (
                        "No implementation errors. "
                        "Method failures remain formal failures."
                    ),

                "trials":
                    pre_results,
            },
            f,
            indent=2,
            default=str
        )

    print()
    print("=" * 110)
    print("PREFLIGHT PASS")
    print("=" * 110)

    if args.preflight_only:

        print(
            "Run full formal benchmark with:"
        )

        print(
            "python benchmark_parallel_trained.py --workers 6"
        )

        return 0

    # --------------------------------------------------------
    # All 1120.
    # Existing valid preflight results are automatically skipped.
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("WHERE2ACT FORMAL 56 x 20")
    print("=" * 110)
    print("targets: 56")
    print("trials/target: 20")
    print("total: 1120")
    print("workers:", args.workers)
    print("=" * 110)

    all_results = run_pool(
        tasks,
        args.workers,
        freeze,
        label="FORMAL",
    )

    errors = [
        x for x in all_results
        if x["status"]
        == "IMPLEMENTATION_ERROR"
    ]

    if errors:

        path = (
            OUT_ROOT
            / "IMPLEMENTATION_ERRORS.json"
        )

        with open(
            path,
            "w"
        ) as f:

            json.dump(
                errors,
                f,
                indent=2,
                default=str
            )

        print()
        print("=" * 110)
        print(
            "FORMAL RUN HAS IMPLEMENTATION ERRORS"
        )
        print("=" * 110)

        print(
            "count:",
            len(errors)
        )

        print(path)

        print(
            "只处理工程错误；"
            "不要根据成功率修改评测规则。"
        )

        return 2

    summary, missing = (
        write_final_summary(
            targets,
            tasks,
            freeze,
            args.workers,
        )
    )

    return (
        0
        if not missing
        else 3
    )


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
