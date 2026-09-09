import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


HOME = Path.home()

CODE_DIR = Path(__file__).resolve().parent
RUN_TRIAL = CODE_DIR / "run_trial.py"
PANDA_CONTROLLER = CODE_DIR / "panda_controller.py"
GRASP_ADAPTER = CODE_DIR / "grasp_pose_adapter.py"
CONTACT_MONITOR = CODE_DIR / "contact_monitor.py"

POSE_CATALOG = (
    HOME
    / "robot_baselines/configs/flowbot3d/eval"
    / "eval_pose_catalog.jsonl"
)

SMOKE_ROOT = (
    HOME
    / "robot_baselines/results/flowbot3d"
    / "conditionA_physical_v2_smoke"
)
PASS_FILE = SMOKE_ROOT / "EXECUTOR_SMOKE_PASS.json"

OUT_ROOT = (
    HOME
    / "robot_baselines/results/flowbot3d"
    / "FINAL_CONDITIONA_PHYSICAL_V2"
)
LOG_ROOT = OUT_ROOT / "logs"

TRIAL_SEEDS = [2026082900 + i for i in range(20)]
PROTOCOL_VERSION = "flowbot3d_conditionA_physical_v2"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="并行 trial 数；当前 RTX 5080 + 16C/32T 推荐 8",
    )
    return parser.parse_args()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def current_hashes():
    return {
        "run_trial_sha256": sha256(RUN_TRIAL),
        "panda_controller_sha256": sha256(PANDA_CONTROLLER),
        "grasp_pose_adapter_sha256": sha256(GRASP_ADAPTER),
        "contact_monitor_sha256": sha256(CONTACT_MONITOR),
    }


def verify_frozen_code():
    if not PASS_FILE.exists():
        raise RuntimeError(
            "找不到 EXECUTOR_SMOKE_PASS.json；请先运行 smoke10.py。"
        )

    with open(PASS_FILE) as f:
        manifest = json.load(f)

    if manifest.get("status") != "PASS":
        raise RuntimeError("smoke manifest 不是 PASS")

    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("smoke protocol_version 不匹配")

    hashes = current_hashes()

    for key, value in hashes.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"{key} 在 smoke10 之后发生变化，正式 benchmark 拒绝启动。"
            )

    return hashes


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
                raise RuntimeError("eval pose catalog 有记录缺少 link 名")

            targets.append(
                {
                    "shape_id": str(row["shape_id"]),
                    "link_name": link,
                    "category": row.get("category", "UNKNOWN"),
                }
            )

    keys = [(x["shape_id"], x["link_name"]) for x in targets]
    if len(targets) != 56:
        raise RuntimeError(f"验证目标应为 56，实际为 {len(targets)}")
    if len(set(keys)) != 56:
        raise RuntimeError("验证目标存在重复 shape/link")

    return targets


def load_result(result_file):
    with open(result_file) as f:
        result = json.load(f)

    required = [
        "method",
        "protocol_version",
        "shape_id",
        "target_link",
        "trial_seed",
        "grasp_success",
        "success",
    ]
    missing = [k for k in required if k not in result]
    if missing:
        raise RuntimeError(f"{result_file}: result.json 缺字段 {missing}")

    if result["protocol_version"] != PROTOCOL_VERSION:
        raise RuntimeError(f"{result_file}: protocol_version 不匹配")

    # 最终成功必须是抓取成功的子集。
    if bool(result["success"]) and not bool(result["grasp_success"]):
        raise RuntimeError(
            f"{result_file}: success=True 但 grasp_success=False，指标逻辑不一致"
        )

    if (
        not bool(result["success"])
        and result.get("failure_reason") is None
    ):
        raise RuntimeError(f"{result_file}: 失败 trial 没有 failure_reason")

    return result


def make_tasks(targets):
    tasks = []
    for target_index, target in enumerate(targets):
        for trial_index, seed in enumerate(TRIAL_SEEDS):
            sid = target["shape_id"]
            link = target["link_name"]
            trial_dir = OUT_ROOT / f"{sid}_{link}_seed_{seed}"
            tasks.append(
                {
                    **target,
                    "target_index": target_index,
                    "trial_index": trial_index,
                    "trial_seed": seed,
                    "trial_dir": trial_dir,
                    "result_file": trial_dir / "result.json",
                    "log_file": LOG_ROOT / f"{sid}_{link}_seed_{seed}.log",
                }
            )
    return tasks


def run_one(task):
    result_file = task["result_file"]

    if result_file.exists():
        try:
            load_result(result_file)
            return {**task, "status": "SKIP"}
        except Exception:
            try:
                result_file.unlink()
            except Exception:
                pass

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
        "--output-root",
        str(OUT_ROOT),
        "--no-video",
    ]

    env = dict(os.environ)
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["FLOWBOT_DEVICE"] = "cuda:0"

    task["log_file"].parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    with open(task["log_file"], "w") as log:
        proc = subprocess.run(
            cmd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    elapsed = time.time() - t0

    if proc.returncode == 0 and result_file.exists():
        try:
            load_result(result_file)
            return {
                **task,
                "status": "OK",
                "elapsed_s": elapsed,
            }
        except Exception as exc:
            return {
                **task,
                "status": "IMPLEMENTATION_ERROR",
                "elapsed_s": elapsed,
                "error": f"invalid result.json: {exc}",
            }

    return {
        **task,
        "status": "IMPLEMENTATION_ERROR",
        "elapsed_s": elapsed,
        "returncode": proc.returncode,
        "error": "process failed or result.json missing",
    }


def scan_results(tasks):
    records = []
    missing = []

    for task in tasks:
        if not task["result_file"].exists():
            missing.append(
                {
                    "shape_id": task["shape_id"],
                    "link_name": task["link_name"],
                    "category": task["category"],
                    "trial_index": task["trial_index"],
                    "trial_seed": task["trial_seed"],
                }
            )
            continue

        result = load_result(task["result_file"])

        records.append(
            {
                "shape_id": task["shape_id"],
                "link_name": task["link_name"],
                "category": task["category"],
                "trial_index": task["trial_index"],
                "trial_seed": task["trial_seed"],
                "requested_initial_progress": result.get(
                    "requested_initial_progress"
                ),
                "actual_initial_progress": result.get(
                    "actual_initial_progress"
                ),
                "pre_pull_progress": result.get("pre_pull_progress"),
                "final_progress": result.get("final_progress"),
                "initial_contact_distance_m": result.get(
                    "initial_contact_distance_m"
                ),
                "approach_position_error_m": result.get(
                    "approach_position_error_m"
                ),
                "grasp_success": bool(result["grasp_success"]),
                "operation_attempted_after_grasp": bool(
                    result.get("operation_attempted_after_grasp", False)
                ),
                "success": bool(result["success"]),
                "failure_reason": result.get("failure_reason"),
            }
        )

    return records, missing


def write_summary_files(targets, tasks, hashes, workers):
    records, missing = scan_results(tasks)

    trial_csv = OUT_ROOT / "trial_results.csv"
    trial_fields = [
        "shape_id",
        "link_name",
        "category",
        "trial_index",
        "trial_seed",
        "requested_initial_progress",
        "actual_initial_progress",
        "pre_pull_progress",
        "final_progress",
        "initial_contact_distance_m",
        "approach_position_error_m",
        "grasp_success",
        "operation_attempted_after_grasp",
        "success",
        "failure_reason",
    ]

    with open(trial_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=trial_fields)
        writer.writeheader()
        writer.writerows(records)

    grouped = defaultdict(list)
    for r in records:
        grouped[(r["shape_id"], r["link_name"], r["category"])].append(r)

    target_rows = []
    for (sid, link, category), xs in sorted(grouped.items()):
        n = len(xs)
        grasp_n = sum(int(x["grasp_success"]) for x in xs)
        final_n = sum(int(x["success"]) for x in xs)
        post_grasp_rate = final_n / grasp_n if grasp_n > 0 else None

        target_rows.append(
            {
                "shape_id": sid,
                "link_name": link,
                "category": category,
                "completed_trials": n,
                "grasp_success_trials": grasp_n,
                "grasp_success_rate": grasp_n / n if n else None,
                "final_success_trials": final_n,
                "post_grasp_operation_success_rate": post_grasp_rate,
                "final_success_rate": final_n / n if n else None,
            }
        )

    target_csv = OUT_ROOT / "target_results.csv"
    with open(target_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "shape_id",
                "link_name",
                "category",
                "completed_trials",
                "grasp_success_trials",
                "grasp_success_rate",
                "final_success_trials",
                "post_grasp_operation_success_rate",
                "final_success_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(target_rows)

    failure_counter = Counter(
        x["failure_reason"]
        for x in records
        if not x["success"]
    )

    grasp_n = sum(int(x["grasp_success"]) for x in records)
    final_n = sum(int(x["success"]) for x in records)

    # 学长要求的三个指标。
    grasp_rate = grasp_n / 1120.0 if len(missing) == 0 else None
    post_grasp_rate = (
        final_n / float(grasp_n)
        if len(missing) == 0 and grasp_n > 0
        else None
    )
    final_rate = final_n / 1120.0 if len(missing) == 0 else None

    summary = {
        "method": "FlowBot3D",
        "protocol_version": PROTOCOL_VERSION,
        "interaction": "physical_two_finger_no_object_drive",
        "expected_targets": 56,
        "trials_per_target": 20,
        "expected_trials": 1120,
        "completed_trials": len(records),
        "missing_trials": len(missing),
        "grasp_success_trials": grasp_n,
        "grasp_success_rate": grasp_rate,
        "final_success_trials": final_n,
        "post_grasp_operation_success_rate": post_grasp_rate,
        "final_success_rate": final_rate,
        "failure_reasons": dict(failure_counter),
        "trial_seeds": TRIAL_SEEDS,
        "workers": int(workers),
        "code_hashes": hashes,
    }

    with open(OUT_ROOT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(OUT_ROOT / "missing_trials.json", "w") as f:
        json.dump(missing, f, indent=2)

    return summary, records, missing


def main():
    args = parse_args()
    workers = int(args.workers)
    if workers < 1:
        raise ValueError("--workers 必须 >= 1")

    hashes = verify_frozen_code()
    targets = load_targets()
    tasks = make_tasks(targets)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    with open(OUT_ROOT / "benchmark_manifest.json", "w") as f:
        json.dump(
            {
                "protocol_version": PROTOCOL_VERSION,
                "interaction": "physical_two_finger_no_object_drive",
                "targets": 56,
                "trials_per_target": 20,
                "expected_trials": 1120,
                "trial_seeds": TRIAL_SEEDS,
                "workers": workers,
                "gpu_device": "cuda:0",
                "code_hashes": hashes,
                "pose_catalog": str(POSE_CATALOG),
            },
            f,
            indent=2,
        )

    print("=" * 110)
    print("FLOWBOT3D CONDITION-A REAL PHYSICAL TWO-FINGER V2")
    print("=" * 110)
    print("interaction: REAL finger collision/contact/friction ONLY; NO suction/weld/object drive")
    print("targets: 56")
    print("trials/target: 20")
    print("total: 1120")
    print("workers:", workers)
    print("output:", OUT_ROOT)
    print("=" * 110)

    errors = []
    completed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, task): task for task in tasks}

        for future in as_completed(futures):
            result = future.result()
            completed += 1

            elapsed = max(time.time() - t0, 1e-6)
            throughput = completed / elapsed * 60.0
            remain = 1120 - completed
            eta = remain / throughput if throughput > 0 else 0.0

            print(
                f"[{completed:04d}/1120] "
                f"{result['status']:<20s} "
                f"{result['shape_id']}/{result['link_name']} "
                f"trial={result['trial_index']:02d} | "
                f"{throughput:.1f} trial/min | ETA≈{eta:.1f} min"
            )

            if result["status"] == "IMPLEMENTATION_ERROR":
                errors.append(
                    {
                        "shape_id": result["shape_id"],
                        "link_name": result["link_name"],
                        "trial_index": result["trial_index"],
                        "trial_seed": result["trial_seed"],
                        "returncode": result.get("returncode"),
                        "error": result.get("error"),
                        "log": str(result["log_file"]),
                    }
                )

    if errors:
        with open(OUT_ROOT / "IMPLEMENTATION_ERRORS.json", "w") as f:
            json.dump(errors, f, indent=2)

        print()
        print("=" * 110)
        print("存在 implementation_error:", len(errors))
        print("请只修工程错误，不根据抓取/任务成功率调评测规则。")
        print(OUT_ROOT / "IMPLEMENTATION_ERRORS.json")
        print("=" * 110)
        return 2

    summary, _, missing = write_summary_files(
        targets,
        tasks,
        hashes,
        workers,
    )

    print()
    print("=" * 110)
    print("FINAL CONDITION-A PHYSICAL V2 BENCHMARK SUMMARY")
    print("=" * 110)
    print("completed:", f"{summary['completed_trials']}/1120")
    print("missing:", summary["missing_trials"])
    print()

    if not missing:
        grasp_n = summary["grasp_success_trials"]
        final_n = summary["final_success_trials"]

        print(
            "抓取成功率: "
            f"{grasp_n}/1120 = "
            f"{100.0 * summary['grasp_success_rate']:.2f}%"
        )

        if grasp_n > 0:
            print(
                "抓取成功后的操作成功率: "
                f"{final_n}/{grasp_n} = "
                f"{100.0 * summary['post_grasp_operation_success_rate']:.2f}%"
            )
        else:
            print("抓取成功后的操作成功率: N/A（抓取成功数为 0）")

        print(
            "最终成功率: "
            f"{final_n}/1120 = "
            f"{100.0 * summary['final_success_rate']:.2f}%"
        )

    print()
    print("failure reasons:")
    for reason, count in Counter(summary["failure_reasons"]).most_common():
        print(f"  {reason}: {count}")

    print()
    print(OUT_ROOT / "summary.json")
    print(OUT_ROOT / "trial_results.csv")
    print(OUT_ROOT / "target_results.csv")
    print("=" * 110)

    return 0 if not missing else 3


if __name__ == "__main__":
    raise SystemExit(main())
