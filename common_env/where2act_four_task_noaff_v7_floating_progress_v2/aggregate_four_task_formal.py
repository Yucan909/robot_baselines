#!/usr/bin/env python3
"""Strict aggregation for the corrected progress-targeted protocol."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from formal_protocol import (
    FAILURE_CATEGORIES,
    TASK_SPECS,
    TRIAL_SEEDS,
    check_unique_targets,
    expected_trials,
    load_jsonl,
    read_json,
    read_valid_result,
    result_path,
    sha256,
    write_json,
)


def percent(numerator: int, denominator: int) -> Any:
    return None if denominator == 0 else 100.0 * numerator / denominator


def rate_payload(numerator: int, denominator: int) -> Dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "percentage": percent(numerator, denominator),
        "display": (
            f"{numerator}/{denominator}=N/A"
            if denominator == 0
            else f"{numerator}/{denominator}={100.0 * numerator / denominator:.3f}%"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-manifest", required=True)
    parser.add_argument("--result-root", required=True)
    args = parser.parse_args()

    protocol_path = Path(args.protocol_manifest).resolve()
    result_root = Path(args.result_root).resolve()
    protocol = read_json(protocol_path)
    if protocol.get("status") != "FROZEN":
        raise RuntimeError("protocol manifest is not FROZEN")
    if protocol.get("trial_seeds") != TRIAL_SEEDS:
        raise RuntimeError("seed freeze mismatch")

    summary_root = result_root / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    metrics: Dict[str, Any] = {
        "title": "WHERE2ACT NO-AFFORDANCE V7 FLOATING PROGRESS-TARGETED V2",
        "protocol_manifest": str(protocol_path),
        "protocol_manifest_sha256": sha256(protocol_path),
        "formal_result_root": str(result_root),
        "completed_episodes": 0,
        "expected_episodes": 3640,
        "tasks": {},
        "secondary_paper_native_metrics": {
            "F-score": "NOT COMPUTED — missing required scoring-only labels",
            "Sample-Succ": "NOT COMPUTED — missing required scoring-only labels",
            "reason": (
                "The local paper-native code expects dedicated offline interaction/scoring "
                "labels that are not present in these formal rollout results; no metric was fabricated."
            ),
        },
    }
    per_target_rows: List[Dict[str, Any]] = []
    metrics_csv_rows: List[Dict[str, Any]] = []
    global_failures: Counter[str] = Counter()
    failures_by_task: Dict[str, Counter[str]] = {}

    for task in TASK_SPECS:
        frozen = protocol["tasks"][task]
        targets = check_unique_targets(task, load_jsonl(Path(frozen["formal_catalog"])))
        results: List[Dict[str, Any]] = []
        target_results: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
        errors: List[str] = []
        for target in targets:
            shape_id = str(target["shape_id"])
            target_link = str(target["target_link"])
            for seed in TRIAL_SEEDS:
                rp = result_path(result_root, task, shape_id, target_link, seed)
                result, reason = read_valid_result(
                    rp,
                    task=task,
                    shape_id=shape_id,
                    target_link=target_link,
                    seed=seed,
                    checkpoint=str(frozen["checkpoint"]),
                    checkpoint_sha256=str(frozen["checkpoint_sha256"]),
                )
                if result is None:
                    errors.append(f"{rp}: {reason}")
                    continue
                results.append(result)
                target_results[(shape_id, target_link)].append(result)
        if errors:
            excerpt = "\n".join(errors[:30])
            raise RuntimeError(f"{task}: {len(errors)} missing/invalid formal result(s):\n{excerpt}")
        expected = expected_trials(task)
        if len(results) != expected:
            raise RuntimeError(f"{task}: expected {expected}, validated {len(results)}")

        grasp = sum(bool(row["grasp_success"]) for row in results)
        final = sum(bool(row["final_operation_success"]) for row in results)
        reach_35 = sum(bool(row["final_state_threshold_met"]) for row in results)
        if final > grasp:
            raise RuntimeError(f"{task}: final successes exceed grasp/engagement successes")
        failures = Counter(
            str(row["failure_reason"])
            for row in results
            if not row["final_operation_success"]
        )
        failures_by_task[task] = failures
        global_failures.update(failures)
        task_metrics = {
            "task": task,
            "primitive": TASK_SPECS[task]["primitive"],
            "goal": TASK_SPECS[task]["goal"],
            "targets": len(targets),
            "total_trials": len(results),
            "final_operation_success": final,
            "final_operation_success_rate": rate_payload(final, len(results)),
            "reach_35": reach_35,
            "reach_35_rate": rate_payload(reach_35, len(results)),
            "reach_35_definition": (
                "ungated final-state reach: Open final_progress>=0.35; "
                "Close final_progress<=0.15"
            ),
            "diagnostic_grasp_or_engagement_success": grasp,
            "failure_breakdown": {name: int(failures.get(name, 0)) for name in FAILURE_CATEGORIES},
        }
        metrics["tasks"][task] = task_metrics
        metrics["completed_episodes"] += len(results)

        for metric_name in ("final_operation_success_rate", "reach_35_rate"):
            rate = task_metrics[metric_name]
            metrics_csv_rows.append(
                {
                    "task": task,
                    "primitive": TASK_SPECS[task]["primitive"],
                    "goal": TASK_SPECS[task]["goal"],
                    "metric": metric_name,
                    "numerator": rate["numerator"],
                    "denominator": rate["denominator"],
                    "percentage": "N/A" if rate["percentage"] is None else f"{rate['percentage']:.6f}",
                }
            )

        for target in targets:
            key = (str(target["shape_id"]), str(target["target_link"]))
            rows = target_results[key]
            if len(rows) != 20:
                raise RuntimeError(f"{task}/{key}: expected 20 trials, got {len(rows)}")
            target_grasp = sum(bool(row["grasp_success"]) for row in rows)
            target_final = sum(bool(row["final_operation_success"]) for row in rows)
            target_reach_35 = sum(bool(row["final_state_threshold_met"]) for row in rows)
            per_target_rows.append(
                {
                    "task": task,
                    "primitive": TASK_SPECS[task]["primitive"],
                    "goal": TASK_SPECS[task]["goal"],
                    "shape_id": key[0],
                    "target_link": key[1],
                    "total_trials": 20,
                    "final_operation_success": target_final,
                    "final_operation_success_percentage": f"{100.0 * target_final / 20:.6f}",
                    "reach_35": target_reach_35,
                    "reach_35_percentage": f"{100.0 * target_reach_35 / 20:.6f}",
                    "diagnostic_grasp_or_engagement_success": target_grasp,
                }
            )

    if metrics["completed_episodes"] != 3640:
        raise RuntimeError(
            f"formal total mismatch: {metrics['completed_episodes']}/3640"
        )

    write_json(summary_root / "FINAL_METRICS.json", metrics)
    failure_payload = {
        "categories": list(FAILURE_CATEGORIES),
        "by_task": {
            task: {name: int(counter.get(name, 0)) for name in FAILURE_CATEGORIES}
            for task, counter in failures_by_task.items()
        },
        "overall": {name: int(global_failures.get(name, 0)) for name in FAILURE_CATEGORIES},
    }
    write_json(summary_root / "failure_breakdown.json", failure_payload)

    with (summary_root / "FINAL_METRICS.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics_csv_rows[0]))
        writer.writeheader()
        writer.writerows(metrics_csv_rows)
    with (summary_root / "per_target_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_target_rows[0]))
        writer.writeheader()
        writer.writerows(per_target_rows)

    lines = [
        "WHERE2ACT NO-AFFORDANCE V7",
        "FLOATING-GRIPPER PROGRESS-TARGETED V2 FORMAL EVALUATION",
        "",
        "completed_episodes:",
        "3640 / 3640",
        "",
    ]
    for task in TASK_SPECS:
        item = metrics["tasks"][task]
        lines.extend(
            [
                "=" * 50,
                task,
                "=" * 50,
                "total_trials:",
                str(item["total_trials"]),
                "final_operation_success_rate:",
                item["final_operation_success_rate"]["display"].replace("=", " = ", 1),
                "reach_35_rate:",
                item["reach_35_rate"]["display"].replace("=", " = ", 1),
                "reach_35_definition:",
                item["reach_35_definition"],
                "diagnostic_grasp_or_engagement_success:",
                str(item["diagnostic_grasp_or_engagement_success"]),
                "",
                "failure_breakdown:",
            ]
        )
        for category in FAILURE_CATEGORIES:
            lines.append(f"{category}: {item['failure_breakdown'][category]}")
        lines.append("")
    lines.extend(
        [
            "=" * 50,
            "SECONDARY PAPER-NATIVE METRICS",
            "=" * 50,
            "F-score: NOT COMPUTED — missing required scoring-only labels",
            "Sample-Succ: NOT COMPUTED — missing required scoring-only labels",
            "",
            f"protocol_manifest: {protocol_path}",
            f"formal_result_root: {result_root}",
        ]
    )
    (summary_root / "FINAL_METRICS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((summary_root / "FINAL_METRICS.txt").read_text(encoding="utf-8"))
    print(f"JSON: {summary_root / 'FINAL_METRICS.json'}")


if __name__ == "__main__":
    main()
