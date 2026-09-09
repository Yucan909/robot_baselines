#!/usr/bin/env python3
"""Compare paired full-arm and floating-gripper formal Where2Act runs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


TASKS = ("door_open", "door_close", "drawer_open", "drawer_close")
RATE_KEYS = ("final_operation_success_rate", "reach_35_rate")


def load_json(path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_totals(metrics):
    total = sum(metrics["tasks"][task]["total_trials"] for task in TASKS)
    final = sum(metrics["tasks"][task]["final_operation_success"] for task in TASKS)
    reach_35 = sum(metrics["tasks"][task]["reach_35"] for task in TASKS)
    return {
        "total_trials": total,
        "final_operation_success": final,
        "final_operation_success_rate": 100.0 * final / total,
        "reach_35": reach_35,
        "reach_35_rate": 100.0 * reach_35 / total,
    }


def display_rate(rate):
    return rate["display"].replace("=", " = ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-arm-metrics", type=Path, required=True)
    parser.add_argument("--floating-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    full = load_json(args.full_arm_metrics)
    floating = load_json(args.floating_metrics)
    for name, metrics in (("full_arm", full), ("floating_gripper", floating)):
        if metrics["completed_episodes"] != 3640 or metrics["expected_episodes"] != 3640:
            raise RuntimeError(f"{name} is not a complete 3640-episode formal run")
        for task in TASKS:
            if metrics["tasks"][task]["total_trials"] != full["tasks"][task]["total_trials"]:
                raise RuntimeError(f"trial-count mismatch for {task}")

    result = {
        "title": "WHERE2ACT NO-AFFORDANCE V7 EXECUTION-CONDITION COMPARISON",
        "paired_design": {
            "same_checkpoints": True,
            "same_formal_targets": True,
            "same_20_seeds": True,
            "same_scene_metadata": True,
            "open_command_final_progress": 0.40,
            "open_success_final_progress": 0.35,
            "close_command_final_progress": 0.10,
            "close_success_final_progress": 0.15,
            "fixed_operation_distance": False,
            "operation_endpoint": "measured absolute articulation progress",
            "reach_35_definition": (
                "ungated final-state reach: Open final_progress>=0.35; "
                "Close final_progress<=0.15"
            ),
        },
        "conditions": {
            "full_arm": {
                "description": "ArticuBot Panda full arm with IK, collision checking and OMPL planning",
                "metrics_path": str(args.full_arm_metrics.resolve()),
                "metrics_sha256": sha256(args.full_arm_metrics),
                "formal_result_root": full["formal_result_root"],
                "protocol_manifest": full["protocol_manifest"],
                "protocol_manifest_sha256": full["protocol_manifest_sha256"],
                "completed_episodes": full["completed_episodes"],
            },
            "floating_gripper": {
                "description": "Original-style free-flying two-finger Panda gripper; no arm IK, OMPL or arm collision path planning",
                "metrics_path": str(args.floating_metrics.resolve()),
                "metrics_sha256": sha256(args.floating_metrics),
                "formal_result_root": floating["formal_result_root"],
                "protocol_manifest": floating["protocol_manifest"],
                "protocol_manifest_sha256": floating["protocol_manifest_sha256"],
                "completed_episodes": floating["completed_episodes"],
            },
        },
        "tasks": {},
        "overall": {
            "full_arm": aggregate_totals(full),
            "floating_gripper": aggregate_totals(floating),
        },
        "interpretation_caveat": (
            "This is an embodiment/execution-condition comparison, not an isolated planner ablation: "
            "the floating condition also removes full-arm geometry/occlusion and uses direct gripper pose control."
        ),
    }

    for task in TASKS:
        f = full["tasks"][task]
        g = floating["tasks"][task]
        task_out = {
            "primitive": f["primitive"],
            "goal": f["goal"],
            "total_trials_per_condition": f["total_trials"],
            "full_arm": f,
            "floating_gripper": g,
            "percentage_point_delta_floating_minus_full_arm": {},
        }
        for key in RATE_KEYS:
            fp = f[key]["percentage"]
            gp = g[key]["percentage"]
            task_out["percentage_point_delta_floating_minus_full_arm"][key] = (
                None if fp is None or gp is None else gp - fp
            )
        result["tasks"][task] = task_out

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "EXECUTION_CONDITION_COMPARISON.json"
    txt_path = args.output_dir / "EXECUTION_CONDITION_COMPARISON.txt"
    csv_path = args.output_dir / "EXECUTION_CONDITION_COMPARISON.csv"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")

    lines = [
        result["title"],
        "",
        "FULL ARM: ArticuBot Panda + IK/collision checking/OMPL",
        "FLOATING: original-style free-flying two-finger Panda gripper; no full-arm planning",
        "Both conditions: same four V7 checkpoints, targets, scenes and seeds.",
        "Open commands final progress 40% and succeeds at >=35%; Close commands 10% and succeeds at <=15%.",
        "Operation uses physical 1 cm segments with measured articulation feedback; there is no fixed 5 cm operation endpoint.",
        "",
    ]
    for task in TASKS:
        f = full["tasks"][task]
        g = floating["tasks"][task]
        lines.extend([
            "=" * 72,
            task,
            "=" * 72,
            f"full_arm final_operation_success_rate: {display_rate(f['final_operation_success_rate'])}",
            f"floating final_operation_success_rate: {display_rate(g['final_operation_success_rate'])}",
            f"full_arm reach_35_rate: {display_rate(f['reach_35_rate'])}",
            f"floating reach_35_rate: {display_rate(g['reach_35_rate'])}",
            "",
        ])
    lines.extend([
        "=" * 72,
        "OVERALL (all four tasks)",
        "=" * 72,
    ])
    for name in ("full_arm", "floating_gripper"):
        totals = result["overall"][name]
        lines.extend([
            f"{name}: {totals['total_trials']} episodes",
            f"  final_operation_success_rate: {totals['final_operation_success']}/{totals['total_trials']} = {totals['final_operation_success_rate']:.3f}%",
            f"  reach_35_rate: {totals['reach_35']}/{totals['total_trials']} = {totals['reach_35_rate']:.3f}%",
        ])
    lines.extend(["", "CAVEAT", result["interpretation_caveat"], ""])
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["task", "condition", "metric", "numerator", "denominator", "percentage"])
        for task in TASKS:
            for condition, metrics in (("full_arm", full), ("floating_gripper", floating)):
                record = metrics["tasks"][task]
                for key in RATE_KEYS:
                    rate = record[key]
                    writer.writerow([task, condition, key, rate["numerator"], rate["denominator"], rate["percentage"]])

    print(json_path)
    print(txt_path)
    print(csv_path)


if __name__ == "__main__":
    main()
