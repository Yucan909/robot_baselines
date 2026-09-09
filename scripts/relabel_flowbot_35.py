import csv
import json
import re
from pathlib import Path
from collections import defaultdict

HOME = Path.home()

THRESHOLD = 0.35

EXPERIMENTS = {
    "PHYSICAL_V2": {
        "root": HOME / "robot_baselines/results/flowbot3d/FINAL_CONDITIONA_PHYSICAL_V2",
        "protocol": "flowbot3d_conditionA_physical_v2",
    },
    "SUCTION_5CM_V3": {
        "root": HOME / "robot_baselines/results/flowbot3d/FINAL_CONDITIONB_SUCTION_5CM_V3",
        "protocol": "flowbot3d_conditionB_suction_5cm_v3",
    },
}

trial_pattern = re.compile(
    r"^(?P<shape_id>\d+)_(?P<link>link_\d+)_seed_(?P<seed>\d+)$"
)


def safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def analyze(name, cfg):
    root = cfg["root"]
    protocol = cfg["protocol"]

    print()
    print("=" * 78)
    print(name)
    print("=" * 78)
    print("source:", root)
    print("new success threshold:", THRESHOLD)

    result_files = sorted(root.glob("*/result.json"))

    print("result.json count:", len(result_files))

    if len(result_files) != 1120:
        print(
            f"WARNING: expected 1120 result files, "
            f"found {len(result_files)}"
        )

    rows = []
    missing_progress = []
    bad_protocol = []

    for path in result_files:
        with open(path, "r") as f:
            result = json.load(f)

        if result.get("protocol_version") != protocol:
            bad_protocol.append(str(path))
            continue

        m = trial_pattern.match(path.parent.name)

        if m:
            shape_id = m.group("shape_id")
            link = m.group("link")
            seed = m.group("seed")
            target = f"{shape_id}_{link}"
        else:
            shape_id = str(result.get("shape_id", "UNKNOWN"))
            link = str(
                result.get(
                    "selected_link",
                    result.get("target_link", "UNKNOWN")
                )
            )
            seed = str(result.get("seed", "UNKNOWN"))
            target = f"{shape_id}_{link}"

        grasp = bool(result.get("grasp_success", False))
        old_success = bool(result.get("success", False))

        final_progress = safe_float(
            result.get("final_progress")
        )

        if final_progress is None:
            missing_progress.append(str(path))

        # 学长的新定义：
        # 必须先抓取成功，并且最终 articulation progress >= 35%
        new_success = (
            grasp
            and final_progress is not None
            and final_progress >= THRESHOLD
        )

        # 仅用于审计：
        # 有些抓取失败 trial 可能因为碰撞导致物体移动到 35%以上。
        # 这些绝对不能计入最终成功。
        progress_only_35 = (
            final_progress is not None
            and final_progress >= THRESHOLD
        )

        rows.append({
            "target": target,
            "shape_id": shape_id,
            "link": link,
            "seed": seed,
            "grasp_success": grasp,
            "final_progress": final_progress,
            "old_success_40": old_success,
            "new_success_35": new_success,
            "progress_ge_35_regardless_of_grasp": progress_only_35,
            "newly_promoted_by_35_threshold": (
                new_success and not old_success
            ),
            "result_json": str(path),
        })

    if bad_protocol:
        print("ERROR: protocol mismatch:", len(bad_protocol))
        for x in bad_protocol[:10]:
            print("  ", x)
        raise SystemExit(1)

    if missing_progress:
        print(
            "WARNING: missing final_progress:",
            len(missing_progress)
        )
        for x in missing_progress[:20]:
            print("  ", x)

    n = len(rows)
    grasp_n = sum(int(r["grasp_success"]) for r in rows)
    old_final_n = sum(int(r["old_success_40"]) for r in rows)
    new_final_n = sum(int(r["new_success_35"]) for r in rows)

    progress_only_n = sum(
        int(r["progress_ge_35_regardless_of_grasp"])
        for r in rows
    )

    false_physical_success_n = sum(
        int(
            r["progress_ge_35_regardless_of_grasp"]
            and not r["grasp_success"]
        )
        for r in rows
    )

    promoted_n = sum(
        int(r["newly_promoted_by_35_threshold"])
        for r in rows
    )

    grasp_rate = grasp_n / n if n else None

    old_post_rate = (
        old_final_n / grasp_n
        if grasp_n else None
    )

    new_post_rate = (
        new_final_n / grasp_n
        if grasp_n else None
    )

    old_final_rate = (
        old_final_n / n
        if n else None
    )

    new_final_rate = (
        new_final_n / n
        if n else None
    )

    print()
    print("------------- OLD 40% -------------")
    print(
        f"grasp_success_rate                 : "
        f"{grasp_n}/{n} = {grasp_rate:.6f} "
        f"({100*grasp_rate:.3f}%)"
    )
    print(
        f"post_grasp_operation_success_rate  : "
        f"{old_final_n}/{grasp_n} = "
        f"{old_post_rate:.6f} "
        f"({100*old_post_rate:.3f}%)"
        if grasp_n
        else "post_grasp_operation_success_rate: N/A"
    )
    print(
        f"final_success_rate                 : "
        f"{old_final_n}/{n} = {old_final_rate:.6f} "
        f"({100*old_final_rate:.3f}%)"
    )

    print()
    print("------------- NEW 35% -------------")
    print(
        f"grasp_success_rate                 : "
        f"{grasp_n}/{n} = {grasp_rate:.6f} "
        f"({100*grasp_rate:.3f}%)"
    )
    print(
        f"post_grasp_operation_success_rate  : "
        f"{new_final_n}/{grasp_n} = "
        f"{new_post_rate:.6f} "
        f"({100*new_post_rate:.3f}%)"
        if grasp_n
        else "post_grasp_operation_success_rate: N/A"
    )
    print(
        f"final_success_rate                 : "
        f"{new_final_n}/{n} = {new_final_rate:.6f} "
        f"({100*new_final_rate:.3f}%)"
    )

    print()
    print("------------- AUDIT -------------")
    print(
        "newly promoted 40% -> 35%:",
        promoted_n,
    )
    print(
        "final_progress >= 35% ignoring grasp:",
        progress_only_n,
    )
    print(
        ">=35% but grasp_success=False "
        "(correctly excluded):",
        false_physical_success_n,
    )
    print(
        "missing final_progress:",
        len(missing_progress),
    )

    # ---------------------------------------------------------
    # Per-target metrics
    # ---------------------------------------------------------
    groups = defaultdict(list)

    for row in rows:
        groups[row["target"]].append(row)

    target_rows = []

    for target in sorted(groups):
        xs = groups[target]

        tn = len(xs)
        tg = sum(int(x["grasp_success"]) for x in xs)
        told = sum(int(x["old_success_40"]) for x in xs)
        tnew = sum(int(x["new_success_35"]) for x in xs)

        target_rows.append({
            "target": target,
            "trials": tn,
            "grasp_success_trials": tg,
            "grasp_success_rate": tg / tn if tn else None,
            "old_40_final_success_trials": told,
            "old_40_post_grasp_operation_success_rate":
                told / tg if tg else None,
            "old_40_final_success_rate":
                told / tn if tn else None,
            "new_35_final_success_trials": tnew,
            "new_35_post_grasp_operation_success_rate":
                tnew / tg if tg else None,
            "new_35_final_success_rate":
                tnew / tn if tn else None,
        })

    out = root / "RELABELED_35_PERCENT"
    out.mkdir(exist_ok=True)

    # trial CSV
    trial_csv = out / "trial_results_35.csv"

    with open(trial_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
            if rows else []
        )
        writer.writeheader()
        writer.writerows(rows)

    # target CSV
    target_csv = out / "target_results_35.csv"

    with open(target_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(target_rows[0].keys())
            if target_rows else []
        )
        writer.writeheader()
        writer.writerows(target_rows)

    summary = {
        "method": "FlowBot3D",
        "source_protocol_version": protocol,
        "source_result_root": str(root),

        "execution_target_ratio": 0.40,
        "relabel_success_threshold": THRESHOLD,

        "completed_trials": n,

        "grasp_success_trials": grasp_n,
        "grasp_success_rate": grasp_rate,

        "old_40_final_success_trials": old_final_n,
        "old_40_post_grasp_operation_success_rate":
            old_post_rate,
        "old_40_final_success_rate": old_final_rate,

        "new_35_final_success_trials": new_final_n,
        "new_35_post_grasp_operation_success_rate":
            new_post_rate,
        "new_35_final_success_rate": new_final_rate,

        "newly_promoted_trials": promoted_n,

        "progress_ge_35_ignoring_grasp": progress_only_n,
        "progress_ge_35_but_grasp_failed":
            false_physical_success_n,

        "missing_final_progress_trials":
            len(missing_progress),

        "success_definition":
            "grasp_success AND final_progress >= 0.35",
    }

    with open(out / "summary_35.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("saved:")
    print(" ", out / "summary_35.json")
    print(" ", trial_csv)
    print(" ", target_csv)

    return summary


all_summaries = {}

for name, cfg in EXPERIMENTS.items():
    all_summaries[name] = analyze(name, cfg)

print()
print("=" * 78)
print("FINAL COMPARISON @ 35%")
print("=" * 78)

for name, s in all_summaries.items():
    n = s["completed_trials"]
    g = s["grasp_success_trials"]
    f = s["new_35_final_success_trials"]

    print()
    print(name)
    print(
        f"1. grasp_success_rate                 : "
        f"{g}/{n} = {100*g/n:.3f}%"
    )
    print(
        f"2. post_grasp_operation_success_rate  : "
        f"{f}/{g} = {100*f/g:.3f}%"
        if g else
        "2. post_grasp_operation_success_rate  : N/A"
    )
    print(
        f"3. final_success_rate                 : "
        f"{f}/{n} = {100*f/n:.3f}%"
    )
