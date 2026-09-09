#!/usr/bin/env python3

import csv
import json
from collections import Counter
from pathlib import Path


HOME = Path.home()

ORIGINAL_ROOT = (
    HOME
    / "robot_baselines/results/where2act"
    / "FINAL_BACKEND_V2_TRAINED_1120"
)

REPAIR_ROOT = (
    HOME
    / "robot_baselines/results/where2act"
    / "BACKEND_V2_1_STARTSYNC_REPAIR4"
)

OUT_ROOT = (
    HOME
    / "robot_baselines/results/where2act"
    / "FINAL_ADJUDICATED_1120"
)

EXPECTED = 1120

COLLISION_MARKER = "OMPL路径包含碰撞state"


def load_json(path):
    return json.loads(
        path.read_text()
    )


def key_of(d):
    return (
        str(d["shape_id"]),
        str(d["target_link"]),
        int(d["trial_seed"]),
    )


def planner_collision_failure(result_path):
    stderr = (
        result_path.parent
        / "planner"
        / "planner_stderr.txt"
    )

    if not stderr.is_file():
        return False

    txt = stderr.read_text(
        errors="replace"
    )

    return (
        COLLISION_MARKER
        in txt
    )


# ============================================================
# Load original 1120
# ============================================================

original_files = sorted(
    ORIGINAL_ROOT.glob(
        "*_link_*/seed_*/result.json"
    )
)

if len(original_files) != EXPECTED:
    raise RuntimeError(
        f"Expected 1120 original results, "
        f"got {len(original_files)}"
    )

original = {}

for p in original_files:
    d = load_json(p)
    k = key_of(d)

    if k in original:
        raise RuntimeError(
            f"duplicate original key: {k}"
        )

    original[k] = (p, d)


# ============================================================
# Load 4 start-sync repair trials
# ============================================================

repair_files = sorted(
    REPAIR_ROOT.glob(
        "*_link_*/seed_*/result.json"
    )
)

if len(repair_files) != 4:
    raise RuntimeError(
        f"Expected 4 repair results, "
        f"got {len(repair_files)}"
    )

repairs = {}

for p in repair_files:
    d = load_json(p)
    repairs[key_of(d)] = (p, d)


# ============================================================
# Adjudication
# ============================================================

rows = []
unresolved = []

adjudication_counter = Counter()
failure_counter = Counter()


def formal_failure(
    *,
    key,
    original_path,
    original_data,
    reason,
    adjudication,
    source,
):
    sid, link, seed = key

    return {
        "shape_id": sid,
        "target_link": link,
        "trial_seed": seed,

        "grasp_success": False,
        "operation_success_given_grasp": False,
        "final_success": False,

        "failure_reason": reason,

        "adjudication": adjudication,
        "result_source": source,

        "raw_implementation_error":
            bool(
                original_data.get(
                    "implementation_error",
                    False
                )
            ),

        "raw_implementation_exception":
            original_data.get(
                "implementation_exception"
            ),

        "original_result":
            str(original_path),

        "repair_result":
            None,
    }


for key, (
    original_path,
    d
) in sorted(original.items()):

    impl = bool(
        d.get(
            "implementation_error",
            False
        )
    )

    exc = str(
        d.get(
            "implementation_exception",
            ""
        )
    )

    # --------------------------------------------------------
    # A. Already-valid formal result
    # --------------------------------------------------------

    if not impl:

        row = {
            "shape_id":
                key[0],

            "target_link":
                key[1],

            "trial_seed":
                key[2],

            "grasp_success":
                bool(
                    d.get(
                        "grasp_success",
                        False
                    )
                ),

            "operation_success_given_grasp":
                bool(
                    d.get(
                        "operation_success_given_grasp",
                        False
                    )
                ),

            "final_success":
                bool(
                    d.get(
                        "final_success",
                        False
                    )
                ),

            "failure_reason":
                d.get(
                    "failure_reason"
                ),

            "adjudication":
                "original_valid",

            "result_source":
                "backend_v2_original",

            "raw_implementation_error":
                False,

            "raw_implementation_exception":
                None,

            "original_result":
                str(original_path),

            "repair_result":
                None,
        }

        rows.append(row)
        adjudication_counter[
            "original_valid"
        ] += 1

        if not row[
            "final_success"
        ]:
            failure_counter[
                str(
                    row[
                        "failure_reason"
                    ]
                )
            ] += 1

        continue

    # --------------------------------------------------------
    # B. target not visible
    # --------------------------------------------------------

    if (
        exc
        == (
            "Where2ActObservationError: "
            "target_not_visible"
        )
    ):

        row = formal_failure(
            key=key,
            original_path=original_path,
            original_data=d,
            reason="target_not_visible",
            adjudication=(
                "observation_failure"
            ),
            source="backend_v2_original",
        )

        rows.append(row)

        adjudication_counter[
            "target_not_visible"
        ] += 1

        failure_counter[
            "target_not_visible"
        ] += 1

        continue

    # --------------------------------------------------------
    # C. Robot failed to track the planned pregrasp trajectory
    # --------------------------------------------------------

    if (
        "SAPIEN_WAYPOINT_TRACKING_FAILED"
        in exc
    ):

        row = formal_failure(
            key=key,
            original_path=original_path,
            original_data=d,
            reason=(
                "pregrasp_execution_failed"
            ),
            adjudication=(
                "execution_failure"
            ),
            source="backend_v2_original",
        )

        rows.append(row)

        adjudication_counter[
            "waypoint_tracking_failure"
        ] += 1

        failure_counter[
            "pregrasp_execution_failed"
        ] += 1

        continue

    # --------------------------------------------------------
    # D. Planner produced trajectory containing collision state
    # --------------------------------------------------------

    if (
        "PLANNER_IMPLEMENTATION_ERROR"
        in exc
        and planner_collision_failure(
            original_path
        )
    ):

        row = formal_failure(
            key=key,
            original_path=original_path,
            original_data=d,
            reason="planning_failed",
            adjudication=(
                "collision_path_rejected"
            ),
            source="backend_v2_original",
        )

        rows.append(row)

        adjudication_counter[
            "collision_path_planning_failure"
        ] += 1

        failure_counter[
            "planning_failed"
        ] += 1

        continue

    # --------------------------------------------------------
    # E. Known v2 start-state synchronization bug
    #    Use exactly one v2.1 repaired trial.
    # --------------------------------------------------------

    if (
        "PLANNER_SAPIEN_START_MISMATCH"
        in exc
    ):

        if key not in repairs:
            unresolved.append(
                (
                    key,
                    "missing repair result"
                )
            )
            continue

        repair_path, r = (
            repairs[key]
        )

        r_impl = bool(
            r.get(
                "implementation_error",
                False
            )
        )

        r_exc = str(
            r.get(
                "implementation_exception",
                ""
            )
        )

        # --------------------------------------------
        # Repair produced an ordinary valid trial
        # --------------------------------------------

        if not r_impl:

            row = {
                "shape_id":
                    key[0],

                "target_link":
                    key[1],

                "trial_seed":
                    key[2],

                "grasp_success":
                    bool(
                        r.get(
                            "grasp_success",
                            False
                        )
                    ),

                "operation_success_given_grasp":
                    bool(
                        r.get(
                            "operation_success_given_grasp",
                            False
                        )
                    ),

                "final_success":
                    bool(
                        r.get(
                            "final_success",
                            False
                        )
                    ),

                "failure_reason":
                    r.get(
                        "failure_reason"
                    ),

                "adjudication":
                    "start_sync_repaired_valid",

                "result_source":
                    "backend_v2_1_startsync",

                "raw_implementation_error":
                    True,

                "raw_implementation_exception":
                    exc,

                "original_result":
                    str(
                        original_path
                    ),

                "repair_result":
                    str(
                        repair_path
                    ),
            }

            rows.append(row)

            adjudication_counter[
                "start_sync_repaired_valid"
            ] += 1

            if not row[
                "final_success"
            ]:
                failure_counter[
                    str(
                        row[
                            "failure_reason"
                        ]
                    )
                ] += 1

            continue

        # --------------------------------------------
        # Start sync fixed, execution subsequently fails
        # --------------------------------------------

        if (
            "SAPIEN_WAYPOINT_TRACKING_FAILED"
            in r_exc
        ):

            row = formal_failure(
                key=key,
                original_path=original_path,
                original_data=d,
                reason=(
                    "pregrasp_execution_failed"
                ),
                adjudication=(
                    "start_sync_repaired_"
                    "execution_failure"
                ),
                source=(
                    "backend_v2_1_startsync"
                ),
            )

            row["repair_result"] = str(
                repair_path
            )

            rows.append(row)

            adjudication_counter[
                "start_sync_repaired_"
                "tracking_failure"
            ] += 1

            failure_counter[
                "pregrasp_execution_failed"
            ] += 1

            continue

        # --------------------------------------------
        # Start sync fixed, collision-free planning fails
        # --------------------------------------------

        if (
            "PLANNER_IMPLEMENTATION_ERROR"
            in r_exc
            and planner_collision_failure(
                repair_path
            )
        ):

            row = formal_failure(
                key=key,
                original_path=original_path,
                original_data=d,
                reason="planning_failed",
                adjudication=(
                    "start_sync_repaired_"
                    "collision_path_rejected"
                ),
                source=(
                    "backend_v2_1_startsync"
                ),
            )

            row["repair_result"] = str(
                repair_path
            )

            rows.append(row)

            adjudication_counter[
                "start_sync_repaired_"
                "planning_failure"
            ] += 1

            failure_counter[
                "planning_failed"
            ] += 1

            continue

        unresolved.append(
            (
                key,
                f"repair unresolved: {r_exc}"
            )
        )

        continue

    # --------------------------------------------------------
    # Unknown implementation error
    # --------------------------------------------------------

    unresolved.append(
        (
            key,
            exc
        )
    )


# ============================================================
# Validation
# ============================================================

if unresolved:
    print()
    print("UNRESOLVED:")

    for x in unresolved:
        print(x)

    raise RuntimeError(
        f"{len(unresolved)} unresolved trials"
    )


if len(rows) != EXPECTED:
    raise RuntimeError(
        f"Expected 1120 adjudicated rows, "
        f"got {len(rows)}"
    )


keys = [
    (
        x["shape_id"],
        x["target_link"],
        x["trial_seed"],
    )
    for x in rows
]

if len(set(keys)) != EXPECTED:
    raise RuntimeError(
        "duplicate adjudicated trials"
    )


# Logical consistency
for row in rows:

    grasp = bool(
        row["grasp_success"]
    )

    op = bool(
        row[
            "operation_success_given_grasp"
        ]
    )

    final = bool(
        row["final_success"]
    )

    if final and not grasp:
        raise RuntimeError(
            f"final without grasp: {row}"
        )

    if final and not op:
        raise RuntimeError(
            f"final without operation: {row}"
        )


# ============================================================
# Senior's three metrics
# ============================================================

G = sum(
    int(
        x["grasp_success"]
    )
    for x in rows
)

F = sum(
    int(
        x["final_success"]
    )
    for x in rows
)

OP = sum(
    int(
        x[
            "operation_success_given_grasp"
        ]
    )
    for x in rows
)

if OP != F:
    raise RuntimeError(
        f"operation count {OP} "
        f"!= final count {F}"
    )


grasp_rate = G / EXPECTED

post_grasp_rate = (
    F / G
    if G > 0
    else None
)

final_rate = F / EXPECTED


# ============================================================
# Save
# ============================================================

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

csv_path = (
    OUT_ROOT
    / "final_trials_1120.csv"
)

fields = [
    "shape_id",
    "target_link",
    "trial_seed",
    "grasp_success",
    "operation_success_given_grasp",
    "final_success",
    "failure_reason",
    "adjudication",
    "result_source",
    "raw_implementation_error",
    "raw_implementation_exception",
    "original_result",
    "repair_result",
]

with open(
    csv_path,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fields
    )

    writer.writeheader()

    writer.writerows(
        sorted(
            rows,
            key=lambda x: (
                x["shape_id"],
                x["target_link"],
                x["trial_seed"],
            )
        )
    )


summary = {
    "method":
        "Where2Act",

    "total_trials":
        EXPECTED,

    "grasp_success_trials":
        G,

    "grasp_success_rate":
        grasp_rate,

    "post_grasp_operation_success_trials":
        F,

    "post_grasp_operation_success_denominator":
        G,

    "post_grasp_operation_success_rate":
        post_grasp_rate,

    "final_success_trials":
        F,

    "final_success_rate":
        final_rate,

    "adjudication_breakdown":
        dict(
            adjudication_counter
        ),

    "failure_reason_breakdown":
        dict(
            failure_counter
        ),

    "original_result_root":
        str(
            ORIGINAL_ROOT
        ),

    "start_sync_repair_root":
        str(
            REPAIR_ROOT
        ),
}

summary_path = (
    OUT_ROOT
    / "final_summary.json"
)

summary_path.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False
    )
    + "\n"
)


# ============================================================
# Print final answer
# ============================================================

print()
print("=" * 100)
print("WHERE2ACT FINAL 1120 RESULTS")
print("=" * 100)

print(
    f"Total trials: {EXPECTED}"
)

print()

print(
    "1. Grasp Success Rate"
)

print(
    f"   {G}/{EXPECTED}"
    f" = {100.0 * grasp_rate:.2f}%"
)

print()

print(
    "2. Post-Grasp Operation Success Rate"
)

if G > 0:

    print(
        f"   {F}/{G}"
        f" = {100.0 * post_grasp_rate:.2f}%"
    )

else:

    print(
        "   N/A"
    )

print()

print(
    "3. Final Success Rate"
)

print(
    f"   {F}/{EXPECTED}"
    f" = {100.0 * final_rate:.2f}%"
)

print()
print("=" * 100)
print("ADJUDICATION BREAKDOWN")
print("=" * 100)

for k, v in (
    adjudication_counter
    .most_common()
):
    print(
        f"{v:4d}  {k}"
    )

print()
print("=" * 100)
print("FAILURE REASON BREAKDOWN")
print("=" * 100)

for k, v in (
    failure_counter
    .most_common()
):
    print(
        f"{v:4d}  {k}"
    )

print()
print("=" * 100)
print("SAVED")
print("=" * 100)
print(summary_path)
print(csv_path)
print("=" * 100)
