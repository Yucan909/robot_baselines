#!/usr/bin/env python3

import csv
import json
from pathlib import Path
from collections import Counter

import numpy as np


CSV = Path(
    "/home/feng/robot_baselines/results/where2act/"
    "FINAL_ADJUDICATED_1120/final_trials_1120.csv"
)

rows = list(
    csv.DictReader(
        CSV.open()
    )
)


def as_bool(v):
    return str(v).strip().lower() in {
        "true", "1", "yes"
    }


def valid_path(v):
    s = str(v or "").strip()

    if (
        not s
        or s.lower() in {
            "none",
            "null",
        }
    ):
        return None

    p = Path(s)

    if p.is_file():
        return p

    return None


def effective_result(row):

    # start-sync repair 的 4 条优先读修复结果
    rp = valid_path(
        row.get(
            "repair_result"
        )
    )

    if rp is None:
        rp = valid_path(
            row.get(
                "original_result"
            )
        )

    if rp is None:
        return None, None

    try:
        d = json.loads(
            rp.read_text()
        )
    except Exception:
        return rp, None

    return rp, d


def describe(name, vals):

    vals = np.asarray(
        [
            float(x)
            for x in vals
            if x is not None
            and np.isfinite(float(x))
        ],
        dtype=np.float64,
    )

    if len(vals) == 0:
        print(
            f"{name:<34s}: N=0"
        )
        return

    print(
        f"{name:<34s}: "
        f"N={len(vals):4d} "
        f"mean={vals.mean():.4f} "
        f"median={np.median(vals):.4f} "
        f"p10={np.quantile(vals, 0.10):.4f} "
        f"p90={np.quantile(vals, 0.90):.4f}"
    )


effective = []

for row in rows:

    p, d = effective_result(
        row
    )

    effective.append(
        (
            row,
            p,
            d,
        )
    )


print(
    "=" * 110
)

print(
    "WHERE2ACT BEFORE-V3 AUDIT"
)

print(
    "=" * 110
)

print(
    "adjudicated rows:",
    len(rows)
)


# =====================================================================
# 1. Pipeline funnel
# =====================================================================

print()
print(
    "=" * 110
)
print(
    "1. PIPELINE FUNNEL"
)
print(
    "=" * 110
)

stage_names = [
    "observation_success",
    "policy_success",
    "planning_success",
    "pregrasp_execution_success",
]

for stage in stage_names:

    n = sum(
        int(
            bool(
                d.get(
                    stage,
                    False
                )
            )
        )
        for _, _, d
        in effective
        if d is not None
    )

    print(
        f"{stage:<34s}: "
        f"{n:4d}/1120 "
        f"({100*n/1120:.2f}%)"
    )


# 真正执行过 close + contact monitor
grasp_attempts = []

for row, p, d in effective:

    if d is None:
        continue

    if isinstance(
        d.get(
            "grasp_diagnostics"
        ),
        dict
    ):
        grasp_attempts.append(
            (
                row,
                p,
                d,
            )
        )


grasp_success_rows = [
    x
    for x in effective
    if as_bool(
        x[0].get(
            "grasp_success"
        )
    )
]


print()
print(
    f"{'physical grasp attempts':<34s}: "
    f"{len(grasp_attempts):4d}/1120 "
    f"({100*len(grasp_attempts)/1120:.2f}%)"
)

print(
    f"{'firm grasp successes':<34s}: "
    f"{len(grasp_success_rows):4d}/1120 "
    f"({100*len(grasp_success_rows)/1120:.2f}%)"
)

if grasp_attempts:

    print(
        f"{'grasp success | attempted':<34s}: "
        f"{len(grasp_success_rows)}/{len(grasp_attempts)} "
        f"({100*len(grasp_success_rows)/len(grasp_attempts):.2f}%)"
    )


# =====================================================================
# 2. Main formal failure distribution
# =====================================================================

print()
print(
    "=" * 110
)
print(
    "2. ADJUDICATED FAILURE REASONS"
)
print(
    "=" * 110
)

failure_counter = Counter()

for row in rows:

    if as_bool(
        row.get(
            "final_success"
        )
    ):
        continue

    failure_counter[
        str(
            row.get(
                "failure_reason"
            )
        )
    ] += 1


for k, v in (
    failure_counter
    .most_common()
):

    print(
        f"{v:4d}  {k}"
    )


# =====================================================================
# 3. Scores: grasp success vs grasp failure after actual attempt
# =====================================================================

print()
print(
    "=" * 110
)
print(
    "3. MODEL SCORE SEPARATION AT REAL GRASP ATTEMPTS"
)
print(
    "=" * 110
)

succ_action = []
fail_action = []

succ_critic = []
fail_critic = []

succ_pos_err = []
fail_pos_err = []

succ_rot_err = []
fail_rot_err = []

succ_bilateral = []
fail_bilateral = []

succ_left = []
fail_left = []

succ_right = []
fail_right = []


for row, p, d in grasp_attempts:

    success = as_bool(
        row.get(
            "grasp_success"
        )
    )

    action = d.get(
        "interaction_score"
    )

    critic = d.get(
        "critic_score"
    )

    pos_err = d.get(
        "contact_position_error"
    )

    rot_err = d.get(
        "contact_rotation_error"
    )

    gd = d.get(
        "grasp_diagnostics",
        {}
    )

    bilateral = gd.get(
        "bilateral_fraction"
    )

    left = gd.get(
        "left_contact_fraction"
    )

    right = gd.get(
        "right_contact_fraction"
    )

    if success:

        succ_action.append(
            action
        )
        succ_critic.append(
            critic
        )
        succ_pos_err.append(
            pos_err
        )
        succ_rot_err.append(
            rot_err
        )
        succ_bilateral.append(
            bilateral
        )
        succ_left.append(
            left
        )
        succ_right.append(
            right
        )

    else:

        fail_action.append(
            action
        )
        fail_critic.append(
            critic
        )
        fail_pos_err.append(
            pos_err
        )
        fail_rot_err.append(
            rot_err
        )
        fail_bilateral.append(
            bilateral
        )
        fail_left.append(
            left
        )
        fail_right.append(
            right
        )


describe(
    "actionability | grasp success",
    succ_action
)

describe(
    "actionability | grasp fail",
    fail_action
)

print()

describe(
    "critic | grasp success",
    succ_critic
)

describe(
    "critic | grasp fail",
    fail_critic
)

print()

describe(
    "contact pos err | success",
    succ_pos_err
)

describe(
    "contact pos err | fail",
    fail_pos_err
)

describe(
    "contact rot err | success",
    succ_rot_err
)

describe(
    "contact rot err | fail",
    fail_rot_err
)

print()

describe(
    "bilateral fraction | success",
    succ_bilateral
)

describe(
    "bilateral fraction | fail",
    fail_bilateral
)

describe(
    "left contact | fail",
    fail_left
)

describe(
    "right contact | fail",
    fail_right
)


# =====================================================================
# 4. Senior requested opening statistics
# =====================================================================

print()
print(
    "=" * 110
)
print(
    "4. OPENING STATISTICS AFTER FIRM GRASP"
)
print(
    "=" * 110
)

ops = []

for row, p, d in grasp_success_rows:

    if d is None:
        continue

    pre = d.get(
        "pre_pull_progress"
    )

    final = d.get(
        "final_progress"
    )

    if (
        pre is None
        or final is None
    ):
        continue

    pre = float(pre)
    final = float(final)

    if (
        not np.isfinite(pre)
        or not np.isfinite(final)
    ):
        continue

    ops.append(
        {
            "pre":
                pre,

            "final":
                final,

            "delta":
                final - pre,

            "grasp_lost":
                bool(
                    (
                        d.get(
                            "post_pull_contact"
                        )
                        or {}
                    ).get(
                        "grasp_lost",
                        False
                    )
                ),

            "shape":
                row[
                    "shape_id"
                ],

            "link":
                row[
                    "target_link"
                ],

            "seed":
                row[
                    "trial_seed"
                ],
        }
    )


print(
    "valid post-grasp pulls:",
    len(ops)
)

if ops:

    finals = np.asarray(
        [
            x["final"]
            for x in ops
        ],
        dtype=np.float64,
    )

    deltas = np.asarray(
        [
            x["delta"]
            for x in ops
        ],
        dtype=np.float64,
    )

    print()
    print(
        f"final progress mean   : {finals.mean():.4f}"
    )

    print(
        f"final progress median : {np.median(finals):.4f}"
    )

    print(
        f"final progress max    : {finals.max():.4f}"
    )

    print()
    print(
        f"delta mean            : {deltas.mean():+.4f}"
    )

    print(
        f"delta median          : {np.median(deltas):+.4f}"
    )

    print(
        f"delta max             : {deltas.max():+.4f}"
    )

    print()

    for threshold in [
        0.20,
        0.25,
        0.30,
        0.40,
    ]:

        n = int(
            (
                finals
                >= threshold
            ).sum()
        )

        print(
            f"final >= {threshold:.2f}: "
            f"{n}/{len(ops)} "
            f"post-grasp="
            f"{100*n/len(ops):.2f}% | "
            f"end-to-end="
            f"{100*n/1120:.2f}%"
        )

    print()

    for threshold in [
        0.00,
        0.01,
        0.02,
        0.05,
        0.10,
    ]:

        n = int(
            (
                deltas
                > threshold
            ).sum()
        )

        print(
            f"delta > {threshold:+.2f}: "
            f"{n}/{len(ops)} "
            f"({100*n/len(ops):.2f}%)"
        )

    lost = sum(
        int(
            x[
                "grasp_lost"
            ]
        )
        for x in ops
    )

    print()
    print(
        f"grasp lost during pull: "
        f"{lost}/{len(ops)} "
        f"({100*lost/len(ops):.2f}%)"
    )


# =====================================================================
# 5. Write machine-readable report
# =====================================================================

OUT = Path(
    "/home/feng/robot_baselines/results/where2act/"
    "where2act_before_v3_audit.json"
)

report = {
    "total":
        len(rows),

    "grasp_attempts":
        len(
            grasp_attempts
        ),

    "grasp_success":
        len(
            grasp_success_rows
        ),

    "post_grasp_valid":
        len(
            ops
        ),

    "failure_reasons":
        dict(
            failure_counter
        ),
}

OUT.write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
    )
    + "\n"
)

print()
print(
    "=" * 110
)
print(
    "saved:",
    OUT
)
print(
    "=" * 110
)
