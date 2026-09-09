#!/usr/bin/env python3

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# 验证冻结代码完全没被改过
from benchmark_parallel_trained import verify_freeze

verify_freeze()

HOME = Path.home()

ROOT = (
    HOME
    / "robot_baselines/results/where2act"
    / "FINAL_BACKEND_V2_TRAINED_1120"
)

ERROR_FILE = (
    ROOT
    / "IMPLEMENTATION_ERRORS.json"
)

ARCHIVE = (
    ROOT
    / "implementation_error_archive_pre_rerun"
)

LOG_ROOT = ROOT / "logs"

RUN_TRIAL = (
    HOME
    / "robot_baselines/common_env"
    / "where2act_backend_v2"
    / "run_trial.py"
)

POSE = (
    HOME
    / "robot_baselines/configs/flowbot3d/eval"
    / "eval_pose_catalog.jsonl"
)

CKPT = (
    HOME
    / "robot_baselines/repos/where2act/logs"
    / "where2act_full_adapt/best-network.pth"
)

errors = json.loads(
    ERROR_FILE.read_text()
)

target_not_visible = []
true_impl = []


def trial_dir(x):
    return (
        ROOT
        / f"{x['shape_id']}_{x['link_name']}"
        / f"seed_{int(x['trial_seed'])}"
    )


for x in errors:

    td = trial_dir(x)
    rp = td / "result.json"

    if not rp.is_file():
        true_impl.append(x)
        continue

    try:
        d = json.loads(
            rp.read_text()
        )
    except Exception:
        true_impl.append(x)
        continue

    exc = str(
        d.get(
            "implementation_exception",
            ""
        )
    )

    if (
        exc
        == "Where2ActObservationError: target_not_visible"
    ):
        target_not_visible.append(x)
    else:
        true_impl.append(x)


print("=" * 100)
print("FORMAL ERROR PARTITION")
print("=" * 100)
print(
    "target_not_visible -> formal failure:",
    len(target_not_visible)
)
print(
    "true implementation errors to rerun:",
    len(true_impl)
)

if len(target_not_visible) != 20:
    raise RuntimeError(
        f"expected 20 target_not_visible, "
        f"got {len(target_not_visible)}"
    )

if len(true_impl) != 11:
    raise RuntimeError(
        f"expected 11 true implementation errors, "
        f"got {len(true_impl)}"
    )

ARCHIVE.mkdir(
    parents=True,
    exist_ok=True
)

audit = []

for idx, x in enumerate(
    true_impl,
    1
):

    sid = str(
        x["shape_id"]
    )

    link = str(
        x["link_name"]
    )

    seed = int(
        x["trial_seed"]
    )

    td = trial_dir(x)

    result_file = (
        td
        / "result.json"
    )

    old_log = (
        LOG_ROOT
        / f"{sid}_{link}_seed_{seed}.log"
    )

    # --------------------------------------------------------
    # Archive the original implementation-error evidence
    # --------------------------------------------------------

    arc = (
        ARCHIVE
        / f"{sid}_{link}_seed_{seed}"
    )

    arc.mkdir(
        parents=True,
        exist_ok=True
    )

    if result_file.is_file():
        shutil.copy2(
            result_file,
            arc / "result_before_rerun.json"
        )

    if old_log.is_file():
        shutil.copy2(
            old_log,
            arc / "console_before_rerun.log"
        )

    # --------------------------------------------------------
    # Remove only this invalid trial
    # --------------------------------------------------------

    if td.exists():
        shutil.rmtree(td)

    cmd = [
        sys.executable,
        str(RUN_TRIAL),

        "--shape-id",
        sid,

        "--target-link",
        link,

        "--pose-catalog",
        str(POSE),

        "--trial-seed",
        str(seed),

        "--checkpoint",
        str(CKPT),

        "--output-root",
        str(ROOT),

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

    env = dict(
        os.environ
    )

    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    env[
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
    ] = "1"

    new_log = (
        LOG_ROOT
        / f"{sid}_{link}_seed_{seed}.log"
    )

    print()
    print("=" * 100)
    print(
        f"RERUN {idx}/11: "
        f"{sid}/{link} seed={seed}"
    )
    print("=" * 100)

    with open(
        new_log,
        "w"
    ) as f:

        proc = subprocess.run(
            cmd,
            cwd=str(
                RUN_TRIAL.parent
            ),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )

    status = "RESULT_MISSING"
    reason = None
    impl_exc = None
    grasp = None
    final = None

    if result_file.is_file():

        d = json.loads(
            result_file.read_text()
        )

        reason = d.get(
            "failure_reason"
        )

        impl_exc = d.get(
            "implementation_exception"
        )

        grasp = d.get(
            "grasp_success"
        )

        final = d.get(
            "final_success"
        )

        if d.get(
            "implementation_error",
            False
        ):
            status = (
                "IMPLEMENTATION_ERROR"
            )
        else:
            status = (
                "VALID_FORMAL_RESULT"
            )

    item = {
        "shape_id":
            sid,

        "link_name":
            link,

        "trial_seed":
            seed,

        "returncode":
            int(
                proc.returncode
            ),

        "status":
            status,

        "failure_reason":
            reason,

        "implementation_exception":
            impl_exc,

        "grasp_success":
            grasp,

        "final_success":
            final,
    }

    audit.append(
        item
    )

    print(
        "status:",
        status
    )

    print(
        "failure_reason:",
        reason
    )

    print(
        "implementation_exception:",
        impl_exc
    )

    print(
        "grasp_success:",
        grasp
    )

    print(
        "final_success:",
        final
    )


out = (
    ROOT
    / "RERUN_TRUE_IMPL_ERRORS_ONCE.json"
)

out.write_text(
    json.dumps(
        audit,
        indent=2
    )
    + "\n"
)

remain = [
    x
    for x in audit
    if x["status"]
    != "VALID_FORMAL_RESULT"
]

print()
print("=" * 100)
print("11-TRIAL RERUN COMPLETE")
print("=" * 100)
print(
    "resolved:",
    len(audit) - len(remain)
)
print(
    "remaining implementation errors:",
    len(remain)
)

if remain:

    print()
    for x in remain:
        print(
            x["shape_id"],
            x["link_name"],
            x["trial_seed"],
            "->",
            x[
                "implementation_exception"
            ]
        )

print()
print(
    "audit:",
    out
)
print(
    "archive:",
    ARCHIVE
)
print("=" * 100)
