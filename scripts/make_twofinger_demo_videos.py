
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


HOME = Path.home()

BASE = (
    HOME
    / "robot_baselines"
)

FINAL_ROOT = (
    BASE
    / "results/flowbot3d/"
      "FINAL_TWOFINGER_PHYSICAL_V1"
)

CSV_FILE = (
    FINAL_ROOT
    / "trial_results.csv"
)

RUN_TRIAL = (
    BASE
    / "common_env/"
      "flowbot3d_twofinger_physical_v1/"
      "run_trial_video.py"
)

POSE_CATALOG = (
    BASE
    / "configs/flowbot3d/eval/"
      "eval_pose_catalog.jsonl"
)

FLOWBOT_ROOT = (
    BASE
    / "repos/flowbot3d"
)

DEMO_ROOT = (
    BASE
    / "results/flowbot3d/"
      "DEMO_TWOFINGER_PHYSICAL_V1"
)

VIDEO_ROOT = (
    DEMO_ROOT
    / "videos"
)

RAW_ROOT = (
    DEMO_ROOT
    / "raw"
)


def as_bool(x):
    return str(x).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


def as_float(x, default=float("nan")):
    try:
        return float(x)
    except Exception:
        return default


if not CSV_FILE.exists():
    raise RuntimeError(
        "找不到正式 trial_results.csv。"
        "请等 1120 benchmark 完成后再运行。"
    )


with open(
    CSV_FILE,
    newline="",
) as f:

    rows = list(
        csv.DictReader(f)
    )


print(
    "正式 trials:",
    len(rows),
)

if len(rows) != 1120:
    raise RuntimeError(
        f"正式结果不是1120条，当前={len(rows)}"
    )


# ============================================================
# 候选池
# ============================================================

success_rows = [
    r
    for r in rows
    if as_bool(r["success"])
]

postgrasp_fail_rows = [
    r
    for r in rows
    if (
        as_bool(
            r["grasp_success"]
        )
        and not as_bool(
            r["success"]
        )
    )
]

grasp_fail_rows = [
    r
    for r in rows
    if (
        not as_bool(
            r["grasp_success"]
        )
        and not as_bool(
            r["success"]
        )
    )
]


# 成功案例：
# 优先 final_progress 更高的，
# 视频通常更直观。
success_rows.sort(
    key=lambda r: as_float(
        r.get(
            "final_progress"
        )
    ),
    reverse=True,
)


# 抓住以后失败：
# 优先 grasp_lost，
# 其次 max_steps。
reason_priority = {
    "grasp_lost": 0,
    "max_steps": 1,
    "object_disturbed_before_pull": 2,
}

postgrasp_fail_rows.sort(
    key=lambda r: (
        reason_priority.get(
            r.get(
                "failure_reason",
                "",
            ),
            99,
        ),
        -as_float(
            r.get(
                "final_progress"
            ),
            0.0,
        ),
    )
)


# 抓取失败：
# 优先 approach error 小的案例，
# 这样更能说明是“到了但没夹住”，
# 而不是机械臂根本没过去。
grasp_fail_rows.sort(
    key=lambda r: as_float(
        r.get(
            "approach_position_error_m"
        ),
        999.0,
    )
)


DEMO_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

VIDEO_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

RAW_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


env = dict(
    os.environ
)

env[
    "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
] = "1"

env[
    "CUDA_VISIBLE_DEVICES"
] = "0"

env[
    "FLOWBOT_DEVICE"
] = "cuda:0"


def trial_dir(
    root,
    row,
):
    return (
        root
        / (
            f"{row['shape_id']}_"
            f"{row['link_name']}_"
            f"seed_{int(row['trial_seed'])}"
        )
    )


def replay(
    label,
    candidates,
    predicate,
    max_tries=10,
):

    if not candidates:
        print(
            f"[SKIP] {label}: 没有候选"
        )
        return None

    label_root = (
        RAW_ROOT
        / label
    )

    label_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for attempt, row in enumerate(
        candidates[:max_tries],
        start=1,
    ):

        sid = row[
            "shape_id"
        ]

        link = row[
            "link_name"
        ]

        seed = int(
            row[
                "trial_seed"
            ]
        )

        initial_ratio = float(
            row[
                "requested_initial_progress"
            ]
        )

        out_dir = trial_dir(
            label_root,
            row,
        )

        if out_dir.exists():
            shutil.rmtree(
                out_dir
            )

        print()
        print(
            "=" * 90
        )

        print(
            f"{label} "
            f"attempt={attempt}"
        )

        print(
            f"{sid}/{link} "
            f"seed={seed}"
        )

        print(
            "formal:",
            "grasp=",
            row[
                "grasp_success"
            ],
            "success=",
            row[
                "success"
            ],
            "reason=",
            row[
                "failure_reason"
            ],
        )

        cmd = [
            sys.executable,
            str(
                RUN_TRIAL
            ),

            "--shape-id",
            sid,

            "--target-link",
            link,

            "--pose-catalog",
            str(
                POSE_CATALOG
            ),

            "--trial-seed",
            str(
                seed
            ),

            "--initial-ratio",
            repr(
                initial_ratio
            ),

            "--output-root",
            str(
                label_root
            ),

            # 注意：
            # 故意没有 --no-video
        ]

        subprocess.run(
            cmd,
            cwd=str(
                FLOWBOT_ROOT
            ),
            env=env,
            check=True,
        )

        result_path = (
            out_dir
            / "result.json"
        )

        video_path = (
            out_dir
            / "trial.mp4"
        )

        if not result_path.exists():
            print(
                "重放没有 result.json，跳过"
            )
            continue

        with open(
            result_path
        ) as f:

            result = json.load(
                f
            )

        if not predicate(
            result
        ):
            print(
                "重放结果类别变化，尝试下一个候选：",
                "grasp=",
                result.get(
                    "grasp_success"
                ),
                "success=",
                result.get(
                    "success"
                ),
                "reason=",
                result.get(
                    "failure_reason"
                ),
            )
            continue

        if (
            not video_path.exists()
            or video_path.stat().st_size
            < 10_000
        ):
            print(
                "视频不存在或过小，尝试下一个候选"
            )
            continue

        final_name = (
            f"{label}__"
            f"{sid}__"
            f"{link}__"
            f"seed_{seed}.mp4"
        )

        final_video = (
            VIDEO_ROOT
            / final_name
        )

        shutil.copy2(
            video_path,
            final_video,
        )

        shutil.copy2(
            result_path,
            VIDEO_ROOT
            / final_name.replace(
                ".mp4",
                ".json",
            ),
        )

        print()
        print(
            "[VIDEO OK]",
            final_video,
        )

        print(
            "replay result:",
            "grasp=",
            result.get(
                "grasp_success"
            ),
            "success=",
            result.get(
                "success"
            ),
            "reason=",
            result.get(
                "failure_reason"
            ),
            "final_progress=",
            result.get(
                "final_progress"
            ),
        )

        return final_video

    print(
        f"[FAILED] {label}: "
        f"前{min(max_tries, len(candidates))}个候选"
        "都没有稳定复现"
    )

    return None


# ============================================================
# 1. 最终成功案例
# ============================================================

replay(
    "SUCCESS",
    success_rows,
    lambda r: bool(
        r.get(
            "success",
            False,
        )
    ),
)


# ============================================================
# 2. 抓取成功、但最终操作失败
#
# 这个最适合展示“真实二指物理交互失败”：
# 能抓住，但是后面滑脱或没有打开到40%。
# ============================================================

replay(
    "POSTGRASP_FAIL",
    postgrasp_fail_rows,
    lambda r: (
        bool(
            r.get(
                "grasp_success",
                False,
            )
        )
        and not bool(
            r.get(
                "success",
                False,
            )
        )
    ),
)


# ============================================================
# 3. 额外生成一个纯抓取失败
#
# 如果你只想给学长成功/失败各一个，
# 前两个视频就够了。
# ============================================================

replay(
    "GRASP_FAIL",
    grasp_fail_rows,
    lambda r: (
        not bool(
            r.get(
                "grasp_success",
                False,
            )
        )
        and not bool(
            r.get(
                "success",
                False,
            )
        )
    ),
)


print()
print(
    "=" * 90
)
print(
    "录像完成"
)
print(
    "=" * 90
)
print(
    VIDEO_ROOT
)
