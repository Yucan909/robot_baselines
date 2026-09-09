
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

from collections import Counter, defaultdict
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import Path


# ============================================================
# 基础配置
# ============================================================

HOME = Path.home()

SOURCE_CODE = (
    HOME
    / "robot_baselines/common_env"
    / "flowbot3d_clean_v1"
)

WORK_CODE = (
    HOME
    / "robot_baselines/common_env"
    / "flowbot3d_1cm_v1"
)

POSE_CATALOG = (
    HOME
    / "robot_baselines/configs/flowbot3d/eval"
    / "eval_pose_catalog.jsonl"
)

RESULT_ROOT = (
    HOME
    / "robot_baselines/results/flowbot3d"
    / "FINAL_1CM_V1"
)

LOG_ROOT = (
    RESULT_ROOT
    / "logs"
)

VIDEO_RESULT_ROOT = (
    HOME
    / "robot_baselines/results/flowbot3d"
    / "DEMO4_1CM_SMOOTH"
)

VIDEO_RAW_ROOT = (
    VIDEO_RESULT_ROOT
    / "raw"
)

VIDEO_ROOT = (
    VIDEO_RESULT_ROOT
    / "videos"
)

VIDEO_LOG_ROOT = (
    VIDEO_RESULT_ROOT
    / "logs"
)


# ============================================================
# 最终1cm协议
# ============================================================

CONTACT_TOLERANCE_M = 0.010

# 保持严格版：
FINAL_CONTACT_STANDOFF_M = 0.0

# 保持严格版低层控制：
CONTROLLER_TOLERANCE_M = 0.002

SUCCESS_PROGRESS = 0.40

TRIAL_SEEDS = [
    2026082900 + i
    for i in range(20)
]


# ============================================================
# 当前机器默认并行配置
#
# Ryzen 9 9955HX3D
# 16C / 32T
#
# RTX 5080 Laptop
# 单张CUDA GPU
#
# 默认6个trial并行共享cuda:0。
# ============================================================

DEFAULT_WORKERS = 6


parser = argparse.ArgumentParser()

parser.add_argument(
    "--workers",
    type=int,
    default=DEFAULT_WORKERS,
)

parser.add_argument(
    "--fresh",
    action="store_true",
    help="删除已有1cm结果，从0重新开始",
)

parser.add_argument(
    "--skip-videos",
    action="store_true",
)

ARGS = parser.parse_args()

N_WORKERS = int(
    ARGS.workers
)


# ============================================================
# 工具函数
# ============================================================

def sha256(path):

    h = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:

        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):

            h.update(
                chunk
            )

    return h.hexdigest()


def patch_number(
    text,
    variable,
    value,
):

    pattern = (
        rf"{re.escape(variable)}"
        rf"\s*=\s*[0-9.]+"
    )

    replacement = (
        f"{variable} = {value}"
    )

    new_text, count = (
        re.subn(
            pattern,
            replacement,
            text,
            count=1,
        )
    )

    if count != 1:

        raise RuntimeError(
            f"没有唯一找到参数：{variable}"
        )

    return new_text


def parse_bool(x):

    return (
        str(x)
        .strip()
        .lower()
        in {
            "true",
            "1",
            "yes",
        }
    )


def to_float(
    x,
    default=None,
):

    try:

        if x in (
            None,
            "",
            "None",
        ):

            return default

        return float(
            x
        )

    except Exception:

        return default


# ============================================================
# 1. 检查源文件
# ============================================================

required_source = [
    "run_trial.py",
    "panda_executor.py",
    "run_trial_smooth_video.py",
    "panda_executor_smooth_video.py",
]


for name in required_source:

    path = (
        SOURCE_CODE
        / name
    )

    if not path.exists():

        raise FileNotFoundError(
            f"找不到：{path}"
        )


if not POSE_CATALOG.exists():

    raise FileNotFoundError(
        POSE_CATALOG
    )


# ============================================================
# 2. --fresh：
# 删除旧的1cm实验。
#
# 不碰2mm和5cm结果。
# ============================================================

if ARGS.fresh:

    print(
        "清除已有1cm结果..."
    )

    for path in [
        WORK_CODE,
        RESULT_ROOT,
        VIDEO_RESULT_ROOT,
    ]:

        if path.exists():

            shutil.rmtree(
                path
            )


# ============================================================
# 3. 创建完全独立的1cm代码目录
# ============================================================

WORK_CODE.mkdir(
    parents=True,
    exist_ok=True,
)


for name in required_source:

    shutil.copy2(
        SOURCE_CODE
        / name,

        WORK_CODE
        / name,
    )


formal_trial = (
    WORK_CODE
    / "run_trial.py"
)

formal_executor = (
    WORK_CODE
    / "panda_executor.py"
)

video_trial = (
    WORK_CODE
    / "run_trial_smooth_video.py"
)

video_executor = (
    WORK_CODE
    / "panda_executor_smooth_video.py"
)


# ============================================================
# 4. 只改变接触接受容差：
#
# 2mm → 1cm
#
# 不修改：
# - standoff = 0
# - controller tolerance = 2mm
# ============================================================

for path in [
    formal_trial,
    video_trial,
]:

    text = (
        path.read_text()
    )

    text = patch_number(
        text,
        "CONTACT_TOLERANCE",
        "0.010",
    )

    # 强制确认仍然没有standoff
    text = patch_number(
        text,
        "FINAL_CONTACT_STANDOFF",
        "0.0",
    )

    path.write_text(
        text
    )


# ============================================================
# 5. 验证低层控制器仍然是2mm
# ============================================================

for path in [
    formal_executor,
    video_executor,
]:

    text = (
        path.read_text()
    )

    match = re.search(
        r"control_tolerance\s*=\s*([0-9.]+)",
        text,
    )

    if match is None:

        raise RuntimeError(
            f"{path.name} 中找不到 "
            "control_tolerance"
        )

    value = float(
        match.group(1)
    )

    if abs(
        value
        - CONTROLLER_TOLERANCE_M
    ) > 1e-9:

        raise RuntimeError(
            f"{path.name} 当前低层控制容差"
            f"不是2mm，而是 {value}"
        )


# ============================================================
# 6. 写实验协议
# ============================================================

protocol = {
    "protocol_name":
        "FlowBot3D_1CM_V1",

    "contact_acceptance_tolerance_m":
        CONTACT_TOLERANCE_M,

    "controller_position_tolerance_m":
        CONTROLLER_TOLERANCE_M,

    "final_contact_standoff_m":
        FINAL_CONTACT_STANDOFF_M,

    "initial_progress_range":
        [0.10, 0.20],

    "success_progress":
        SUCCESS_PROGRESS,

    "trial_seeds":
        TRIAL_SEEDS,

    "number_of_targets":
        56,

    "trials_per_target":
        20,

    "expected_trials":
        1120,

    "parallel_workers":
        N_WORKERS,

    "cuda_device":
        "cuda:0",
}


with open(
    WORK_CODE
    / "protocol_1cm.json",
    "w",
) as f:

    json.dump(
        protocol,
        f,
        indent=2,
    )


# ============================================================
# 7. 代码hash
# ============================================================

code_hashes = {
    "run_trial.py":
        sha256(
            formal_trial
        ),

    "panda_executor.py":
        sha256(
            formal_executor
        ),
}


print(
    "=" * 110
)

print(
    "FLOWBOT3D 1CM PROTOCOL"
)

print(
    "=" * 110
)

print(
    "接触接受容差:",
    "1 cm",
)

print(
    "Panda低层控制容差:",
    "2 mm",
)

print(
    "最终standoff:",
    "0 mm",
)

print(
    "success threshold:",
    "40%",
)

print(
    "workers:",
    N_WORKERS,
)

print(
    "GPU:",
    "cuda:0",
)

print()

print(
    "正式代码hash:"
)

for k, v in (
    code_hashes.items()
):

    print(
        f"  {k}: {v}"
    )


# ============================================================
# 8. 读取56个验证目标
# ============================================================

targets = []


with open(
    POSE_CATALOG
) as f:

    for line in f:

        if not line.strip():
            continue

        row = json.loads(
            line
        )

        link = (
            row.get(
                "link_name"
            )
            or row.get(
                "target_link"
            )
            or row.get(
                "link"
            )
        )

        if link is None:

            raise RuntimeError(
                "pose catalog缺少link_name"
            )

        targets.append(
            {
                "shape_id":
                    str(
                        row[
                            "shape_id"
                        ]
                    ),

                "link_name":
                    link,

                "category":
                    row.get(
                        "category",
                        "UNKNOWN",
                    ),
            }
        )


unique_targets = {
    (
        x[
            "shape_id"
        ],
        x[
            "link_name"
        ],
    )
    for x in targets
}


if len(targets) != 56:

    raise RuntimeError(
        f"验证目标应为56，实际={len(targets)}"
    )


if len(
    unique_targets
) != 56:

    raise RuntimeError(
        "验证目标存在重复"
    )


# ============================================================
# 9. 正式结果目录
# ============================================================

RESULT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


with open(
    RESULT_ROOT
    / "benchmark_manifest.json",
    "w",
) as f:

    json.dump(
        {
            **protocol,

            "code_hashes":
                code_hashes,

            "pose_catalog":
                str(
                    POSE_CATALOG
                ),
        },
        f,
        indent=2,
    )


# ============================================================
# 10. 构建1120个任务
# ============================================================

tasks = []


for target in targets:

    for trial_index, seed in enumerate(
        TRIAL_SEEDS
    ):

        sid = target[
            "shape_id"
        ]

        link = target[
            "link_name"
        ]


        trial_dir = (
            RESULT_ROOT
            / (
                f"{sid}_"
                f"{link}_"
                f"seed_{seed}"
            )
        )

        result_file = (
            trial_dir
            / "result.json"
        )

        log_file = (
            LOG_ROOT
            / (
                f"{sid}_"
                f"{link}_"
                f"seed_{seed}.log"
            )
        )


        tasks.append(
            {
                **target,

                "trial_index":
                    trial_index,

                "trial_seed":
                    seed,

                "trial_dir":
                    trial_dir,

                "result_file":
                    result_file,

                "log_file":
                    log_file,
            }
        )


# ============================================================
# 11. 单个trial
# ============================================================

def run_one(
    task,
):

    result_file = (
        task[
            "result_file"
        ]
    )


    # 断点续跑
    if result_file.exists():

        try:

            with open(
                result_file
            ) as f:

                json.load(
                    f
                )

            return {
                **task,
                "status":
                    "SKIP",
            }

        except Exception:

            try:
                result_file.unlink()
            except Exception:
                pass


    sid = task[
        "shape_id"
    ]

    link = task[
        "link_name"
    ]

    seed = task[
        "trial_seed"
    ]


    cmd = [
        sys.executable,

        str(
            formal_trial
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

        "--output-root",
        str(
            RESULT_ROOT
        ),

        "--no-video",
    ]


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


    start = time.time()


    with open(
        task[
            "log_file"
        ],
        "w",
    ) as log:

        proc = subprocess.run(
            cmd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )


    elapsed = (
        time.time()
        - start
    )


    if (
        proc.returncode == 0
        and result_file.exists()
    ):

        try:

            with open(
                result_file
            ) as f:

                json.load(
                    f
                )

            return {
                **task,

                "status":
                    "OK",

                "elapsed_s":
                    elapsed,
            }

        except Exception:

            pass


    return {
        **task,

        "status":
            "IMPLEMENTATION_ERROR",

        "returncode":
            proc.returncode,

        "elapsed_s":
            elapsed,
    }


# ============================================================
# 12. 六路并行正式benchmark
# ============================================================

print()
print(
    "=" * 110
)

print(
    "START 1CM FINAL BENCHMARK"
)

print(
    "=" * 110
)

print(
    "targets: 56"
)

print(
    "trials/target: 20"
)

print(
    "total: 1120"
)

print(
    "workers:",
    N_WORKERS,
)

print(
    "result:",
    RESULT_ROOT,
)

print(
    "=" * 110
)


t0 = time.time()

completed = 0

implementation_errors = []


with ThreadPoolExecutor(
    max_workers=N_WORKERS
) as pool:

    futures = {
        pool.submit(
            run_one,
            task,
        ):
        task

        for task
        in tasks
    }


    for future in as_completed(
        futures
    ):

        result = (
            future.result()
        )

        completed += 1


        elapsed = (
            time.time()
            - t0
        )

        throughput = (
            completed
            / elapsed
            * 60.0
        )


        remaining = (
            1120
            - completed
        )

        eta_min = (
            remaining
            / throughput
            if throughput > 0
            else 0
        )


        print(
            f"[{completed:04d}/1120] "
            f"{result['status']:<20s} "
            f"{result['shape_id']}/"
            f"{result['link_name']} "
            f"trial="
            f"{result['trial_index']:02d} "
            f"| "
            f"{throughput:.1f} trial/min "
            f"| ETA≈{eta_min:.1f} min"
        )


        if (
            result[
                "status"
            ]
            == "IMPLEMENTATION_ERROR"
        ):

            implementation_errors.append(
                result
            )


# ============================================================
# 13. 如果出现真正工程异常，停止视频阶段
# ============================================================

if implementation_errors:

    error_file = (
        RESULT_ROOT
        / "implementation_errors.json"
    )

    clean = []

    for x in implementation_errors:

        clean.append(
            {
                "shape_id":
                    x[
                        "shape_id"
                    ],

                "link_name":
                    x[
                        "link_name"
                    ],

                "trial_index":
                    x[
                        "trial_index"
                    ],

                "trial_seed":
                    x[
                        "trial_seed"
                    ],

                "log":
                    str(
                        x[
                            "log_file"
                        ]
                    ),
            }
        )


    with open(
        error_file,
        "w",
    ) as f:

        json.dump(
            clean,
            f,
            indent=2,
        )


    print()
    print(
        "存在 implementation error：",
        len(
            implementation_errors
        ),
    )

    print(
        error_file
    )

    raise SystemExit(
        2
    )


# ============================================================
# 14. 扫描1120个正式结果
# ============================================================

records = []

missing = []


for task in tasks:

    result_file = (
        task[
            "result_file"
        ]
    )


    if not result_file.exists():

        missing.append(
            task
        )

        continue


    with open(
        result_file
    ) as f:

        result = json.load(
            f
        )


    success = bool(
        result.get(
            "success",
            False,
        )
    )


    failure_reason = (
        result.get(
            "failure_reason"
        )
    )


    if success:

        failure_reason = None

    elif failure_reason is None:

        failure_reason = (
            "max_steps"
        )


    contact_distance = (
        result.get(
            "contact_distance_m"
        )
    )

    if contact_distance is None:

        contact_distance = (
            result.get(
                "contact_distance"
            )
        )


    records.append(
        {
            "shape_id":
                task[
                    "shape_id"
                ],

            "link_name":
                task[
                    "link_name"
                ],

            "category":
                task[
                    "category"
                ],

            "trial_index":
                task[
                    "trial_index"
                ],

            "trial_seed":
                task[
                    "trial_seed"
                ],

            "initial_progress":
                result.get(
                    "initial_progress"
                ),

            "pre_pull_progress":
                result.get(
                    "pre_pull_progress"
                ),

            "final_progress":
                result.get(
                    "final_progress"
                ),

            "contact_distance":
                contact_distance,

            "success":
                success,

            "failure_reason":
                failure_reason,
        }
    )


# ============================================================
# 15. trial_results.csv
# ============================================================

trial_csv = (
    RESULT_ROOT
    / "trial_results.csv"
)


fields = [
    "shape_id",
    "link_name",
    "category",
    "trial_index",
    "trial_seed",
    "initial_progress",
    "pre_pull_progress",
    "final_progress",
    "contact_distance",
    "success",
    "failure_reason",
]


with open(
    trial_csv,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fields,
    )

    writer.writeheader()

    writer.writerows(
        records
    )


# ============================================================
# 16. target_results.csv
# ============================================================

groups = defaultdict(
    list
)


for row in records:

    groups[
        (
            row[
                "shape_id"
            ],
            row[
                "link_name"
            ],
            row[
                "category"
            ],
        )
    ].append(
        row
    )


target_rows = []


for (
    sid,
    link,
    category,
), xs in sorted(
    groups.items()
):

    n_success = sum(
        int(
            x[
                "success"
            ]
        )
        for x in xs
    )


    target_rows.append(
        {
            "shape_id":
                sid,

            "link_name":
                link,

            "category":
                category,

            "completed_trials":
                len(
                    xs
                ),

            "success_trials":
                n_success,

            "success_rate":
                (
                    n_success
                    / len(xs)
                ),
        }
    )


target_csv = (
    RESULT_ROOT
    / "target_results.csv"
)


with open(
    target_csv,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "shape_id",
            "link_name",
            "category",
            "completed_trials",
            "success_trials",
            "success_rate",
        ],
    )

    writer.writeheader()

    writer.writerows(
        target_rows
    )


# ============================================================
# 17. summary.json
# ============================================================

success_count = sum(
    int(
        x[
            "success"
        ]
    )
    for x in records
)


failure_counter = Counter(
    x[
        "failure_reason"
    ]
    for x in records
    if not x[
        "success"
    ]
)


summary = {
    "method":
        "FlowBot3D",

    "protocol":
        "1CM_V1",

    "contact_tolerance_m":
        CONTACT_TOLERANCE_M,

    "controller_tolerance_m":
        CONTROLLER_TOLERANCE_M,

    "final_contact_standoff_m":
        FINAL_CONTACT_STANDOFF_M,

    "completed_trials":
        len(
            records
        ),

    "missing_trials":
        len(
            missing
        ),

    "success_trials":
        success_count,

    "success_rate":
        (
            success_count
            / len(records)
            if records
            else None
        ),

    "failure_reasons":
        dict(
            failure_counter
        ),

    "code_hashes":
        code_hashes,
}


with open(
    RESULT_ROOT
    / "summary.json",
    "w",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


print()
print(
    "=" * 110
)

print(
    "1CM FINAL BENCHMARK SUMMARY"
)

print(
    "=" * 110
)

print(
    "completed:",
    f"{len(records)}/1120"
)

print(
    "missing:",
    len(
        missing
    )
)

print(
    "success:",
    success_count
)

print(
    "success rate:",
    f"{100.0 * success_count / len(records):.2f}%"
)

print()

print(
    "failure reasons:"
)

for k, v in (
    failure_counter
    .most_common()
):

    print(
        f"  {k}: {v}"
    )

print()

print(
    RESULT_ROOT
    / "summary.json"
)

print(
    trial_csv
)

print(
    target_csv
)


if missing:

    print(
        "存在missing trial，停止视频生成。"
    )

    raise SystemExit(
        2
    )


# ============================================================
# 18. 视频阶段
# ============================================================

if ARGS.skip_videos:

    print(
        "已跳过视频生成。"
    )

    raise SystemExit(
        0
    )


VIDEO_RAW_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

VIDEO_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

VIDEO_LOG_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 19. 自动选择4个正式案例
#
# A：成功
# B：另一个不同target成功
# C：max_steps
#    若1cm版本没有max_steps，
#    自动改为最接近1cm阈值的失败。
#
# D：典型contact_unreachable
# ============================================================

success_rows = [
    x
    for x in records
    if x[
        "success"
    ]
]


selected_success = []

used_targets = set()


for row in success_rows:

    key = (
        row[
            "shape_id"
        ],
        row[
            "link_name"
        ],
    )

    if key in used_targets:

        continue

    selected_success.append(
        row
    )

    used_targets.add(
        key
    )

    if len(
        selected_success
    ) == 2:

        break


# 不同target不足2个时，
# 用不同seed补。
if len(
    selected_success
) < 2:

    for row in success_rows:

        if row in selected_success:

            continue

        selected_success.append(
            row
        )

        if len(
            selected_success
        ) == 2:

            break


if len(
    selected_success
) < 2:

    raise RuntimeError(
        "正式1cm结果中成功trial不足2个，"
        "无法生成2个成功视频。"
    )


max_steps_rows = [
    x
    for x in records
    if (
        not x[
            "success"
        ]
        and
        x[
            "failure_reason"
        ]
        == "max_steps"
    )
]


unreachable_rows = [
    x
    for x in records
    if (
        not x[
            "success"
        ]
        and
        x[
            "failure_reason"
        ]
        == "contact_unreachable"
        and
        x[
            "contact_distance"
        ]
        is not None
    )
]


if not unreachable_rows:

    raise RuntimeError(
        "没有contact_unreachable案例"
    )


unreachable_rows.sort(
    key=lambda x: float(
        x[
            "contact_distance"
        ]
    )
)


if max_steps_rows:

    third_case = max(
        max_steps_rows,

        key=lambda x: (
            to_float(
                x[
                    "final_progress"
                ],
                -1,
            )
        ),
    )

    third_label = (
        "03_failure_max_steps"
    )

else:

    # 没有max_steps：
    # 选择最接近1cm阈值但仍失败的正式案例
    third_case = (
        unreachable_rows[0]
    )

    third_label = (
        "03_failure_near_1cm_threshold"
    )


typical_unreachable = (
    unreachable_rows[
        len(
            unreachable_rows
        )
        // 2
    ]
)


cases = [
    {
        "label":
            "01_success_A",

        "row":
            selected_success[0],
    },

    {
        "label":
            "02_success_B",

        "row":
            selected_success[1],
    },

    {
        "label":
            third_label,

        "row":
            third_case,
    },

    {
        "label":
            "04_failure_contact_unreachable_typical",

        "row":
            typical_unreachable,
    },
]


print()
print(
    "=" * 110
)

print(
    "GENERATING 1CM SMOOTH DEMOS"
)

print(
    "=" * 110
)


for i, case in enumerate(
    cases,
    1,
):

    row = case[
        "row"
    ]

    print(
        f"{i}. "
        f"{case['label']}"
    )

    print(
        "   target:",
        f"{row['shape_id']}/"
        f"{row['link_name']}"
    )

    print(
        "   seed:",
        row[
            "trial_seed"
        ]
    )

    print(
        "   formal success:",
        row[
            "success"
        ]
    )

    print(
        "   reason:",
        row[
            "failure_reason"
        ]
    )

    print(
        "   final progress:",
        row[
            "final_progress"
        ]
    )

    print(
        "   contact distance:",
        row[
            "contact_distance"
        ]
    )

    print()


# ============================================================
# 20. 回放四个视频
# ============================================================

for i, case in enumerate(
    cases,
    1,
):

    label = case[
        "label"
    ]

    row = case[
        "row"
    ]

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


    case_root = (
        VIDEO_RAW_ROOT
        / label
    )


    if case_root.exists():

        shutil.rmtree(
            case_root
        )


    case_root.mkdir(
        parents=True,
        exist_ok=True,
    )


    log_file = (
        VIDEO_LOG_ROOT
        / f"{label}.log"
    )


    print(
        f"[VIDEO {i}/4] "
        f"{sid}/{link}, "
        f"seed={seed}"
    )


    cmd = [
        sys.executable,

        str(
            video_trial
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

        "--output-root",
        str(
            case_root
        ),

        # 故意不加 --no-video
    ]


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


    with open(
        log_file,
        "w",
    ) as log:

        proc = subprocess.run(
            cmd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )


    if proc.returncode != 0:

        print(
            "  -> 视频回放异常"
        )

        print(
            "     log:",
            log_file
        )

        continue


    videos = list(
        case_root.rglob(
            "trial.mp4"
        )
    )


    if not videos:

        print(
            "  -> 没找到trial.mp4"
        )

        print(
            "     log:",
            log_file
        )

        continue


    source = (
        videos[0]
    )


    destination = (
        VIDEO_ROOT
        / (
            f"{label}"
            f"__{sid}_{link}"
            f"__seed_{seed}"
            f"__1CM.mp4"
        )
    )


    shutil.copy2(
        source,
        destination,
    )


    print(
        "  ->",
        destination
    )


# ============================================================
# 21. 最终视频汇总
# ============================================================

videos = sorted(
    VIDEO_ROOT.glob(
        "*.mp4"
    )
)


print()
print(
    "=" * 110
)

print(
    "1CM VIDEO SUMMARY"
)

print(
    "=" * 110
)


for video in videos:

    mb = (
        video.stat().st_size
        / 1024
        / 1024
    )

    print(
        video.name,
        f"({mb:.2f} MB)"
    )


print()

print(
    "生成视频:",
    f"{len(videos)}/4"
)

print(
    "视频目录:"
)

print(
    VIDEO_ROOT
)

print(
    "=" * 110
)


# ============================================================
# 22. 最后检查正式代码没有被运行过程中修改
# ============================================================

current_hashes = {
    "run_trial.py":
        sha256(
            formal_trial
        ),

    "panda_executor.py":
        sha256(
            formal_executor
        ),
}


if (
    current_hashes
    != code_hashes
):

    raise RuntimeError(
        "正式benchmark运行过程中代码hash发生变化！"
    )


print()
print(
    "正式代码hash检查：PASS"
)

print(
    "1CM benchmark + videos 全部完成。"
)
