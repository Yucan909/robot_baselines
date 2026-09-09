import hashlib
import json
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


HOME = Path.home()

TWO_FINGER_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

RUN_TRIAL = (
    TWO_FINGER_DIR
    / "run_trial.py"
)

PANDA_CONTROLLER = (
    TWO_FINGER_DIR
    / "panda_controller.py"
)

GRASP_ADAPTER = (
    TWO_FINGER_DIR
    / "grasp_pose_adapter.py"
)

CONTACT_MONITOR = (
    TWO_FINGER_DIR
    / "contact_monitor.py"
)

TRAIN_SPLIT = (
    HOME
    / "robot_baselines/splits/train_split.json"
)

POSE_CATALOG = (
    HOME
    / "robot_baselines/configs/flowbot3d"
    / "train_pose_catalog.jsonl"
)

OUT_ROOT = (
    HOME
    / "robot_baselines/results/flowbot3d"
    / "twofinger_physical_v1_smoke10"
)

N_TARGETS = 10

SELECTION_SEED = 202608290600

# 已经参与此前 executor 调试的训练目标全部排除。
USED_TARGETS = {
    ("9277", "link_2"),
    ("7263", "link_0"),
    ("10627", "link_1"),
    ("9386", "link_1"),
    ("7273", "link_0"),

    ("102301", "link_0"),
    ("103521", "link_0"),
    ("12248", "link_1"),
    ("27267", "link_0"),
    ("7310", "link_0"),
    ("47742", "link_0"),
    ("102309", "link_0"),
    ("103361", "link_0"),
    ("10685", "link_1"),
    ("31249", "link_3"),
    ("7304", "link_1"),
    ("45594", "link_0"),
    ("102389", "link_16"),
    ("103778", "link_0"),
    ("10751", "link_2"),
    ("32566", "link_0"),
    ("45661", "link_0"),
    ("101605", "link_0"),
    ("103775", "link_0"),
    ("10627", "link_2"),
}


def sha256(
    path,
):
    h = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:
        while True:
            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(
                chunk
            )

    return h.hexdigest()


def select_targets():
    pose_targets = set()

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

            pose_targets.add(
                (
                    str(
                        row["shape_id"]
                    ),
                    link,
                )
            )

    with open(
        TRAIN_SPLIT
    ) as f:
        split = json.load(
            f
        )

    by_category = defaultdict(
        list
    )

    for shape_id in split[
        "ids"
    ]:
        obj = split[
            "objects"
        ][
            shape_id
        ]

        category = obj.get(
            "category",
            "UNKNOWN",
        )

        for link_info in obj[
            "selected_links"
        ]:
            link = link_info[
                "link_name"
            ]

            key = (
                str(
                    shape_id
                ),
                link,
            )

            if key in USED_TARGETS:
                continue

            if key not in pose_targets:
                continue

            by_category[
                category
            ].append(
                {
                    "shape_id":
                        str(
                            shape_id
                        ),

                    "link_name":
                        link,

                    "category":
                        category,
                }
            )

    rng = random.Random(
        SELECTION_SEED
    )

    for xs in (
        by_category.values()
    ):
        rng.shuffle(
            xs
        )

    categories = sorted(
        by_category.keys()
    )

    rng.shuffle(
        categories
    )

    selected = []
    round_idx = 0

    while (
        len(
            selected
        )
        < N_TARGETS
    ):
        added = False

        for category in categories:
            xs = by_category[
                category
            ]

            if (
                round_idx
                < len(
                    xs
                )
            ):
                selected.append(
                    xs[
                        round_idx
                    ]
                )
                added = True

                if (
                    len(
                        selected
                    )
                    >= N_TARGETS
                ):
                    break

        if not added:
            break

        round_idx += 1

    if (
        len(
            selected
        )
        != N_TARGETS
    ):
        raise RuntimeError(
            f"只能选择到 {len(selected)} 个新训练目标"
        )

    return selected


def main():
    if not RUN_TRIAL.exists():
        raise FileNotFoundError(
            RUN_TRIAL
        )

    for required in [
        PANDA_CONTROLLER,
        GRASP_ADAPTER,
        CONTACT_MONITOR,
    ]:
        if not required.exists():
            raise FileNotFoundError(required)

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    pass_file = (
        OUT_ROOT
        / "EXECUTOR_SMOKE_PASS.json"
    )

    if pass_file.exists():
        pass_file.unlink()

    targets = select_targets()

    with open(
        OUT_ROOT
        / "targets.json",
        "w",
    ) as f:
        json.dump(
            targets,
            f,
            indent=2,
        )

    print("=" * 100)
    print(
        "FlowBot3D PHYSICAL TWO-FINGER V1 - 10 个训练目标工程检查"
    )
    print("=" * 100)

    for i, target in enumerate(
        targets,
        1,
    ):
        print(
            f"{i:02d}. "
            f"{target['shape_id']}/"
            f"{target['link_name']} "
            f"{target['category']}"
        )

    print("=" * 100)

    summary = []

    for i, target in enumerate(
        targets
    ):
        sid = target[
            "shape_id"
        ]

        link = target[
            "link_name"
        ]

        # 一目标一固定 seed。
        trial_seed = (
            2026083100
            + i
        )

        trial_dir = (
            OUT_ROOT
            / (
                f"{sid}_{link}_"
                f"seed_{trial_seed}"
            )
        )

        result_file = (
            trial_dir
            / "result.json"
        )

        log_file = (
            OUT_ROOT
            / (
                f"{sid}_{link}_"
                f"seed_{trial_seed}.log"
            )
        )

        if trial_dir.exists():
            # 只删除该 trial 旧 result，
            # 不动整个结果根目录。
            if result_file.exists():
                result_file.unlink()

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
                trial_seed
            ),

            "--output-root",
            str(
                OUT_ROOT
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
            "FLOWBOT_DEVICE"
        ] = "cuda:0"

        print()
        print(
            f"[{i + 1:02d}/10] "
            f"{sid}/{link}"
        )

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

        item = {
            **target,

            "trial_seed":
                trial_seed,

            "returncode":
                proc.returncode,

            "log":
                str(
                    log_file
                ),
        }

        if (
            proc.returncode == 0
            and result_file.exists()
        ):
            with open(
                result_file
            ) as f:
                result = json.load(
                    f
                )

            item[
                "implementation_ok"
            ] = True

            item[
                "grasp_success"
            ] = bool(
                result.get(
                    "grasp_success",
                    False,
                )
            )

            item[
                "success"
            ] = bool(
                result.get(
                    "success",
                    False,
                )
            )

            item[
                "failure_reason"
            ] = result.get(
                "failure_reason"
            )

            item[
                "final_progress"
            ] = result.get(
                "final_progress"
            )

            item[
                "contact_distance_m"
            ] = result.get(
                "contact_distance_m"
            )

        else:
            item[
                "implementation_ok"
            ] = False

            item[
                "grasp_success"
            ] = False

            item[
                "success"
            ] = False

            item[
                "failure_reason"
            ] = (
                "implementation_error"
            )

        summary.append(
            item
        )

    with open(
        OUT_ROOT
        / "summary.json",
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 110)
    print(
        "PHYSICAL TWO-FINGER V1 SMOKE10 汇总"
    )
    print("=" * 110)

    for x in summary:
        print(
            f"{x['shape_id']}/"
            f"{x['link_name']:<8s} "
            f"{x['category']:<20s} "
            f"impl="
            f"{str(x['implementation_ok']):<5s} "
            f"grasp="
            f"{str(x.get('grasp_success', False)):<5s} "
            f"success="
            f"{str(x['success']):<5s} "
            f"reason="
            f"{x['failure_reason']}"
        )

    implementation_ok = sum(
        int(
            x[
                "implementation_ok"
            ]
        )
        for x in summary
    )

    grasp_success = sum(
        int(
            x.get(
                "grasp_success",
                False,
            )
        )
        for x in summary
    )

    task_success = sum(
        int(
            x[
                "success"
            ]
        )
        for x in summary
    )

    distribution = Counter()

    for x in summary:
        if x[
            "implementation_ok"
        ]:
            if x[
                "success"
            ]:
                distribution[
                    "success"
                ] += 1
            else:
                distribution[
                    x[
                        "failure_reason"
                    ]
                ] += 1
        else:
            distribution[
                "implementation_error"
            ] += 1

    print()
    print(
        "合法 result.json:",
        f"{implementation_ok}/10",
    )

    print(
        "implementation_error:",
        10 - implementation_ok,
    )

    print(
        "grasp success:",
        f"{grasp_success}/10",
    )

    print(
        "task success:",
        f"{task_success}/10",
    )

    print()
    print(
        "结果分布:"
    )

    for key, value in (
        distribution
        .most_common()
    ):
        print(
            f"  {key}: {value}"
        )

    if (
        implementation_ok
        == 10
    ):
        manifest = {
            "status":
                "PASS",

            "protocol_version":
                "flowbot3d_twofinger_physical_v1",

            "criterion":
                "10/10 trials produced valid result.json; task failures are allowed",

            "run_trial_sha256":
                sha256(
                    RUN_TRIAL
                ),

            "panda_controller_sha256":
                sha256(
                    PANDA_CONTROLLER
                ),

            "grasp_pose_adapter_sha256":
                sha256(
                    GRASP_ADAPTER
                ),

            "contact_monitor_sha256":
                sha256(
                    CONTACT_MONITOR
                ),

            "targets_file":
                str(
                    OUT_ROOT
                    / "targets.json"
                ),

            "summary_file":
                str(
                    OUT_ROOT
                    / "summary.json"
                ),
        }

        with open(
            pass_file,
            "w",
        ) as f:
            json.dump(
                manifest,
                f,
                indent=2,
            )

        print()
        print("=" * 110)
        print(
            "ENGINEERING CHECK: PASS"
        )
        print(
            "已写入代码 hash；后续 benchmark 若代码发生变化会拒绝启动。"
        )
        print(
            pass_file
        )
        print("=" * 110)

        return 0

    print()
    print("=" * 110)
    print(
        "ENGINEERING CHECK: FAIL"
    )
    print(
        "只检查 implementation_error；"
        "不要根据 task success 调 executor。"
    )
    print("=" * 110)

    return 2


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
