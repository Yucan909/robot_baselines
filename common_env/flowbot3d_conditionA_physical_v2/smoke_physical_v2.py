
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


HOME = Path.home()

CODE_DIR = Path(__file__).resolve().parent

RUN_TRIAL = CODE_DIR / "run_trial.py"
PANDA_CONTROLLER = CODE_DIR / "panda_controller.py"
GRASP_ADAPTER = CODE_DIR / "grasp_pose_adapter.py"
CONTACT_MONITOR = CODE_DIR / "contact_monitor.py"

POSE_CATALOG = (
    HOME
    / "robot_baselines"
    / "configs"
    / "flowbot3d"
    / "eval"
    / "eval_pose_catalog.jsonl"
)

OLD_ROOT = (
    HOME
    / "robot_baselines"
    / "results"
    / "flowbot3d"
    / "FINAL_TWOFINGER_PHYSICAL_V1"
)

OUT_ROOT = (
    HOME
    / "robot_baselines"
    / "results"
    / "flowbot3d"
    / "conditionA_physical_v2_smoke"
)

PROTOCOL_VERSION = (
    "flowbot3d_conditionA_physical_v2"
)


def sha256(path):

    h = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:

        while True:

            x = f.read(
                1024 * 1024
            )

            if not x:
                break

            h.update(
                x
            )

    return h.hexdigest()


def eval_keys():

    rows = []

    with open(
        POSE_CATALOG
    ) as f:

        for line in f:

            if not line.strip():
                continue

            d = json.loads(
                line
            )

            link = (
                d.get(
                    "link_name"
                )
                or d.get(
                    "target_link"
                )
                or d.get(
                    "link"
                )
            )

            rows.append(
                (
                    str(
                        d[
                            "shape_id"
                        ]
                    ),
                    str(
                        link
                    ),
                )
            )

    return rows


def candidates():

    valid = set(
        eval_keys()
    )

    final_success = []
    grasp_success = []

    if OLD_ROOT.is_dir():

        for p in sorted(
            OLD_ROOT.glob(
                "*_seed_*/result.json"
            )
        ):

            try:

                d = json.loads(
                    p.read_text()
                )

            except Exception:
                continue

            sid = str(
                d.get(
                    "shape_id"
                )
            )

            link = str(
                d.get(
                    "target_link"
                )
            )

            if (
                sid,
                link,
            ) not in valid:
                continue

            if not bool(
                d.get(
                    "grasp_success",
                    False,
                )
            ):
                continue

            try:

                seed = int(
                    d[
                        "trial_seed"
                    ]
                )

            except Exception:
                continue

            row = {
                "shape_id":
                    sid,

                "target_link":
                    link,

                "trial_seed":
                    seed,

                "old_success":
                    bool(
                        d.get(
                            "success",
                            False,
                        )
                    ),
            }

            if row[
                "old_success"
            ]:

                final_success.append(
                    row
                )

            else:

                grasp_success.append(
                    row
                )

    # Old final successes first, then all old real-grasp successes.
    return (
        final_success
        + grasp_success
    )


def main():

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

    picks = candidates()

    if not picks:

        raise RuntimeError(
            "No previous physical-grasp-success candidates found"
        )

    print(
        "=" * 110
    )

    print(
        "FLOWBOT CONDITION-A PHYSICAL V2 STRONG SMOKE"
    )

    print(
        "=" * 110
    )

    print(
        "old real-grasp-success candidates:",
        len(
            picks
        ),
    )

    attempts = []

    # Try enough candidates to exercise the physical operation branch.
    for index, item in enumerate(
        picks[:20],
        1,
    ):

        sid = item[
            "shape_id"
        ]

        link = item[
            "target_link"
        ]

        seed = int(
            item[
                "trial_seed"
            ]
        )

        trial_dir = (
            OUT_ROOT
            / f"{sid}_{link}_seed_{seed}"
        )

        result_file = (
            trial_dir
            / "result.json"
        )

        log_file = (
            OUT_ROOT
            / (
                f"{index:02d}_"
                f"{sid}_{link}_seed_{seed}.log"
            )
        )

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
                seed
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
            "CUDA_VISIBLE_DEVICES"
        ] = "0"

        env[
            "FLOWBOT_DEVICE"
        ] = "cuda:0"

        print()
        print(
            f"[{index:02d}] "
            f"{sid}/{link} "
            f"seed={seed} "
            f"old_final_success={item['old_success']}"
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

        attempt = {
            **item,

            "returncode":
                int(
                    proc.returncode
                ),

            "log":
                str(
                    log_file
                ),

            "result":
                str(
                    result_file
                ),
        }

        # Engineering errors are never hidden by moving to another object.
        if (
            proc.returncode != 0
            or not result_file.exists()
        ):

            attempt[
                "implementation_ok"
            ] = False

            attempts.append(
                attempt
            )

            print()
            print(
                "IMPLEMENTATION ERROR"
            )

            print(
                log_file
            )

            break

        d = json.loads(
            result_file.read_text()
        )

        if (
            d.get(
                "protocol_version"
            )
            != PROTOCOL_VERSION
        ):

            raise RuntimeError(
                "protocol mismatch"
            )

        for field in [
            "grasp_success",
            "operation_attempted_after_grasp",
            "operation_success_given_grasp",
            "success",
            "failure_reason",
        ]:

            if field not in d:

                raise RuntimeError(
                    f"result missing field: {field}"
                )

        if (
            bool(
                d[
                    "success"
                ]
            )
            and not bool(
                d[
                    "grasp_success"
                ]
            )
        ):

            raise RuntimeError(
                "final success without physical grasp success"
            )

        attempt[
            "implementation_ok"
        ] = True

        attempt[
            "grasp_success"
        ] = bool(
            d[
                "grasp_success"
            ]
        )

        attempt[
            "operation_attempted_after_grasp"
        ] = bool(
            d[
                "operation_attempted_after_grasp"
            ]
        )

        attempt[
            "history_steps"
        ] = len(
            d.get(
                "history",
                [],
            )
        )

        attempt[
            "operation_success_given_grasp"
        ] = bool(
            d[
                "operation_success_given_grasp"
            ]
        )

        attempt[
            "success"
        ] = bool(
            d[
                "success"
            ]
        )

        attempt[
            "failure_reason"
        ] = d.get(
            "failure_reason"
        )

        attempt[
            "final_progress"
        ] = d.get(
            "final_progress"
        )

        attempt[
            "grasp_diagnostics"
        ] = d.get(
            "grasp_diagnostics"
        )

        attempts.append(
            attempt
        )

        print(
            "  implementation:",
            True
        )

        print(
            "  physical grasp:",
            attempt[
                "grasp_success"
            ]
        )

        print(
            "  operation:",
            attempt[
                "operation_attempted_after_grasp"
            ]
        )

        print(
            "  history steps:",
            attempt[
                "history_steps"
            ]
        )

        print(
            "  final success:",
            attempt[
                "success"
            ]
        )

        print(
            "  reason:",
            attempt[
                "failure_reason"
            ]
        )

        exercised = bool(
            attempt[
                "grasp_success"
            ]
            and attempt[
                "operation_attempted_after_grasp"
            ]
            and attempt[
                "history_steps"
            ] > 0
        )

        if exercised:

            manifest = {
                "status":
                    "PASS",

                "protocol_version":
                    PROTOCOL_VERSION,

                "criterion":
                    (
                        "real bilateral physical grasp succeeded "
                        "and FlowBot physical operation branch executed; "
                        "task success itself not required"
                    ),

                "physical_operation_branch_exercised":
                    True,

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

                "passing_trial":
                    attempt,

                "attempts":
                    attempts,
            }

            pass_file.write_text(
                json.dumps(
                    manifest,
                    indent=2,
                )
            )

            print()
            print(
                "=" * 110
            )

            print(
                "PHYSICAL ENGINEERING SMOKE: PASS"
            )

            print(
                "=" * 110
            )

            print(
                pass_file
            )

            return 0

    failure_file = (
        OUT_ROOT
        / "smoke_attempts.json"
    )

    failure_file.write_text(
        json.dumps(
            attempts,
            indent=2,
        )
    )

    print()
    print(
        "=" * 110
    )

    print(
        "PHYSICAL ENGINEERING SMOKE: NOT FROZEN"
    )

    print(
        "=" * 110
    )

    print(
        "No candidate exercised the real physical operation branch."
    )

    print(
        "Do NOT launch 1120."
    )

    print(
        failure_file
    )

    if attempts:

        print(
            attempts[-1][
                "log"
            ]
        )

    return 2


if __name__ == "__main__":

    raise SystemExit(
        main()
    )

