#!/usr/bin/env python3
"""Load every ID through ArticuBot's unmodified environment before formal runs."""

from __future__ import annotations

import json
import time
import traceback

import pybullet as p

from run_official_idswap import (
    RESULT_ROOT,
    SOURCE_SOLUTION,
    SOURCE_TRIAL,
    TASK_NAME,
    build_task_manifest,
    generated_config,
    write_json_atomic,
)
from manipulation.utils import build_up_env


def main() -> None:
    manifest = build_task_manifest(20)
    first_task_for_id = {}
    for spec in manifest["tasks"]:
        first_task_for_id.setdefault(str(spec["object_id"]), spec["task_mode"])
    rows = []
    for index, (object_id, task_mode) in enumerate(first_task_for_id.items(), start=1):
        started = time.time()
        simulator = None
        try:
            config = generated_config(object_id)
            simulator, _ = build_up_env(
                str(config), SOURCE_SOLUTION, TASK_NAME, None, render=False, horizon=600
            )
            simulator.reset()
            info = simulator._get_info()
            joint = p.getJointInfo(
                simulator.urdf_ids["storagefurniture"],
                int(simulator.handle_joint),
                physicsClientId=simulator.id,
            )
            row = {
                "object_id": object_id,
                "first_task_mode": task_mode,
                "status": "pass",
                "handle_joint": int(simulator.handle_joint),
                "joint_name": joint[1].decode("utf-8"),
                "joint_type": int(joint[2]),
                "handle_points": int(len(simulator.all_handle_points)),
                "initial_angle": float(info["opened_joint_angle"]),
                "runtime_sec": time.time() - started,
            }
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            lowered = message.lower()
            if "argmin of an empty sequence" in lowered:
                category = "unsupported_handle_semantics"
            elif "parts_render" in lowered and "handle.obj" in lowered:
                category = "missing_official_handle_asset"
            else:
                category = "other_setup_failure"
            row = {
                "object_id": object_id,
                "first_task_mode": task_mode,
                "status": "fail",
                "category": category,
                "exception": message,
                "traceback": traceback.format_exc(),
                "runtime_sec": time.time() - started,
            }
        finally:
            if simulator is not None:
                try:
                    simulator.close()
                except Exception:
                    pass
        rows.append(row)
        print(
            f"{index}/{len(first_task_for_id)} object={object_id} "
            f"status={row['status']} {row.get('category', '')}",
            flush=True,
        )
    payload = {
        "schema": "articubot_official_idswap_preflight_v1",
        "source_config": str(SOURCE_TRIAL),
        "total_unique_ids": len(rows),
        "pass": sum(row["status"] == "pass" for row in rows),
        "fail": sum(row["status"] == "fail" for row in rows),
        "rows": rows,
    }
    write_json_atomic(RESULT_ROOT / "official_id_preflight.json", payload)
    print(json.dumps({key: payload[key] for key in ("total_unique_ids", "pass", "fail")}, indent=2))


if __name__ == "__main__":
    main()
