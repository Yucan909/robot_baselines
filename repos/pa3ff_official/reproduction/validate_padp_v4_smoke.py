#!/usr/bin/env python3
"""Structural-only smoke checks; no success-rate criterion is permitted."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


TASKS = ("door_open", "door_close", "drawer_open", "drawer_close")


def finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-root", type=Path, required=True)
    args = parser.parse_args()
    failures, rows = [], []
    for task in TASKS:
        paths = sorted(
            path for path in (args.smoke_root / task).rglob("result.json")
            if path.parent.name.startswith("trial_")
        )
        if len(paths) not in (1, 2):
            failures.append(f"{task}: result count {len(paths)} not in [1,2]")
        for path in paths:
            result = json.loads(path.read_text(encoding="utf-8"))
            local = []
            if result.get("episode_status") != "complete":
                local.append(f"episode_status={result.get('episode_status')}")
            if result.get("exception") is not None:
                local.append("exception is not null")
            for key in ("initial_progress", "final_progress", "directional_task_progress"):
                if not finite(result.get(key)):
                    local.append(f"{key} is not finite")
            if not isinstance(result.get("initial_policy_output_diagnostics"), dict):
                local.append("policy diagnostics missing")
            # A normal policy failure at the engagement gate intentionally
            # stops before operation.  Require operation telemetry only when
            # that gate passed; otherwise the structural smoke would silently
            # turn grasp success into a pass/fail criterion.
            if result.get("grasp_success") is True and not isinstance(
                result.get("operation_control"), list
            ):
                local.append("operation_control missing after engagement")
            if result.get("forbidden_formal_fields_used_by_policy") != []:
                local.append("forbidden policy input usage")
            if local:
                failures.append(f"{path}: {', '.join(local)}")
            rows.append({
                "task": task, "path": str(path), "checks": local,
                # Recorded only; never used as a smoke pass/fail condition.
                "grasp_success_diagnostic_only": result.get("grasp_success"),
                "progress_diagnostic_only": result.get("directional_task_progress"),
            })
    report = {
        "schema": "pa3ff_padp_v4_structural_smoke_v1",
        "status": "PASS" if not failures else "FAIL",
        "success_or_progress_threshold_used_for_pass": False,
        "rows": rows, "failures": failures,
    }
    output = args.smoke_root / "SMOKE_VALIDATION.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
