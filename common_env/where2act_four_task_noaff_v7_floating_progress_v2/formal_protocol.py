#!/usr/bin/env python3
"""Frozen four-task protocol constants and result validation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


TASK_SPECS: Dict[str, Dict[str, Any]] = {
    "door_open": {"primitive": "pull", "goal": "open", "targets": 56},
    "door_close": {"primitive": "push", "goal": "close", "targets": 56},
    "drawer_open": {"primitive": "pull", "goal": "open", "targets": 36},
    "drawer_close": {"primitive": "push", "goal": "close", "targets": 34},
}

TRIAL_SEEDS: List[int] = list(range(2026082900, 2026082920))
TRIALS_PER_TARGET = 20
OPEN_COMMAND_PROGRESS = 0.40
OPEN_SUCCESS_PROGRESS = 0.35
CLOSE_COMMAND_PROGRESS = 0.10
CLOSE_SUCCESS_PROGRESS = 0.15
FAILURE_CATEGORIES = (
    "model_no_valid_action",
    "planning_failed",
    "approach_failed",
    "no_target_contact",
    "grasp_failed",
    "lost_contact",
    "operation_motion_failed",
    "insufficient_directional_progress",
    "runtime_error",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: JSON row must be an object")
        rows.append(row)
    return rows


def result_path(root: Path, task: str, shape_id: str, target_link: str, seed: int) -> Path:
    return root / task / f"{shape_id}_{target_link}" / f"seed_{seed}" / "result.json"


def episode_log_path(
    root: Path, task: str, shape_id: str, target_link: str, seed: int
) -> Path:
    return root / task / f"{shape_id}_{target_link}" / f"seed_{seed}" / "log.txt"


def expected_trials(task: str) -> int:
    return int(TASK_SPECS[task]["targets"]) * TRIALS_PER_TARGET


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def command_target(goal: str) -> float:
    return OPEN_COMMAND_PROGRESS if goal == "open" else CLOSE_COMMAND_PROGRESS


def success_threshold(goal: str) -> float:
    return OPEN_SUCCESS_PROGRESS if goal == "open" else CLOSE_SUCCESS_PROGRESS


def final_state_success(goal: str, progress: float) -> bool:
    if goal == "open":
        return float(progress) >= OPEN_SUCCESS_PROGRESS
    return float(progress) <= CLOSE_SUCCESS_PROGRESS


def validate_result(
    result: Dict[str, Any],
    *,
    task: str,
    shape_id: str,
    target_link: str,
    seed: int,
    checkpoint: str,
    checkpoint_sha256: str,
) -> Tuple[bool, Optional[str]]:
    expected = {
        "task": task,
        "primitive": TASK_SPECS[task]["primitive"],
        "goal": TASK_SPECS[task]["goal"],
        "shape_id": str(shape_id),
        "target_link": str(target_link),
        "seed": int(seed),
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            return False, f"{key} mismatch: {result.get(key)!r} != {value!r}"
    if result.get("trial_seed") != int(seed):
        return False, "trial_seed mismatch"
    goal = TASK_SPECS[task]["goal"]
    if result.get("command_progress_target") != command_target(goal):
        return False, "command_progress_target mismatch"
    if result.get("success_progress_threshold") != success_threshold(goal):
        return False, "success_progress_threshold mismatch"
    # A scene may legitimately fail at observation construction before the
    # policy object is instantiated (for example, zero visible target pixels).
    # The frozen checkpoint path/SHA above still proves which trained model was
    # assigned.  Do not relabel that physical observation failure as a runtime
    # error, and do not rerun it until a more favorable observation occurs.
    policy_not_reached = bool(
        result.get("failure_reason") == "model_no_valid_action"
        and result.get("policy_success") is False
        and result.get("predicted_contact_point") is None
        and result.get("predicted_qidx") is None
    )
    if result.get("network_trained") is not True and not policy_not_reached:
        return False, "network_trained is false outside a pre-policy observation failure"
    if result.get("engineering_allow_untrained") is not False:
        return False, "engineering_allow_untrained is not false"
    if result.get("implementation_error") is not False:
        return False, "implementation_error is not false"
    for key in ("scene_source", "camera_source", "base_pose_source", "articulation_state_source"):
        if not isinstance(result.get(key), str) or not result[key]:
            return False, f"missing provenance field: {key}"
    for key in ("initial_articulation_q", "final_articulation_q"):
        value = result.get(key)
        if not isinstance(value, list) or not value or not all(_finite_number(x) for x in value):
            return False, f"invalid articulation field: {key}"
    for key in ("initial_progress", "final_progress", "directional_task_progress"):
        if not _finite_number(result.get(key)):
            return False, f"non-finite progress field: {key}"
    for key in (
        "grasp_success",
        "operation_executed",
        "operation_success",
        "final_operation_success",
        "final_success",
        "final_state_threshold_met",
        "command_target_reached",
    ):
        if not isinstance(result.get(key), bool):
            return False, f"non-boolean outcome: {key}"
    state_success = final_state_success(goal, float(result["final_progress"]))
    if result["final_state_threshold_met"] != state_success:
        return False, "final_state_threshold_met inconsistent with final progress"
    expected_command = (
        float(result["final_progress"]) >= OPEN_COMMAND_PROGRESS
        if goal == "open"
        else float(result["final_progress"]) <= CLOSE_COMMAND_PROGRESS
    )
    if result["command_target_reached"] != expected_command:
        return False, "command_target_reached inconsistent with final progress"
    expected_final = bool(result["operation_executed"] and state_success)
    if result["final_success"] != expected_final:
        return False, "final_success inconsistent with executed operation and final state"
    if result["operation_success"] != result["final_success"]:
        return False, "operation_success differs from final_success"
    if result["final_operation_success"] != result["final_success"]:
        return False, "final_operation_success differs from final_success"
    failure = result.get("failure_reason")
    if result["final_success"]:
        if failure is not None:
            return False, "successful result has a failure_reason"
    elif failure not in FAILURE_CATEGORIES:
        return False, f"invalid failure category: {failure!r}"
    for key in (
        "predicted_contact_point",
        "predicted_qidx",
        "predicted_action_orientation",
        "predicted_action_direction",
    ):
        if key not in result:
            return False, f"missing prediction field: {key}"
    return True, None


def read_valid_result(
    path: Path,
    *,
    task: str,
    shape_id: str,
    target_link: str,
    seed: int,
    checkpoint: str,
    checkpoint_sha256: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not path.exists():
        return None, "missing"
    try:
        result = read_json(path)
    except Exception as exc:  # malformed/incomplete file is rerunnable
        return None, f"unreadable: {type(exc).__name__}: {exc}"
    valid, reason = validate_result(
        result,
        task=task,
        shape_id=shape_id,
        target_link=target_link,
        seed=seed,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
    )
    return (result, None) if valid else (None, reason)


def check_unique_targets(task: str, rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = list(rows)
    expected = int(TASK_SPECS[task]["targets"])
    if len(rows) != expected:
        raise ValueError(f"{task}: expected {expected} targets, found {len(rows)}")
    seen = set()
    for row in rows:
        if row.get("task") != task:
            raise ValueError(f"{task}: task mismatch in scene catalog")
        if row.get("primitive") != TASK_SPECS[task]["primitive"]:
            raise ValueError(f"{task}: primitive mismatch in scene catalog")
        if row.get("goal") != TASK_SPECS[task]["goal"]:
            raise ValueError(f"{task}: goal mismatch in scene catalog")
        key = (str(row["shape_id"]), str(row["target_link"]))
        if key in seen:
            raise ValueError(f"{task}: duplicate target {key}")
        seen.add(key)
    return rows
