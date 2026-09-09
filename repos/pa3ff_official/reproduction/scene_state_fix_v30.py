"""Scene-construction repair for supplied articulated-object states.

The legacy PA3FF runner converted every non-finite joint limit to ``[0, 0]``
before clipping a supplied ``initial_object_qpos``.  PartNet continuous joints
therefore lost their recorded state.  This adapter calls the unchanged scene
builder and then restores only supplied finite values for unbounded joints.
It does not inspect the target action, trajectory, progress, or success.
"""
from __future__ import annotations

import numpy as np

import formal_worker as legacy

_LEGACY_CREATE_SCENE = legacy.create_scene


def create_scene(case: dict) -> dict:
    resources = _LEGACY_CREATE_SCENE(case)
    supplied = case.get("initial_object_qpos")
    if supplied is None:
        resources["scene_state_repair"] = {
            "version": "continuous_joint_preservation_v30",
            "applied": False,
            "reason": "ratio_initialized_scene",
        }
        return resources

    supplied = np.asarray(supplied, dtype=np.float64)
    obj = resources["obj"]
    joints = list(obj.get_active_joints())
    if supplied.shape != (len(joints),):
        raise RuntimeError(
            f"supplied initial_object_qpos {supplied.shape}, expected {(len(joints),)}"
        )
    qpos = np.asarray(obj.get_qpos(), dtype=np.float64)
    restored = []
    for index, joint in enumerate(joints):
        limit = np.asarray(joint.get_limits()[0], dtype=np.float64)
        if np.isfinite(limit).all():
            continue
        value = float(supplied[index])
        if not np.isfinite(value):
            raise RuntimeError(f"nonfinite supplied object qpos at joint {index}")
        qpos[index] = value
        joint.set_drive_target(value)
        joint.set_drive_velocity_target(0.0)
        restored.append({
            "joint_index": index,
            "joint_name": joint.get_name(),
            "child_link": joint.get_child_link().get_name(),
            "supplied_value": value,
            "raw_limits": limit.tolist(),
        })
    if restored:
        obj.set_qpos(qpos)
    resources["scene_state_repair"] = {
        "version": "continuous_joint_preservation_v30",
        "applied": bool(restored),
        "restored": restored,
        "uses_policy_or_action_fields": False,
        "uses_progress_or_success": False,
    }
    return resources
