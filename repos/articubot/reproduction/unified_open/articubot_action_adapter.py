"""Decode official 10D ArticuBot EEF actions into the frozen PHYSICAL_V2 controller."""

from __future__ import annotations

import numpy as np


def rotation_6d_to_matrix(orient) -> np.ndarray:
    orient = np.asarray(orient, dtype=np.float64).reshape(2, 3)
    a1, a2 = orient
    n1 = float(np.linalg.norm(a1))
    if n1 < 1e-8:
        raise RuntimeError("degenerate 6D rotation axis 1")
    b1 = a1 / n1
    b2 = a2 - np.dot(a2, b1) * b1
    n2 = float(np.linalg.norm(b2))
    if n2 < 1e-8:
        raise RuntimeError("degenerate 6D rotation axis 2")
    b2 /= n2
    return np.asarray([b1, b2, np.cross(b1, b2)], dtype=np.float64).T


def set_parallel_jaw_target(controller, target: float) -> float:
    limits = np.asarray(controller.finger_joints[0].get_limits()[0], dtype=np.float64)
    target = float(np.clip(target, limits[0], limits[1]))
    for joint in controller.finger_joints:
        joint.set_drive_velocity_target(0.0)
        joint.set_drive_target(target)
    return target


def execute_articubot_action(
    controller,
    action,
    *,
    num_steps: int = 250,
    finger_target_override: float | None = None,
) -> dict:
    """Official semantics: world delta XYZ, current_R @ delta_R, finger delta."""
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if action.shape != (10,) or not np.all(np.isfinite(action)):
        raise RuntimeError(f"invalid ArticuBot action: shape={action.shape}")
    current = controller.get_grasp_pose_matrix()
    target = current.copy()
    target[:3, 3] = current[:3, 3] + action[:3]
    target[:3, :3] = current[:3, :3] @ rotation_6d_to_matrix(action[3:9])
    current_finger = float(np.asarray(controller.get_finger_qpos()).mean())
    target_finger = set_parallel_jaw_target(
        controller,
        current_finger + float(action[9])
        if finger_target_override is None
        else float(finger_target_override),
    )
    control = controller.move_grasp_pose_to(target, int(num_steps))
    return {
        "action": action.astype(np.float32),
        "target_finger": target_finger,
        "position_delta_norm": float(np.linalg.norm(action[:3])),
        "control": control,
    }
