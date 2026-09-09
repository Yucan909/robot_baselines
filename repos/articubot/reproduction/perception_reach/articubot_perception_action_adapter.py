"""Official ArticuBot 10-D action to the senior joint-PD executor."""

from __future__ import annotations

import numpy as np

from articubot_action_adapter import rotation_6d_to_matrix, set_parallel_jaw_target


def execute_articubot_joint_pd_action(
    controller,
    action,
    *,
    num_steps: int,
    finger_target_override: float | None = None,
) -> dict:
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
    control = controller.move_operation_pose_to(target, int(num_steps))
    return {
        "action": action.astype(np.float32),
        "target_finger": target_finger,
        "position_delta_norm": float(np.linalg.norm(action[:3])),
        "control": control,
    }
