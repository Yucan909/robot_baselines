"""Robot-base model-frame bridge for ArticuBot in the frozen SAPIEN scenes.

The official PyBullet setup keeps the Panda root at the identity world pose, so
its recorded "world" point clouds/actions are numerically in the robot-base
frame.  Frozen unified scenes move and yaw the robot root.  Feeding SAPIEN world
coordinates directly therefore changes the checkpoint's coordinate convention.

This module preserves every scene/camera/robot state and applies only a rigid
interface transform:

* observations: SAPIEN world -> Panda root/model frame
* action translation: Panda root/model frame -> SAPIEN world
* action delta rotation: unchanged (official current_R @ delta_R convention)
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from articubot_action_adapter import execute_articubot_action as execute_world_action
from articubot_observation_adapter import capture_articubot_observation as capture_world_observation


MODEL_FRAME = "panda_root_equivalent_to_official_world"


def _root_transform(panda):
    matrix = np.asarray(
        panda.robot.get_root_pose().to_transformation_matrix(), dtype=np.float64
    )
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise RuntimeError("invalid Panda root transform")
    return matrix[:3, :3], matrix[:3, 3]


def _world_points_to_model(points, root_rotation, root_translation):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    # Row-vector form of p_B = R_WB^T (p_W - t_WB).
    return ((points - root_translation) @ root_rotation).astype(np.float32)


def capture_articubot_observation_robot_frame(
    *, scene, camera, obj, panda, num_points: int, rng: np.random.Generator
):
    world = capture_world_observation(
        scene=scene,
        camera=camera,
        obj=obj,
        panda=panda,
        num_points=num_points,
        rng=rng,
    )
    root_rotation, root_translation = _root_transform(panda)
    point_cloud = _world_points_to_model(
        world["point_cloud"], root_rotation, root_translation
    )
    gripper_pcd = _world_points_to_model(
        world["gripper_pcd"], root_rotation, root_translation
    )

    grasp_world = np.asarray(panda.get_grasp_pose_matrix(), dtype=np.float64)
    grasp_position = (grasp_world[:3, 3] - root_translation) @ root_rotation
    grasp_rotation = root_rotation.T @ grasp_world[:3, :3]
    orient_6d = np.asarray(
        [grasp_rotation[:, 0], grasp_rotation[:, 1]], dtype=np.float64
    ).reshape(6)
    finger = float(np.asarray(panda.get_finger_qpos(), dtype=np.float64).mean())
    agent_pos = np.concatenate([grasp_position, orient_6d, [finger]]).astype(np.float32)

    nearest = point_cloud[np.argmin(cdist(gripper_pcd, point_cloud), axis=1)]
    displacement = (nearest - gripper_pcd).astype(np.float32)
    if not all(
        np.all(np.isfinite(value))
        for value in (point_cloud, gripper_pcd, agent_pos, displacement)
    ):
        raise RuntimeError("robot-base observation contains NaN/Inf")
    return {
        "point_cloud": point_cloud,
        "agent_pos": agent_pos,
        "gripper_pcd": gripper_pcd,
        "displacement_gripper_to_object": displacement,
    }


def execute_articubot_action_robot_frame(
    controller,
    action,
    *,
    num_steps: int = 250,
    finger_target_override: float | None = None,
):
    model_action = np.asarray(action, dtype=np.float64).reshape(-1)
    if model_action.shape != (10,) or not np.all(np.isfinite(model_action)):
        raise RuntimeError(f"invalid ArticuBot model-frame action: {model_action.shape}")
    root_rotation, _ = _root_transform(controller)
    world_action = model_action.copy()
    world_action[:3] = root_rotation @ model_action[:3]
    result = execute_world_action(
        controller,
        world_action,
        num_steps=num_steps,
        finger_target_override=finger_target_override,
    )
    result["model_frame_action"] = model_action.astype(np.float32)
    result["world_translation_delta"] = world_action[:3].astype(np.float32)
    result["model_frame"] = MODEL_FRAME
    return result


def audit_dict():
    return {
        "model_frame": MODEL_FRAME,
        "observation_transform": "SAPIEN_world_to_Panda_root",
        "action_translation_transform": "Panda_root_to_SAPIEN_world",
        "action_delta_rotation": "unchanged_current_R_world_times_delta_R",
        "changes_scene_pose": False,
        "changes_camera": False,
        "changes_robot_initial_state": False,
        "uses_ground_truth_handle": False,
        "uses_ground_truth_axis": False,
    }
