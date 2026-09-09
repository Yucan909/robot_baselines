"""Training-domain scene point-cloud observation for PA3FF reproduction.

The reconstructed TRAIN/DEV ``initial_point_cloud_*`` arrays contain both the
articulated object and the Panda.  The previous formal adapter retained only
object actor IDs, which changed the cloud support and, critically, its centroid
(the translation origin used by PADP actions).  This adapter uses the same
frozen camera but retains object *and robot* pixels.  It does not use a target
link mask, action, progress, success, or trajectory field.
"""
from __future__ import annotations

import numpy as np

from articubot_observation_adapter import sample_point_cloud


def _actor_id(actor) -> int:
    return int(actor.get_id()) if hasattr(actor, "get_id") else int(actor.id)


def capture_scene_observation(
    *, scene, camera, obj, panda, num_points: int, rng: np.random.Generator
):
    scene.update_render()
    camera.take_picture()
    position = np.asarray(camera.get_float_texture("Position"))
    segmentation = np.asarray(camera.get_uint32_texture("Segmentation"))
    actor_seg = segmentation[..., 1].astype(np.int64)
    scene_ids = np.asarray(
        [_actor_id(link) for link in obj.get_links()]
        + [_actor_id(link) for link in panda.robot.get_links()],
        dtype=np.int64,
    )
    keep = (position[..., 3] > 0) & np.isin(actor_seg, scene_ids)
    points_camera = position[..., :3][keep].astype(np.float64)
    if len(points_camera) == 0:
        raise RuntimeError("object_and_robot_not_visible")
    model_matrix = np.asarray(camera.get_model_matrix(), dtype=np.float64)
    points_world = (
        points_camera @ model_matrix[:3, :3].T + model_matrix[:3, 3]
    )
    point_cloud = sample_point_cloud(points_world, int(num_points), rng)

    grasp_pose = panda.get_grasp_pose_matrix()
    rot = grasp_pose[:3, :3]
    orient_6d = np.asarray([rot[:, 0], rot[:, 1]], dtype=np.float64).reshape(6)
    finger = float(np.asarray(panda.get_finger_qpos(), dtype=np.float64).mean())
    agent_pos = np.concatenate(
        [grasp_pose[:3, 3], orient_6d, [finger]]
    ).astype(np.float32)
    gripper_pcd = np.asarray(
        [
            panda.hand_link.get_pose().p,
            panda.right_finger_link.get_pose().p,
            panda.left_finger_link.get_pose().p,
            panda.grasp_link.get_pose().p,
        ],
        dtype=np.float32,
    ).reshape(4, 3)
    # Displacement is retained for interface parity; PA3FF does not consume it.
    object_ids = np.asarray(
        [_actor_id(link) for link in obj.get_links()], dtype=np.int64
    )
    object_keep = (position[..., 3] > 0) & np.isin(actor_seg, object_ids)
    object_camera = position[..., :3][object_keep].astype(np.float64)
    if len(object_camera):
        object_world = (
            object_camera @ model_matrix[:3, :3].T + model_matrix[:3, 3]
        )
        object_point_cloud = sample_point_cloud(
            object_world, int(num_points), rng
        )
        nearest = np.asarray([
            object_world[np.argmin(np.linalg.norm(object_world - point, axis=1))]
            for point in gripper_pcd
        ])
        displacement = (nearest - gripper_pcd).astype(np.float32)
    else:
        object_point_cloud = point_cloud.copy()
        displacement = np.zeros((4, 3), dtype=np.float32)
    return {
        "point_cloud": point_cloud,
        "agent_pos": agent_pos,
        "gripper_pcd": gripper_pcd,
        "displacement_gripper_to_object": displacement,
        "point_cloud_actor_scope": "articulated_object_plus_panda",
        "rank_point_cloud": object_point_cloud,
    }
