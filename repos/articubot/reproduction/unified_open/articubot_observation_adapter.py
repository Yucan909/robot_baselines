"""Runtime-only bridge from the frozen SAPIEN camera to ArticuBot observations."""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable

import numpy as np
from scipy.spatial.distance import cdist


def _fps_indices(points: np.ndarray, count: int, start_idx: int) -> np.ndarray:
    # This is the exact FPS package/API used by the official RobogenPointCloudWrapper.
    import fpsample

    try:
        h = min(9, np.log2(count))
        indices = fpsample.bucket_fps_kdline_sampling(
            points[:, :3], count, h=h, start_idx=int(start_idx)
        )
    except Exception:
        indices = fpsample.fps_npdu_kdtree_sampling(
            points[:, :3], count, start_idx=int(start_idx)
        )
    return np.asarray(sorted(np.asarray(indices, dtype=np.int64).tolist()))


def sample_point_cloud(points: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    points = np.ascontiguousarray(np.asarray(points, dtype=np.float64).reshape(-1, 3))
    if len(points) == 0:
        raise RuntimeError("object point cloud is empty")
    if not np.all(np.isfinite(points)):
        raise RuntimeError("object point cloud contains NaN/Inf")
    if len(points) < count:
        extra = rng.choice(len(points), size=count - len(points), replace=True)
        points = np.concatenate([points, points[extra]], axis=0)
    # fpsample otherwise chooses its own random starting point and bypasses the
    # formal episode RNG.  Draw it explicitly from the supplied Generator so
    # identical scene/seed pairs produce identical observations.
    start_idx = int(rng.integers(0, len(points)))
    sampled = points[_fps_indices(points, count, start_idx)].astype(np.float32)
    if sampled.shape != (count, 3):
        raise RuntimeError(f"FPS shape mismatch: {sampled.shape}, expected {(count, 3)}")
    return sampled


def capture_articubot_observation(
    *, scene, camera, obj, panda, num_points: int, rng: np.random.Generator
) -> Dict[str, np.ndarray]:
    """Use only the frozen benchmark camera and convert its object pixels to world XYZ."""
    scene.update_render()
    camera.take_picture()
    position = np.asarray(camera.get_float_texture("Position"))
    segmentation = np.asarray(camera.get_uint32_texture("Segmentation"))
    actor_seg = segmentation[..., 1].astype(np.int64)
    object_ids = np.asarray([int(link.get_id()) for link in obj.get_links()], dtype=np.int64)
    keep = (position[..., 3] > 0) & np.isin(actor_seg, object_ids)
    points_camera = position[..., :3][keep].astype(np.float64)
    if len(points_camera) == 0:
        raise RuntimeError("object_not_visible")

    model_matrix = np.asarray(camera.get_model_matrix(), dtype=np.float64)
    points_world = points_camera @ model_matrix[:3, :3].T + model_matrix[:3, 3]
    point_cloud = sample_point_cloud(points_world, int(num_points), rng)

    grasp_pose = panda.get_grasp_pose_matrix()
    rot = grasp_pose[:3, :3]
    orient_6d = np.asarray([rot[:, 0], rot[:, 1]], dtype=np.float64).reshape(6)
    finger = float(np.asarray(panda.get_finger_qpos(), dtype=np.float64).mean())
    agent_pos = np.concatenate([grasp_pose[:3, 3], orient_6d, [finger]]).astype(np.float32)

    # Exact official order in RobogenPointCloudWrapper.get_gripper_pc().
    gripper_pcd = np.asarray(
        [
            panda.hand_link.get_pose().p,
            panda.right_finger_link.get_pose().p,
            panda.left_finger_link.get_pose().p,
            panda.grasp_link.get_pose().p,
        ],
        dtype=np.float32,
    ).reshape(4, 3)
    nearest = point_cloud[np.argmin(cdist(gripper_pcd, point_cloud), axis=1)]
    displacement = (nearest - gripper_pcd).astype(np.float32)

    scale = np.ptp(point_cloud, axis=0)
    if float(np.linalg.norm(scale)) < 1e-4 or float(np.linalg.norm(scale)) > 10.0:
        raise RuntimeError(f"implausible point-cloud scale: {scale.tolist()}")
    return {
        "point_cloud": point_cloud,
        "agent_pos": agent_pos,
        "gripper_pcd": gripper_pcd,
        "displacement_gripper_to_object": displacement,
    }


class ObservationHistory:
    def __init__(self, n_obs_steps: int):
        self.n_obs_steps = int(n_obs_steps)
        self._items = deque(maxlen=self.n_obs_steps)

    def reset(self, observation: Dict[str, np.ndarray]) -> None:
        self._items.clear()
        for _ in range(self.n_obs_steps):
            self._items.append({k: np.asarray(v).copy() for k, v in observation.items()})

    def append(self, observation: Dict[str, np.ndarray]) -> None:
        if not self._items:
            self.reset(observation)
        else:
            self._items.append({k: np.asarray(v).copy() for k, v in observation.items()})

    def stack(self) -> Dict[str, np.ndarray]:
        if len(self._items) != self.n_obs_steps:
            raise RuntimeError("observation history is not initialized")
        keys: Iterable[str] = self._items[0].keys()
        return {key: np.stack([item[key] for item in self._items], axis=0) for key in keys}
