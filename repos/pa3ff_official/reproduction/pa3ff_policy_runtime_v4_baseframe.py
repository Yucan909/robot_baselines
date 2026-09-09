"""Inference runtime for time-indexed, Panda-base-frame PADP V4."""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import torch


HOME = Path("/home/feng")
CODE = HOME / "robot_baselines/repos/pa3ff_official/reproduction"
OFFICIAL = HOME / "robot_baselines/repos/pa3ff_official"
REPRESENTATION = HOME / "robot_baselines/results/pa3ff/representation_native5_balanced_v1/instance_net_snapshots/instance_net_step10000.pth"
ACTION_NORM = HOME / "robot_baselines/results/pa3ff/padp_data_v4_timeindexed_baseframe_fourtask/ACTION_NORMALIZATION_V4.npz"
TEXT = HOME / "robot_baselines/results/pa3ff/reproduction_v1/frozen_text_embeddings.npz"
TASKS = {
    "door_open": ("door", "open door"),
    "door_close": ("door", "close door"),
    "drawer_open": ("drawer", "open drawer"),
    "drawer_close": ("drawer", "close drawer"),
}


def rotation_6d_to_matrix(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64).reshape(6)
    first = value[:3] / max(float(np.linalg.norm(value[:3])), 1e-12)
    second_raw = value[3:] - np.dot(first, value[3:]) * first
    second = second_raw / max(float(np.linalg.norm(second_raw)), 1e-12)
    third = np.cross(first, second)
    rotation = np.column_stack([first, second, third])
    if not np.isfinite(rotation).all() or abs(np.linalg.det(rotation) - 1.0) > 1e-3:
        raise RuntimeError("invalid rotation 6D")
    return rotation


def base_to_world(base_pose: np.ndarray) -> np.ndarray:
    x, y, yaw, z = np.asarray(base_pose, dtype=np.float64).reshape(4)
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    transform[:3, 3] = [x, y, z]
    return transform


class PA3FFPADPRuntimeV4BaseFrame:
    CHECKPOINT_STAGE = "PA3FF_PADP_V4_TIMEINDEXED_BASEFRAME"
    POSITIONAL_POINT_TOKENS = False
    N_CANDIDATES = 32
    # Bound chosen before rollout from V4 TRAIN normalized-action quantiles.
    # The old +/-1.5 limit truncated valid xyz/rotation and Close gripper data.
    CLIP_PRED_X0 = 3.0
    NOISE_SCALE = 1.0
    DDIM_STEPS = 10
    START_TIMESTEP = 99
    ACTIVE_BASE_POSE: np.ndarray | None = None

    @classmethod
    def set_active_base_pose(cls, base_pose) -> None:
        value = np.asarray(base_pose, dtype=np.float64).reshape(4)
        if not np.isfinite(value).all():
            raise RuntimeError("nonfinite active Panda base pose")
        cls.ACTIVE_BASE_POSE = value.copy()

    def __init__(self, checkpoint: Path, device: str = "cuda"):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        sys.path.insert(0, str(CODE))
        sys.path.insert(0, str(OFFICIAL))
        sys.path.insert(0, str(OFFICIAL / "libs/sonata"))
        os.chdir(OFFICIAL)
        import open3d as o3d
        from pointcept.datasets.transform import Compose
        from pointcept.models.PA3FF import PA3FF
        if self.POSITIONAL_POINT_TOKENS:
            from padp_model_fourtask_v5 import FourTaskPADPPolicyV5 as PolicyClass
            from padp_model_fourtask_v5 import FourTaskX0Diffusion
        else:
            from padp_model_fourtask import FourTaskPADPPolicy as PolicyClass
            from padp_model_fourtask import FourTaskX0Diffusion

        self.o3d = o3d
        self.device = torch.device(device)
        self.checkpoint = Path(checkpoint).resolve()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.set_float32_matmul_precision("high")
        self.transform = Compose([
            dict(type="CenterShift", apply_z=True),
            dict(type="GridSample", grid_size=0.02, hash_type="fnv", mode="train",
                 return_grid_coord=True, return_inverse=True),
            dict(type="NormalizeColor"), dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "grid_coord", "color", "inverse"),
                 feat_keys=("coord", "color", "normal")),
        ])
        self.pa3ff = PA3FF(backbone_dim=1088, output_dim=768, freeze_backbone=True,
                           max_grouping_scale=2, use_hierarchy_losses=True,
                           backbone=None).to(self.device).eval()
        representation = torch.load(REPRESENTATION, map_location="cpu")
        self.pa3ff.instance_net.load_state_dict(representation["instance_net"], strict=True)
        del representation
        for parameter in self.pa3ff.parameters():
            parameter.requires_grad_(False)

        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        if payload.get("stage") != self.CHECKPOINT_STAGE:
            raise RuntimeError(f"not a V4 checkpoint: {payload.get('stage')}")
        self.training_step = int(payload["step"])
        self.model = PolicyClass(
            action_dim=10, d_model=256, scene_layers=4, scene_heads=8,
            scene_ff=1024, unet_down_dims=(256, 512, 1024),
        ).to(self.device).eval()
        self.model.load_state_dict(payload["model"], strict=True)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.diffusion = FourTaskX0Diffusion(100).to(self.device)
        del payload

        with np.load(ACTION_NORM, allow_pickle=False) as z:
            self.action_mean = np.asarray(z["mean"], dtype=np.float32)
            self.action_std = np.asarray(z["std"], dtype=np.float32)
            frame = str(np.asarray(z["coordinate_frame"]).item())
        if frame != "panda_base":
            raise RuntimeError(f"unexpected action frame {frame}")
        with np.load(TEXT, allow_pickle=False) as z:
            texts = np.asarray(z["texts"]).astype(str).tolist()
            embeddings = np.asarray(z["embeddings"], dtype=np.float32)
            normalized = bool(np.asarray(z["normalized"]).item())
        if not normalized or embeddings.shape != (6, 768):
            raise RuntimeError("bad frozen text embeddings")
        self.text_map = {text: embeddings[row] for row, text in enumerate(texts)}

    def _normal(self, point_base: np.ndarray, camera_base: np.ndarray) -> np.ndarray:
        cloud = self.o3d.geometry.PointCloud()
        cloud.points = self.o3d.utility.Vector3dVector(point_base.astype(np.float64))
        cloud.estimate_normals(
            search_param=self.o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30)
        )
        cloud.orient_normals_towards_camera_location(camera_base[:3, 3])
        normal = np.asarray(cloud.normals, dtype=np.float32)
        if normal.shape != (1024, 3) or not np.isfinite(normal).all():
            raise RuntimeError(f"bad normals {normal.shape}")
        return normal

    @torch.inference_mode()
    def predict(self, *, point_cloud_world, camera_pose_world, robot_qpos, task,
                seed, base_pose=None, **kwargs) -> dict:
        if kwargs:
            raise ValueError(f"unexpected V4 inference options {sorted(kwargs)}")
        if base_pose is None:
            base_pose = type(self).ACTIVE_BASE_POSE
        if base_pose is None:
            raise RuntimeError("Panda base pose was not supplied to V4 PADP")
        world_from_base = base_to_world(base_pose)
        base_from_world = np.linalg.inv(world_from_base)
        point_world = np.asarray(point_cloud_world, dtype=np.float32)
        camera_world = np.asarray(camera_pose_world, dtype=np.float64)
        proprio = np.asarray(robot_qpos, dtype=np.float32).reshape(9)
        if point_world.shape != (1024, 3) or not np.isfinite(point_world).all():
            raise RuntimeError(f"bad policy point cloud {point_world.shape}")
        point_base = (
            point_world.astype(np.float64) @ base_from_world[:3, :3].T
            + base_from_world[:3, 3]
        ).astype(np.float32)
        camera_base = base_from_world @ camera_world
        normal = self._normal(point_base, camera_base)
        point = (point_base * 10.0).astype(np.float32)
        color = np.zeros((1024, 3), dtype=np.float32)
        pcd = self.transform({
            "coord": point.copy(), "color": color.copy(), "normal": normal,
            "segment": np.zeros(1024, dtype=np.int64),
        })
        feature, _ = self.pa3ff([{
            "point": point, "color": color, "pcd": pcd,
            "obj_id": "pa3ff_padp_v4_formal",
        }])
        part_text, instruction_text = TASKS[str(task)]
        cls = torch.from_numpy(self.text_map[part_text][None]).to(self.device)
        instruction = torch.from_numpy(self.text_map[instruction_text][None]).to(self.device)
        prop = torch.from_numpy(proprio[None]).to(self.device)
        if self.POSITIONAL_POINT_TOKENS:
            coords = torch.from_numpy(point_base[None]).to(self.device)
            condition = self.model.condition(
                feature[None].float(), coords, cls, instruction, prop
            )
        else:
            condition = self.model.condition(feature[None].float(), cls, instruction, prop)

        noises = []
        for candidate in range(self.N_CANDIDATES):
            generator = torch.Generator(device=self.device).manual_seed(
                int(seed) + candidate * 1000003
            )
            noises.append(torch.randn((1, 16, 10), generator=generator, device=self.device))
        x = torch.cat(noises) * float(self.NOISE_SCALE)
        condition = condition.repeat(self.N_CANDIDATES, 1)
        timesteps = torch.linspace(self.START_TIMESTEP, 0, self.DDIM_STEPS,
                                   device=self.device).round().long()
        timesteps = torch.unique_consecutive(timesteps)
        for index, timestep_value in enumerate(timesteps.tolist()):
            timestep = torch.full((self.N_CANDIDATES,), int(timestep_value),
                                  device=self.device, dtype=torch.long)
            predicted_x0 = self.model.action_head(x, timestep, condition)
            if self.CLIP_PRED_X0 is not None:
                predicted_x0 = predicted_x0.clamp(-self.CLIP_PRED_X0, self.CLIP_PRED_X0)
            previous = int(timesteps[index + 1]) if index + 1 < len(timesteps) else -1
            x = self.diffusion.ddim_step(x, predicted_x0, int(timestep_value), previous)
        normalized_batch = x.float().cpu().numpy()

        relevance = feature.float().cpu().numpy() @ self.text_map[part_text]
        semantic_order = np.argsort(-relevance, kind="stable")
        semantic_clouds = {
            size: point_world[semantic_order[:size]].astype(np.float64)
            for size in (64, 128, 256)
        }
        poses_batch, grippers, diagnostics = [], [], []
        for candidate, normalized_action in enumerate(normalized_batch):
            action = normalized_action * self.action_std[None] + self.action_mean[None]
            poses = np.tile(np.eye(4, dtype=np.float64), (16, 1, 1))
            for row, encoded in enumerate(action):
                pose_base = np.eye(4, dtype=np.float64)
                pose_base[:3, :3] = rotation_6d_to_matrix(encoded[3:9])
                pose_base[:3, 3] = encoded[:3]
                poses[row] = world_from_base @ pose_base
            gripper = np.clip(action[:, 9], 0.0, 0.04).astype(np.float64)
            first = poses[0, :3, 3]
            diagnostics.append({
                "candidate_index": candidate,
                "candidate_seed": int(seed) + candidate * 1000003,
                "surface_distance_m": float(np.min(np.linalg.norm(
                    point_world.astype(np.float64) - first, axis=1))),
                "semantic_surface_distance_m": {
                    str(size): float(np.min(np.linalg.norm(cloud - first, axis=1)))
                    for size, cloud in semantic_clouds.items()
                },
                "displacement_m": float(np.linalg.norm(
                    poses[-1, :3, 3] - poses[0, :3, 3])),
                # Base-frame action coordinates are part of the policy output,
                # not simulator/test metadata.  Expose them so candidate
                # selection can apply an action prior fitted only on TRAIN.
                "first_position_base": action[0, :3].astype(float).tolist(),
                "normalized_min": float(normalized_action.min()),
                "normalized_max": float(normalized_action.max()),
            })
            poses_batch.append(poses)
            grippers.append(gripper)
        poses_batch = np.stack(poses_batch)
        grippers = np.stack(grippers)
        if not np.isfinite(poses_batch).all() or not np.isfinite(grippers).all():
            raise RuntimeError("nonfinite V4 PADP output")
        return {
            "poses_world_grasptarget": poses_batch[0], "gripper": grippers[0],
            "candidate_poses_world_grasptarget": poses_batch,
            "candidate_gripper": grippers,
            "candidate_rank_order": list(range(self.N_CANDIDATES)),
            "candidate_diagnostics": diagnostics,
            "candidate_count": self.N_CANDIDATES,
            "candidate_inference_batch_size": self.N_CANDIDATES,
            "candidate_seed_stride": 1000003,
            "num_inference_steps": self.DDIM_STEPS,
            "clip_pred_x0": self.CLIP_PRED_X0,
            "start_timestep": self.START_TIMESTEP,
            "noise_scale": self.NOISE_SCALE,
            "action_coordinate_frame": "panda_base_decoded_to_world",
            "pointcloud_coordinate_frame": "world_transformed_to_panda_base",
            "positional_point_tokens": self.POSITIONAL_POINT_TOKENS,
            "pa3ff_part_relevance": {
                "part": part_text, "min": float(relevance.min()),
                "median": float(np.median(relevance)), "max": float(relevance.max()),
            },
        }


# Alias expected by wrappers that patch the V17 runtime symbol.
PA3FFPADPRuntimeManyCandidatesV4 = PA3FFPADPRuntimeV4BaseFrame
