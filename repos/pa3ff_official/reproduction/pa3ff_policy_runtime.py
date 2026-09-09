from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import numpy as np
import torch

HOME = Path('/home/feng')
CODE = HOME / 'robot_baselines/repos/pa3ff_official/reproduction'
OFFICIAL = HOME / 'robot_baselines/repos/pa3ff_official'
REP_CKPT = HOME / 'robot_baselines/results/pa3ff/representation_native5_balanced_v1/instance_net_snapshots/instance_net_step10000.pth'
ACTION_NORM = HOME / 'robot_baselines/results/pa3ff/padp_data_v3_initial_state_aligned_fourtask/ACTION_NORMALIZATION_V3.npz'
TEXT_EMBEDDINGS = HOME / 'robot_baselines/results/pa3ff/reproduction_v1/frozen_text_embeddings.npz'
TASKS = {
    'door_open': ('door', 'open door'),
    'door_close': ('door', 'close door'),
    'drawer_open': ('drawer', 'open drawer'),
    'drawer_close': ('drawer', 'close drawer'),
}


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def rotation_6d_to_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(6)
    b1 = x[:3] / max(float(np.linalg.norm(x[:3])), 1e-12)
    a2 = x[3:]
    b2raw = a2 - float(np.dot(b1, a2)) * b1
    b2 = b2raw / max(float(np.linalg.norm(b2raw)), 1e-12)
    b3 = np.cross(b1, b2)
    R = np.column_stack([b1, b2, b3])
    if not np.isfinite(R).all() or abs(np.linalg.det(R) - 1.0) > 1e-3:
        raise RuntimeError('invalid policy rotation6d')
    return R


class PA3FFPADPRuntime:
    """Frozen PA3FF representation plus the DEV-selected reconstructed PADP policy."""

    def __init__(self, checkpoint: Path, device: str = 'cuda'):
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
        sys.path.insert(0, str(CODE))
        sys.path.insert(0, str(OFFICIAL))
        sys.path.insert(0, str(OFFICIAL / 'libs/sonata'))
        os.chdir(OFFICIAL)

        import open3d as o3d
        from pointcept.datasets.transform import Compose
        from pointcept.models.PA3FF import PA3FF
        from padp_model_fourtask import FourTaskPADPPolicy, FourTaskX0Diffusion

        self.o3d = o3d
        self.device = torch.device(device)
        self.checkpoint = Path(checkpoint).resolve()
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        torch.set_float32_matmul_precision('high')

        with np.load(TEXT_EMBEDDINGS, allow_pickle=False) as z:
            texts = [str(x) for x in z['texts'].tolist()]
            raw = np.asarray(z['embeddings'], dtype=np.float32)
            if not bool(np.asarray(z['normalized']).item()):
                raise RuntimeError('frozen SigLIP embeddings are not normalized')
        if raw.shape != (6, 768) or not np.isfinite(raw).all():
            raise RuntimeError(f'bad frozen SigLIP embeddings {raw.shape}')
        self.text_map = {s: raw[i] for i, s in enumerate(texts)}

        self.transform = Compose([
            dict(type='CenterShift', apply_z=True),
            dict(type='GridSample', grid_size=0.02, hash_type='fnv', mode='train', return_grid_coord=True, return_inverse=True),
            dict(type='NormalizeColor'), dict(type='ToTensor'),
            dict(type='Collect', keys=('coord', 'grid_coord', 'color', 'inverse'), feat_keys=('coord', 'color', 'normal')),
        ])
        self.pa3ff = PA3FF(backbone_dim=1088, output_dim=768, freeze_backbone=True,
                           max_grouping_scale=2, use_hierarchy_losses=True, backbone=None).to(self.device).eval()
        rep = torch.load(REP_CKPT, map_location='cpu')
        self.pa3ff.instance_net.load_state_dict(rep['instance_net'], strict=True)
        for p in self.pa3ff.parameters():
            p.requires_grad_(False)
        del rep

        payload = torch.load(self.checkpoint, map_location='cpu', weights_only=False)
        self.training_step = int(payload['step'])
        self.model = FourTaskPADPPolicy(action_dim=10, d_model=256, scene_layers=4, scene_heads=8,
                                        scene_ff=1024, unet_down_dims=(256, 512, 1024)).to(self.device).eval()
        self.model.load_state_dict(payload['model'], strict=True)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.diffusion = FourTaskX0Diffusion(100).to(self.device)
        del payload

        with np.load(ACTION_NORM, allow_pickle=False) as z:
            self.action_mean = np.asarray(z['mean'], dtype=np.float32)
            self.action_std = np.asarray(z['std'], dtype=np.float32)

    def _normals(self, pc_world: np.ndarray, camera_pose: np.ndarray) -> np.ndarray:
        cloud = self.o3d.geometry.PointCloud()
        cloud.points = self.o3d.utility.Vector3dVector(pc_world.astype(np.float64))
        cloud.estimate_normals(search_param=self.o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))
        cloud.orient_normals_towards_camera_location(camera_pose[:3, 3].astype(np.float64))
        normal = np.asarray(cloud.normals, dtype=np.float32)
        if normal.shape != (1024, 3) or not np.isfinite(normal).all():
            raise RuntimeError(f'bad normal array {normal.shape}')
        return normal

    @torch.inference_mode()
    def predict(self, *, point_cloud_world: np.ndarray, camera_pose_world: np.ndarray,
                robot_qpos: np.ndarray, task: str, seed: int,
                num_inference_steps: int = 10,
                clip_pred_x0: float | None = None,
                start_timestep: int | None = None,
                noise_scale: float = 1.0) -> dict:
        pc = np.asarray(point_cloud_world, dtype=np.float32)
        camera = np.asarray(camera_pose_world, dtype=np.float32)
        proprio = np.asarray(robot_qpos, dtype=np.float32).reshape(9)
        if pc.shape != (1024, 3) or not np.isfinite(pc).all():
            raise RuntimeError(f'policy point cloud shape {pc.shape}')
        normal = self._normals(pc, camera)
        point = (pc * 10.0).astype(np.float32)
        color = np.zeros((1024, 3), dtype=np.float32)
        pcd = self.transform({'coord': point.copy(), 'color': color.copy(), 'normal': normal,
                              'segment': np.zeros(1024, dtype=np.int64)})
        feature, _ = self.pa3ff([{'point': point, 'color': color, 'pcd': pcd, 'obj_id': 'formal_runtime'}])
        part, instruction = TASKS[task]
        cls = torch.from_numpy(self.text_map[part][None]).to(self.device)
        instr = torch.from_numpy(self.text_map[instruction][None]).to(self.device)
        prop = torch.from_numpy(proprio[None]).to(self.device)
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        initial_noise = torch.randn((1, 16, 10), generator=gen, device=self.device) * float(noise_scale)
        normalized = self.diffusion.sample(self.model, feature[None].float(), cls, instr, prop,
                                           horizon=16, action_dim=10,
                                           num_inference_steps=int(num_inference_steps),
                                           clip_pred_x0=clip_pred_x0,
                                           start_timestep=start_timestep,
                                           initial_noise=initial_noise)[0].float().cpu().numpy()
        action = normalized * self.action_std[None, :] + self.action_mean[None, :]
        centroid = pc.astype(np.float64).mean(axis=0)
        poses = []
        for a in action:
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = rotation_6d_to_matrix(a[3:9])
            T[:3, 3] = centroid + a[:3].astype(np.float64)
            poses.append(T)
        poses = np.stack(poses)
        if not np.isfinite(poses).all() or not np.isfinite(action).all():
            raise RuntimeError('nonfinite PADP action')
        return {
            'poses_world_grasptarget': poses,
            'gripper': np.clip(action[:, 9], 0.0, 0.04).astype(np.float64),
            'normalized_action_min': float(normalized.min()),
            'normalized_action_max': float(normalized.max()),
            'num_inference_steps': int(num_inference_steps),
            'clip_pred_x0': None if clip_pred_x0 is None else float(clip_pred_x0),
            'start_timestep': None if start_timestep is None else int(start_timestep),
            'noise_scale': float(noise_scale),
            'decoded_xyz_min': action[:, :3].min(axis=0).tolist(),
            'decoded_xyz_max': action[:, :3].max(axis=0).tolist(),
        }
