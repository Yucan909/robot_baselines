"""Efficient many-candidate PADP sampling with one shared scene condition.

The original runtime recomputes the identical frozen Transformer condition at
every DDIM step and for every candidate.  This implementation computes that
condition once, then applies the unchanged diffusion action head to a batch of
independently seeded noises.  Candidate actions and seeds are unchanged.
"""
from __future__ import annotations

import numpy as np
import torch

from pa3ff_policy_runtime import TASKS, rotation_6d_to_matrix
from pa3ff_policy_runtime_devselected_v3 import PA3FFPADPRuntimeDEVSelectedV3


class PA3FFPADPRuntimeManyCandidatesV17(PA3FFPADPRuntimeDEVSelectedV3):
    N_CANDIDATES = 128

    @torch.inference_mode()
    def sample_normalized_from_condition(
        self,
        feature: torch.Tensor,
        cls: torch.Tensor,
        instruction: torch.Tensor,
        proprio: torch.Tensor,
        *,
        base_seed: int,
        candidate_count: int,
    ) -> torch.Tensor:
        count = int(candidate_count)
        if count <= 0:
            raise ValueError(candidate_count)
        condition = self.model.condition(feature, cls, instruction, proprio)
        noises = []
        for candidate_i in range(count):
            generator = torch.Generator(device=self.device).manual_seed(
                int(base_seed) + candidate_i * 1000003
            )
            noises.append(torch.randn(
                (1, 16, 10), generator=generator, device=self.device
            ))
        x = torch.cat(noises, dim=0) * self.NOISE_SCALE
        condition = condition.repeat(count, 1)
        timesteps = self.diffusion.inference_timesteps(self.DDIM_STEPS, self.device)
        for index, timestep_value in enumerate(timesteps.tolist()):
            timestep = torch.full(
                (count,), int(timestep_value), device=self.device, dtype=torch.long
            )
            pred_x0 = self.model.action_head(x, timestep, condition)
            pred_x0 = pred_x0.clamp(-self.CLIP_PRED_X0, self.CLIP_PRED_X0)
            previous = (
                int(timesteps[index + 1]) if index + 1 < len(timesteps) else -1
            )
            x = self.diffusion.ddim_step(
                x, pred_x0, int(timestep_value), previous
            )
        return x

    @torch.inference_mode()
    def predict(self, **kwargs):
        forbidden = {
            "clip_pred_x0", "noise_scale", "num_inference_steps",
            "start_timestep", "candidate_count",
        } & set(kwargs)
        if forbidden:
            raise ValueError(f"frozen sampler options cannot be overridden: {sorted(forbidden)}")
        pc = np.asarray(kwargs["point_cloud_world"], dtype=np.float32)
        camera = np.asarray(kwargs["camera_pose_world"], dtype=np.float32)
        proprio = np.asarray(kwargs["robot_qpos"], dtype=np.float32).reshape(9)
        task = str(kwargs["task"])
        base_seed = int(kwargs["seed"])
        if pc.shape != (1024, 3) or not np.isfinite(pc).all():
            raise RuntimeError(f"policy point cloud shape {pc.shape}")

        normal = self._normals(pc, camera)
        point = (pc * 10.0).astype(np.float32)
        color = np.zeros((1024, 3), dtype=np.float32)
        pcd = self.transform({
            "coord": point.copy(), "color": color.copy(), "normal": normal,
            "segment": np.zeros(1024, dtype=np.int64),
        })
        feature, _ = self.pa3ff([{
            "point": point, "color": color, "pcd": pcd,
            "obj_id": "many_candidate_runtime_v17",
        }])
        part, instruction_text = TASKS[task]
        cls = torch.from_numpy(self.text_map[part][None]).to(self.device)
        instruction = torch.from_numpy(
            self.text_map[instruction_text][None]
        ).to(self.device)
        prop = torch.from_numpy(proprio[None]).to(self.device)
        normalized_batch = self.sample_normalized_from_condition(
            feature[None].float(), cls, instruction, prop,
            base_seed=base_seed, candidate_count=self.N_CANDIDATES,
        ).float().cpu().numpy()

        centroid = pc.astype(np.float64).mean(axis=0)
        relevance = feature.float().cpu().numpy() @ self.text_map[part]
        semantic_order = np.argsort(-relevance, kind="stable")
        semantic_clouds = {
            size: pc[semantic_order[:size]].astype(np.float64)
            for size in (64, 128, 256)
        }
        poses_batch = []
        gripper_batch = []
        diagnostics = []
        for candidate_i, normalized in enumerate(normalized_batch):
            action = normalized * self.action_std[None, :] + self.action_mean[None, :]
            poses = np.tile(np.eye(4, dtype=np.float64), (16, 1, 1))
            for action_i, row in enumerate(action):
                poses[action_i, :3, :3] = rotation_6d_to_matrix(row[3:9])
                poses[action_i, :3, 3] = centroid + row[:3].astype(np.float64)
            poses_batch.append(poses)
            gripper_batch.append(np.clip(action[:, 9], 0.0, 0.04).astype(np.float64))
            first = poses[0, :3, 3]
            diagnostics.append({
                "candidate_index": candidate_i,
                "candidate_seed": base_seed + candidate_i * 1000003,
                "surface_distance_m": float(np.min(np.linalg.norm(
                    pc.astype(np.float64) - first, axis=1
                ))),
                "semantic_surface_distance_m": {
                    str(size): float(np.min(np.linalg.norm(cloud - first, axis=1)))
                    for size, cloud in semantic_clouds.items()
                },
                "displacement_m": float(np.linalg.norm(
                    poses[-1, :3, 3] - poses[0, :3, 3]
                )),
                "normalized_min": float(normalized.min()),
                "normalized_max": float(normalized.max()),
            })
        poses_batch = np.stack(poses_batch)
        gripper_batch = np.stack(gripper_batch)
        if not np.isfinite(poses_batch).all():
            raise RuntimeError("nonfinite many-candidate PADP action")
        return {
            "poses_world_grasptarget": poses_batch[0],
            "gripper": gripper_batch[0],
            "candidate_poses_world_grasptarget": poses_batch,
            "candidate_gripper": gripper_batch,
            "candidate_rank_order": list(range(self.N_CANDIDATES)),
            "candidate_diagnostics": diagnostics,
            "candidate_count": self.N_CANDIDATES,
            "candidate_inference_batch_size": self.N_CANDIDATES,
            "scene_condition_computed_once": True,
            "candidate_seed_stride": 1000003,
            "num_inference_steps": self.DDIM_STEPS,
            "clip_pred_x0": self.CLIP_PRED_X0,
            "start_timestep": self.START_TIMESTEP,
            "noise_scale": self.NOISE_SCALE,
            "pa3ff_part_relevance": {
                "part": part,
                "min": float(relevance.min()),
                "max": float(relevance.max()),
                "median": float(np.median(relevance)),
            },
        }
