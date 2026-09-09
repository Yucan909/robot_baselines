"""DEV-frozen PA3FF/PADP inference configuration.

The numbers are selected in INFERENCE_CONFIG_DEV_SELECTION.json using only
object-level DEV demonstrations.  This wrapper keeps the base model and action
representation unchanged and makes the selected sampler settings impossible
to omit accidentally in a formal worker.
"""
from __future__ import annotations

import numpy as np
import torch

from pa3ff_policy_runtime import PA3FFPADPRuntime, TASKS, rotation_6d_to_matrix


class PA3FFPADPRuntimeDEVSelectedV3(PA3FFPADPRuntime):
    CLIP_PRED_X0 = 1.5
    NOISE_SCALE = 0.5
    DDIM_STEPS = 10
    START_TIMESTEP = 99
    N_CANDIDATES = 32
    DESIRED_SURFACE_DISTANCE_M = 0.025
    DISPLACEMENT_PRIOR_WEIGHT = 0.25
    TRAIN_DISPLACEMENT_MEDIAN_M = {
        "door_open": 0.2950354516506195,
        "drawer_open": 0.16538287699222565,
    }

    @torch.inference_mode()
    def predict(self, **kwargs):
        forbidden = {
            "clip_pred_x0", "noise_scale", "num_inference_steps", "start_timestep"
        } & set(kwargs)
        if forbidden:
            raise ValueError(f"DEV-frozen sampler options cannot be overridden: {sorted(forbidden)}")
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
            "point": point, "color": color, "pcd": pcd, "obj_id": "formal_runtime_multisample",
        }])
        part, instruction = TASKS[task]
        cls = torch.from_numpy(self.text_map[part][None]).to(self.device)
        instr = torch.from_numpy(self.text_map[instruction][None]).to(self.device)
        prop = torch.from_numpy(proprio[None]).to(self.device)
        centroid = pc.astype(np.float64).mean(axis=0)

        # The diffusion implementation is batch-native.  Construct the exact
        # per-candidate seeded initial noises first, then denoise them in
        # one call.  This changes only floating-point kernel grouping (validated
        # below 0.11 mm / 0.001 rad on the frozen diagnostic), not the sampler,
        # seeds, conditioning, action representation, or ranking rule.
        initial_noises = []
        for candidate_i in range(self.N_CANDIDATES):
            candidate_seed = base_seed + candidate_i * 1000003
            gen = torch.Generator(device=self.device).manual_seed(candidate_seed)
            initial_noises.append(
                torch.randn((1, 16, 10), generator=gen, device=self.device)
            )
        n = self.N_CANDIDATES
        normalized_batch = self.diffusion.sample(
            self.model, feature[None].float().repeat(n, 1, 1),
            cls.repeat(n, 1), instr.repeat(n, 1), prop.repeat(n, 1),
            horizon=16, action_dim=10, num_inference_steps=self.DDIM_STEPS,
            clip_pred_x0=self.CLIP_PRED_X0, start_timestep=self.START_TIMESTEP,
            initial_noise=torch.cat(initial_noises, dim=0) * self.NOISE_SCALE,
        ).float().cpu().numpy()

        candidates = []
        for candidate_i, normalized in enumerate(normalized_batch):
            candidate_seed = base_seed + candidate_i * 1000003
            action = normalized * self.action_std[None, :] + self.action_mean[None, :]
            poses = []
            for row in action:
                pose = np.eye(4, dtype=np.float64)
                pose[:3, :3] = rotation_6d_to_matrix(row[3:9])
                pose[:3, 3] = centroid + row[:3].astype(np.float64)
                poses.append(pose)
            poses = np.stack(poses)
            surface = float(np.min(np.linalg.norm(pc.astype(np.float64) - poses[0, :3, 3], axis=1)))
            displacement_m = float(np.linalg.norm(poses[-1, :3, 3] - poses[0, :3, 3]))
            displacement_prior_error_m = abs(
                displacement_m - self.TRAIN_DISPLACEMENT_MEDIAN_M[task]
            )
            candidates.append({
                "candidate_index": candidate_i,
                "candidate_seed": candidate_seed,
                "poses": poses,
                "gripper": np.clip(action[:, 9], 0.0, 0.04).astype(np.float64),
                "surface_distance_m": surface,
                "surface_score": abs(surface - self.DESIRED_SURFACE_DISTANCE_M),
                "displacement_m": displacement_m,
                "train_displacement_median_m": self.TRAIN_DISPLACEMENT_MEDIAN_M[task],
                "displacement_prior_error_m": displacement_prior_error_m,
                "observation_only_rank_score": (
                    abs(surface - self.DESIRED_SURFACE_DISTANCE_M)
                    + self.DISPLACEMENT_PRIOR_WEIGHT * displacement_prior_error_m
                ),
                "normalized_min": float(normalized.min()),
                "normalized_max": float(normalized.max()),
                "decoded_xyz_min": action[:, :3].min(axis=0).tolist(),
                "decoded_xyz_max": action[:, :3].max(axis=0).tolist(),
            })
        order = sorted(
            range(self.N_CANDIDATES),
            key=lambda i: (candidates[i]["observation_only_rank_score"], i),
        )
        selected = candidates[order[0]]
        if not np.isfinite(selected["poses"]).all():
            raise RuntimeError("nonfinite PADP multi-sample action")
        return {
            "poses_world_grasptarget": selected["poses"],
            "gripper": selected["gripper"],
            "candidate_poses_world_grasptarget": np.stack([x["poses"] for x in candidates]),
            "candidate_gripper": np.stack([x["gripper"] for x in candidates]),
            "candidate_rank_order": order,
            "candidate_diagnostics": [
                {k: v for k, v in x.items() if k not in {"poses", "gripper"}}
                for x in candidates
            ],
            "candidate_count": self.N_CANDIDATES,
            "candidate_inference_batch_size": self.N_CANDIDATES,
            "candidate_ranking": (
                f"abs(first_nearest_object_surface_m-{self.DESIRED_SURFACE_DISTANCE_M})"
                f"+{self.DISPLACEMENT_PRIOR_WEIGHT}*abs(predicted_displacement_m-TRAIN_task_median_m)"
            ),
            "normalized_action_min": selected["normalized_min"],
            "normalized_action_max": selected["normalized_max"],
            "num_inference_steps": self.DDIM_STEPS,
            "clip_pred_x0": self.CLIP_PRED_X0,
            "start_timestep": self.START_TIMESTEP,
            "noise_scale": self.NOISE_SCALE,
            "decoded_xyz_min": selected["decoded_xyz_min"],
            "decoded_xyz_max": selected["decoded_xyz_max"],
        }
