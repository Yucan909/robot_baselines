"""Load and run the two unmodified official pretrained ArticuBot policies."""

from __future__ import annotations

import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict

import numpy as np
import torch


ARTICUBOT_ROOT = Path("/home/feng/robot_baselines/repos/articubot")
DP_ROOT = ARTICUBOT_ROOT / "3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy"
HIGH_CKPT = ARTICUBOT_ROOT / "data/high_level_200_obj_ckpt.pth"
LOW_EXP = ARTICUBOT_ROOT / "data/low-level-ckpt"
LOW_CKPT = LOW_EXP / "checkpoints/low-level.ckpt"


class ArticuBotPolicyAdapter:
    def __init__(self, device: str = "cuda:0"):
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        os.environ.setdefault("PROJECT_DIR", str(DP_ROOT))
        for path in (str(DP_ROOT), str(ARTICUBOT_ROOT)):
            if path not in sys.path:
                sys.path.insert(0, path)
        if not HIGH_CKPT.is_file():
            raise FileNotFoundError(HIGH_CKPT)
        if not LOW_CKPT.is_file():
            raise FileNotFoundError(LOW_CKPT)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.high_level_policy = self._load_high_level()
        self.low_level_policy, self.cfg = self._load_low_level()
        self._batch_generators = None
        self._install_batched_noise_adapter()
        self.n_obs_steps = int(self.cfg.n_obs_steps)
        self.n_action_steps = int(self.cfg.n_action_steps)
        self.num_points = int(self.cfg.task.env_runner.num_point_in_pc)
        self.observation_mode = str(self.cfg.task.env_runner.observation_mode)

    def _install_batched_noise_adapter(self):
        """Preserve per-episode official RNG streams while batching inference."""
        adapter = self
        original = self.low_level_policy.conditional_sample

        def conditional_sample(
            condition_data, condition_mask, condition_data_pc=None, condition_mask_pc=None,
            local_cond=None, global_cond=None, generator=None, **kwargs
        ):
            if adapter._batch_generators is None:
                return original(
                    condition_data, condition_mask,
                    condition_data_pc=condition_data_pc, condition_mask_pc=condition_mask_pc,
                    local_cond=local_cond, global_cond=global_cond, generator=generator, **kwargs,
                )
            policy = adapter.low_level_policy
            if len(adapter._batch_generators) != condition_data.shape[0]:
                raise RuntimeError("batch RNG stream count mismatch")
            parts = [
                torch.randn(
                    size=(1, *condition_data.shape[1:]), dtype=condition_data.dtype,
                    device=condition_data.device, generator=item,
                )
                for item in adapter._batch_generators
            ]
            trajectory = torch.cat(parts, dim=0)
            policy.noise_scheduler.set_timesteps(policy.num_inference_steps)
            for timestep in policy.noise_scheduler.timesteps:
                trajectory[condition_mask] = condition_data[condition_mask]
                model_output = policy.model(
                    sample=trajectory, timestep=timestep, local_cond=local_cond,
                    global_cond=global_cond, **kwargs,
                )
                trajectory = policy.noise_scheduler.step(
                    model_output, timestep, trajectory
                ).prev_sample
            trajectory[condition_mask] = condition_data[condition_mask]
            return trajectory

        self.low_level_policy.conditional_sample = conditional_sample

    def _load_high_level(self):
        from weighted_displacement_model.model_invariant import PointNet2_super

        model = PointNet2_super(num_classes=13, input_channel=3)
        state = torch.load(str(HIGH_CKPT), map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if any(str(key).startswith("module.") for key in state):
            state = {str(key).removeprefix("module."): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
        return model.eval().to(self.device)

    def _load_low_level(self):
        import hydra
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf
        from train_ddp import TrainDP3Workspace

        override_file = LOW_EXP / ".hydra/overrides.yaml"
        if not override_file.is_file():
            raise FileNotFoundError(override_file)
        GlobalHydra.instance().clear()
        old_cwd = Path.cwd()
        try:
            os.chdir(DP_ROOT)
            with hydra.initialize_config_dir(
                config_dir=str(DP_ROOT / "diffusion_policy_3d/config"), version_base=None
            ):
                cfg = hydra.compose(
                    config_name="dp3.yaml",
                    overrides=list(OmegaConf.load(str(override_file))),
                )
        finally:
            os.chdir(old_cwd)
        workspace = TrainDP3Workspace(cfg)
        workspace.load_checkpoint(path=LOW_CKPT)
        policy = deepcopy(workspace.ema_model if workspace.cfg.training.use_ema else workspace.model)
        policy.eval()
        policy.reset()
        policy.to(self.device)
        return policy, cfg

    @torch.no_grad()
    def high_level_infer(self, history: Dict[str, np.ndarray]) -> np.ndarray:
        return self.high_level_infer_batch([history])

    @torch.no_grad()
    def high_level_infer_batch(self, histories) -> np.ndarray:
        pointcloud = torch.from_numpy(
            np.stack([history["point_cloud"][-1] for history in histories], axis=0)
        ).to(self.device)
        gripper_pcd = torch.from_numpy(
            np.stack([history["gripper_pcd"][-1] for history in histories], axis=0)
        ).to(self.device)
        inputs = torch.cat([pointcloud, gripper_pcd], dim=1)
        outputs = self.high_level_policy(inputs.permute(0, 2, 1))
        weights = outputs[:, :-4, -1]
        batch_size = len(histories)
        displacements = outputs[:, :-4, :-1].reshape(batch_size, -1, 4, 3)
        goals = displacements + inputs[:, :-4, :3].unsqueeze(2)
        goals = (goals * torch.softmax(weights, dim=1)[..., None, None]).sum(dim=1)
        goals = goals.unsqueeze(1)
        if goals.shape != (batch_size, 1, 4, 3) or not torch.isfinite(goals).all():
            raise RuntimeError(f"invalid high-level output {tuple(goals.shape)}")
        return goals.detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def low_level_infer(self, history: Dict[str, np.ndarray], goal: np.ndarray) -> np.ndarray:
        return self.low_level_infer_batch([history], goal)[0]

    @torch.no_grad()
    def low_level_infer_batch(self, histories, goal: np.ndarray, noise_generators=None) -> np.ndarray:
        batch = {
            key: torch.from_numpy(np.stack([np.asarray(history[key]) for history in histories], axis=0)).to(self.device)
            for key in histories[0]
        }
        goal_tensor = torch.from_numpy(np.asarray(goal, dtype=np.float32)).to(self.device)
        batch["goal_gripper_pcd"] = goal_tensor.repeat(1, self.n_obs_steps, 1, 1)
        self._batch_generators = noise_generators
        try:
            result = self.low_level_policy.predict_action(batch)
        finally:
            self._batch_generators = None
        actions = result["action"].detach().cpu().numpy()
        if actions.ndim != 3 or actions.shape[0] != len(histories) or actions.shape[-1] != 10:
            raise RuntimeError(f"invalid low-level output {actions.shape}")
        if not np.all(np.isfinite(actions)):
            raise RuntimeError("low-level output contains NaN/Inf")
        return actions.astype(np.float32)

    def reset(self) -> None:
        self.low_level_policy.reset()

    def infer(self, history: Dict[str, np.ndarray]):
        goal = self.high_level_infer(history)
        return goal, self.low_level_infer(history, goal)

    def make_noise_generator(self, seed: int):
        return torch.Generator(device=self.device).manual_seed(int(seed))

    def infer_batch(self, histories, noise_generators=None):
        goals = self.high_level_infer_batch(histories)
        return goals, self.low_level_infer_batch(histories, goals, noise_generators=noise_generators)

    def audit(self) -> dict:
        return {
            "device": str(self.device),
            "n_obs_steps": self.n_obs_steps,
            "n_action_steps": self.n_action_steps,
            "num_points": self.num_points,
            "observation_mode": self.observation_mode,
            "action_dim": 10,
        }
