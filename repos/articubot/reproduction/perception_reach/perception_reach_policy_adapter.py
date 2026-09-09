"""Policy loader for the PerceptionReach variant.

The released DDP workspace unconditionally moves the low-level model to CUDA
while loading a checkpoint.  That is unnecessary for inference and makes CPU
validation impossible.  This adapter loads the exact same state dictionaries
on CPU, without changing or re-saving either official checkpoint.
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path

import dill
import torch

from articubot_policy_adapter import (
    ArticuBotPolicyAdapter,
    DP_ROOT,
    LOW_CKPT,
    LOW_EXP,
)


class PerceptionReachPolicyAdapter(ArticuBotPolicyAdapter):
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
                config_dir=str(DP_ROOT / "diffusion_policy_3d/config"),
                version_base=None,
            ):
                cfg = hydra.compose(
                    config_name="dp3.yaml",
                    overrides=list(OmegaConf.load(str(override_file))),
                )
        finally:
            os.chdir(old_cwd)

        workspace = TrainDP3Workspace(cfg)
        if self.device.type == "cpu":
            payload = torch.load(
                LOW_CKPT.open("rb"), pickle_module=dill, map_location="cpu",
                weights_only=False,
            )
            state_dicts = payload.get("state_dicts", {})
            if "model" not in state_dicts:
                raise RuntimeError("official low-level checkpoint has no model state")
            workspace.model.load_state_dict(state_dicts["model"], strict=True)
            if workspace.cfg.training.use_ema:
                if "ema_model" not in state_dicts:
                    raise RuntimeError("official low-level checkpoint has no EMA state")
                workspace.ema_model.load_state_dict(state_dicts["ema_model"], strict=True)
            del payload
        else:
            workspace.load_checkpoint(path=LOW_CKPT)

        policy = deepcopy(
            workspace.ema_model if workspace.cfg.training.use_ema else workspace.model
        )
        policy.eval()
        policy.reset()
        policy.to(self.device)
        return policy, cfg

    def audit(self) -> dict:
        result = super().audit()
        result["checkpoint_loader"] = (
            "exact_state_dict_cpu_map_location"
            if self.device.type == "cpu"
            else "official_workspace_loader"
        )
        result["checkpoint_modified"] = False
        return result
