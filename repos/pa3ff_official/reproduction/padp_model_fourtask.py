from __future__ import annotations

import torch
import torch.nn as nn

# Reuse the exact Stage7 diffusion head and diffusion algebra.
# stage8c1_fourtask_smoke.py prepends the frozen Stage7 bundle to sys.path.
from padp_model import ConditionalUnet1D, X0Diffusion


class FourTaskSceneEncoder(nn.Module):
    """
    Minimal paper-consistent extension of the frozen Stage7 SceneEncoder.

    Preserved from Stage7 / PA3FF Sec. 3.2:
      - frozen PA3FF per-point features (B,N,768)
      - task-critical semantic embedding as Transformer CLS (B,768)
      - trainable Transformer aggregation
      - robot proprioception
      - two-layer MLP producing a compact d_model condition

    Added solely to realize the paper's explicit language-conditioning path
    (Fig. 2) for joint multi-task training:
      - frozen SigLIP instruction embedding (B,768)
      - concatenated with global scene + proprioception immediately before
        the same two-layer conditioning MLP.

    There is deliberately no extra language projection/attention block.
    """

    def __init__(
        self,
        point_feature_dim: int = 768,
        proprio_dim: int = 9,
        instruction_dim: int = 768,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
    ):
        super().__init__()
        self.point_proj = nn.Linear(point_feature_dim, d_model)
        self.cls_proj = nn.Linear(point_feature_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.cls_norm = nn.LayerNorm(d_model)
        self.state_fusion = nn.Sequential(
            nn.Linear(d_model + proprio_dim + instruction_dim, 512),
            nn.GELU(),
            nn.Linear(512, d_model),
            nn.LayerNorm(d_model),
        )
        self.d_model = int(d_model)
        self.proprio_dim = int(proprio_dim)
        self.instruction_dim = int(instruction_dim)

    def forward(
        self,
        point_features: torch.Tensor,
        cls_embedding: torch.Tensor,
        instruction_embedding: torch.Tensor,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        if point_features.ndim != 3 or point_features.shape[-1] != 768:
            raise ValueError(f"point_features shape={tuple(point_features.shape)}")
        b = point_features.shape[0]
        if cls_embedding.ndim == 1:
            cls_embedding = cls_embedding.unsqueeze(0).expand(b, -1)
        if instruction_embedding.ndim == 1:
            instruction_embedding = instruction_embedding.unsqueeze(0).expand(b, -1)
        if cls_embedding.shape != (b, 768):
            raise ValueError(f"cls_embedding shape={tuple(cls_embedding.shape)}")
        if instruction_embedding.shape != (b, 768):
            raise ValueError(f"instruction_embedding shape={tuple(instruction_embedding.shape)}")
        if proprio.shape != (b, 9):
            raise ValueError(f"proprio shape={tuple(proprio.shape)}")

        point_tokens = self.point_proj(point_features)
        cls_token = self.cls_proj(cls_embedding).unsqueeze(1)
        tokens = torch.cat([cls_token, point_tokens], dim=1)
        tokens = self.transformer(tokens)
        global_scene = self.cls_norm(tokens[:, 0])
        fusion_input = torch.cat(
            [global_scene, proprio, instruction_embedding], dim=-1
        )
        return self.state_fusion(fusion_input)


class FourTaskPADPPolicy(nn.Module):
    def __init__(
        self,
        action_dim: int = 10,
        d_model: int = 256,
        scene_layers: int = 4,
        scene_heads: int = 8,
        scene_ff: int = 1024,
        unet_down_dims=(256, 512, 1024),
    ):
        super().__init__()
        self.scene_encoder = FourTaskSceneEncoder(
            point_feature_dim=768,
            proprio_dim=9,
            instruction_dim=768,
            d_model=d_model,
            nhead=scene_heads,
            num_layers=scene_layers,
            dim_feedforward=scene_ff,
        )
        self.action_head = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=d_model,
            diffusion_step_embed_dim=256,
            down_dims=unet_down_dims,
            kernel_size=5,
            n_groups=8,
        )

    def condition(
        self,
        point_features: torch.Tensor,
        cls_embedding: torch.Tensor,
        instruction_embedding: torch.Tensor,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        return self.scene_encoder(
            point_features, cls_embedding, instruction_embedding, proprio
        )

    def forward(
        self,
        point_features: torch.Tensor,
        cls_embedding: torch.Tensor,
        instruction_embedding: torch.Tensor,
        proprio: torch.Tensor,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        cond = self.condition(
            point_features, cls_embedding, instruction_embedding, proprio
        )
        return self.action_head(noisy_action, timestep, cond)


class FourTaskX0Diffusion(X0Diffusion):
    """
    Exact Stage7 X0Diffusion algebra with only the sampler call signature
    extended to forward the frozen instruction embedding to FourTaskPADPPolicy.
    q_sample, inference_timesteps, and ddim_step are inherited unchanged.
    """

    @torch.no_grad()
    def sample(
        self,
        model: FourTaskPADPPolicy,
        point_features: torch.Tensor,
        cls_embedding: torch.Tensor,
        instruction_embedding: torch.Tensor,
        proprio: torch.Tensor,
        horizon: int = 16,
        action_dim: int = 10,
        num_inference_steps: int = 10,
        initial_noise: torch.Tensor | None = None,
        clip_pred_x0: float | None = None,
        start_timestep: int | None = None,
    ):
        b = point_features.shape[0]
        device = point_features.device
        if initial_noise is None:
            x = torch.randn(b, horizon, action_dim, device=device)
        else:
            x = initial_noise.to(device=device).clone()
        if start_timestep is None:
            ts = self.inference_timesteps(num_inference_steps, device)
        else:
            start = int(start_timestep)
            if not 0 <= start < self.num_train_steps:
                raise ValueError(f"invalid start_timestep={start}")
            ts = torch.linspace(start, 0, int(num_inference_steps), device=device).round().long()
            ts = torch.unique_consecutive(ts)
        for i, t in enumerate(ts.tolist()):
            timestep = torch.full((b,), int(t), device=device, dtype=torch.long)
            pred_x0 = model(
                point_features,
                cls_embedding,
                instruction_embedding,
                proprio,
                x,
                timestep,
            )
            if clip_pred_x0 is not None:
                bound = float(clip_pred_x0)
                if not bound > 0:
                    raise ValueError(f"clip_pred_x0 must be positive, got {bound}")
                pred_x0 = pred_x0.clamp(-bound, bound)
            prev_t = int(ts[i + 1]) if i + 1 < len(ts) else -1
            x = self.ddim_step(x, pred_x0, int(t), prev_t)
        return x
