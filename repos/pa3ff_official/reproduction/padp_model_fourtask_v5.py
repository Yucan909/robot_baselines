"""Four-task PADP with explicit 3D positional tokens.

The PA3FF preprocessing centers coordinates before its frozen backbone.  A
policy that predicts absolute Panda-base end-effector poses must therefore
receive the point positions as well as f(p); otherwise global translation is
unidentifiable.  V5 adds only a per-point coordinate projection to the same
Transformer aggregator and retains the V4 action diffusion head and loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from padp_model import ConditionalUnet1D
from padp_model_fourtask import FourTaskX0Diffusion


class FourTaskSceneEncoderV5(nn.Module):
    def __init__(self, point_feature_dim=768, proprio_dim=9,
                 instruction_dim=768, d_model=256, nhead=8,
                 num_layers=4, dim_feedforward=1024):
        super().__init__()
        self.point_proj = nn.Linear(point_feature_dim, d_model)
        self.point_coord_proj = nn.Sequential(
            nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.cls_proj = nn.Linear(point_feature_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=0.0, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.cls_norm = nn.LayerNorm(d_model)
        self.state_fusion = nn.Sequential(
            nn.Linear(d_model + proprio_dim + instruction_dim, 512),
            nn.GELU(), nn.Linear(512, d_model), nn.LayerNorm(d_model),
        )

    def forward(self, point_features, point_coords_base, cls_embedding,
                instruction_embedding, proprio):
        if point_features.ndim != 3 or point_features.shape[-1] != 768:
            raise ValueError(f"point_features shape={tuple(point_features.shape)}")
        if point_coords_base.shape != point_features.shape[:2] + (3,):
            raise ValueError(f"point_coords_base shape={tuple(point_coords_base.shape)}")
        batch = point_features.shape[0]
        if cls_embedding.ndim == 1:
            cls_embedding = cls_embedding.unsqueeze(0).expand(batch, -1)
        if instruction_embedding.ndim == 1:
            instruction_embedding = instruction_embedding.unsqueeze(0).expand(batch, -1)
        if cls_embedding.shape != (batch, 768):
            raise ValueError(f"cls_embedding shape={tuple(cls_embedding.shape)}")
        if instruction_embedding.shape != (batch, 768):
            raise ValueError(f"instruction_embedding shape={tuple(instruction_embedding.shape)}")
        if proprio.shape != (batch, 9):
            raise ValueError(f"proprio shape={tuple(proprio.shape)}")
        point_tokens = (
            self.point_proj(point_features)
            + self.point_coord_proj(point_coords_base.float())
        )
        cls_token = self.cls_proj(cls_embedding).unsqueeze(1)
        tokens = self.transformer(torch.cat([cls_token, point_tokens], dim=1))
        global_scene = self.cls_norm(tokens[:, 0])
        return self.state_fusion(torch.cat([
            global_scene, proprio, instruction_embedding
        ], dim=-1))


class FourTaskPADPPolicyV5(nn.Module):
    def __init__(self, action_dim=10, d_model=256, scene_layers=4,
                 scene_heads=8, scene_ff=1024,
                 unet_down_dims=(256, 512, 1024)):
        super().__init__()
        self.scene_encoder = FourTaskSceneEncoderV5(
            point_feature_dim=768, proprio_dim=9, instruction_dim=768,
            d_model=d_model, nhead=scene_heads, num_layers=scene_layers,
            dim_feedforward=scene_ff,
        )
        self.action_head = ConditionalUnet1D(
            input_dim=action_dim, global_cond_dim=d_model,
            diffusion_step_embed_dim=256, down_dims=unet_down_dims,
            kernel_size=5, n_groups=8,
        )

    def condition(self, point_features, point_coords_base, cls_embedding,
                  instruction_embedding, proprio):
        return self.scene_encoder(
            point_features, point_coords_base, cls_embedding,
            instruction_embedding, proprio,
        )

    def forward(self, point_features, point_coords_base, cls_embedding,
                instruction_embedding, proprio, noisy_action, timestep):
        condition = self.condition(
            point_features, point_coords_base, cls_embedding,
            instruction_embedding, proprio,
        )
        return self.action_head(noisy_action, timestep, condition)


__all__ = ["FourTaskPADPPolicyV5", "FourTaskX0Diffusion"]
