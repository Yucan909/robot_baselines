from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000.0) / max(1, half - 1)
        freq = torch.exp(
            torch.arange(half, device=x.device, dtype=torch.float32) * (-scale)
        )
        emb = x.float()[:, None] * freq[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return emb


class Conv1dBlock(nn.Module):
    def __init__(self, inp: int, out: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        if out % n_groups != 0:
            raise ValueError(f"out={out} must be divisible by n_groups={n_groups}")
        self.block = nn.Sequential(
            nn.Conv1d(inp, out, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        inp: int,
        out: int,
        cond_dim: int,
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(inp, out, kernel_size, n_groups),
                Conv1dBlock(out, out, kernel_size, n_groups),
            ]
        )
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out * 2),
        )
        self.residual = nn.Conv1d(inp, out, 1) if inp != out else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        y = self.blocks[0](x)
        emb = self.cond_encoder(cond).reshape(cond.shape[0], 2, -1, 1)
        scale = emb[:, 0]
        bias = emb[:, 1]
        y = scale * y + bias
        y = self.blocks[1](y)
        return y + self.residual(x)


class Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SceneEncoder(nn.Module):
    """
    PADP perception/conditioning side:
      frozen PA3FF point features (B,N,768)
      + frozen task-critical SigLIP CLS embedding (768)
      -> trainable Transformer aggregation
      -> global feature + robot proprioception
      -> two-layer MLP compact condition.
    """

    def __init__(
        self,
        point_feature_dim: int = 768,
        proprio_dim: int = 9,
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
            nn.Linear(d_model + proprio_dim, 512),
            nn.GELU(),
            nn.Linear(512, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        point_features: torch.Tensor,
        cls_embedding: torch.Tensor,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        if point_features.ndim != 3 or point_features.shape[-1] != 768:
            raise ValueError(f"point_features shape={tuple(point_features.shape)}")
        b = point_features.shape[0]
        if cls_embedding.ndim == 1:
            cls_embedding = cls_embedding.unsqueeze(0).expand(b, -1)
        if cls_embedding.shape != (b, 768):
            raise ValueError(f"cls_embedding shape={tuple(cls_embedding.shape)}")
        if proprio.shape != (b, 9):
            raise ValueError(f"proprio shape={tuple(proprio.shape)}")

        point_tokens = self.point_proj(point_features)
        cls_token = self.cls_proj(cls_embedding).unsqueeze(1)
        tokens = torch.cat([cls_token, point_tokens], dim=1)
        tokens = self.transformer(tokens)
        global_scene = self.cls_norm(tokens[:, 0])
        return self.state_fusion(torch.cat([global_scene, proprio], dim=-1))


class ConditionalUnet1D(nn.Module):
    """
    Diffusion-Policy-style conditional 1D U-Net action head.
    The PA3FF paper does not publish PADP's denoiser micro-architecture;
    this is a transparent reproduction choice based on the Diffusion Policy family.
    """

    def __init__(
        self,
        input_dim: int = 10,
        global_cond_dim: int = 256,
        diffusion_step_embed_dim: int = 256,
        down_dims=(256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )
        cond_dim = diffusion_step_embed_dim + global_cond_dim

        all_dims = [input_dim] + list(down_dims)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        self.down_modules = nn.ModuleList([])
        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = i >= len(in_out) - 1
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in, dim_out, cond_dim, kernel_size, n_groups
                        ),
                        ConditionalResidualBlock1D(
                            dim_out, dim_out, cond_dim, kernel_size, n_groups
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim, mid_dim, cond_dim, kernel_size, n_groups
                ),
                ConditionalResidualBlock1D(
                    mid_dim, mid_dim, cond_dim, kernel_size, n_groups
                ),
            ]
        )

        self.up_modules = nn.ModuleList([])
        for dim_in, dim_out in reversed(in_out[1:]):
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2, dim_in, cond_dim, kernel_size, n_groups
                        ),
                        ConditionalResidualBlock1D(
                            dim_in, dim_in, cond_dim, kernel_size, n_groups
                        ),
                        Upsample1d(dim_in),
                    ]
                )
            )

        start_dim = down_dims[0]
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size, n_groups),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        if sample.ndim != 3:
            raise ValueError(f"sample shape={tuple(sample.shape)}")
        x = sample.transpose(1, 2)
        if not torch.is_tensor(timestep):
            timestep = torch.tensor(timestep, device=x.device, dtype=torch.long)
        if timestep.ndim == 0:
            timestep = timestep[None]
        timestep = timestep.to(device=x.device, dtype=torch.long).expand(x.shape[0])

        time_emb = self.diffusion_step_encoder(timestep)
        cond = torch.cat([time_emb, global_cond], dim=-1)

        skips = []
        for res1, res2, down in self.down_modules:
            x = res1(x, cond)
            x = res2(x, cond)
            skips.append(x)
            x = down(x)

        for mid in self.mid_modules:
            x = mid(x, cond)

        for res1, res2, up in self.up_modules:
            skip = skips.pop()
            if x.shape[-1] != skip.shape[-1]:
                raise RuntimeError(
                    f"UNet temporal mismatch x={tuple(x.shape)} skip={tuple(skip.shape)}"
                )
            x = torch.cat([x, skip], dim=1)
            x = res1(x, cond)
            x = res2(x, cond)
            x = up(x)

        x = self.final_conv(x)
        x = x.transpose(1, 2)
        if x.shape != sample.shape:
            raise RuntimeError(f"UNet output={tuple(x.shape)} input={tuple(sample.shape)}")
        return x


class PADPPolicy(nn.Module):
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
        self.scene_encoder = SceneEncoder(
            point_feature_dim=768,
            proprio_dim=9,
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
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        return self.scene_encoder(point_features, cls_embedding, proprio)

    def forward(
        self,
        point_features: torch.Tensor,
        cls_embedding: torch.Tensor,
        proprio: torch.Tensor,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        cond = self.condition(point_features, cls_embedding, proprio)
        return self.action_head(noisy_action, timestep, cond)


def squared_cosine_beta_schedule(num_train_steps: int, s: float = 0.008):
    steps = torch.arange(num_train_steps + 1, dtype=torch.float64)
    f = torch.cos(((steps / num_train_steps + s) / (1.0 + s)) * math.pi / 2.0) ** 2
    alpha_bar = f / f[0]
    beta = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return beta.clamp(1e-5, 0.999).float()


class X0Diffusion:
    """
    DDPM training noise + deterministic DDIM-style inference for x0 prediction.

    PA3FF paper Eq. (5) trains D_theta(o_t, a_tilde, k) against clean action a_t,
    so this reproduction uses prediction_type='sample' (x0), not epsilon.
    """

    def __init__(self, num_train_steps: int = 100):
        self.num_train_steps = int(num_train_steps)
        beta = squared_cosine_beta_schedule(self.num_train_steps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.beta = beta
        self.alpha_bar = alpha_bar

    def to(self, device):
        self.beta = self.beta.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        return self

    def q_sample(self, x0: torch.Tensor, timestep: torch.Tensor, noise: torch.Tensor):
        ab = self.alpha_bar[timestep].view(-1, 1, 1)
        return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise

    def inference_timesteps(self, num_inference_steps: int, device):
        ts = torch.linspace(
            self.num_train_steps - 1,
            0,
            int(num_inference_steps),
            device=device,
        ).round().long()
        ts = torch.unique_consecutive(ts)
        if int(ts[0]) != self.num_train_steps - 1:
            raise RuntimeError("DDIM schedule must start at final train timestep")
        if int(ts[-1]) != 0:
            raise RuntimeError("DDIM schedule must end at timestep 0")
        return ts

    def ddim_step(self, x_t: torch.Tensor, pred_x0: torch.Tensor, t: int, prev_t: int):
        ab_t = self.alpha_bar[int(t)]
        if prev_t >= 0:
            ab_prev = self.alpha_bar[int(prev_t)]
        else:
            ab_prev = torch.ones((), device=x_t.device, dtype=x_t.dtype)
        eps = (x_t - ab_t.sqrt() * pred_x0) / (1.0 - ab_t).sqrt().clamp_min(1e-8)
        return ab_prev.sqrt() * pred_x0 + (1.0 - ab_prev).sqrt() * eps

    @torch.no_grad()
    def sample(
        self,
        model: PADPPolicy,
        point_features: torch.Tensor,
        cls_embedding: torch.Tensor,
        proprio: torch.Tensor,
        horizon: int = 16,
        action_dim: int = 10,
        num_inference_steps: int = 10,
        initial_noise: torch.Tensor | None = None,
    ):
        b = point_features.shape[0]
        device = point_features.device
        if initial_noise is None:
            x = torch.randn(b, horizon, action_dim, device=device)
        else:
            x = initial_noise.to(device=device).clone()
        ts = self.inference_timesteps(num_inference_steps, device)
        for i, t in enumerate(ts.tolist()):
            timestep = torch.full((b,), int(t), device=device, dtype=torch.long)
            pred_x0 = model(point_features, cls_embedding, proprio, x, timestep)
            prev_t = int(ts[i + 1]) if i + 1 < len(ts) else -1
            x = self.ddim_step(x, pred_x0, int(t), prev_t)
        return x
