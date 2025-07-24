from typing import Optional

import numpy as np
import torch
import xformers.ops as xops
from rotary_embedding_torch import RotaryEmbedding
from torch import nn

from models.cdt_config import CDTModelConfig


def crop_image(img: np.array, H_new: int, W_new: int) -> np.array:
    H, W = img.shape[:2]
    H_diff = (H - H_new) // 2
    W_diff = (W - W_new) // 2
    return img[H_diff : H_diff + H_new, W_diff : W_diff + W_new, :]


class ConditionalEncoding(nn.Module):
    def __init__(self, encoding_dim: int, pe_dim: int = 65) -> None:
        assert (pe_dim - 1) % 2 == 0, "Encoding Dim Invalid Size"
        super().__init__()
        self.length = (pe_dim - 1) // 2
        self.ffn = nn.Sequential(
            nn.Linear(pe_dim, 2 * encoding_dim),
            nn.GELU(),
            nn.Linear(2 * encoding_dim, encoding_dim),
        )

    def positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        B, _ = x.shape
        pe = torch.zeros((B, 2 * self.length + 1), device=device)
        pe[:, 0] = x.T
        xh = 2 ** torch.arange(self.length, device=device) * torch.pi * x
        pe[:, 1::2] = torch.sin(xh)
        pe[:, 2::2] = torch.cos(xh)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            pe = self.positional_encoding(x)
        return self.ffn(pe)


class AdaLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, cond_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, elementwise_affine=False, eps=eps)
        self.modulation = nn.Linear(cond_dim, 2 * normalized_shape)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        gamma_beta = self.modulation(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return gamma * x + beta


class ScaledLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, cond_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, elementwise_affine=False, eps=eps)
        self.modulation = nn.Linear(cond_dim, normalized_shape)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        alpha = self.modulation(cond)
        return alpha * x


class ConditionalDiffusionTransformer(nn.Module):
    def __init__(self, config: CDTModelConfig) -> None:
        super().__init__()

        # Encodings
        self.x_action_encoding = ConditionalEncoding(176)
        self.y_action_encoding = ConditionalEncoding(176)
        self.yaw_action_encoding = ConditionalEncoding(176)
        self.time_encoding = ConditionalEncoding(528)
        self.denoise_encoding = ConditionalEncoding(528)
        self.rope = RotaryEmbedding(dim=32)

        # Layer Norms
        self.aln1 = AdaLayerNorm(512, 528)
        self.sln1 = ScaledLayerNorm(512, 528)
        self.aln2 = AdaLayerNorm(512, 528)
        self.aln3 = AdaLayerNorm(512, 528)
        self.sln2 = ScaledLayerNorm(512, 528)
        self.aln4 = AdaLayerNorm(512, 528)
        self.sln3 = ScaledLayerNorm(512, 528)

        # Feed Forward Networks
        self.vae_ffn = self.latent_project = nn.Sequential(
            nn.Linear(4, 128), nn.GELU(), nn.Linear(128, 512)
        )
        self.sa_ffn = nn.Sequential(
            nn.Linear(512, 4 * 512),
            nn.GELU(),
            nn.Linear(4 * 512, 512),
        )
        self.ca_ffn = nn.Sequential(
            nn.Linear(512, 4 * 512),
            nn.GELU(),
            nn.Linear(4 * 512, 512),
        )
        self.pw_ffn = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
        )

    def forward(
        self,
        s_t: torch.Tensor,
        k: torch.Tensor,
        t: torch.Tensor,
        a: Optional[torch.Tensor] = None,
        s_prev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Condition encoding
        cond = self.time_encoding(k)
        cond += self.denoise_encoding(t)
        if a:
            x_cond = self.x_action_encoding(a[:, [0]])
            y_cond = self.y_action_encoding(a[:, [1]])
            yaw_cond = self.yaw_action_encoding(a[:, [2]])
            cond += torch.hstack([x_cond, y_cond, yaw_cond])

        # Self attention block
        sa_in = self.vae_ffn(s_t)
        x = self.aln1(sa_in, cond)
        x = self.rope.rotate_queries_or_keys(x)
        x = xops.memory_efficient_attention(x, x, x)
        x = self.sa_ffn(x)

        # Cross attention block
        ca_in = self.sln1(x, cond) + sa_in

        # TODO: Get prev state conditions for cross attention
        y = self.vae_ffn(s_prev)
        y = self.aln2(y, cond)
        y = self.rope.rotate_queries_or_keys(y)
        x = self.aln3(ca_in, cond)
        x = self.rope.rotate_queries_or_keys(x)
        x = xops.memory_efficient_attention(x, y, y)
        x = self.ca_ffn(x)

        # Pointwise feedforward block
        pw_in = self.sln2(x, cond) + ca_in
        x = self.aln4(pw_in, cond)
        x = self.pw_ffn(x)
        x = self.sln3(x, cond) + pw_in
        return x
