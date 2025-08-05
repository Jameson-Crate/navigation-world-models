from typing import Tuple

import torch
from pydantic import BaseModel
from torch import nn
from transformers import Mamba2Config, Mamba2Model


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


class DiSSMConfig(BaseModel):
    hidden_size: int = 512
    num_mamba_layers: int = 4
    mamba_head_dim: int = 64
    num_mamba_heads: int = 16
    latent_dim: int = 4
    latent_height: int = 64
    latent_width: int = 64
    patch_height: int = 8
    patch_width: int = 8
    max_sequence_length: int = 128


class DiSSM(nn.Module):
    """
    Latent  :  (B , T , 4 , 64 , 64)
    Tokens  :  8×8 patchify  → 64 tokens / frame
    Hidden  :  512-d   (config.hidden_size)
    Output  :  noise prediction (B , 4 , 64 , 64) for the *last* frame
    """

    def __init__(self, config: DiSSMConfig) -> None:
        super().__init__()
        self.config = config
        pH, pW = config.patch_height, config.patch_width  # 8×8
        H_p, W_p = config.latent_height // pH, config.latent_width // pW
        self.num_patches = H_p * W_p  # 64

        # ------------------------------------------------------------------#
        # Positional embeddings (64 patches) – fixed sin-cos
        # ------------------------------------------------------------------#
        pe = get_2d_sincos_pos_embed(
            embed_dim=config.hidden_size,
            grid_size=(H_p, W_p),
        )  # (64 , hidden)
        self.register_buffer("pos_embed", pe, persistent=False)  # not trained
        ce = get_1d_sincos_pos_embed_from_grid(
            embed_dim=config.hidden_size,
            pos=torch.tensor(1),
        )
        self.register_buffer("cond_embed", ce, persistent=False)  # not trained

        # ------------------------------------------------------------------#
        # Patchify   (Conv3D: kernel & stride 1×8×8)
        # ------------------------------------------------------------------#
        self.patch_proj = nn.Conv3d(
            in_channels=config.latent_dim,
            out_channels=config.hidden_size,
            kernel_size=(1, pH, pW),
            stride=(1, pH, pW),
        )  # → (B , hid , T , 8 , 8)

        # ------------------------------------------------------------------#
        # Timestep → (γ,β) for AdaLayerNorm
        # ------------------------------------------------------------------#
        self.timestep_ffn = nn.Sequential(
            nn.Linear(config.hidden_size, 4 * config.hidden_size),
            nn.SiLU(),
            nn.Linear(4 * config.hidden_size, config.hidden_size),
        )
        self.adaln = AdaLayerNorm(config.hidden_size, config.hidden_size)

        # ------------------------------------------------------------------#
        # Mamba2 backbone
        # ------------------------------------------------------------------#
        mb_cfg = Mamba2Config(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_mamba_layers,
            num_heads=config.num_mamba_heads,
            head_dim=config.mamba_head_dim,
        )
        self.mamba = Mamba2Model(mb_cfg)

        # ------------------------------------------------------------------#
        # Patch-wise decoder (token → patch latent)
        # ------------------------------------------------------------------#
        self.patch_decoder = nn.Linear(
            config.hidden_size,
            config.latent_dim * pH * pW,  # 4×8×8 = 256
        )

    # ======================================================================#
    def forward(self, video_latents: torch.Tensor, denoise_step: torch.Tensor) -> torch.Tensor:
        """
        video_latents : (B , T , 4 , 64 , 64)
        denoise_step  : (B,)  – scalar timestep *for target frame only* (last frame)
        Returns       : (B , 4 , 64 , 64)  noise prediction for last frame
        """
        B, T, C, H, W = video_latents.shape
        assert (C, H, W) == (
            self.config.latent_dim,
            self.config.latent_height,
            self.config.latent_width,
        ), "latent dims mismatch"

        # ------------------------------------------------------------------#
        # 1. Patchify     (B , hid , T , 8 , 8) → (B , T , 64 , hid)
        # ------------------------------------------------------------------#
        patches = video_latents.permute(0, 2, 1, 3, 4).contiguous()
        patches = self.patch_proj(patches)  # (B , hid , T , 8 , 8)
        patches = patches.permute(0, 2, 3, 4, 1).contiguous()  # (B , T , 8 , 8 , hid)
        patches = patches.view(B, T * self.num_patches, -1)  # (B , T·64 , hid)

        # ------------------------------------------------------------------#
        # 2. Add spatial positional embedding (repeat for T frames)
        # ------------------------------------------------------------------#
        pos = self.pos_embed.unsqueeze(0).repeat(1, T, 1)  # (1 , T·64 , hid)
        tokens = patches + pos  # (B , T·64, hid)

        # ------------------------------------------------------------------#
        # 3. Ada-LayerNorm on the 64 noisy-frame tokens only
        # ------------------------------------------------------------------#
        tgt_start = (T - 1) * self.num_patches
        tgt_slice = slice(tgt_start, tgt_start + self.num_patches)  # last 64 tokens
        # timestep embedding  (use fixed sin-cos vector stored as buffer)
        t_vec = self.cond_embed.expand(B, -1)  # (B , hidden)
        t_vec = self.timestep_ffn(t_vec)  # (B , hidden)
        tokens[:, tgt_slice, :] = self.adaln(tokens[:, tgt_slice, :], t_vec)

        # ------------------------------------------------------------------#
        # 4. Temporal modelling with Mamba2
        # ------------------------------------------------------------------#
        h = self.mamba(inputs_embeds=tokens).last_hidden_state  # (B , T·64 , hid)

        # ------------------------------------------------------------------#
        # 5. Decode *target-frame* tokens back to latent patches
        # ------------------------------------------------------------------#
        h_tgt = h[:, tgt_slice, :]  # (B , 64 , hid)
        patches = self.patch_decoder(h_tgt)  # (B , 64 , 256)
        pH, pW = self.config.patch_height, self.config.patch_width
        patches = patches.view(B, 8, 8, C, pH, pW)  # (B , 8 , 8 , 4 , 8 , 8)
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()  # (B , 4 , 8 , 8 , 8 , 8)
        noise = patches.view(B, C, H, W)  # (B , 4 , 64 , 64)

        return noise


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: Tuple[int, int], cls_token: bool = False, extra_tokens: int = 0, device: str = None
) -> torch.Tensor:
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = torch.arange(grid_size[0], dtype=torch.float32, device=device)
    grid_w = torch.arange(grid_size[1], dtype=torch.float32, device=device)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")  # here w goes first
    grid = torch.stack(grid, dim=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = torch.cat([torch.zeros([extra_tokens, embed_dim], device=device), pos_embed], dim=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: torch.Tensor) -> torch.Tensor:
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


device = "cuda"
config = DiSSMConfig()
model = DiSSM(config).to(device)
B = 4
data = torch.randn((B, 10, 4, 64, 64), device=device)
step = torch.randint(1, 1001, (B, 1), device=device)
output = model(data, step)
print("DEBUG: Testing Finsihed.")
