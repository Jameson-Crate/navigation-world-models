import torch
from torch import nn
import cv2

from diffusers.models import AutoencoderKL
import xformers.ops as xops
from rotary_embedding_torch import RotaryEmbedding

def crop_image(img, H_new, W_new):
    H, W = img.shape[:2]
    H_diff = (H - H_new) // 2
    W_diff = (W - W_new) // 2
    return img[H_diff:H_diff + H_new, W_diff:W_diff + W_new, :]

class ConditionalEncoding(nn.Module):
    def __init__(self, encoding_dim, hidden_dim=512):
        assert (encoding_dim - 1) % 2 == 0, "Encoding Dim Invalid Size"
        super().__init__()
        self.length = (encoding_dim - 1) // 2
        self.ffn = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, encoding_dim),
        )

    def positional_encoding(self, x):
        device = x.device
        B, _ = x.shape
        pe = torch.zeros((B, 2 * self.length + 1), device=device)
        pe[:, 0] = x.T
        xh = 2 ** torch.arange(self.length, device=device) * torch.pi * x
        pe[:, 1::2] = torch.sin(xh)
        pe[:, 2::2] = torch.cos(xh)
        return pe
    
    def forward(self, x):
        with torch.no_grad():
            pe = self.positional_encoding(x)
        return self.ffn(pe)


class AdaLayerNorm(nn.Module):
    def __init__(self, normalized_shape, cond_dim, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, elementwise_affine=False, eps=eps)
        self.modulation = nn.Linear(cond_dim, 2 * normalized_shape)

    def forward(self, x, cond):
        x = self.norm(x)
        gamma_beta = self.modulation(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return gamma * x + beta
    

class ScaledLayerNorm(nn.Module):
    def __init__(self, normalized_shape, cond_dim, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, elementwise_affine=False, eps=eps)
        self.modulation = nn.Linear(cond_dim, normalized_shape)

    def forward(self, x, cond):
        x = self.norm(x)
        alpha = self.modulation(cond)
        return alpha * x


class ConditionalDiffusionTransformer(nn.Module):
    def __init__(self):
        # Encodings
        self.x_action_encoding = ConditionalEncoding()
        self.y_action_encoding = ConditionalEncoding()
        self.yaw_action_encoding = ConditionalEncoding()
        self.time_encoding = ConditionalEncoding()
        self.denoise_encoding = ConditionalEncoding()
        self.rope = RotaryEmbedding()

        # Layer Norms
        self.aln1 = AdaLayerNorm()
        self.sln1 = ScaledLayerNorm()
        self.aln2 = AdaLayerNorm()
        self.aln3 = AdaLayerNorm()
        self.sln2 = ScaledLayerNorm()        
        self.aln4 = AdaLayerNorm()
        self.sln3 = ScaledLayerNorm()

        # Feed Forward Networks
        self.sa_ffn1 = nn.Sequential(
            nn.Linear(4, ),
        )
        self.sa_ffn2 = nn.Sequential(
            nn.Linear(, ),
            nn.GELU(),
            nn.Linear(, ),
        )        
        self.ca_ffn = nn.Sequential(
            nn.Linear(, ),
            nn.GELU(),
            nn.Linear(, ),
        )
        self.pw_ffn = nn.Sequential(
            nn.Linear(),
            nn.GELU(),
            nn.Linear(),
            nn.GELU(),
            nn.Linear(),
        )

    def forward(self, s_t, k, t, a=None, s_prev=None):
        cond = self.time_encoding(k)
        cond += self.denoise_encoding(t)
        
        sa_in = self.sa_ffn1(s_t)
        x = self.aln1(sa_in, cond)
        x = self.rope.rotate_queries_or_keys(x)
        x = xops.memory_efficient_attention(x, x, x)
        x = self.sa_ffn2(x)
        ca_in = self.sln1(x, cond) + sa_in
        x = self.aln2(ca_in, cond)
        x = self.rope.rotate_queries_or_keys(x)

        # TODO: Get prev state conditions for cross attention

        x = xops.memory_efficient_attention(x, k, x)
        x = self.ca_ffn(x)
        pw_in = self.aln2(x) + ca_in

# Load VAE
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
img = torch.from_numpy(cv2.imread('data/arnolfini-portrait-512.png'))
data = (img[None, :, :, :].permute((0, 3, 2, 1)) / 255.0).to(device)
enc_x = vae.encode(data)
z = enc_x.latent_dist.sample()
data_sample = vae.decode(z)

print('Ended Program')