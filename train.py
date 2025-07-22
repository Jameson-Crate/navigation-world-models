import cv2
import torch
from diffusers.models import AutoencoderKL

# Load VAE
device = "cuda" if torch.cuda.is_available() else "cpu"
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
img = torch.from_numpy(cv2.imread("data/arnolfini-portrait-512.png"))
data = (img[None, :, :, :].permute((0, 3, 2, 1)) / 255.0).to(device)
enc_x = vae.encode(data)
z = enc_x.latent_dist.sample()
data_sample = vae.decode(z)
