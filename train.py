import glob
import os
from pathlib import Path
from typing import List

import cv2
import hydra
import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKL
from diffusers.schedulers import DDPMScheduler
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.cdit_config import CDiTModelConfig
from models.cdit_model import CDiTModel


class BirdFrameDataset(Dataset):
    def __init__(self, data_dir: str, sequence_length: int = 4) -> None:
        """
        Dataset for loading sequential bird frames.

        Args:
            data_dir: Path to bird frame directory
            sequence_length: Number of consecutive frames to load
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length

        # Get all frame files and sort them
        frame_files = sorted(glob.glob(str(self.data_dir / "frame_*.png")))
        self.frame_paths = [Path(f) for f in frame_files]

        # Extract frame numbers for proper sequencing
        self.frame_numbers: List[int] = []
        for path in self.frame_paths:
            frame_num = int(path.stem.split("_")[1])
            self.frame_numbers.append(frame_num)

        print(
            f"Found {len(self.frame_paths)} frames from {min(self.frame_numbers)} \
                to {max(self.frame_numbers)}"
        )

    def __len__(self) -> int:
        # We can create sequences up to the point where we have enough frames
        return max(0, len(self.frame_paths) - self.sequence_length + 1)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Load sequence of frames starting from idx
        frames = []
        for i in range(self.sequence_length):
            frame_path = self.frame_paths[idx + i]

            # Load image using cv2
            img = cv2.imread(str(frame_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize to 512x512 for the VAE
            img = cv2.resize(img, (512, 512))

            # Convert to tensor and normalize to [0, 1]
            img_tensor = torch.from_numpy(img).float() / 255.0
            # Permute to CHW format
            img_tensor = img_tensor.permute(2, 0, 1)

            frames.append(img_tensor)

        return torch.stack(frames)  # Shape: (sequence_length, 3, 512, 512)


def encode_frames_with_vae(
    frames: torch.Tensor, vae: AutoencoderKL, device: str
) -> torch.Tensor:
    """
    Encode a batch of frames using the VAE.

    Args:
        frames: Tensor of shape (B, T, 3, H, W) where T is sequence length
        vae: VAE model
        device: Device to run on

    Returns:
        Encoded latents of shape (B, T, 4, H//8, W//8)
    """
    B, T, C, H, W = frames.shape

    # Reshape to (B*T, C, H, W) for batch processing
    frames_flat = frames.view(B * T, C, H, W).to(device)

    # Encode with VAE
    with torch.no_grad():
        encoded = vae.encode(frames_flat)
        latents = encoded.latent_dist.sample()
        # Scale latents as done in stable diffusion
        latents = latents * vae.config.scaling_factor

    # Reshape back to (B, T, 4, H//8, W//8)
    latent_shape = latents.shape
    latents = latents.view(B, T, latent_shape[1], latent_shape[2], latent_shape[3])

    return latents


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize VAE
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()  # Set to eval mode

    # Initialize diffusion scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
    )

    # Create dataset and dataloader
    print("Creating dataset...")
    dataset = BirdFrameDataset("data/bird", sequence_length=5)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)

    # Initialize model
    print("Initializing model...")
    cfg_dict = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg = CDiTModelConfig(**cfg_dict)
    model = CDiTModel(model_cfg).to(device)

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=1e-4)

    # Training parameters
    num_epochs = 10

    print("Starting training...")
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, frames in enumerate(progress_bar):
            # frames shape: (B, T, 3, 512, 512)
            batch_size, seq_len = frames.shape[:2]

            # Encode frames with VAE
            latents = encode_frames_with_vae(frames, vae, device)
            # latents shape: (B, T, 4, 64, 64)

            # For diffusion training, we'll use current frame as target and
            # previous frames as conditioning
            # Take the last frame as target and earlier frames as conditioning
            target_latents = latents[:, -1]  # (B, 4, 64, 64)
            prev_latents = latents[:, :-1]  # (B, T-1, 4, 64, 64)

            # Flatten spatial dimensions for the model
            # Model expects (B, 4096, 4) format based on the example
            B, C, H, W = target_latents.shape
            target_flat = target_latents.view(B, H * W, C)  # (B, 4096, 4)

            # Also flatten previous latents
            B_prev, T_prev, C_prev, H_prev, W_prev = prev_latents.shape
            prev_flat = prev_latents.view(
                B_prev, T_prev * H_prev * W_prev, C_prev
            )  # (B, T*4096, 4)

            # Add noise to target latents (diffusion forward process)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=device,
            )
            noise = torch.randn_like(target_flat)
            noisy_latents = noise_scheduler.add_noise(target_flat, noise, timesteps)

            # Prepare model inputs
            t = (
                timesteps.float().unsqueeze(1)
                / noise_scheduler.config.num_train_timesteps
            )  # Normalized time

            # Forward pass
            noise_pred = model(noisy_latents, t, None, None, prev_flat)

            # Compute loss (predict the noise)
            loss = F.mse_loss(noise_pred, noise)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{epoch_loss/num_batches:.4f}",
                }
            )

        print(f"Epoch {epoch+1} completed. Average loss: {epoch_loss/num_batches:.4f}")

        # Save checkpoint every few epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"outputs/checkpoint_epoch_{epoch+1}.pt"
            os.makedirs("outputs", exist_ok=True)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_loss / num_batches,
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved to {checkpoint_path}")

    print("Training completed!")


if __name__ == "__main__":
    main()
