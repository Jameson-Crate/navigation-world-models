import glob
import os
from typing import Dict

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKL
from diffusers.schedulers import DDPMScheduler
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm
from train import BirdFrameDataset, encode_frames_with_vae

from models.cdit_config import CDiTModelConfig
from models.cdit_model import CDiTModel


def decode_latents_with_vae(latents: torch.Tensor, vae: AutoencoderKL, device: str) -> torch.Tensor:
    """
    Decode latents back to images using the VAE decoder.

    Args:
        latents: Tensor of shape (B, 4, H//8, W//8)
        vae: VAE model
        device: Device to run on

    Returns:
        Decoded images of shape (B, 3, H, W)
    """
    # Unscale latents
    latents = latents / vae.config.scaling_factor

    with torch.no_grad():
        decoded = vae.decode(latents).sample
        decoded = torch.clamp(decoded, 0.0, 1.0)

    return decoded


def diffusion_sampling(
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    prev_latents: torch.Tensor,
    batch_size: int,
    device: str,
    num_inference_steps: int = 50,
) -> torch.Tensor:
    """
    Run diffusion sampling to generate a new frame.

    Args:
        model: Trained CDiT model
        scheduler: DDPM scheduler
        prev_latents: Previous frames for conditioning (B, T*4096, 4)
        batch_size: Batch size
        device: Device to run on
        num_inference_steps: Number of denoising steps

    Returns:
        Generated latents of shape (B, 4096, 4)
    """
    # Set the scheduler timesteps
    scheduler.set_timesteps(num_inference_steps, device=device)

    # Start from pure noise
    latents_shape = (batch_size, 4096, 4)  # 64*64 = 4096
    latents = torch.randn(latents_shape, device=device)

    # Denoise iteratively
    for i, timestep in enumerate(tqdm(scheduler.timesteps, desc="Sampling")):
        # Prepare inputs
        t = torch.full(
            (batch_size, 1),
            timestep.item() / scheduler.config.num_train_timesteps,
            device=device,
        )
        k = torch.ones((batch_size, 1), device=device) * 0.5  # Fixed denoise strength

        # Predict noise
        with torch.no_grad():
            noise_pred = model(latents, k, t, None, prev_latents)

        # Remove the predicted noise
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    return latents


def visualize_results(original_frames: torch.Tensor, generated_frames: torch.Tensor, save_path: str) -> None:
    """
    Create a visualization comparing original and generated frames.

    Args:
        original_frames: Original frames tensor (B, 3, H, W)
        generated_frames: Generated frames tensor (B, 3, H, W)
        save_path: Path to save the visualization
    """
    batch_size = original_frames.shape[0]

    fig, axes = plt.subplots(2, batch_size, figsize=(batch_size * 4, 8))
    if batch_size == 1:
        axes = axes.reshape(2, 1)

    for i in range(batch_size):
        # Original frame
        orig_img = original_frames[i].permute(1, 2, 0).cpu().numpy()
        axes[0, i].imshow(orig_img)
        axes[0, i].set_title(f"Original Frame {i+1}")
        axes[0, i].axis("off")

        # Generated frame
        gen_img = generated_frames[i].permute(1, 2, 0).cpu().numpy()
        axes[1, i].imshow(gen_img)
        axes[1, i].set_title(f"Generated Frame {i+1}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_individual_frames(frames: torch.Tensor, save_dir: str, prefix: str = "frame") -> None:
    """
    Save individual frames as PNG files.

    Args:
        frames: Tensor of shape (B, 3, H, W)
        save_dir: Directory to save frames
        prefix: Filename prefix
    """
    os.makedirs(save_dir, exist_ok=True)

    for i, frame in enumerate(frames):
        # Convert to PIL Image
        frame_np = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(frame_np)

        # Save
        filename = f"{prefix}_{i:04d}.png"
        img.save(os.path.join(save_dir, filename))


def compute_metrics(original_frames: torch.Tensor, generated_frames: torch.Tensor) -> Dict[str, float]:
    """
    Compute evaluation metrics between original and generated frames.

    Args:
        original_frames: Original frames tensor (B, 3, H, W)
        generated_frames: Generated frames tensor (B, 3, H, W)

    Returns:
        Dictionary of metrics
    """
    # Mean Squared Error
    mse = F.mse_loss(generated_frames, original_frames).item()

    # Peak Signal-to-Noise Ratio
    psnr = 20 * torch.log10(1.0 / torch.sqrt(torch.tensor(mse))).item()

    # Structural Similarity Index (simplified version)
    # Convert to grayscale for SSIM calculation
    orig_gray = torch.mean(original_frames, dim=1, keepdim=True)
    gen_gray = torch.mean(generated_frames, dim=1, keepdim=True)

    # Compute means
    mu1 = torch.mean(orig_gray)
    mu2 = torch.mean(gen_gray)

    # Compute variances and covariance
    var1 = torch.var(orig_gray)
    var2 = torch.var(gen_gray)
    cov = torch.mean((orig_gray - mu1) * (gen_gray - mu2))

    # SSIM formula (simplified)
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mu1 * mu2 + c1) * (2 * cov + c2)) / ((mu1**2 + mu2**2 + c1) * (var1 + var2 + c2))

    return {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim.item() if torch.is_tensor(ssim) else ssim,
    }


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Configuration
    checkpoint_path = "outputs/checkpoint_epoch_5.pt"  # Adjust as needed
    output_dir = "eval_results"
    num_eval_samples = 5
    num_inference_steps = 50

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Available checkpoints:")
        checkpoint_dir = "outputs"
        if os.path.exists(checkpoint_dir):
            checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
            for cp in checkpoints:
                print(f"  {cp}")
        else:
            print("  No checkpoints found. Please train the model first.")
        return

    # Initialize VAE
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()

    # Initialize diffusion scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
    )

    # Initialize model
    print("Initializing model...")
    cfg_dict = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg = CDiTModelConfig(**cfg_dict)
    model = CDiTModel(model_cfg).to(device)

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(
        f"Loaded model from epoch {checkpoint['epoch']} with loss \
            {checkpoint['loss']:.4f}"
    )

    # Create evaluation dataset
    print("Creating evaluation dataset...")
    eval_dataset = BirdFrameDataset("data/bird", sequence_length=4)

    # Use a subset for evaluation
    eval_indices = np.linspace(0, len(eval_dataset) - 1, num_eval_samples, dtype=int)

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "generated"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "original"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "comparisons"), exist_ok=True)

    all_metrics = []

    print("Running evaluation...")
    for i, idx in enumerate(tqdm(eval_indices, desc="Evaluating samples")):
        # Get frames
        frames = eval_dataset[idx].unsqueeze(0)  # Add batch dimension
        batch_size = frames.shape[0]

        # Encode frames with VAE
        latents = encode_frames_with_vae(frames, vae, device)

        # Use first 3 frames as conditioning, predict the 4th
        prev_latents = latents[:, :-1]  # (B, 3, 4, 64, 64)
        target_latents = latents[:, -1]  # (B, 4, 64, 64)

        # Flatten previous latents for model input
        B_prev, T_prev, C_prev, H_prev, W_prev = prev_latents.shape
        prev_flat = prev_latents.view(B_prev, T_prev * H_prev * W_prev, C_prev)

        # Generate new frame using diffusion sampling
        generated_latents_flat = diffusion_sampling(
            model, scheduler, prev_flat, batch_size, device, num_inference_steps
        )

        # Reshape generated latents back to spatial format
        generated_latents = generated_latents_flat.view(batch_size, 4, 64, 64)

        # Decode both target and generated latents to images
        original_images = decode_latents_with_vae(target_latents, vae, device)
        generated_images = decode_latents_with_vae(generated_latents, vae, device)

        # Compute metrics
        metrics = compute_metrics(original_images, generated_images)
        all_metrics.append(metrics)

        print(
            f"Sample {i+1}: MSE={metrics['mse']:.4f}, PSNR={metrics['psnr']:.2f}, \
                SSIM={metrics['ssim']:.4f}"
        )

        # Save individual frames
        save_individual_frames(
            original_images,
            os.path.join(output_dir, "original"),
            f"original_sample_{i}",
        )
        save_individual_frames(
            generated_images,
            os.path.join(output_dir, "generated"),
            f"generated_sample_{i}",
        )

        # Create comparison visualization
        comparison_path = os.path.join(output_dir, "comparisons", f"comparison_sample_{i}.png")
        visualize_results(original_images, generated_images, comparison_path)

    # Compute average metrics
    avg_metrics = {metric: np.mean([m[metric] for m in all_metrics]) for metric in all_metrics[0].keys()}

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Average MSE: {avg_metrics['mse']:.6f}")
    print(f"Average PSNR: {avg_metrics['psnr']:.2f} dB")
    print(f"Average SSIM: {avg_metrics['ssim']:.4f}")
    print("=" * 50)

    # Save metrics to file
    metrics_file = os.path.join(output_dir, "metrics.txt")
    with open(metrics_file, "w") as f:
        f.write("Evaluation Metrics\n")
        f.write("==================\n\n")
        f.write(f"Model checkpoint: {checkpoint_path}\n")
        f.write(f"Number of samples: {num_eval_samples}\n")
        f.write(f"Inference steps: {num_inference_steps}\n\n")
        f.write("Average Metrics:\n")
        f.write(f"MSE: {avg_metrics['mse']:.6f}\n")
        f.write(f"PSNR: {avg_metrics['psnr']:.2f} dB\n")
        f.write(f"SSIM: {avg_metrics['ssim']:.4f}\n\n")
        f.write("Per-sample Metrics:\n")
        for i, metrics in enumerate(all_metrics):
            f.write(
                f"Sample {i+1}: MSE={metrics['mse']:.6f}, PSNR={metrics['psnr']:.2f}, \
                    SSIM={metrics['ssim']:.4f}\n"
            )

    print(f"\nResults saved to {output_dir}/")
    print(f"Metrics saved to {metrics_file}")
    print("Evaluation completed!")


if __name__ == "__main__":
    main()
