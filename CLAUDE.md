# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch implementation recreating "Navigation World Models" - a machine learning project for training conditional diffusion transformer (CDiT) models on sequential visual data. The project uses a VAE encoder/decoder with diffusion models for generating navigation-based visual sequences.

## Development Commands

### Environment Setup
```bash
pip install -r requirements.txt
```

### Training
```bash
python train.py
# Uses Hydra configuration from config/config.yaml
# Trains on bird frame sequences in data/bird/
# Saves checkpoints to outputs/ directory every 5 epochs
```

### Code Quality
```bash
# Formatting
black .
isort .

# Linting
ruff check .
```

## Architecture

### Core Model Components
- **CDiTModel** (`models/cdit_model.py`): Main conditional diffusion transformer
- **CDiTBlock** (`models/cdit_model.py`): Individual transformer block with self-attention, cross-attention, and feed-forward layers
- **ConditionalEncoding**: Positional encoding for actions (x, y, yaw) and temporal information
- **AdaLayerNorm/ScaledLayerNorm**: Adaptive normalization layers conditioned on temporal/action inputs

### Data Flow
1. Sequential image frames loaded via `BirdFrameDataset`
2. Images encoded to latent space using Stable Diffusion VAE (`stabilityai/sd-vae-ft-mse`)
3. Latents reshaped from (B, C, H, W) to (B, H*W, C) for transformer processing
4. Model predicts noise in diffusion training process
5. Uses DDPM scheduler with 1000 timesteps

### Key Architecture Details
- Input: 512x512 RGB images → 64x64x4 VAE latents → 4096x4 flattened tokens
- Model processes previous frames as conditioning via cross-attention
- Time and action conditioning via positional encodings and adaptive normalization
- Uses xFormers memory-efficient attention and rotary position embeddings

### Configuration
- Model config via Hydra/OmegaConf in `config/config.yaml`
- Pydantic-based config classes in `models/cdit_config.py`
- Default: 4 CDiT blocks, various encoding dimensions specified in config

### Training Data
- Expects sequential frame data in `data/bird/` directory
- Frame naming: `frame_XXXX.png` format
- Sequences of 5 frames used for training (4 conditioning + 1 target)