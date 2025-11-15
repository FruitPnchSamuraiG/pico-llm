# pico-llm
CSCI-GA.2565 — Pico-LLM

## Overview
This repository contains implementations of three language modeling approaches:
1. **K-gram MLP**: Fixed-window feedforward model with optimized vectorized operations
2. **LSTM**: Recurrent neural network with long short-term memory
3. **Transformer**: Attention-based decoder-only model (GPT-style)

## Branch Merge Information
This branch (`copilot/merge-kgram-mlp-section`) merges the best components from:
- **siva/pico-llm**: Optimized K-gram MLP using vectorized operations (unfold)
- **keshav-pico**: Better documentation, helper scripts, and utilities

### Key Features
- **Optimized K-gram MLP**: Uses `unfold()` for vectorized sliding window processing (much faster than nested loops)
- **Helper Scripts**:
  - `inference.py`: Run inference on trained models
  - `plot_losses.py`: Plot training curves
  - `train_all_models.sh`: Convenient training script for all models
- **Well-documented code**: Comprehensive comments explaining architecture and algorithms

## Quick Start

### Training with Multiple Configurations
The `train_all_models.sh` script runs 4 different hyperparameter configurations automatically:

```bash
# Quick CPU test - runs all 4 configurations
bash train_all_models.sh

# GPU training - runs all 4 configurations on GPU
bash train_all_models.sh --gpu
```

**Configurations:**
1. **baseline**: Small model (k=3, embed=256) - fastest, good for testing
2. **large_embed**: Larger embeddings (k=3, embed=512) - better capacity
3. **wide_context**: Wider context window (k=5, embed=384) - longer history
4. **deep_model**: Deeper architecture (k=4, 4 blocks, embed=384) - more layers

Each configuration generates:
- `training_losses_[config].png` - Loss comparison plot
- `loss_histories_[config].pkl` - Raw loss data
- `*_epoch*_[config].pt` - Model checkpoints

### Custom Training
```bash
python pico-llm.py --enable_kgram --enable_transformer \
    --batch_size 16 --num_epochs 3 --device_id cuda:0
```

### Inference
```bash
python inference.py --model transformer \
    --checkpoint transformer_epoch3_baseline.pt \
    --prompt "Once upon a time" \
    --max_tokens 100
```

### Plot Results
```bash
# Plot a specific configuration
python plot_losses.py --input loss_histories_baseline.pkl

# With smoothing and log scale
python plot_losses.py --input loss_histories_deep_model.pkl --smooth 20 --log
```

## Requirements
- Python 3.8+
- PyTorch
- tiktoken
- datasets
- matplotlib

