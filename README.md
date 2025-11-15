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

### Training
```bash
# Quick CPU test
bash train_all_models.sh

# GPU training
bash train_all_models.sh --gpu

# Custom training
python pico-llm.py --enable_kgram --enable_transformer \
    --batch_size 16 --num_epochs 3 --device_id cuda:0
```

### Inference
```bash
python inference.py --model transformer \
    --checkpoint transformer_epoch3.pt \
    --prompt "Once upon a time" \
    --max_tokens 100
```

### Plot Results
```bash
python plot_losses.py --input loss_histories.pkl
```

## Requirements
- Python 3.8+
- PyTorch
- tiktoken
- datasets
- matplotlib

