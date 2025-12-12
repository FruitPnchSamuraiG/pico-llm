# pico-llm
CSCI-GA.2565 — Pico-LLM

## Overview
This repository contains implementations of three language modeling approaches:
1. **K-gram MLP**: Fixed-window feedforward model with optimized vectorized operations
2. **LSTM**: Recurrent neural network with long short-term memory
3. **Transformer**: Attention-based decoder-only model (GPT-style)

### Key Features
- **Optimized K-gram MLP**: Uses `unfold()` for vectorized sliding window processing (much faster than nested loops)
- **Helper Scripts**:
  - `inference.py`: Run inference / decoding experiments
  - `plot_losses.py`: Plot training curves
  - `train_all_models.sh`: Convenient training script for all models

## Environment / constraints (project settings)
- GPU: 2× GTX TITAN X (12GB); project runs on **`cuda:0`** by default.
- Venv: `/scratch/kk6081/ml_fall25/venv/`
- Checkpoints/artifacts: **`/scratch/kk6081/picollm_extend/`** (see `--checkpoint_dir`)
- Focus: **Transformer-only improvements** (test-time search + reasoning finetune)

## High-level flow (Transformer-only)

```mermaid
flowchart TD
  A[Train base Transformer<br/>TinyStories + optional text] -->|checkpoint| B[Decode / Test-Time Search<br/>greedy / nucleus / beam / lookahead (LNS)]
  B --> C[Decode metrics<br/>distinct-1/2, rep-4, wall time]
  A -->|init_from| D[Reasoning finetune<br/>OpenThoughts-114k → text lines]
  D --> E[Reasoning eval<br/>task-specific accuracy]
```

## Quick Start

Activate the required environment:

```bash
source /scratch/kk6081/ml_fall25/venv/bin/activate
```

### 1) Fast dev: train a small Transformer checkpoint

```bash
bash scripts/train_transformer_fast.sh
```

Outputs (default):
- `/scratch/kk6081/picollm_extend/transformer_epoch*.pt`
- `/scratch/kk6081/picollm_extend/loss_histories.pkl`

### 2) Test-time decoding/search experiments

`inference.py` supports these decoding modes (Transformer):
- `--decode greedy`
- `--decode nucleus`
- `--decode beam`
- `--decode lookahead` (Lookahead Nucleus Search / LNS)

Example grid (fast settings):

```bash
bash scripts/run_tts_grid.sh transformer_epoch1.pt "Once upon a time" --fast
```

### 3) Reasoning finetune using an existing dataset (Hugging Face)

Default reasoning dataset:
- `open-thoughts/OpenThoughts-114k`

Finetune script:

```bash
bash scripts/train_transformer_reasoning.sh
```

Notes:
- Uses `--init_from /scratch/kk6081/picollm_extend/transformer_epoch1.pt` by default.
- Writes finetune checkpoints in an isolated subdir then copies them back as:
  - `/scratch/kk6081/picollm_extend/transformer_reasoning_transformer_epoch*.pt`

### 4) Reasoning evaluation (accuracy)

```bash
python scripts/eval_reasoning.py \
  --checkpoint /scratch/kk6081/picollm_extend/transformer_reasoning_transformer_epoch1.pt \
  --data data/open_thoughts_val.txt \
  --decode greedy
```

You can compare test-time strategies on the reasoning prompts too:
- `--decode nucleus`
- `--decode beam`
- `--decode lookahead`

## What is Lookahead Nucleus Search (LNS)?

At each generation step:
1. Compute the model distribution for the next token.
2. Take the top-`K` candidate tokens by log-probability.
3. For each candidate, do a short rollout of length `H` (sampling with nucleus/top-p).
4. Score each candidate:

**Score = average logprob(rollout) − rep_penalty * repetition(rollout)**

Then pick the candidate with the best score.

## Normalization
1. Toggle Pre/Post-Normalization by setting `--norm_type=pre` (default) or `--norm_type=post`
2. Enable SGD with no warmup by setting `--warmup=no`
3. Enable SGD with warmup by setting `--warmup=yes`

## Interpretability & Analysis

### Attention Head Visualization

Visualize what Transformer attention heads are learning by saving heatmaps for any prompt:

```bash
python3 pico-llm.py \
  --enable_transformer --disable_lstm \
  --device_id cpu \
  --tinystories_weight 0.0 --input_files 3seqs.txt \
  --block_size 128 --batch_size 4 --num_epochs 1 --max_steps_per_epoch 2 \
  --save_attention_for_prompt --attention_outdir attn_plots_pos \
  --prompt "Once upon a time"
```

## Requirements
- Python 3.8+
- PyTorch
- tiktoken
- datasets
- matplotlib

