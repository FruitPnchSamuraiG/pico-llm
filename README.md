# pico-llm

A small transformer for math reasoning, trained on Orca-Math and fine-tuned on GSM8K.

Includes interpretability tools for analyzing attention patterns and neuron activations. Supports models from 10M to 1.5B parameters.

## Quick Start

```bash
# 1. Train base model on Orca-Math (~8-10 hours)
bash scripts/train.sh orca

# 2. Fine-tune on GSM8K with RL (~12-14 hours, auto-detects checkpoint)
bash scripts/train.sh gsm8k

# 3. Evaluate
python scripts/evaluation/eval_reasoning.py \
  --checkpoint /scratch/kk6081/picollm_extend/gsm8k_transformer_epoch8.pt
```

### Model Sizes

| Size | Parameters | Memory | Fits on GTX TITAN X (12GB) |
|------|-----------|--------|---------------------------|
| `small` | 10M | ~2GB | ✅ Yes |
| `medium` (default) | 40M | ~4GB | ✅ Yes |
| `gpt2-small` | 117M | ~8GB | ✅ Yes (tight) |

**Recommended for your hardware:** `small`, `medium`, or `gpt2-small` with reduced batch size.

```bash
# Train with gpt2-small (max for 12GB GPU)
BATCH=8 TRANSFORMER_SIZE=gpt2-small bash scripts/train.sh orca
```

## Configuration

You can control training parameters using environment variables:

```bash
# Model & Data
TRANSFORMER_SIZE=medium          # small, medium, gpt2-*
MAX_SAMPLES=100000              # Orca-Math examples to use

# Training
BATCH=16                        # Batch size (reduce if OOM)
EPOCHS=8                        # Number of epochs
LR=3e-4                         # Learning rate
DEVICE=cuda:0                   # GPU device

# Example: Train larger model with more epochs
TRANSFORMER_SIZE=gpt2-small EPOCHS=10 bash scripts/train.sh orca
```

## Training Modes

The unified `scripts/train.sh` handles all training workflows:

```bash
# Base training on Orca-Math
bash scripts/train.sh orca [model_size]

# Fine-tune on GSM8K (auto-detects latest checkpoint)
bash scripts/train.sh gsm8k [model_size]

# Train GPT-2 models
bash scripts/train.sh gpt2 [gpt2-small|gpt2-medium|gpt2-large|gpt2-xl]
```

### RL Refinement

GSM8K training includes RL refinement by default:
- Best-of-n sampling (generates 8 solutions per problem)
- Outcome-based rewards (correct answer gets positive reward)
- Runs for 400 steps after supervised training

```bash
# Skip RL refinement
RUN_RL=0 bash scripts/train.sh gsm8k

# More RL steps
RL_STEPS=600 bash scripts/train.sh gsm8k
```

## Interpretability

Analyze attention patterns, neuron activations, and internal representations:

```bash
# Generate interpretability data
python scripts/utils/interpret_transformer.py \
  --checkpoint /scratch/kk6081/picollm_extend/transformer_epoch8.pt \
  --analysis attention,logit_lens,neurons \
  --out_dir /scratch/kk6081/picollm_extend/interpretability

# View results in browser
python scripts/utils/interpretability_viewer.py \
  --root /scratch/kk6081/picollm_extend/interpretability \
  --port 8000
```

Outputs:
- Attention heatmaps per layer/head
- Logit lens (token predictions at each layer)
- Top neurons with max-activating contexts

Inspired by [Transformer Circuits](https://transformer-circuits.pub/)

## Repository Structure

```
pico-llm/
├── pico-llm.py              # Main training script
├── inference.py             # Text generation
├── plot_losses.py           # Visualize training curves
├── scripts/
│   ├── train.sh            # Unified training script (orca/gsm8k/gpt2)
│   ├── data_prep/          # Dataset preparation
│   │   ├── prepare_orca_math_data.py
│   │   └── prepare_hf_gsm8k_data.sh
│   ├── evaluation/         # Model evaluation
│   │   ├── eval_reasoning.py
│   │   └── rl_reasoning_outcome.py
│   └── utils/              # Utilities
│       ├── check_checkpoint_arch.py
│       ├── interpret_transformer.py
│       └── interpretability_viewer.py
└── data/                   # Training data (auto-downloaded)
```

## Additional Resources

- **Quick reference**: See `QUICKSTART.txt` for common commands
- **Datasets**: Orca-Math-200k, GSM8K
- **Inspiration**: [Transformer Circuits](https://transformer-circuits.pub/), [DeepSeek-R1](https://arxiv.org/abs/2501.12948)

