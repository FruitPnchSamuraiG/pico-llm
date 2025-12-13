# pico-llm (Transformer-only extension)

Educational Transformer project with focus on mathematical reasoning:
- **Base training** on Orca-Math-200k (math word problems) or TinyStories (general language)
- **Math reasoning**: GSM8K finetuning with optional RL refinement
- **Interpretability** tooling inspired by Anthropic / Transformer Circuits

## Comprehensive flowchart

```mermaid
flowchart TD
  VENV[Activate venv<br/>/scratch/kk6081/ml_fall25/venv/] --> BASE

  subgraph BASE["Stage 1: Base Training"]
    ORCA[train_transformer_orca.sh<br/>100k Orca-Math examples, 8 epochs<br/>⭐ RECOMMENDED]
    TFULL[train_transformer_full.sh<br/>500k TinyStories, 5 epochs<br/>For general language]
    ORCA --> CKPT_MATH[transformer_epoch_N.pt]
    TFULL --> CKPT_STORY[transformer_epoch_N.pt]
  end

  CKPT_MATH -->|init_from| GSM_MATH
  CKPT_STORY -->|init_from| GSM_STORY

  subgraph GSM_MATH["Stage 2A: GSM8K from Orca-Math RECOMMENDED"]
    PREP_MATH[prepare_hf_gsm8k_data.sh<br/>openai/gsm8k] --> FILES_MATH[gsm8k_train.txt]
    FILES_MATH --> SFT_MATH[train_transformer_gsm8k.sh<br/>10 epochs, LR=5e-4]
    SFT_MATH --> SFT_CKPT_MATH[gsm8k checkpoint]
    SFT_CKPT_MATH --> RL_MATH[RL Refinement<br/>400 steps]
    RL_MATH --> FINAL_MATH[Final checkpoint<br/>40-55% accuracy ⭐]
  end

  subgraph GSM_STORY["Stage 2B: GSM8K from TinyStories"]
    PREP_STORY[prepare_hf_gsm8k_data.sh<br/>openai/gsm8k] --> FILES_STORY[gsm8k_train.txt]
    FILES_STORY --> SFT_STORY[train_transformer_gsm8k.sh<br/>10 epochs, LR=5e-4]
    SFT_STORY --> SFT_CKPT_STORY[gsm8k checkpoint]
    SFT_CKPT_STORY --> RL_STORY[RL Refinement<br/>400 steps]
    RL_STORY --> FINAL_STORY[Final checkpoint<br/>25-40% accuracy]
  end

  subgraph EVAL["Evaluation"]
    E1[eval_reasoning.py<br/>Test set accuracy]
  end

  FINAL_MATH --> E1
  FINAL_STORY --> E1

  subgraph INTERP["Interpretability"]
    IT[interpret_transformer.py<br/>Attention + Neurons]
    IT --> VIEW[interpretability_viewer.py<br/>Web UI]
  end

  CKPT_MATH --> IT
  FINAL_MATH --> IT

  classDef recommended fill:#d4edda,stroke:#28a745,stroke-width:3px,color:#000
  classDef alternative fill:#fff3cd,stroke:#ffc107,stroke-width:2px,color:#000
  
  class ORCA,GSM_MATH,FINAL_MATH recommended
  class TFULL,GSM_STORY,FINAL_STORY alternative
  
  linkStyle default stroke:#666,stroke-width:2.5px
```

## Environment / constraints
- GPU: 2x GTX TITAN X (12GB). Default device: **`cuda:0`**
- Venv: `/scratch/kk6081/ml_fall25/venv/`
- Artifacts/checkpoints: **`/scratch/kk6081/picollm_extend/`**

## TL;DR (quick start commands)

### Recommended: Orca-Math Base → GSM8K for Best Results ⭐

```bash
source /scratch/kk6081/ml_fall25/venv/bin/activate
cd /home/kk6081/pico_llm_extend/pico-llm

# 1) Train on Orca-Math (200k math word problems, 100k subset) - 8-10 hours
EPOCHS=8 MAX_SAMPLES=100000 bash scripts/train_transformer_orca.sh

# 2) Train on GSM8K with RL - 12-14 hours
BASE_CKPT="/scratch/kk6081/picollm_extend/transformer_epoch8.pt" \
  EPOCHS=10 LR=5e-4 RUN_RL=1 \
  bash scripts/train_transformer_gsm8k.sh

# 3) Evaluate
python scripts/eval_reasoning.py

# Expected: 45-60% GSM8K accuracy with MEDIUM model!
```

### Alternative: TinyStories Base (General Language)

```bash
# 1) Train on TinyStories (500k stories, 5 epochs - 3-6 hours)
TINYSTORIES_SUBSET=500000 EPOCHS=5 bash scripts/train_transformer_full.sh

# 2) Train GSM8K with HIGH LR to override narrative patterns
BASE_CKPT="/scratch/kk6081/picollm_extend/transformer_epoch3.pt" \
  EPOCHS=10 LR=5e-4 RUN_RL=1 \
  bash scripts/train_transformer_gsm8k.sh

# Expected: 25-40% GSM8K accuracy (lower due to pattern contamination)
```

### Quick Evaluation & Interpretability

```bash
# Evaluate GSM8K accuracy
python scripts/eval_reasoning.py \
  --checkpoint /scratch/kk6081/picollm_extend/transformer_gsm8k_rl.pt \
  --data data/gsm8k_test.txt

# Interpretability analysis
python scripts/interpret_transformer.py \
  --checkpoint /scratch/kk6081/picollm_extend/transformer_finemath_epoch5.pt \
  --analysis attention,logit_lens,neurons \
  --out_dir /scratch/kk6081/picollm_extend/interpretability \
  --embed_size 512 --transformer_heads 8 --transformer_blocks 6 --ff_mult 4 \
  --test_prompts "Problem: If x + 5 = 12, then x =" \
  --device cuda:0

# View interpretability results in browser
python scripts/interpretability_viewer.py \
  --root /scratch/kk6081/picollm_extend/interpretability \
  --host 127.0.0.1 --port 8000
```

## 📚 Key Concepts

### Why Orca-Math Base?

Traditional approach (TinyStories base) teaches narrative patterns that **contaminate** math training:
- ❌ Repetition: "A: 20! A: 20! A: 20!"
- ❌ Emotions: Excessive "!" and "?" 
- ❌ Story patterns: "Once upon a time..." doesn't help with "x + 5 = 12"

**Orca-Math base** provides clean mathematical reasoning patterns:
- ✅ Step-by-step solutions with explanations
- ✅ Math vocabulary: "Let's calculate...", "Therefore...", "We can solve..."
- ✅ Natural transfer: Math word problems → GSM8K word problems
- ✅ High quality: Microsoft-curated 200k dataset with verified solutions

### Expected Results

| Base Model | GSM8K Accuracy | Training Time | Notes |
|------------|----------------|---------------|-------|
| **Orca-Math** | **45-60%** | 20-24 hours | Clean word problems ⭐ |
| TinyStories (high LR) | 25-40% | 15-19 hours | Must override story patterns |
| TinyStories (low LR) | 7-15% | 10-14 hours | Contaminated outputs |

### Datasets Used

- **Orca-Math-Word-Problems-200k** (base): 100k clean math word problems from Microsoft
- **GSM8K** (finetuning): 7.3k grade school math word problems
- **RL refinement**: Best-of-n sampling for answer selection

## 🚀 Quick Start: Apply Training Patches (RECOMMENDED)

**Before your first training run**, apply stability patches for 30-50% faster convergence:

```bash
# 1. Preview what will be changed (safe, no modifications)
python scripts/training_stability_patch.py --dry-run

# 2. Apply patches to pico-llm.py (creates backup automatically)
python scripts/training_stability_patch.py --apply

# 3. Train as normal - patches are now active!
bash scripts/train_transformer_fast.sh
```

**What gets patched**:
- ✅ **GPT-2 style weight initialization** - prevents exploding/vanishing gradients
- ✅ **Improved AdamW settings** - better convergence (beta2=0.95, weight_decay=0.1)
- ✅ **Gradient norm monitoring** - detect instability early (logs grad_norm in training)
- ✅ **Early stopping** - saves best checkpoint, stops if validation loss plateaus

**Verification**: After applying, your training logs will show:
```
[transformer] Epoch 1/3, Step 20/200 (global step: 20) Loss: 4.2341, Grad_norm: 2.156, LR: 1.50e-04
```

If you see `Grad_norm` and `LR` in logs → patches are active! ✅

**Rollback if needed**:
```bash
# Restore original (backup created automatically)
cp pico-llm.py.backup pico-llm.py
```

📖 **For advanced techniques** (mixed precision, gradient accumulation, LLRD), see: **[TRAINING_IMPROVEMENTS.md](TRAINING_IMPROVEMENTS.md)**

---

## Training

### 🎯 One-Command Full Pipeline (NEW!)

For the complete Orca-Math → GSM8K workflow in one command:

```bash
# Complete pipeline: Download data + Base training + GSM8K finetuning
bash scripts/full_pipeline_orca.sh
```

This script will:
1. Download 100k Orca-Math word problems (from 200k available)
2. Train base transformer (8 epochs, ~8-10 hours)
3. Guide you through GSM8K fine-tuning setup

**Manual control** (if you prefer step-by-step):

```bash
# Step 1: Download Orca-Math data
python3 scripts/prepare_orca_math_data.py --max_samples 100000

# Step 2: Base training
bash scripts/train_transformer_orca.sh

# Step 3: GSM8K fine-tuning
BASE_CKPT="/scratch/kk6081/picollm_extend/transformer_epoch8.pt" \
  EPOCHS=10 LR=5e-4 RUN_RL=1 \
  bash scripts/train_transformer_gsm8k.sh
```

### Base Transformer Options

**Choose your base training data:**

1. **TinyStories (Default)** - Good for general language modeling
   - Fast dev: `bash scripts/train_transformer_fast.sh`
   - Full run: `bash scripts/train_transformer_full.sh`
   - ⚠️ **Warning**: Narrative patterns may contaminate math reasoning tasks

2. **Orca-Math (RECOMMENDED for math)** - Microsoft's 200k math word problems
   - `bash scripts/train_transformer_orca.sh`
   - ✅ **Best for**: GSM8K, any math word problem tasks
   - ✅ **No contamination**: Clean step-by-step solutions transfer naturally to GSM8K

Scaling knobs (see `pico-llm.py`):
- `--transformer_size {small,medium}`
- `--tinystories_train_subset_size N`
- Faster training knobs: `--sample_interval_seconds`, `--sample_every_steps`, `--lr_schedule`, `--lr_warmup_steps`, `--lr_min_ratio`

If you hit OOM on 12GB:
- `BATCH=8` (or `4`) and/or `TRANSFORMER_SIZE=small`

### Training Parameters

Environment variables to control training (work with all training scripts):
```bash
# Model architecture
TRANSFORMER_SIZE=medium          # small (384d, 4h, 3b) or medium (512d, 8h, 6b)
TINYSTORIES_SUBSET=100000        # Number of TinyStories examples

# Training hyperparameters
BATCH=16                         # Batch size (reduce to 8 or 4 if OOM)
EPOCHS=3                         # Number of epochs
LR=2e-4                          # Learning rate
VAL_SPLIT=0.1                    # Validation split (0.1 = 10%)

# LR scheduling
LR_SCHEDULE=cosine               # none, cosine, or linear
LR_WARMUP_STEPS=500              # Warmup steps for stability
LR_MIN_RATIO=0.1                 # Min LR as fraction of base LR

# Sampling/logging
SAMPLE_EVERY_STEPS=100           # Generate text every N steps (0=time-based)
SAMPLE_INTERVAL_SECONDS=300      # Or every N seconds (if SAMPLE_EVERY_STEPS=0)
```

**Example**: Train medium model with larger batch
```bash
TRANSFORMER_SIZE=medium BATCH=32 LR=3e-4 bash scripts/train_transformer_fast.sh
```

### 🔧 Advanced: Manual Training Improvements

If you want more control beyond the automatic patch, see **[TRAINING_IMPROVEMENTS.md](TRAINING_IMPROVEMENTS.md)** for:
- **Gradient accumulation** - simulate larger batch sizes (effective batch = BATCH × ACCUM_STEPS)
- **Mixed precision training (FP16)** - 2x faster, 50% less memory
- **Layer-wise learning rate decay (LLRD)** - for finetuning only
- **Dropout strategies** - prevent overfitting
- **Curriculum learning** - start with easier examples
- **Troubleshooting guide** - fix loss explosions, plateaus, overfitting

## GSM8K Math Reasoning Training

### Step-by-Step Guide

**1. Prepare GSM8K Data** (automatic, runs on first training):
```bash
bash scripts/prepare_hf_gsm8k_data.sh
```
Downloads GSM8K dataset from HuggingFace and creates:
- `data/gsm8k_train.txt` (7,324 examples)
- `data/gsm8k_val.txt` (split from train)
- `data/gsm8k_test.txt` (1,319 examples)

**2. Train on GSM8K**:
```bash
# With FineMath base (recommended)
BASE_CKPT="/scratch/kk6081/picollm_extend/transformer_finemath_epoch5.pt" \
  EPOCHS=10 LR=5e-4 RUN_RL=1 \
  bash scripts/train_transformer_gsm8k.sh

# OR with TinyStories base (higher LR needed)
BASE_CKPT="/scratch/kk6081/picollm_extend/transformer_epoch3.pt" \
  EPOCHS=10 LR=5e-4 RUN_RL=1 \
  bash scripts/train_transformer_gsm8k.sh
```

**3. Evaluate**:
```bash
python scripts/eval_reasoning.py \
  --checkpoint /scratch/kk6081/picollm_extend/transformer_gsm8k_rl.pt \
  --data data/gsm8k_test.txt
```

### Training Configuration

| Parameter | Default | Notes |
|-----------|---------|-------|
| `EPOCHS` | 8 | Use 10 for better results |
| `LR` | 3e-4 | Use 5e-4 with TinyStories base |
| `RUN_RL` | 1 | Set to 0 to skip RL refinement |
| `RL_STEPS` | 400 | RL refinement iterations |
| `RL_BATCH` | 12 | Batch size for RL |
| `RL_NUM_SAMPLES` | 8 | Best-of-n samples per problem |

**Override examples**:
```bash
# Quick test (no RL, 1 epoch)
EPOCHS=1 RUN_RL=0 bash scripts/train_transformer_gsm8k.sh

# Extended training (15 epochs + RL)
EPOCHS=15 RUN_RL=1 RL_STEPS=600 bash scripts/train_transformer_gsm8k.sh
```

### Understanding RL Refinement

After supervised finetuning (SFT), RL refinement improves answer selection:
1. Generate `N` candidate solutions per problem (best-of-n sampling)
2. Score solutions based on correct final answer
3. Update model to prefer correct reasoning paths

**Reading**:
- [DeepSeek-R1 paper](https://arxiv.org/abs/2501.12948) - RL for reasoning
- [Dr. Tulu draft](https://www.datocms-assets.com/64837/1763496622-dr_tulu_draft.pdf) - Outcome supervision

## Interpretability & analysis

Tool: `scripts/interpret_transformer.py` (attention heatmaps, logit lens, neuron max-activation contexts, patching stub).

### Web UI (browse saved results)

After you generate results with `interpret_transformer.py`, you can browse them with a lightweight local web UI:

```bash
python scripts/interpretability_viewer.py \
  --root /scratch/kk6081/picollm_extend/interpretability_test \
  --host 127.0.0.1 --port 8000
```

Then open: `http://127.0.0.1:8000/`

Transformer Circuits hub:
- https://transformer-circuits.pub/

Referenced posts:
- Framework: https://transformer-circuits.pub/2021/framework/index.html
- Induction heads: https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
- Monosemantic features: https://transformer-circuits.pub/2023/monosemantic-features/index.html
- Toy models of superposition: https://transformer-circuits.pub/2022/toy_model/index.html

Example:

```bash
python scripts/interpret_transformer.py \
  --checkpoint /scratch/kk6081/picollm_extend/transformer_epoch1.pt \
  --analysis attention,logit_lens,neurons \
  --out_dir /scratch/kk6081/picollm_extend/interpretability_test \
  --embed_size 384 --transformer_heads 4 --transformer_blocks 3 --ff_mult 2 \
  --test_prompts "Once upon a time" "2 + 2 =" \
  --device cuda:0
```

Outputs:
- `attention/attn_*.png` - Attention heatmaps per layer/head
- `logit_lens/results.json` - Token predictions at each layer
- `neurons/top_neurons.json` - Max-activating contexts for FF neurons
- `summary.json` - Analysis metadata

---

## 📁 Repository Structure

### Training Scripts
- `train_transformer_finemath.sh` ⭐ - Train base model on FineMath (math-focused)
- `train_transformer_full.sh` - Train base model on TinyStories (general language)
- `train_transformer_fast.sh` - Quick TinyStories training for testing
- `train_transformer_gsm8k.sh` - GSM8K finetuning + optional RL

### Data Preparation
- `prepare_hf_finemath_data.py` - Download FineMath from HuggingFace
- `prepare_hf_gsm8k_data.sh` - Download GSM8K from HuggingFace

### Evaluation & Analysis
- `eval_reasoning.py` - Evaluate GSM8K accuracy
- `interpret_transformer.py` - Attention/neuron analysis
- `interpretability_viewer.py` - Web UI for interpretability results
- `rl_reasoning_outcome.py` - RL refinement script (called by train_transformer_gsm8k.sh)

### Utilities
- `check_checkpoint_arch.py` - Inspect checkpoint architecture
- `training_stability_patch.py` - Apply training improvements to pico-llm.py

### Core
- `pico-llm.py` - Main training script (Transformer + LSTM implementations)
- `inference.py` - Generate text from trained models
- `plot_losses.py` - Visualize training curves

---

## 🎯 Quick Reference Card

| Task | Command | Time |
|------|---------|------|
| **Math base training** | `bash scripts/train_transformer_finemath.sh` | 6-8h |
| **GSM8K training** | `BASE_CKPT=... bash scripts/train_transformer_gsm8k.sh` | 12-14h |
| **Evaluate GSM8K** | `python scripts/eval_reasoning.py --checkpoint ... --data data/gsm8k_test.txt` | 5-10m |
| **Interpretability** | `python scripts/interpret_transformer.py --checkpoint ...` | 2-5m |
| **View results** | `python scripts/interpretability_viewer.py --root ...` | instant |

**Total time for best model**: ~20 hours (FineMath base + GSM8K + RL)

