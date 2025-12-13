#!/usr/bin/env bash
set -euo pipefail

# Curriculum learning for math reasoning:
# 1. Base training on TinyStories (5 epochs)
# 2. Simple arithmetic reasoning (3 epochs)
# 3. GSM8K full training (8 epochs)
# 4. RL refinement

cd "$(dirname "$0")/.."
source /scratch/kk6081/ml_fall25/venv/bin/activate

DEVICE=${DEVICE:-cuda:0}
OUTDIR=${OUTDIR:-/scratch/kk6081/picollm_extend}

echo "=========================================="
echo "📚 Curriculum Math Training Pipeline"
echo "Device: $DEVICE"
echo "Output: $OUTDIR"
echo "=========================================="

# Stage 1: Base training (5 epochs, 500k stories for better foundation)
echo ""
echo "🔹 Stage 1: Base Training (5 epochs on TinyStories)"
echo "   Using 500k stories for stronger language modeling foundation"
EPOCHS=5 TINYSTORIES_SUBSET=500000 MAX_STEPS=999999 \
  bash scripts/train_transformer_full.sh

# Stage 2: Simple reasoning warm-up (3 epochs)
echo ""
echo "🔹 Stage 2: Simple Arithmetic Reasoning (3 epochs)"
DATA_DIR=data
mkdir -p "$DATA_DIR"

# Download arithmetic curriculum data from HuggingFace
ARITH_TRAIN="$DATA_DIR/curriculum_arith_train.txt"
ARITH_VAL="$DATA_DIR/curriculum_arith_val.txt"

if [[ ! -f "$ARITH_TRAIN" || ! -f "$ARITH_VAL" ]]; then
  echo "⚠️  Arithmetic curriculum data not found. Downloading from HuggingFace..."
  python3 scripts/prepare_hf_arithmetic_data.py \
    --output_dir "$DATA_DIR" \
    --max_samples 5000 \
    --datasets "asdiv,simple"
fi

FT_SUBDIR="$OUTDIR/finetune_arith"
mkdir -p "$FT_SUBDIR"

# Find latest base checkpoint
BASE_CKPT=$(ls -1 "$OUTDIR"/transformer_epoch*.pt 2>/dev/null | \
  sed 's/.*transformer_epoch\([0-9]*\)\.pt/\1 &/' | \
  sort -rn | head -1 | awk '{print $2}')

if [[ -z "$BASE_CKPT" ]]; then
  echo "❌ No base checkpoint found!" >&2
  exit 1
fi

echo "Using base checkpoint: $BASE_CKPT"

python pico-llm.py \
  --enable_transformer --disable_lstm \
  --device_id "$DEVICE" \
  --checkpoint_dir "$FT_SUBDIR" \
  --init_from "$BASE_CKPT" \
  --tinystories_weight 0.0 \
  --custom_train_data "$ARITH_TRAIN" \
  --custom_val_data "$ARITH_VAL" \
  --batch_size 16 --num_epochs 3 --max_steps_per_epoch 999999 \
  --block_size 256 \
  --transformer_size medium \
  --learning_rate 1e-4 \
  --prompt "Q: What is 5 + 3? A:" \
  --sample_interval_seconds 600 \
  --lr_schedule cosine \
  --lr_warmup_steps 100 \
  --lr_min_ratio 0.1

# Copy checkpoint for next stage
ARITH_CKPT=$(ls -1 "$FT_SUBDIR"/transformer_epoch*.pt | tail -1)
cp "$ARITH_CKPT" "$OUTDIR/transformer_arith_final.pt"

# Stage 3: GSM8K training (8 epochs) + RL refinement (400 steps)
echo ""
echo "🔹 Stage 3: GSM8K Training (8 epochs SFT + 400 steps RL)"
BASE_CKPT="$OUTDIR/transformer_arith_final.pt" EPOCHS=8 RUN_RL=1 \
  bash scripts/train_transformer_gsm8k.sh

echo ""
echo "=========================================="
echo "✅ Curriculum training complete!"
echo ""
echo "Checkpoints created:"
echo "  - Arithmetic checkpoint: $OUTDIR/transformer_arith_final.pt"
echo "  - GSM8K SFT checkpoints: $OUTDIR/transformer_gsm8k_transformer_epoch*.pt"
echo "  - GSM8K RL checkpoint: $OUTDIR/transformer_gsm8k_rl.pt"
echo ""
echo "Next steps:"
echo "  1. Evaluate: python scripts/eval_reasoning.py"
echo "  2. Use best checkpoint for inference"
echo "=========================================="
