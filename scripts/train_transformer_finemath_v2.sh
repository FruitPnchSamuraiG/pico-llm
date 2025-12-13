#!/usr/bin/env bash
set -euo pipefail

# IMPROVED FineMath training with better hyperparameters
# Fixes: stagnation, low diversity, repetition issues

cd "$(dirname "$0")/.."
source /scratch/kk6081/ml_fall25/venv/bin/activate

DEVICE=${DEVICE:-cuda:0}
OUTDIR=${OUTDIR:-/scratch/kk6081/picollm_extend}
DATA_DIR=${DATA_DIR:-data}

# 🔧 IMPROVED HYPERPARAMETERS
TRANSFORMER_SIZE=${TRANSFORMER_SIZE:-medium}
BATCH=${BATCH:-16}
EPOCHS=${EPOCHS:-8}              # More epochs with better data
MAX_SAMPLES=${MAX_SAMPLES:-100000}
LR=${LR:-3e-4}                   # Higher LR (was 2e-4)
BLOCK_SIZE=${BLOCK_SIZE:-256}
VAL_SPLIT=${VAL_SPLIT:-0.05}

# Better LR scheduling
LR_SCHEDULE=${LR_SCHEDULE:-cosine}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-1000}  # More warmup (was 500)
LR_MIN_RATIO=${LR_MIN_RATIO:-0.2}         # Higher min LR (was 0.1)

# More frequent sampling to monitor progress
SAMPLE_EVERY_STEPS=${SAMPLE_EVERY_STEPS:-0}
SAMPLE_INTERVAL_SECONDS=${SAMPLE_INTERVAL_SECONDS:-300}  # Every 5min (was 10min)

echo "=========================================="
echo "🔢 IMPROVED FineMath Training"
echo "Device: $DEVICE"
echo "Output: $OUTDIR"
echo "Transformer size: $TRANSFORMER_SIZE"
echo "Batch: $BATCH, Epochs: $EPOCHS"
echo "LR: $LR (min ratio: $LR_MIN_RATIO)"
echo "Max samples: $MAX_SAMPLES"
echo "=========================================="

# Download simple FineMath data
TRAIN_FILE="$DATA_DIR/finemath_train.txt"
VAL_FILE="$DATA_DIR/finemath_val.txt"

if [[ ! -f "$TRAIN_FILE" || ! -f "$VAL_FILE" ]]; then
  echo ""
  echo "📥 Downloading SIMPLE FineMath examples (max 512 chars)..."
  python3 scripts/prepare_hf_finemath_data.py \
    --output_dir "$DATA_DIR" \
    --max_samples "$MAX_SAMPLES" \
    --min_length 100 \
    --max_length 512 \
    --val_split "$VAL_SPLIT"
  echo ""
fi

# Check if data looks reasonable
AVG_LEN=$(head -100 "$TRAIN_FILE" | awk '{print length}' | awk '{sum+=$1; count+=1} END {print int(sum/count)}')
echo "✓ Data check: Average line length = $AVG_LEN chars"
if [[ $AVG_LEN -gt 600 ]]; then
  echo "⚠️  WARNING: Data seems too complex (avg $AVG_LEN chars)"
  echo "   Consider regenerating with: bash scripts/regenerate_finemath_simple.sh"
  read -p "Continue anyway? (y/N) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
  fi
fi

TRAIN_COUNT=$(wc -l < "$TRAIN_FILE")
VAL_COUNT=$(wc -l < "$VAL_FILE")

echo "✓ FineMath data ready:"
echo "  Train: $TRAIN_COUNT examples (avg $AVG_LEN chars)"
echo "  Val:   $VAL_COUNT examples"
echo ""

PROMPT="Problem: If x + 5 = 12, then x ="

echo "🚀 Starting improved training..."
echo "   (monitoring: loss should drop below 2.0 by epoch 4)"
echo ""

python pico-llm.py \
  --enable_transformer --disable_lstm \
  --device_id "$DEVICE" \
  --checkpoint_dir "$OUTDIR" \
  --input_files "$TRAIN_FILE" "$VAL_FILE" \
  --tinystories_weight 0.0 \
  --batch_size "$BATCH" \
  --num_epochs "$EPOCHS" \
  --block_size "$BLOCK_SIZE" \
  --transformer_size "$TRANSFORMER_SIZE" \
  --learning_rate "$LR" \
  --val_split "$VAL_SPLIT" \
  --prompt "$PROMPT" \
  --sample_interval_seconds "$SAMPLE_INTERVAL_SECONDS" \
  --sample_every_steps "$SAMPLE_EVERY_STEPS" \
  --lr_schedule "$LR_SCHEDULE" \
  --lr_warmup_steps "$LR_WARMUP_STEPS" \
  --lr_min_ratio "$LR_MIN_RATIO"

echo ""
echo "=========================================="
echo "✅ Improved FineMath training complete!"
echo ""
echo "Expected results:"
echo "  - Final loss: < 2.0 (vs previous 2.35)"
echo "  - Better generations (no repetition)"
echo ""
echo "Checkpoints: $OUTDIR/transformer_finemath_epoch*.pt"
echo ""
echo "Next: Train GSM8K"
echo "  BASE_CKPT=$OUTDIR/transformer_finemath_epoch8.pt bash scripts/train_transformer_gsm8k.sh"
echo "=========================================="
