#!/usr/bin/env bash
set -euo pipefail

# Train transformer base model on FineMath instead of TinyStories
# This provides a better foundation for math reasoning tasks (GSM8K, arithmetic, etc.)

cd "$(dirname "$0")/.."
source /scratch/kk6081/ml_fall25/venv/bin/activate

DEVICE=${DEVICE:-cuda:0}
OUTDIR=${OUTDIR:-/scratch/kk6081/picollm_extend}
DATA_DIR=${DATA_DIR:-data}

# Training hyperparameters
TRANSFORMER_SIZE=${TRANSFORMER_SIZE:-medium}  # small or medium
BATCH=${BATCH:-16}
EPOCHS=${EPOCHS:-5}
MAX_SAMPLES=${MAX_SAMPLES:-100000}  # 100k FineMath examples
LR=${LR:-2e-4}
BLOCK_SIZE=${BLOCK_SIZE:-256}
VAL_SPLIT=${VAL_SPLIT:-0.05}

# LR scheduling
LR_SCHEDULE=${LR_SCHEDULE:-cosine}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-500}
LR_MIN_RATIO=${LR_MIN_RATIO:-0.1}

# Sampling/logging
SAMPLE_EVERY_STEPS=${SAMPLE_EVERY_STEPS:-0}
SAMPLE_INTERVAL_SECONDS=${SAMPLE_INTERVAL_SECONDS:-600}

echo "=========================================="
echo "🔢 Transformer Base Training on FineMath"
echo "Device: $DEVICE"
echo "Output: $OUTDIR"
echo "Transformer size: $TRANSFORMER_SIZE"
echo "Batch: $BATCH, Epochs: $EPOCHS"
echo "Max samples: $MAX_SAMPLES"
echo "=========================================="

# Download FineMath data if not exists
TRAIN_FILE="$DATA_DIR/finemath_train.txt"
VAL_FILE="$DATA_DIR/finemath_val.txt"

if [[ ! -f "$TRAIN_FILE" || ! -f "$VAL_FILE" ]]; then
  echo ""
  echo "📥 Downloading FineMath-4plus dataset from HuggingFace..."
  python3 scripts/prepare_hf_finemath_data.py \
    --output_dir "$DATA_DIR" \
    --max_samples "$MAX_SAMPLES" \
    --val_split "$VAL_SPLIT"
  echo ""
fi

# Count examples
TRAIN_COUNT=$(wc -l < "$TRAIN_FILE")
VAL_COUNT=$(wc -l < "$VAL_FILE")

echo "✓ FineMath data ready:"
echo "  Train: $TRAIN_COUNT examples"
echo "  Val:   $VAL_COUNT examples"
echo ""

# Sample prompt for math
PROMPT="Problem: If x + 5 = 12, then x ="

echo "Starting training..."
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
echo "✅ FineMath base training complete!"
echo ""
echo "Checkpoints saved to: $OUTDIR/transformer_finemath_epoch*.pt"
echo ""
echo "Next steps:"
echo "  1. Train on GSM8K: BASE_CKPT=$OUTDIR/transformer_finemath_epoch5.pt bash scripts/train_transformer_gsm8k.sh"
echo "  2. Or use curriculum: Update train_curriculum_math.sh to use FineMath checkpoint"
echo "=========================================="
