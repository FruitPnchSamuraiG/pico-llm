#!/usr/bin/env bash
set -euo pipefail

# Train transformer base model on Orca-Math-Word-Problems-200k
# This dataset provides clean, structured math word problems perfect for GSM8K transfer

cd "$(dirname "$0")/.."
source /scratch/kk6081/ml_fall25/venv/bin/activate

DEVICE=${DEVICE:-cuda:0}
OUTDIR=${OUTDIR:-/scratch/kk6081/picollm_extend}
DATA_DIR=${DATA_DIR:-data}

# Training hyperparameters
TRANSFORMER_SIZE=${TRANSFORMER_SIZE:-medium}
BATCH=${BATCH:-16}
EPOCHS=${EPOCHS:-8}              # 8 epochs for good coverage
MAX_SAMPLES=${MAX_SAMPLES:-100000}  # 100k from 200k available
LR=${LR:-3e-4}                   # Standard LR for math data
BLOCK_SIZE=${BLOCK_SIZE:-256}
VAL_SPLIT=${VAL_SPLIT:-0.05}

# LR scheduling
LR_SCHEDULE=${LR_SCHEDULE:-cosine}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-1000}
LR_MIN_RATIO=${LR_MIN_RATIO:-0.2}  # Don't decay too low

# Sampling/logging
SAMPLE_EVERY_STEPS=${SAMPLE_EVERY_STEPS:-0}
SAMPLE_INTERVAL_SECONDS=${SAMPLE_INTERVAL_SECONDS:-300}

echo "=========================================="
echo "🧮 Orca-Math Base Training"
echo "=========================================="
echo "Device: $DEVICE"
echo "Output: $OUTDIR"
echo "Transformer: $TRANSFORMER_SIZE"
echo "Batch: $BATCH, Epochs: $EPOCHS"
echo "LR: $LR (min ratio: $LR_MIN_RATIO)"
echo "Max samples: $MAX_SAMPLES from 200k available"
echo "=========================================="
echo ""

# Download Orca-Math data
TRAIN_FILE="$DATA_DIR/orca_math_train.txt"
VAL_FILE="$DATA_DIR/orca_math_val.txt"

if [[ ! -f "$TRAIN_FILE" || ! -f "$VAL_FILE" ]]; then
  echo "📥 Downloading Orca-Math-Word-Problems dataset..."
  echo "   (200k high-quality math word problems from Microsoft)"
  echo ""
  python3 scripts/prepare_orca_math_data.py \
    --output_dir "$DATA_DIR" \
    --max_samples "$MAX_SAMPLES" \
    --min_length 50 \
    --max_length 1024 \
    --val_split "$VAL_SPLIT"
  echo ""
fi

# Verify data quality
if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "❌ Error: Failed to download Orca-Math data"
  exit 1
fi

TRAIN_COUNT=$(wc -l < "$TRAIN_FILE")
VAL_COUNT=$(wc -l < "$VAL_FILE")
AVG_LEN=$(head -100 "$TRAIN_FILE" | awk '{print length}' | awk '{sum+=$1; count+=1} END {print int(sum/count)}')

echo "✓ Orca-Math data ready:"
echo "  Train: $TRAIN_COUNT examples"
echo "  Val:   $VAL_COUNT examples"
echo "  Avg length: $AVG_LEN chars"
echo ""

if [[ $AVG_LEN -lt 100 ]]; then
  echo "⚠️  WARNING: Examples seem too short (avg $AVG_LEN chars)"
  echo "   This might indicate a data format issue"
fi

PROMPT="Q: A bakery sells 5 cupcakes for \$3. How much would 20 cupcakes cost? A:"

echo "🚀 Starting Orca-Math training..."
echo "   Expected: Loss should drop below 1.5 by epoch 6"
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
echo "✅ Orca-Math training complete!"
echo "=========================================="
echo ""
echo "Checkpoints: $OUTDIR/transformer_epoch*.pt"
echo ""
echo "Next steps:"
echo "  1. Verify quality:"
echo "     python inference.py --checkpoint $OUTDIR/transformer_epoch8.pt \\"
echo "       --prompt 'Q: If x + 5 = 12, then x = ' \\"
echo "       --device $DEVICE --max_new_tokens 50"
echo ""
echo "  2. Train on GSM8K:"
echo "     BASE_CKPT=$OUTDIR/transformer_epoch8.pt \\"
echo "       EPOCHS=10 LR=5e-4 RUN_RL=1 \\"
echo "       bash scripts/train_transformer_gsm8k.sh"
echo ""
echo "Expected GSM8K accuracy: 45-60% (better than FineMath!)"
echo "=========================================="
