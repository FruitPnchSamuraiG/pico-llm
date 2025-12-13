#!/usr/bin/env bash
set -euo pipefail

# Full training (Transformer only) on cuda:0.
# This runs WITHOUT --max_steps_per_epoch so each epoch is a full pass through the dataset.

cd "$(dirname "$0")/.."

source /scratch/kk6081/ml_fall25/venv/bin/activate

DEVICE=${DEVICE:-cuda:0}
OUTDIR=${OUTDIR:-/scratch/kk6081/picollm_extend}

# Larger defaults (override via env vars)
# IMPORTANT: For good GSM8K reasoning, use 500k-1M stories with 5-8 epochs
# Old defaults (200k×3) were too small for strong language modeling
TRANSFORMER_SIZE=${TRANSFORMER_SIZE:-medium}
TINYSTORIES_SUBSET=${TINYSTORIES_SUBSET:-500000}  # Increased from 200k to 500k

EPOCHS=${EPOCHS:-5}          # Increased from 3 to 5 for better base model
BATCH=${BATCH:-16}
LR=${LR:-2e-4}
VAL_SPLIT=${VAL_SPLIT:-0.05}
BLOCK_SIZE=${BLOCK_SIZE:-256}

PROMPT=${PROMPT:-"Once upon a time"}

# Faster training knobs (override via env vars)
SAMPLE_INTERVAL_SECONDS=${SAMPLE_INTERVAL_SECONDS:-600}
SAMPLE_EVERY_STEPS=${SAMPLE_EVERY_STEPS:-0}
LR_SCHEDULE=${LR_SCHEDULE:-cosine}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-500}
LR_MIN_RATIO=${LR_MIN_RATIO:-0.1}

echo "=========================================="
echo "🏋️  Full Transformer Training"
echo "Device: $DEVICE"
echo "outdir=$OUTDIR"
echo "transformer_size=$TRANSFORMER_SIZE tinystories_subset=$TINYSTORIES_SUBSET"
echo "batch=$BATCH epochs=$EPOCHS lr=$LR"
echo "val_split=$VAL_SPLIT"
echo "(no max_steps_per_epoch)"
echo "sample_interval_seconds=$SAMPLE_INTERVAL_SECONDS sample_every_steps=$SAMPLE_EVERY_STEPS"
echo "lr_schedule=$LR_SCHEDULE lr_warmup_steps=$LR_WARMUP_STEPS lr_min_ratio=$LR_MIN_RATIO"
echo "=========================================="

python pico-llm.py \
  --enable_transformer --disable_lstm \
  --device_id "$DEVICE" \
  --checkpoint_dir "$OUTDIR" \
  --transformer_size "$TRANSFORMER_SIZE" \
  --tinystories_train_subset_size "$TINYSTORIES_SUBSET" \
  --batch_size "$BATCH" --num_epochs "$EPOCHS" \
  --block_size "$BLOCK_SIZE" \
  --learning_rate "$LR" \
  --val_split "$VAL_SPLIT" \
  --prompt "$PROMPT" \
  --sample_interval_seconds "$SAMPLE_INTERVAL_SECONDS" \
  --sample_every_steps "$SAMPLE_EVERY_STEPS" \
  --lr_schedule "$LR_SCHEDULE" \
  --lr_warmup_steps "$LR_WARMUP_STEPS" \
  --lr_min_ratio "$LR_MIN_RATIO"

echo "\n✅ Done. Checkpoints: $OUTDIR/transformer_epoch*.pt"
