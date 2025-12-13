#!/usr/bin/env bash
set -euo pipefail

# Fast dev training (Transformer only) on cuda:0.
# Produces a checkpoint quickly so you can iterate on decoding/search.

cd "$(dirname "$0")/.."

source /scratch/kk6081/ml_fall25/venv/bin/activate

DEVICE=${DEVICE:-cuda:0}
OUTDIR=${OUTDIR:-/scratch/kk6081/picollm_extend}

# Larger defaults (override via env vars)
TRANSFORMER_SIZE=${TRANSFORMER_SIZE:-medium}
TINYSTORIES_SUBSET=${TINYSTORIES_SUBSET:-200000}  # Increased from 100k to 200k for better base

EPOCHS=${EPOCHS:-5}              # Increased from 1 to 5 for stronger base model
MAX_STEPS=${MAX_STEPS:-999999}   # Removed limit - train on full subset
BATCH=${BATCH:-16}
LR=${LR:-2e-4}
VAL_SPLIT=${VAL_SPLIT:-0.05}
BLOCK_SIZE=${BLOCK_SIZE:-256}

# Faster training knobs (override via env vars)
SAMPLE_INTERVAL_SECONDS=${SAMPLE_INTERVAL_SECONDS:-300}
SAMPLE_EVERY_STEPS=${SAMPLE_EVERY_STEPS:-0}
LR_SCHEDULE=${LR_SCHEDULE:-cosine}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-200}
LR_MIN_RATIO=${LR_MIN_RATIO:-0.1}

PROMPT=${PROMPT:-"Once upon a time"}

echo "=========================================="
echo "⚡ Fast Transformer Training"
echo "Device: $DEVICE"
echo "transformer_size=$TRANSFORMER_SIZE tinystories_subset=$TINYSTORIES_SUBSET"
echo "batch=$BATCH epochs=$EPOCHS max_steps_per_epoch=$MAX_STEPS lr=$LR"
echo "val_split=$VAL_SPLIT"
echo "sample_interval_seconds=$SAMPLE_INTERVAL_SECONDS sample_every_steps=$SAMPLE_EVERY_STEPS"
echo "lr_schedule=$LR_SCHEDULE lr_warmup_steps=$LR_WARMUP_STEPS lr_min_ratio=$LR_MIN_RATIO"
echo "outdir=$OUTDIR"
echo "=========================================="

python pico-llm.py \
  --enable_transformer --disable_lstm \
  --device_id "$DEVICE" \
  --checkpoint_dir "$OUTDIR" \
  --transformer_size "$TRANSFORMER_SIZE" \
  --tinystories_train_subset_size "$TINYSTORIES_SUBSET" \
  --batch_size "$BATCH" --num_epochs "$EPOCHS" --max_steps_per_epoch "$MAX_STEPS" \
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
