#!/usr/bin/env bash
set -euo pipefail

# Full training (Transformer only) on cuda:0.
# This runs WITHOUT --max_steps_per_epoch so each epoch is a full pass through the dataset.

cd "$(dirname "$0")/.."

source /scratch/kk6081/ml_fall25/venv/bin/activate

DEVICE=${DEVICE:-cuda:0}
OUTDIR=${OUTDIR:-/scratch/kk6081/picollm_extend}

# Larger defaults (override via env vars)
TRANSFORMER_SIZE=${TRANSFORMER_SIZE:-medium}
TINYSTORIES_SUBSET=${TINYSTORIES_SUBSET:-200000}

EPOCHS=${EPOCHS:-3}
BATCH=${BATCH:-16}
LR=${LR:-2e-4}
VAL_SPLIT=${VAL_SPLIT:-0.05}
BLOCK_SIZE=${BLOCK_SIZE:-256}

PROMPT=${PROMPT:-"Once upon a time"}

echo "=========================================="
echo "🏋️  Full Transformer Training"
echo "Device: $DEVICE"
echo "outdir=$OUTDIR"
echo "transformer_size=$TRANSFORMER_SIZE tinystories_subset=$TINYSTORIES_SUBSET"
echo "batch=$BATCH epochs=$EPOCHS lr=$LR"
echo "val_split=$VAL_SPLIT"
echo "(no max_steps_per_epoch)"
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
  --prompt "$PROMPT"

echo "\n✅ Done. Checkpoints: $OUTDIR/transformer_epoch*.pt"
