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
TINYSTORIES_SUBSET=${TINYSTORIES_SUBSET:-100000}

EPOCHS=${EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:-200}
BATCH=${BATCH:-16}
LR=${LR:-2e-4}
VAL_SPLIT=${VAL_SPLIT:-0.05}
BLOCK_SIZE=${BLOCK_SIZE:-256}

PROMPT=${PROMPT:-"Once upon a time"}

echo "=========================================="
echo "⚡ Fast Transformer Training"
echo "Device: $DEVICE"
echo "transformer_size=$TRANSFORMER_SIZE tinystories_subset=$TINYSTORIES_SUBSET"
echo "batch=$BATCH epochs=$EPOCHS max_steps_per_epoch=$MAX_STEPS lr=$LR"
echo "val_split=$VAL_SPLIT"
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
  --prompt "$PROMPT"

echo "\n✅ Done. Checkpoints: $OUTDIR/transformer_epoch*.pt"
