#!/usr/bin/env bash
set -euo pipefail

# Full training (Transformer only) on cuda:0.
# This runs WITHOUT --max_steps_per_epoch so each epoch is a full pass through the dataset.

cd "$(dirname "$0")/.."

source /scratch/kk6081/ml_fall25/venv/bin/activate

DEVICE=${DEVICE:-cuda:0}
OUTDIR=${OUTDIR:-/scratch/kk6081/picollm_extend}

# Larger/standard defaults (still 12GB-friendly; override via env vars)
BLOCK_SIZE=${BLOCK_SIZE:-256}
EMBED=${EMBED:-384}
HEADS=${HEADS:-4}
BLOCKS=${BLOCKS:-4}
FF_MULT=${FF_MULT:-2}
BATCH=${BATCH:-16}
EPOCHS=${EPOCHS:-3}
LR=${LR:-2e-4}
TINYSTORIES_WEIGHT=${TINYSTORIES_WEIGHT:-0.5}
VAL_SPLIT=${VAL_SPLIT:-0.1}

PROMPT=${PROMPT:-"Once upon a time"}

echo "=========================================="
echo "🏋️  Full Transformer Training"
echo "Device: $DEVICE"
echo "outdir=$OUTDIR"
echo "block_size=$BLOCK_SIZE embed=$EMBED heads=$HEADS blocks=$BLOCKS ff_mult=$FF_MULT"
echo "batch=$BATCH epochs=$EPOCHS lr=$LR"
echo "tinystories_weight=$TINYSTORIES_WEIGHT val_split=$VAL_SPLIT"
echo "(no max_steps_per_epoch)"
echo "=========================================="

python pico-llm.py \
  --enable_transformer --disable_lstm \
  --device_id "$DEVICE" \
  --checkpoint_dir "$OUTDIR" \
  --batch_size "$BATCH" --num_epochs "$EPOCHS" \
  --block_size "$BLOCK_SIZE" --embed_size "$EMBED" \
  --transformer_heads "$HEADS" --transformer_blocks "$BLOCKS" --ff_mult "$FF_MULT" \
  --learning_rate "$LR" \
  --tinystories_weight "$TINYSTORIES_WEIGHT" \
  --val_split "$VAL_SPLIT" \
  --prompt "$PROMPT"

echo "\n✅ Done. Checkpoints: $OUTDIR/transformer_epoch*.pt"
