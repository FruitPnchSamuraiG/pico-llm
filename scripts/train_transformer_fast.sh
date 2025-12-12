#!/usr/bin/env bash
set -euo pipefail

# Fast dev training (Transformer only) on cuda:0.
# Produces a checkpoint quickly so you can iterate on decoding/search.

cd "$(dirname "$0")/.."

source /scratch/kk6081/ml_fall25/venv/bin/activate

DEVICE=${DEVICE:-cuda:0}
OUTDIR=${OUTDIR:-/scratch/kk6081/picollm_extend}

# Fast defaults (12GB TITAN X friendly)
BLOCK_SIZE=${BLOCK_SIZE:-256}
EMBED=${EMBED:-384}
HEADS=${HEADS:-4}
BLOCKS=${BLOCKS:-3}
FF_MULT=${FF_MULT:-2}
BATCH=${BATCH:-16}
EPOCHS=${EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:-200}
LR=${LR:-2e-4}
TINYSTORIES_WEIGHT=${TINYSTORIES_WEIGHT:-0.5}
VAL_SPLIT=${VAL_SPLIT:-0.05}

PROMPT=${PROMPT:-"Once upon a time"}

echo "=========================================="
echo "⚡ Fast Transformer Training"
echo "Device: $DEVICE"
echo "block_size=$BLOCK_SIZE embed=$EMBED heads=$HEADS blocks=$BLOCKS ff_mult=$FF_MULT"
echo "batch=$BATCH epochs=$EPOCHS max_steps_per_epoch=$MAX_STEPS lr=$LR"
echo "tinystories_weight=$TINYSTORIES_WEIGHT val_split=$VAL_SPLIT"
echo "outdir=$OUTDIR"
echo "=========================================="

python pico-llm.py \
  --enable_transformer --disable_lstm \
  --device_id "$DEVICE" \
  --checkpoint_dir "$OUTDIR" \
  --batch_size "$BATCH" --num_epochs "$EPOCHS" --max_steps_per_epoch "$MAX_STEPS" \
  --block_size "$BLOCK_SIZE" --embed_size "$EMBED" \
  --transformer_heads "$HEADS" --transformer_blocks "$BLOCKS" --ff_mult "$FF_MULT" \
  --learning_rate "$LR" \
  --tinystories_weight "$TINYSTORIES_WEIGHT" \
  --val_split "$VAL_SPLIT" \
  --prompt "$PROMPT"

echo "\n✅ Done. Checkpoints: $OUTDIR/transformer_epoch*.pt"
