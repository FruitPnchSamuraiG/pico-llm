#!/usr/bin/env bash
set -euo pipefail

# Train a LARGE transformer for better GSM8K performance
# LARGE: 768d, 12h, 12 blocks, ff_mult=4 (~50M params)

cd "$(dirname "$0")/.."
source /scratch/kk6081/ml_fall25/venv/bin/activate

DEVICE=${DEVICE:-cuda:0}
OUTDIR=${OUTDIR:-/scratch/kk6081/picollm_extend_large}

EPOCHS=${EPOCHS:-5}
MAX_STEPS=${MAX_STEPS:-999999}
BATCH=${BATCH:-8}  # Smaller batch for larger model
LR=${LR:-2e-4}
BLOCK_SIZE=${BLOCK_SIZE:-256}
TINYSTORIES_SUBSET=${TINYSTORIES_SUBSET:-200000}

echo "=========================================="
echo "🚀 LARGE Transformer Training"
echo "Architecture: 768d, 12h, 12 blocks (CUSTOM)"
echo "Device: $DEVICE"
echo "Output: $OUTDIR"
echo "=========================================="

mkdir -p "$OUTDIR"

python pico-llm.py \
  --enable_transformer --disable_lstm \
  --device_id "$DEVICE" \
  --checkpoint_dir "$OUTDIR" \
  --tinystories_train_subset_size "$TINYSTORIES_SUBSET" \
  --batch_size "$BATCH" --num_epochs "$EPOCHS" --max_steps_per_epoch "$MAX_STEPS" \
  --block_size "$BLOCK_SIZE" \
  --embed_dim 768 \
  --n_heads 12 \
  --n_blocks 12 \
  --ff_mult 4 \
  --learning_rate "$LR" \
  --val_split 0.05 \
  --prompt "Once upon a time" \
  --sample_interval_seconds 600 \
  --lr_schedule cosine \
  --lr_warmup_steps 200 \
  --lr_min_ratio 0.1

echo ""
echo "✅ LARGE model training complete!"
echo "Checkpoints: $OUTDIR/transformer_epoch*.pt"
echo ""
echo "Next steps:"
echo "  1. Use curriculum training: OUTDIR=$OUTDIR bash scripts/train_curriculum_math.sh"
echo "  2. Or directly finetune: BASE_CKPT=\$(ls -1 $OUTDIR/transformer_epoch*.pt | tail -1) \\"
echo "     EMBED=768 HEADS=12 BLOCKS=12 FF_MULT=4 bash scripts/train_transformer_gsm8k.sh"
