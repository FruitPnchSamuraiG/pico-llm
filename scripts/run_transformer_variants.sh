#!/usr/bin/env bash
# Train Transformer twice: with positional embeddings and without.
# Produce individual loss histories and a combined plot using plot_losses.py --multi.

set -e

DEVICE=${DEVICE:-cuda:1}
EPOCHS=${EPOCHS:-3}
BATCH=${BATCH:-16}
BLOCK_SIZE=${BLOCK_SIZE:-256}
EMBED=${EMBED:-512}
HEADS=${HEADS:-4}
BLOCKS=${BLOCKS:-4}
FF_MULT=${FF_MULT:-4}
PROMPT=${PROMPT:-"Once upon a time"}
TINYSTORIES_WEIGHT=${TINYSTORIES_WEIGHT:-0.5}
MAX_STEPS=""
if [[ "$1" == "--fast" ]]; then
  echo "⚡ Fast mode: 1 epoch, few steps, no TinyStories"
  EPOCHS=1
  MAX_STEPS="--max_steps_per_epoch 10"
  TINYSTORIES_WEIGHT=0.0
fi

echo "=========================================="
echo "Training Transformer dual variants (pos emb & no pos emb)"
echo "Device: $DEVICE"
echo "=========================================="

rm -f transformer_*epoch*.pt loss_histories*.pkl

train_variant() {
  local TAG=$1      # pos_emb | no_pos_emb
  local EXTRA=$2    # '' or '--no_pos_emb'
  echo "\n--- Training $TAG ---"
  python pico-llm.py \
    --enable_transformer --disable_lstm \
    --device_id $DEVICE \
    --batch_size $BATCH \
    --num_epochs $EPOCHS \
    --block_size $BLOCK_SIZE \
    --embed_size $EMBED \
    --transformer_heads $HEADS \
    --transformer_blocks $BLOCKS \
    --ff_mult $FF_MULT \
    --prompt "$PROMPT" \
    --tinystories_weight $TINYSTORIES_WEIGHT \
    $MAX_STEPS \
    $EXTRA

  # Rename transformer checkpoints
  for f in transformer_epoch*.pt; do
    [[ -f "$f" ]] || continue
    mv "$f" "${f/transformer_epoch/transformer_${TAG}_epoch}"
  done

  if [[ -f loss_histories.pkl ]]; then
    mv loss_histories.pkl loss_histories_${TAG}.pkl
  fi
}

train_variant pos_emb ""
train_variant no_pos_emb "--no_pos_emb"

if [[ -f loss_histories_pos_emb.pkl && -f loss_histories_no_pos_emb.pkl ]]; then
  python plot_losses.py --multi \
    loss_histories_pos_emb.pkl:transformer_pos_emb \
    loss_histories_no_pos_emb.pkl:transformer_no_pos_emb \
    --output training_losses_transformer_variants.png --smooth 10 || echo "Plot failed"
  echo "✅ Combined plot saved: training_losses_transformer_variants.png"
else
  echo "❌ Missing one of the variant loss history files; combined plot skipped"
fi

echo "\n=========================================="
echo "✅ Done. Files:"
echo "  transformer_pos_emb_epoch*.pt transformer_no_pos_emb_epoch*.pt"
echo "  loss_histories_pos_emb.pkl loss_histories_no_pos_emb.pkl"
echo "  training_losses_transformer_variants.png (combined)"
echo "Run 'python plot_losses.py --multi loss_histories_pos_emb.pkl:transformer_pos_emb loss_histories_no_pos_emb.pkl:transformer_no_pos_emb --output custom.png' for custom plot."
echo "=========================================="
