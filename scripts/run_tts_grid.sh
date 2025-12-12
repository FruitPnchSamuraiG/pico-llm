#!/usr/bin/env bash
set -euo pipefail

# Run a small decoding grid for the transformer checkpoint.
# Usage:
#   bash scripts/run_tts_grid.sh transformer_epoch3.pt "Once upon a time"

CKPT_IN=${1:?need checkpoint}
PROMPT=${2:-"Once upon a time"}
DEVICE=${DEVICE:-cuda:0}

# If user passes just a filename, default to OUTDIR
OUTDIR=${OUTDIR:-/scratch/kk6081/picollm_extend}
if [[ "$CKPT_IN" == /* ]]; then
  CKPT="$CKPT_IN"
else
  CKPT="$OUTDIR/$CKPT_IN"
fi

# Fast dev mode: bash scripts/run_tts_grid.sh ckpt "prompt" --fast
FAST=false
if [[ "${3:-}" == "--fast" ]]; then
  FAST=true
fi

# These MUST match the checkpoint's training hyperparams:
BLOCK_SIZE=${BLOCK_SIZE:-256}
EMBED=${EMBED:-384}
HEADS=${HEADS:-4}
BLOCKS=${BLOCKS:-3}
FF_MULT=${FF_MULT:-2}

MAXTOK=${MAXTOK:-128}
TOPP=${TOPP:-0.95}

if [[ "$FAST" == "true" ]]; then
  echo "⚡ FAST mode enabled"
  MAXTOK=${MAXTOK:-64}
fi

echo "Checkpoint: $CKPT"
echo "Prompt: $PROMPT"
echo "Device: $DEVICE"

echo "\n== nucleus =="
python inference.py --model transformer --checkpoint "$CKPT" --prompt "$PROMPT" \
  --device "$DEVICE" --decode nucleus --top_p "$TOPP" \
  --max_tokens "$MAXTOK" --block_size "$BLOCK_SIZE" --embed_size "$EMBED" \
  --transformer_heads "$HEADS" --transformer_blocks "$BLOCKS" --ff_mult "$FF_MULT"

echo "\n== beam =="
python inference.py --model transformer --checkpoint "$CKPT" --prompt "$PROMPT" \
  --device "$DEVICE" --decode beam --beam_width 4 \
  --max_tokens "$MAXTOK" --block_size "$BLOCK_SIZE" --embed_size "$EMBED" \
  --transformer_heads "$HEADS" --transformer_blocks "$BLOCKS" --ff_mult "$FF_MULT"

if [[ "$FAST" == "true" ]]; then
  # smaller grid for quick turnaround
  echo "\n== lookahead (FAST k=8,h=4) =="
  python inference.py --model transformer --checkpoint "$CKPT" --prompt "$PROMPT" \
    --device "$DEVICE" --decode lookahead --top_p "$TOPP" --lookahead_k 8 --lookahead_h 4 --rep_penalty 0.2 \
    --max_tokens "$MAXTOK" --block_size "$BLOCK_SIZE" --embed_size "$EMBED" \
    --transformer_heads "$HEADS" --transformer_blocks "$BLOCKS" --ff_mult "$FF_MULT"

  echo "\n== lookahead (FAST k=12,h=4) =="
  python inference.py --model transformer --checkpoint "$CKPT" --prompt "$PROMPT" \
    --device "$DEVICE" --decode lookahead --top_p "$TOPP" --lookahead_k 12 --lookahead_h 4 --rep_penalty 0.2 \
    --max_tokens "$MAXTOK" --block_size "$BLOCK_SIZE" --embed_size "$EMBED" \
    --transformer_heads "$HEADS" --transformer_blocks "$BLOCKS" --ff_mult "$FF_MULT"

  exit 0
fi

echo "\n== lookahead (k=8,h=8) =="
python inference.py --model transformer --checkpoint "$CKPT" --prompt "$PROMPT" \
  --device "$DEVICE" --decode lookahead --top_p "$TOPP" --lookahead_k 8 --lookahead_h 8 --rep_penalty 0.2 \
  --max_tokens "$MAXTOK" --block_size "$BLOCK_SIZE" --embed_size "$EMBED" \
  --transformer_heads "$HEADS" --transformer_blocks "$BLOCKS" --ff_mult "$FF_MULT"

echo "\n== lookahead (k=16,h=8) =="
python inference.py --model transformer --checkpoint "$CKPT" --prompt "$PROMPT" \
  --device "$DEVICE" --decode lookahead --top_p "$TOPP" --lookahead_k 16 --lookahead_h 8 --rep_penalty 0.2 \
  --max_tokens "$MAXTOK" --block_size "$BLOCK_SIZE" --embed_size "$EMBED" \
  --transformer_heads "$HEADS" --transformer_blocks "$BLOCKS" --ff_mult "$FF_MULT"
