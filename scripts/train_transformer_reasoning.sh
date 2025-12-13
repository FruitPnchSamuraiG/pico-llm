#!/usr/bin/env bash
set -euo pipefail

# Post-train (finetune) the Transformer on a reasoning dataset.
# Default: Hugging Face OpenThoughts-114k (no local synthetic generation).

cd "$(dirname "$0")/.."

source /scratch/kk6081/ml_fall25/venv/bin/activate

DEVICE=${DEVICE:-cuda:0}
OUTDIR=${OUTDIR:-/scratch/kk6081/picollm_extend}
DATA_DIR=${DATA_DIR:-data}

# NEW: dataset selection
HF_DATASET=${HF_DATASET:-open-thoughts/OpenThoughts-114k}
HF_SPLIT_TRAIN=${HF_SPLIT_TRAIN:-"train[1%:]"}
HF_SPLIT_VAL=${HF_SPLIT_VAL:-"train[:1%]"}

# NEW: where we write converted text files (one line = one example)
TRAIN_TXT=${TRAIN_TXT:-"$DATA_DIR/open_thoughts_train.txt"}
VAL_TXT=${VAL_TXT:-"$DATA_DIR/open_thoughts_val.txt"}

# NEW: isolate finetune outputs to avoid overwriting base checkpoints
FT_SUBDIR=${FT_SUBDIR:-"$OUTDIR/finetune_reasoning"}
mkdir -p "$FT_SUBDIR" "$DATA_DIR"

# Model hyperparams MUST match the base checkpoint
# Default to MEDIUM architecture (512d, 8h, 6b, ff=4x) - matches train_transformer_fast.sh default
# If your base checkpoint is SMALL (384d, 4h, 3b, ff=2x), override with:
#   EMBED=384 HEADS=4 BLOCKS=3 FF_MULT=2 bash scripts/train_transformer_reasoning.sh
BLOCK_SIZE=${BLOCK_SIZE:-256}
EMBED=${EMBED:-512}
HEADS=${HEADS:-8}
BLOCKS=${BLOCKS:-6}
FF_MULT=${FF_MULT:-4}

# Finetune knobs (small + stable)
BATCH=${BATCH:-16}
EPOCHS=${EPOCHS:-2}              # Increased from 1 to 2 for better learning
MAX_STEPS=${MAX_STEPS:-999999}   # Removed limit - train on full dataset
LR=${LR:-2e-4}
VAL_SPLIT=${VAL_SPLIT:-0.05}

# Faster training knobs (override via env vars)
SAMPLE_INTERVAL_SECONDS=${SAMPLE_INTERVAL_SECONDS:-600}
SAMPLE_EVERY_STEPS=${SAMPLE_EVERY_STEPS:-0}
LR_SCHEDULE=${LR_SCHEDULE:-cosine}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-200}
LR_MIN_RATIO=${LR_MIN_RATIO:-0.1}

PROMPT=${PROMPT:-"Q: Compute ( 3 + 2 ) - 1. Let's think step by step. A:"}

# Base checkpoint to finetune from
# Auto-detect latest epoch checkpoint if not specified
if [[ -z "${BASE_CKPT:-}" ]]; then
  # Find the highest epoch number checkpoint
  LATEST_CKPT=$(ls -1 "$OUTDIR"/transformer_epoch*.pt 2>/dev/null | \
    sed 's/.*transformer_epoch\([0-9]*\)\.pt/\1 &/' | \
    sort -rn | \
    head -1 | \
    awk '{print $2}')
  
  if [[ -n "$LATEST_CKPT" ]]; then
    BASE_CKPT="$LATEST_CKPT"
    EPOCH_NUM=$(basename "$BASE_CKPT" .pt | sed 's/transformer_epoch//')
    echo "✓ Auto-detected latest base checkpoint: $BASE_CKPT (epoch $EPOCH_NUM)"
  else
    echo "❌ No base checkpoints found in $OUTDIR/transformer_epoch*.pt" >&2
    echo "   Run 'bash scripts/train_transformer_fast.sh' first to create a base checkpoint" >&2
    exit 2
  fi
fi

# Ensure we start from an existing base checkpoint
if [[ ! -f "$BASE_CKPT" ]]; then
  echo "❌ Base checkpoint not found: $BASE_CKPT" >&2
  echo "   Set BASE_CKPT=/path/to/transformer_epochK.pt or create one with train_transformer_fast.sh" >&2
  exit 2
fi

# Auto-detect checkpoint architecture and warn if mismatch
echo "🔍 Checking base checkpoint architecture..."
DETECTED_ARCH=$(python3 -c "
import torch
try:
    ckpt = torch.load('$BASE_CKPT', map_location='cpu')
    embed = ckpt['embed.weight'].shape[1]
    blocks = max([int(k.split('.')[1]) for k in ckpt.keys() if k.startswith('blocks.')]) + 1
    ff_mult = ckpt['blocks.0.ff.0.weight'].shape[0] // embed
    # Note: Cannot reliably detect n_heads from checkpoint (q_proj is d_model->d_model)
    # User must specify HEADS manually
    print(f'{embed},{blocks},{ff_mult}')
except Exception as e:
    print('DETECTION_FAILED')
" 2>/dev/null)

if [[ "$DETECTED_ARCH" != "DETECTION_FAILED" && -n "$DETECTED_ARCH" ]]; then
  IFS=',' read -r DET_EMBED DET_BLOCKS DET_FF <<< "$DETECTED_ARCH"
  echo "✓ Detected: embed=$DET_EMBED, blocks=$DET_BLOCKS, ff_mult=$DET_FF"
  echo "  Note: n_heads cannot be auto-detected from checkpoint"
  echo "  Using: heads=$HEADS (from env var or script default)"
  
  # Warn if mismatch (only check embed, blocks, ff_mult - NOT heads)
  if [[ "$DET_EMBED" != "$EMBED" || "$DET_BLOCKS" != "$BLOCKS" || "$DET_FF" != "$FF_MULT" ]]; then
    echo "⚠️  WARNING: Architecture mismatch detected!"
    echo "   Checkpoint: embed=$DET_EMBED, blocks=$DET_BLOCKS, ff_mult=$DET_FF"
    echo "   Script:     embed=$EMBED, blocks=$BLOCKS, ff_mult=$FF_MULT"
    echo ""
    echo "   To fix, run with:"
    echo "   EMBED=$DET_EMBED HEADS=<match_your_training> BLOCKS=$DET_BLOCKS FF_MULT=$DET_FF bash scripts/train_transformer_reasoning.sh"
    echo ""
    exit 1
  fi
  echo "✓ Architecture matches checkpoint (embed/blocks/ff_mult)!"
else
  echo "⚠️  Could not auto-detect architecture - proceeding with manual settings"
fi

echo "=========================================="
echo "🧠 Transformer Reasoning Finetune"
echo "Device: $DEVICE"
echo "outdir=$OUTDIR"
echo "ft_subdir=$FT_SUBDIR"
echo "data_dir=$DATA_DIR"
echo "hf_dataset=$HF_DATASET"
echo "train_txt=$TRAIN_TXT"
echo "val_txt=$VAL_TXT"
echo "base_ckpt=$BASE_CKPT"
echo "block_size=$BLOCK_SIZE embed=$EMBED heads=$HEADS blocks=$BLOCKS ff_mult=$FF_MULT"
echo "batch=$BATCH epochs=$EPOCHS max_steps_per_epoch=$MAX_STEPS lr=$LR"
echo "sample_interval_seconds=$SAMPLE_INTERVAL_SECONDS sample_every_steps=$SAMPLE_EVERY_STEPS"
echo "lr_schedule=$LR_SCHEDULE lr_warmup_steps=$LR_WARMUP_STEPS lr_min_ratio=$LR_MIN_RATIO"
echo "=========================================="

# Convert HF dataset -> plain text files (only if missing)
if [[ ! -f "$TRAIN_TXT" ]]; then
  echo "Converting HF dataset to $TRAIN_TXT / $VAL_TXT ..."
  python scripts/prepare_hf_reasoning_data.py \
    --dataset "$HF_DATASET" \
    --train_split "$HF_SPLIT_TRAIN" \
    --val_split "$HF_SPLIT_VAL" \
    --out_train "$TRAIN_TXT" \
    --out_val "$VAL_TXT" \
    --limit_train 50000 \
    --limit_val 2000
fi

# Determine which preset to use based on detected architecture
# NOTE: pico-llm.py's --transformer_size preset OVERRIDES explicit flags, so we must use the matching preset
if [[ "$EMBED" -eq 512 && "$HEADS" -eq 8 && "$BLOCKS" -eq 6 && "$FF_MULT" -eq 4 ]]; then
  TRANSFORMER_SIZE="medium"
elif [[ "$EMBED" -eq 384 && "$HEADS" -eq 4 && "$BLOCKS" -eq 3 && "$FF_MULT" -eq 2 ]]; then
  TRANSFORMER_SIZE="small"
else
  # Custom architecture - cannot use preset, must override pico-llm.py default
  # Workaround: pick closest preset and warn
  if [[ "$EMBED" -ge 512 ]]; then
    TRANSFORMER_SIZE="medium"
  else
    TRANSFORMER_SIZE="small"
  fi
  echo "⚠️  WARNING: Custom architecture detected (embed=$EMBED, heads=$HEADS, blocks=$BLOCKS, ff_mult=$FF_MULT)"
  echo "   Using --transformer_size $TRANSFORMER_SIZE as base, but architecture may not match exactly"
  echo "   Consider modifying pico-llm.py to fix preset override behavior"
fi

# Train on the converted reasoning lines only (tinystories_weight=0)
python pico-llm.py \
  --enable_transformer --disable_lstm \
  --device_id "$DEVICE" \
  --checkpoint_dir "$FT_SUBDIR" \
  --init_from "$BASE_CKPT" \
  --tinystories_weight 0.0 \
  --input_files "$TRAIN_TXT" \
  --batch_size "$BATCH" --num_epochs "$EPOCHS" --max_steps_per_epoch "$MAX_STEPS" \
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

# Copy the produced checkpoints into OUTDIR with a clear prefix
shopt -s nullglob
for f in "$FT_SUBDIR"/transformer_epoch*.pt; do
  bn=$(basename "$f")
  cp -f "$f" "$OUTDIR/transformer_reasoning_${bn}"
done
shopt -u nullglob

echo "\n✅ Done. Reasoning checkpoints: $OUTDIR/transformer_reasoning_transformer_epoch*.pt"
