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
BLOCK_SIZE=${BLOCK_SIZE:-256}
EMBED=${EMBED:-384}
HEADS=${HEADS:-4}
BLOCKS=${BLOCKS:-3}
FF_MULT=${FF_MULT:-2}

# Finetune knobs (small + stable)
BATCH=${BATCH:-16}
EPOCHS=${EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:-300}
LR=${LR:-2e-4}
VAL_SPLIT=${VAL_SPLIT:-0.05}

# Faster training knobs (override via env vars)
SAMPLE_INTERVAL_SECONDS=${SAMPLE_INTERVAL_SECONDS:-600}
SAMPLE_EVERY_STEPS=${SAMPLE_EVERY_STEPS:-0}
LR_SCHEDULE=${LR_SCHEDULE:-cosine}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-200}
LR_MIN_RATIO=${LR_MIN_RATIO:-0.1}

BASE_CKPT=${BASE_CKPT:-$OUTDIR/transformer_epoch1.pt}
PROMPT=${PROMPT:-"Q: Compute ( 3 + 2 ) - 1. Let's think step by step. A:"}

# Ensure we start from an existing base checkpoint
if [[ ! -f "$BASE_CKPT" ]]; then
  echo "❌ Base checkpoint not found: $BASE_CKPT" >&2
  echo "Set BASE_CKPT=/path/to/transformer_epochK.pt" >&2
  exit 2
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

# Train on the converted reasoning lines only (tinystories_weight=0)
python pico-llm.py \
  --enable_transformer --disable_lstm \
  --device_id "$DEVICE" \
  --checkpoint_dir "$FT_SUBDIR" \
  --init_from "$BASE_CKPT" \
  --tinystories_weight 0.0 \
  --input_files "$TRAIN_TXT" \
  --batch_size "$BATCH" --num_epochs "$EPOCHS" --max_steps_per_epoch "$MAX_STEPS" \
  --block_size "$BLOCK_SIZE" --embed_size "$EMBED" \
  --transformer_heads "$HEADS" --transformer_blocks "$BLOCKS" --ff_mult "$FF_MULT" \
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
