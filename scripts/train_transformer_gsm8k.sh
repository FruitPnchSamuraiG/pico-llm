#!/usr/bin/env bash
set -euo pipefail

# Finetune the Transformer on GSM8K (openai/gsm8k).
# Optionally run RL-style outcome post-training (best-of-n) afterward.

cd "$(dirname "$0")/.."
source /scratch/kk6081/ml_fall25/venv/bin/activate

DEVICE=${DEVICE:-cuda:0}
OUTDIR=${OUTDIR:-/scratch/kk6081/picollm_extend}
DATA_DIR=${DATA_DIR:-data}
mkdir -p "$DATA_DIR"

# Base checkpoint to finetune from
BASE_CKPT=${BASE_CKPT:-$OUTDIR/transformer_epoch1.pt}

# Isolate finetune outputs
FT_SUBDIR=${FT_SUBDIR:-"$OUTDIR/finetune_gsm8k"}
mkdir -p "$FT_SUBDIR"

# Optional RL stage
RUN_RL=${RUN_RL:-1}                     # set to 0 to disable
RL_OUTDIR=${RL_OUTDIR:-"$OUTDIR/rl_gsm8k"}
RL_STEPS=${RL_STEPS:-200}
RL_BATCH=${RL_BATCH:-8}
RL_NUM_SAMPLES=${RL_NUM_SAMPLES:-4}
RL_MAX_NEW_TOKENS=${RL_MAX_NEW_TOKENS:-64}
RL_LR=${RL_LR:-2e-5}

# Model hyperparams MUST match the base checkpoint
BLOCK_SIZE=${BLOCK_SIZE:-256}
EMBED=${EMBED:-384}
HEADS=${HEADS:-4}
BLOCKS=${BLOCKS:-3}
FF_MULT=${FF_MULT:-2}

# Finetune knobs
BATCH=${BATCH:-16}
EPOCHS=${EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:-600}
LR=${LR:-2e-4}
VAL_SPLIT=${VAL_SPLIT:-0.05}

# Faster training knobs
SAMPLE_INTERVAL_SECONDS=${SAMPLE_INTERVAL_SECONDS:-600}
SAMPLE_EVERY_STEPS=${SAMPLE_EVERY_STEPS:-0}
LR_SCHEDULE=${LR_SCHEDULE:-cosine}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-200}
LR_MIN_RATIO=${LR_MIN_RATIO:-0.1}

TRAIN_TXT=${TRAIN_TXT:-"$DATA_DIR/gsm8k_train.txt"}
VAL_TXT=${VAL_TXT:-"$DATA_DIR/gsm8k_val.txt"}
TEST_TXT=${TEST_TXT:-"$DATA_DIR/gsm8k_test.txt"}

# Ensure base checkpoint exists
if [[ ! -f "$BASE_CKPT" ]]; then
  echo "❌ Base checkpoint not found: $BASE_CKPT" >&2
  exit 2
fi

echo "=========================================="
echo "🧮 Transformer GSM8K Finetune"
echo "Device: $DEVICE"
echo "outdir=$OUTDIR"
echo "ft_subdir=$FT_SUBDIR"
echo "base_ckpt=$BASE_CKPT"
echo "train_txt=$TRAIN_TXT"
echo "val_txt=$VAL_TXT"
echo "test_txt=$TEST_TXT"
echo "block_size=$BLOCK_SIZE embed=$EMBED heads=$HEADS blocks=$BLOCKS ff_mult=$FF_MULT"
echo "batch=$BATCH epochs=$EPOCHS max_steps_per_epoch=$MAX_STEPS lr=$LR"
echo "sample_interval_seconds=$SAMPLE_INTERVAL_SECONDS sample_every_steps=$SAMPLE_EVERY_STEPS"
echo "lr_schedule=$LR_SCHEDULE lr_warmup_steps=$LR_WARMUP_STEPS lr_min_ratio=$LR_MIN_RATIO"
echo "run_rl=$RUN_RL rl_outdir=$RL_OUTDIR"
echo "=========================================="

# Prepare data (only if missing)
if [[ ! -f "$TRAIN_TXT" || ! -f "$VAL_TXT" || ! -f "$TEST_TXT" ]]; then
  echo "Preparing GSM8K text files..."
  bash scripts/prepare_hf_gsm8k_data.sh
fi

PROMPT=${PROMPT:-"Q: If you have 3 apples and buy 2 more, how many apples do you have? A:"}

# -----------------
# Stage 1: SFT
# -----------------
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

# Copy produced checkpoints into OUTDIR with a clear prefix
shopt -s nullglob
SFT_CKPT=""
for f in "$FT_SUBDIR"/transformer_epoch*.pt; do
  bn=$(basename "$f")
  cp -f "$f" "$OUTDIR/transformer_gsm8k_${bn}"
  SFT_CKPT="$f"
done
shopt -u nullglob

if [[ -z "$SFT_CKPT" ]]; then
  echo "❌ No SFT checkpoint produced in $FT_SUBDIR" >&2
  exit 3
fi

echo "✅ SFT done. Latest checkpoint: $SFT_CKPT"

# -----------------
# Stage 2 (optional): RL-ish outcome post-training
# -----------------
if [[ "$RUN_RL" == "1" ]]; then
  mkdir -p "$RL_OUTDIR"
  echo "Running RL-style outcome post-training on GSM8K (best-of-n)..."

  python scripts/rl_reasoning_outcome.py \
    --init_from "$SFT_CKPT" \
    --train_data "$TRAIN_TXT" \
    --val_data "$VAL_TXT" \
    --out_dir "$RL_OUTDIR" \
    --device "$DEVICE" \
    --block_size "$BLOCK_SIZE" \
    --embed_size "$EMBED" \
    --transformer_heads "$HEADS" \
    --transformer_blocks "$BLOCKS" \
    --ff_mult "$FF_MULT" \
    --num_steps "$RL_STEPS" \
    --batch_size "$RL_BATCH" \
    --num_samples "$RL_NUM_SAMPLES" \
    --max_new_tokens "$RL_MAX_NEW_TOKENS" \
    --lr "$RL_LR"

  # Copy RL output with a clear prefix (script writes transformer_rl_reasoning.pt)
  if [[ -f "$RL_OUTDIR/transformer_rl_reasoning.pt" ]]; then
    cp -f "$RL_OUTDIR/transformer_rl_reasoning.pt" "$OUTDIR/transformer_gsm8k_rl.pt"
    echo "✅ RL done. RL checkpoint: $OUTDIR/transformer_gsm8k_rl.pt"
  else
    echo "⚠️ RL finished but did not find $RL_OUTDIR/transformer_rl_reasoning.pt" >&2
  fi
fi

echo "\n✅ Done. GSM8K checkpoints: $OUTDIR/transformer_gsm8k_transformer_epoch*.pt"
