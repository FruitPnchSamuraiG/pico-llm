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

# Isolate finetune outputs
FT_SUBDIR=${FT_SUBDIR:-"$OUTDIR/finetune_gsm8k"}
mkdir -p "$FT_SUBDIR"

# Optional RL stage
RUN_RL=${RUN_RL:-1}                     # set to 0 to disable
RL_OUTDIR=${RL_OUTDIR:-"$OUTDIR/rl_gsm8k"}
RL_STEPS=${RL_STEPS:-400}               # Increased from 200 to 400 for better RL training
RL_BATCH=${RL_BATCH:-12}                # Increased from 8 to 12 for more stable updates
RL_NUM_SAMPLES=${RL_NUM_SAMPLES:-8}     # Increased from 4 to 8 for better exploration
RL_MAX_NEW_TOKENS=${RL_MAX_NEW_TOKENS:-64}
RL_LR=${RL_LR:-1e-5}                    # Reduced from 2e-5 to 1e-5 for stability

# Model hyperparams MUST match the base checkpoint
# Default to MEDIUM architecture (512d, 8h, 6b, ff=4x) - matches train_transformer_fast.sh default
# If your base checkpoint is SMALL (384d, 4h, 3b, ff=2x), override with:
#   EMBED=384 HEADS=4 BLOCKS=3 FF_MULT=2 bash scripts/train_transformer_gsm8k.sh
BLOCK_SIZE=${BLOCK_SIZE:-256}
EMBED=${EMBED:-512}
HEADS=${HEADS:-8}
BLOCKS=${BLOCKS:-6}
FF_MULT=${FF_MULT:-4}

# Finetune knobs
BATCH=${BATCH:-16}
EPOCHS=${EPOCHS:-8}              # Increased to 8 for proper GSM8K reasoning (was 3)
MAX_STEPS=${MAX_STEPS:-999999}   # Removed limit - train on full dataset
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
    echo "   EMBED=$DET_EMBED HEADS=<match_your_training> BLOCKS=$DET_BLOCKS FF_MULT=$DET_FF bash scripts/train_transformer_gsm8k.sh"
    echo ""
    exit 1
  fi
  echo "✓ Architecture matches checkpoint (embed/blocks/ff_mult)!"
else
  echo "⚠️  Could not auto-detect architecture - proceeding with manual settings"
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
