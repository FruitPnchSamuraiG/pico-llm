#!/usr/bin/env bash
set -euo pipefail

# Unified training script for Pico-LLM transformers
# Usage:
#   bash scripts/train.sh [base|gsm8k|orca|gpt2] [options]
#
# Examples:
#   bash scripts/train.sh orca                    # Train on Orca-Math (medium model)
#   bash scripts/train.sh gpt2 gpt2-small         # Train GPT-2 Small on Orca-Math
#   bash scripts/train.sh gsm8k                   # Fine-tune on GSM8K (auto-detect checkpoint)
#   BASE_CKPT=model.pt bash scripts/train.sh gsm8k  # Fine-tune with specific checkpoint

cd "$(dirname "$0")/.."
source /scratch/kk6081/ml_fall25/venv/bin/activate

# ============================================================================
# Configuration
# ============================================================================

DEVICE=${DEVICE:-cuda:0}
OUTDIR=${OUTDIR:-/scratch/kk6081/picollm_extend}
DATA_DIR=${DATA_DIR:-data}
mkdir -p "$DATA_DIR" "$OUTDIR"

# Parse command-line arguments
MODE=${1:-orca}  # orca, gsm8k, gpt2
TRANSFORMER_SIZE=${2:-${TRANSFORMER_SIZE:-medium}}

# Training hyperparameters (auto-adjust for model size)
if [[ "$TRANSFORMER_SIZE" == "gpt2-small" ]]; then
  BATCH=${BATCH:-8}   # Reduced for 12GB VRAM
else
  BATCH=${BATCH:-16}
fi
EPOCHS=${EPOCHS:-8}
LR=${LR:-3e-4}
BLOCK_SIZE=${BLOCK_SIZE:-256}
VAL_SPLIT=${VAL_SPLIT:-0.05}
MAX_SAMPLES=${MAX_SAMPLES:-100000}

# LR scheduling
LR_SCHEDULE=${LR_SCHEDULE:-cosine}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-1000}
LR_MIN_RATIO=${LR_MIN_RATIO:-0.2}

# Training stability
GRAD_CLIP=${GRAD_CLIP:-1.0}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}

# Sampling/logging
SAMPLE_EVERY_STEPS=${SAMPLE_EVERY_STEPS:-0}
SAMPLE_INTERVAL_SECONDS=${SAMPLE_INTERVAL_SECONDS:-300}

# RL options (for GSM8K)
RUN_RL=${RUN_RL:-1}
RL_STEPS=${RL_STEPS:-400}
RL_BATCH=${RL_BATCH:-12}
RL_NUM_SAMPLES=${RL_NUM_SAMPLES:-8}
RL_MAX_NEW_TOKENS=${RL_MAX_NEW_TOKENS:-64}
RL_LR=${RL_LR:-1e-5}

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
  echo ""
  echo "=========================================="
  echo "$1"
  echo "=========================================="
}

print_config() {
  echo "Device: $DEVICE"
  echo "Output: $OUTDIR"
  echo "Transformer: $TRANSFORMER_SIZE"
  echo "Batch: $BATCH, Epochs: $EPOCHS"
  echo "LR: $LR (schedule: $LR_SCHEDULE, warmup: $LR_WARMUP_STEPS, min_ratio: $LR_MIN_RATIO)"
  echo "=========================================="
  echo ""
}

estimate_vram() {
  local size=$1
  case "$size" in
    small) echo "~2GB" ;;
    medium) echo "~4GB" ;;
    gpt2-small) echo "~8GB" ;;
    gpt2-medium) echo "~16GB" ;;
    gpt2-large) echo "~32GB" ;;
    gpt2-xl) echo "~64GB" ;;
    *) echo "Unknown" ;;
  esac
}

prepare_orca_data() {
  local train_file="$DATA_DIR/orca_math_train.txt"
  local val_file="$DATA_DIR/orca_math_val.txt"
  
  if [[ ! -f "$train_file" || ! -f "$val_file" ]]; then
    print_header "📥 Downloading Orca-Math Dataset"
    python3 scripts/data_prep/prepare_orca_math_data.py \
      --output_dir "$DATA_DIR" \
      --max_samples "$MAX_SAMPLES" \
      --min_length 50 \
      --max_length 1024 \
      --val_split "$VAL_SPLIT"
    echo ""
  fi
  
  # Verify data
  if [[ ! -f "$train_file" ]]; then
    echo "❌ Error: Failed to download Orca-Math data"
    exit 1
  fi
  
  local train_count=$(wc -l < "$train_file")
  local val_count=$(wc -l < "$val_file")
  echo "✓ Orca-Math data ready:"
  echo "  Train: $train_count examples"
  echo "  Val:   $val_count examples"
  echo ""
}

prepare_gsm8k_data() {
  local train_txt="$DATA_DIR/gsm8k_train.txt"
  local val_txt="$DATA_DIR/gsm8k_val.txt"
  local test_txt="$DATA_DIR/gsm8k_test.txt"
  
  if [[ ! -f "$train_txt" || ! -f "$val_txt" || ! -f "$test_txt" ]]; then
    echo "📥 Downloading GSM8K dataset..."
    bash scripts/data_prep/prepare_hf_gsm8k_data.sh
    echo ""
  fi
}

find_latest_checkpoint() {
  local pattern=${1:-transformer_epoch*.pt}
  local latest=$(ls -1 "$OUTDIR"/$pattern 2>/dev/null | \
    sed 's/.*epoch\([0-9]*\)\.pt/\1 &/' | \
    sort -rn | \
    head -1 | \
    awk '{print $2}')
  echo "$latest"
}

# ============================================================================
# Training Modes
# ============================================================================

train_orca() {
  print_header "🧮 Training on Orca-Math"
  print_config
  
  prepare_orca_data
  
  local vram=$(estimate_vram "$TRANSFORMER_SIZE")
  echo "Model: $TRANSFORMER_SIZE (estimated VRAM: $vram)"
  echo ""
  
  # Hardware check and warnings
  case "$TRANSFORMER_SIZE" in
    gpt2-small)
      if [ "$BATCH" -gt 8 ]; then
        echo "⚠️  WARNING: gpt2-small with batch=$BATCH may exceed 12GB VRAM"
        echo "   Recommended: BATCH=8 or lower for GTX TITAN X"
        echo ""
      fi
      ;;
    gpt2-medium|gpt2-large|gpt2-xl)
      echo "❌ ERROR: $TRANSFORMER_SIZE requires >12GB VRAM"
      echo "   Your GPUs: 2x GTX TITAN X (12GB each)"
      echo "   Required: 24GB+ for gpt2-medium, 80GB+ for gpt2-large/xl"
      echo ""
      echo "   Recommended models for your hardware:"
      echo "   - small (10M params, ~2GB)"
      echo "   - medium (40M params, ~4GB)"
      echo "   - gpt2-small (117M params, ~8GB with BATCH=8)"
      echo ""
      exit 1
      ;;
  esac
  
  local prompt="Q: A bakery sells 5 cupcakes for \$3. How much would 20 cupcakes cost? A:"
  
  print_header "🚀 Starting Training"
  python pico-llm.py \
    --enable_transformer --disable_lstm \
    --device_id "$DEVICE" \
    --checkpoint_dir "$OUTDIR" \
    --input_files "$DATA_DIR/orca_math_train.txt" "$DATA_DIR/orca_math_val.txt" \
    --tinystories_weight 0.0 \
    --batch_size "$BATCH" \
    --num_epochs "$EPOCHS" \
    --block_size "$BLOCK_SIZE" \
    --transformer_size "$TRANSFORMER_SIZE" \
    --learning_rate "$LR" \
    --val_split "$VAL_SPLIT" \
    --prompt "$prompt" \
    --grad_clip "$GRAD_CLIP" \
    --weight_decay "$WEIGHT_DECAY" \
    --sample_interval_seconds "$SAMPLE_INTERVAL_SECONDS" \
    --sample_every_steps "$SAMPLE_EVERY_STEPS" \
    --lr_schedule "$LR_SCHEDULE" \
    --lr_warmup_steps "$LR_WARMUP_STEPS" \
    --lr_min_ratio "$LR_MIN_RATIO"
  
  print_header "✅ Training Complete!"
  echo "Checkpoints: $OUTDIR/transformer_epoch*.pt"
  echo ""
  echo "Next steps:"
  echo "  1. Test inference:"
  echo "     python inference.py --checkpoint $OUTDIR/transformer_epoch${EPOCHS}.pt \\"
  echo "       --prompt 'Q: If x + 5 = 12, then x = ' --device $DEVICE"
  echo ""
  echo "  2. Fine-tune on GSM8K:"
  echo "     bash scripts/train.sh gsm8k"
  echo ""
}

train_gsm8k() {
  print_header "🧮 Fine-tuning on GSM8K"
  
  prepare_gsm8k_data
  
  # Hardware check
  case "$TRANSFORMER_SIZE" in
    gpt2-small)
      if [ "$BATCH" -gt 8 ]; then
        echo "⚠️  WARNING: gpt2-small with batch=$BATCH may exceed 12GB VRAM"
        BATCH=8
        echo "   Auto-adjusted to BATCH=8 for GTX TITAN X"
        echo ""
      fi
      ;;
    gpt2-medium|gpt2-large|gpt2-xl)
      echo "❌ ERROR: $TRANSFORMER_SIZE requires >12GB VRAM"
      echo "   Your GPUs: 2x GTX TITAN X (12GB each)"
      echo ""
      exit 1
      ;;
  esac
  
  # Auto-detect base checkpoint
  if [[ -z "${BASE_CKPT:-}" ]]; then
    BASE_CKPT=$(find_latest_checkpoint)
    if [[ -z "$BASE_CKPT" ]]; then
      echo "❌ No base checkpoints found in $OUTDIR/transformer_epoch*.pt"
      echo "   Run 'bash scripts/train.sh orca' first to create a base checkpoint"
      exit 1
    fi
    local epoch_num=$(basename "$BASE_CKPT" .pt | sed 's/transformer_epoch//')
    echo "✓ Auto-detected base checkpoint: $BASE_CKPT (epoch $epoch_num)"
  else
    if [[ ! -f "$BASE_CKPT" ]]; then
      echo "❌ Base checkpoint not found: $BASE_CKPT"
      exit 1
    fi
  fi
  
  echo "Base checkpoint: $BASE_CKPT"
  print_config
  
  # Create finetune output directory
  local ft_dir="$OUTDIR/finetune_gsm8k"
  mkdir -p "$ft_dir"
  
  # Adjust hyperparameters for GSM8K
  EPOCHS=${EPOCHS_OVERRIDE:-8}
  LR=${LR_OVERRIDE:-2e-4}
  LR_WARMUP_STEPS=${WARMUP_OVERRIDE:-200}
  LR_MIN_RATIO=0.1
  
  local prompt="Q: If you have 3 apples and buy 2 more, how many apples do you have? A:"
  
  print_header "🚀 Stage 1: Supervised Fine-tuning"
  python pico-llm.py \
    --enable_transformer --disable_lstm \
    --device_id "$DEVICE" \
    --checkpoint_dir "$ft_dir" \
    --init_from "$BASE_CKPT" \
    --tinystories_weight 0.0 \
    --input_files "$DATA_DIR/gsm8k_train.txt" \
    --batch_size "$BATCH" \
    --num_epochs "$EPOCHS" \
    --block_size "$BLOCK_SIZE" \
    --transformer_size "$TRANSFORMER_SIZE" \
    --learning_rate "$LR" \
    --val_split "$VAL_SPLIT" \
    --prompt "$prompt" \
    --grad_clip "$GRAD_CLIP" \
    --weight_decay "$WEIGHT_DECAY" \
    --sample_interval_seconds "$SAMPLE_INTERVAL_SECONDS" \
    --sample_every_steps "$SAMPLE_EVERY_STEPS" \
    --lr_schedule "$LR_SCHEDULE" \
    --lr_warmup_steps "$LR_WARMUP_STEPS" \
    --lr_min_ratio "$LR_MIN_RATIO"
  
  # Copy checkpoints with clear naming
  local sft_ckpt=""
  for f in "$ft_dir"/transformer_epoch*.pt; do
    [[ -f "$f" ]] || continue
    local bn=$(basename "$f")
    cp -f "$f" "$OUTDIR/gsm8k_${bn}"
    sft_ckpt="$f"
  done
  
  if [[ -z "$sft_ckpt" ]]; then
    echo "❌ No SFT checkpoint produced"
    exit 1
  fi
  
  echo "✅ SFT complete: $sft_ckpt"
  
  # Optional RL stage
  if [[ "$RUN_RL" == "1" ]]; then
    local rl_dir="$OUTDIR/rl_gsm8k"
    mkdir -p "$rl_dir"
    
    print_header "🚀 Stage 2: RL Outcome Training"
    echo "Steps: $RL_STEPS, Batch: $RL_BATCH, Samples: $RL_NUM_SAMPLES"
    echo "LR: $RL_LR, Max tokens: $RL_MAX_NEW_TOKENS"
    echo ""
    
    python scripts/rl_reasoning_outcome.py \
      --init_from "$sft_ckpt" \
      --train_data "$DATA_DIR/gsm8k_train.txt" \
      --val_data "$DATA_DIR/gsm8k_val.txt" \
      --out_dir "$rl_dir" \
      --device "$DEVICE" \
      --block_size "$BLOCK_SIZE" \
      --num_steps "$RL_STEPS" \
      --batch_size "$RL_BATCH" \
      --num_samples "$RL_NUM_SAMPLES" \
      --max_new_tokens "$RL_MAX_NEW_TOKENS" \
      --lr "$RL_LR"
    
    if [[ -f "$rl_dir/transformer_rl_reasoning.pt" ]]; then
      cp -f "$rl_dir/transformer_rl_reasoning.pt" "$OUTDIR/gsm8k_rl.pt"
      echo "✅ RL complete: $OUTDIR/gsm8k_rl.pt"
    fi
  fi
  
  print_header "✅ GSM8K Training Complete!"
  echo "Checkpoints:"
  echo "  SFT: $OUTDIR/gsm8k_transformer_epoch*.pt"
  [[ "$RUN_RL" == "1" ]] && echo "  RL:  $OUTDIR/gsm8k_rl.pt"
  echo ""
  echo "Evaluate with:"
  echo "  python scripts/eval_reasoning.py \\"
  echo "    --checkpoint $OUTDIR/gsm8k_transformer_epoch${EPOCHS}.pt \\"
  echo "    --device $DEVICE"
  echo ""
}

# ============================================================================
# Main
# ============================================================================

case "$MODE" in
  orca|base)
    train_orca
    ;;
  gpt2)
    # For GPT models, use second argument as size
    if [[ ! "$TRANSFORMER_SIZE" =~ ^gpt2- ]]; then
      echo "❌ Error: For GPT mode, specify size: gpt2-small, gpt2-medium, gpt2-large, or gpt2-xl"
      echo "   Example: bash scripts/train.sh gpt2 gpt2-small"
      exit 1
    fi
    train_orca
    ;;
  gsm8k|finetune)
    train_gsm8k
    ;;
  *)
    echo "Usage: bash scripts/train.sh [orca|gsm8k|gpt2] [options]"
    echo ""
    echo "Modes:"
    echo "  orca    - Train base model on Orca-Math dataset"
    echo "  gsm8k   - Fine-tune on GSM8K (requires base checkpoint)"
    echo "  gpt2    - Train GPT-2 scale model (specify size as 2nd arg)"
    echo ""
    echo "Examples:"
    echo "  bash scripts/train.sh orca                    # Train medium model"
    echo "  bash scripts/train.sh orca small              # Train small model"
    echo "  bash scripts/train.sh gpt2 gpt2-small         # Train GPT-2 Small (max for 12GB)"
    echo "  bash scripts/train.sh gsm8k                   # Fine-tune on GSM8K"
    echo "  EPOCHS=10 bash scripts/train.sh orca          # Train for 10 epochs"
    echo ""
    echo "Environment variables:"
    echo "  TRANSFORMER_SIZE  - Model size (small/medium/gpt2-small)"
    echo "  BATCH            - Batch size (default: 16, auto 8 for gpt2-small)"
    echo "  EPOCHS           - Number of epochs (default: 8)"
    echo "  LR               - Learning rate (default: 3e-4)"
    echo "  GRAD_CLIP        - Gradient clipping norm (default: 1.0)"
    echo "  WEIGHT_DECAY     - AdamW weight decay (default: 0.01)"
    echo "  DEVICE           - Device (default: cuda:0)"
    echo "  BASE_CKPT        - Base checkpoint for GSM8K mode"
    echo "  RUN_RL           - Run RL stage for GSM8K (default: 1)"
    echo ""
    echo "Hardware: 2x GTX TITAN X (12GB VRAM each)"
    echo "Supported models: small, medium, gpt2-small"
    echo "Not supported: gpt2-medium/large/xl (require >12GB VRAM)"
    echo ""
    exit 1
    ;;
esac
