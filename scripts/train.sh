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
#
# Cleanup options:
#   CLEAN_GSM8K_CKPT=1 (default)  # Remove old Stage 2 (GSM8K) checkpoints before training
#   CLEAN_GSM8K_CKPT=0            # Keep old Stage 2 checkpoints (will overwrite)

cd "$(dirname "$0")/.."
source /scratch/kk6081/ml_fall25/venv/bin/activate

# ============================================================================
# Configuration
# ============================================================================

DEVICE=${DEVICE:-cuda:0}
OUTDIR=${OUTDIR:-/scratch/kk6081/picollm_extend}
DATA_DIR=${DATA_DIR:-data}
CLEAN_DATA=${CLEAN_DATA:-0}  # Set to 1 to force regenerate Orca data
CLEAN_GSM8K_CKPT=${CLEAN_GSM8K_CKPT:-1}  # Set to 1 to remove old GSM8K checkpoints (default: enabled)
mkdir -p "$DATA_DIR" "$OUTDIR"

# Clean old data if requested
if [[ "$CLEAN_DATA" == "1" ]]; then
  echo "🧹 Cleaning old Orca-Math data files..."
  rm -f "$DATA_DIR"/orca_math_*.txt 2>/dev/null || true
  echo "   ✓ Removed old Orca data"
  echo ""
fi

# Parse command-line arguments
MODE=${1:-orca}  # orca, gsm8k, gpt2
TRANSFORMER_SIZE=${2:-${TRANSFORMER_SIZE:-medium}}

# Training hyperparameters (auto-adjust for model size)
if [[ "$TRANSFORMER_SIZE" == "gpt2-small" ]]; then
  BATCH=${BATCH:-8}    # Conservative for gpt2-small
  BLOCK_SIZE=${BLOCK_SIZE:-512}  # Longer sequences for gpt2-small
  MAX_SAMPLES=${MAX_SAMPLES:-0}  # Use all 200k examples
elif [[ "$TRANSFORMER_SIZE" == "medium" ]]; then
  BATCH=${BATCH:-8}    # Balanced for medium model
  BLOCK_SIZE=${BLOCK_SIZE:-256}
  MAX_SAMPLES=${MAX_SAMPLES:-100000}  # 100k for medium
else
  BATCH=${BATCH:-8}    # Small model can use larger batch
  BLOCK_SIZE=${BLOCK_SIZE:-256}
  MAX_SAMPLES=${MAX_SAMPLES:-100000}  # 100k for small
fi
EPOCHS=${EPOCHS:-8}
LR=${LR:-3e-4}
VAL_SPLIT=${VAL_SPLIT:-0.05}

# GSM8K is small and can overfit quickly.
# Supported knobs to reduce overfitting:
# - Lower epochs (use GSM8K_EPOCHS / EPOCHS_OVERRIDE)
# - Lower LR (use LR_OVERRIDE or LR)
# - Increase BATCH if VRAM allows

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
  
  # Set max_length based on model size (longer for gpt2-small)
  local max_len
  if [[ "$TRANSFORMER_SIZE" == "gpt2-small" || "$TRANSFORMER_SIZE" =~ ^gpt2- ]]; then
    max_len=2048  # Longer sequences for GPT-2 models
  else
    max_len=1024  # Standard for smaller models
  fi
  
  # Check if we need to regenerate (wrong settings or missing files)
  local needs_regen=0
  if [[ ! -f "$train_file" || ! -f "$val_file" ]]; then
    needs_regen=1
  else
    # Check if existing data is too small (old 20k subset instead of 200k)
    local line_count=$(wc -l < "$train_file" 2>/dev/null || echo 0)
    if [ "$MAX_SAMPLES" -eq 0 ] && [ "$line_count" -lt 100000 ]; then
      echo "⚠️  Existing Orca data has only $line_count examples (expected ~190k for full dataset)"
      echo "   Regenerating with full dataset..."
      needs_regen=1
    fi
  fi
  
  if [ "$needs_regen" -eq 1 ]; then
    print_header "📥 Downloading Orca-Math Dataset"
    echo "Length filtering: DISABLED (keeping ALL sequence lengths)"
    echo "Max samples: $([ "$MAX_SAMPLES" -eq 0 ] && echo "ALL (~200k)" || echo "$MAX_SAMPLES")"
    echo ""
    
    # Remove old files to force fresh download
    rm -f "$train_file" "$val_file" 2>/dev/null || true
    
    python3 scripts/data_prep/prepare_orca_math_data.py \
      --output_dir "$DATA_DIR" \
      --max_samples "$MAX_SAMPLES" \
      --min_length 0 \
      --max_length 0 \
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

train_gsm8k_only() {
  print_header "🧮 Training on GSM8K from Scratch (SFT + RL)"
  print_config
  
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
  
  local vram=$(estimate_vram "$TRANSFORMER_SIZE")
  echo "Model: $TRANSFORMER_SIZE (estimated VRAM: $vram)"
  echo ""
  
  # Check GSM8K data
  local train_count=$(wc -l < "$DATA_DIR/gsm8k_train.txt")
  local val_count=$(wc -l < "$DATA_DIR/gsm8k_val.txt")
  echo "✓ GSM8K data ready:"
  echo "  Train: $train_count examples"
  echo "  Val:   $val_count examples"
  echo ""
  
  local prompt="Q: If you have 3 apples and buy 2 more, how many apples do you have? A:"
  
  print_header "🚀 Stage 1: Supervised Fine-tuning (from scratch)"
  # GSM8K tends to overfit quickly: default to fewer epochs unless user overrides.
  local gsm8k_epochs=${GSM8K_EPOCHS:-${EPOCHS:-4}}
  python pico-llm.py \
    --enable_transformer --disable_lstm \
    --device_id "$DEVICE" \
    --checkpoint_dir "$OUTDIR" \
    --input_files "$DATA_DIR/gsm8k_train.txt" \
    --tinystories_weight 0.0 \
    --batch_size "$BATCH" \
    --num_epochs "$gsm8k_epochs" \
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
  
  # Find latest SFT checkpoint
  local sft_ckpt=$(find_latest_checkpoint)
  if [[ -z "$sft_ckpt" ]]; then
    echo "❌ No SFT checkpoint produced"
    exit 1
  fi
  
  echo "✅ SFT complete: $sft_ckpt"
  
  print_header "✅ GSM8K Training Complete!"
  echo "Checkpoints:"
  echo "  SFT: $OUTDIR/transformer_epoch*.pt"
  echo ""
  echo "Next steps:"
  echo "  1. Evaluate SFT model:"
  echo "     python scripts/evaluation/eval_reasoning.py \\"
  echo "       --checkpoint $OUTDIR/transformer_epoch${gsm8k_epochs}.pt \\"
  echo "       --device $DEVICE"
  echo ""
  echo "  2. Run DPO/GRPO post-training (Stage 3):"
  echo "     bash scripts/train_dpo_grpo.sh dpo $TRANSFORMER_SIZE"
  echo ""
}

train_gsm8k() {
  print_header "🧮 Fine-tuning on GSM8K (from Orca checkpoint)"
  
  # Clean old GSM8K checkpoints (default enabled)
  if [[ "$CLEAN_GSM8K_CKPT" == "1" ]]; then
    echo "🧹 Cleaning old Stage-2 (GSM8K) checkpoints..."
    rm -rf "$OUTDIR/finetune_gsm8k" 2>/dev/null || true
    rm -f "$OUTDIR"/gsm8k_transformer_epoch*.pt 2>/dev/null || true
    rm -rf "$OUTDIR/rl_gsm8k" 2>/dev/null || true
    echo "   ✓ Removed old GSM8K checkpoints"
    echo ""
  fi
  
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
  # Priority order:
  # 1. BASE_CKPT environment variable
  # 2. Known Orca checkpoint: $OUTDIR/transformer_epoch8.pt (Orca-Math trained)
  # 3. Latest transformer_epoch*.pt in $OUTDIR
  # 4. Direct Orca model checkpoints: $OUTDIR/orca_model_*.pt
  if [[ -z "${BASE_CKPT:-}" ]]; then
    # First, check for the known Orca checkpoint (transformer_epoch8.pt)
    if [[ -f "$OUTDIR/transformer_epoch8.pt" ]]; then
      BASE_CKPT="$OUTDIR/transformer_epoch8.pt"
      echo "✓ Using Orca checkpoint: $BASE_CKPT"
    else
      # Fallback to finding latest transformer_epoch*.pt
      BASE_CKPT=$(find_latest_checkpoint "transformer_epoch*.pt")
      
      # If not found, try orca_model_*.pt (direct Orca checkpoints)
      if [[ -z "$BASE_CKPT" ]]; then
        BASE_CKPT=$(find_latest_checkpoint "orca_model_*.pt")
        if [[ -n "$BASE_CKPT" ]]; then
          echo "✓ Found Orca model checkpoint: $BASE_CKPT"
        fi
      else
        local epoch_num=$(basename "$BASE_CKPT" .pt | sed 's/transformer_epoch//')
        echo "✓ Auto-detected transformer checkpoint: $BASE_CKPT (epoch $epoch_num)"
      fi
      
      # If still not found, train from scratch
      if [[ -z "$BASE_CKPT" ]]; then
        echo "⚠️  No base checkpoints found in:"
        echo "   - $OUTDIR/transformer_epoch8.pt (Orca checkpoint)"
        echo "   - $OUTDIR/transformer_epoch*.pt"
        echo "   - $OUTDIR/orca_model_*.pt"
        echo "   Training from scratch on GSM8K (use 'gsm8k-sft' mode explicitly for clarity)"
        echo ""
        train_gsm8k_only
        return
      fi
    fi
  else
    if [[ ! -f "$BASE_CKPT" ]]; then
      echo "❌ Base checkpoint not found: $BASE_CKPT"
      exit 1
    fi
    echo "✓ Using specified checkpoint: $BASE_CKPT"
  fi
  
  echo ""
  echo "📊 Training Configuration:"
  echo "  Base checkpoint: $BASE_CKPT"
  echo "  Model size: $TRANSFORMER_SIZE"
  echo "  Device: $DEVICE"
  echo ""
  
  # Create finetune output directory
  local ft_dir="$OUTDIR/finetune_gsm8k"
  mkdir -p "$ft_dir"
  
  # Improved hyperparameters for GSM8K fine-tuning from Orca
  # These values are optimized to prevent overfitting while ensuring the model learns GSM8K reasoning
  # WARNING: GSM8K is small (7.5k examples) - model overfits badly after epoch 5-6!
  # NOTE: Using block_size=512 (instead of 256) to accommodate full <thinking> blocks (~200-300 tokens)
  #       This allows the model to learn complete reasoning chains without truncation
  EPOCHS=${EPOCHS_OVERRIDE:-${GSM8K_EPOCHS:-5}}  # REDUCED from 10 to 5 to prevent overfitting
  LR=${LR_OVERRIDE:-1e-4}  # LOWERED from 3e-4 to 1e-4 to preserve Orca knowledge
  LR_WARMUP_STEPS=${WARMUP_OVERRIDE:-500}  # Longer warmup for stability
  LR_MIN_RATIO=0.1  # Drop to 10% of peak LR by end
  
  local prompt="Q: If you have 3 apples and buy 2 more, how many apples do you have? A:"
  
  echo "📋 Fine-tuning Hyperparameters:"
  echo "  Epochs: $EPOCHS"
  echo "  Learning rate: $LR"
  echo "  LR warmup steps: $LR_WARMUP_STEPS"
  echo "  LR min ratio: $LR_MIN_RATIO"
  echo "  Batch size: $BATCH"
  echo "  Gradient clip: $GRAD_CLIP"
  echo "  Weight decay: $WEIGHT_DECAY"
  echo ""
  
  print_header "🚀 Starting Supervised Fine-tuning on GSM8K (with <thinking> blocks)"
  # NOTE: Using block_size=256 to match base checkpoint architecture
  # Some long thinking chains may be truncated, but model will still learn the format
  python pico-llm.py \
    --enable_transformer --disable_lstm \
    --device_id "$DEVICE" \
    --checkpoint_dir "$ft_dir" \
    --init_from "$BASE_CKPT" \
    --tinystories_weight 0.0 \
    --input_files "$DATA_DIR/gsm8k_train_reasoning_structured.txt" \
    --batch_size "$BATCH" \
    --num_epochs "$EPOCHS" \
    --block_size 256 \
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
  local sft_final_ckpt=""
  for f in "$ft_dir"/transformer_epoch*.pt; do
    [[ -f "$f" ]] || continue
    local bn=$(basename "$f")
    local target="$OUTDIR/gsm8k_${bn}"
    cp -f "$f" "$target"
    sft_ckpt="$f"
    sft_final_ckpt="$target"
  done
  
  if [[ -z "$sft_ckpt" ]]; then
    echo "❌ No SFT checkpoint produced"
    exit 1
  fi
  
  echo ""
  echo "✅ SFT training complete!"
  echo "   Final checkpoint: $sft_final_ckpt"
  echo ""
  
  print_header "✅ GSM8K Fine-tuning Complete!"
  echo "📁 Checkpoints saved to:"
  echo "   $OUTDIR/gsm8k_transformer_epoch*.pt"
  echo ""
  echo "🧪 Next Steps:"
  echo ""
  echo "1. Quick test inference (verify model generates sensible output):"
  echo "   python scripts/inference_dpo.py \\"
  echo "     --checkpoint $OUTDIR/gsm8k_transformer_epoch${EPOCHS}.pt \\"
  echo "     --prompt 'Q: If 1 apple costs \$1, how much do 2 apples cost? A:'"
  echo ""
  echo "2. Full evaluation on GSM8K test set:"
  echo "   python scripts/evaluation/eval_reasoning.py \\"
  echo "     --checkpoint $OUTDIR/gsm8k_transformer_epoch${EPOCHS}.pt \\"
  echo "     --device $DEVICE"
  echo ""
  echo "3. Run optimized DPO training (if SFT looks good):"
  echo "   bash scripts/fast_dpo_train.sh dpo $TRANSFORMER_SIZE"
  echo ""
  echo "💡 Tips:"
  echo "   - Check that generated text makes mathematical sense"
  echo "   - Look for proper step-by-step reasoning"
  echo "   - If model still generates nonsense, try:"
  echo "     * Lower LR: LR_OVERRIDE=1e-4 bash scripts/train.sh gsm8k"
  echo "     * More epochs: EPOCHS_OVERRIDE=15 bash scripts/train.sh gsm8k"
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
  gsm8k-sft|gsm8k-only|sft)
    # Train from scratch on GSM8K only (no Orca-Math)
    train_gsm8k_only
    ;;
  gsm8k|finetune)
    # Fine-tune on GSM8K from base checkpoint (backward compatible)
    train_gsm8k
    ;;
  *)
    echo "Usage: bash scripts/train.sh [MODE] [options]"
    echo ""
    echo "🎯 GSM8K-Only Modes (NO Orca-Math):"
    echo "  gsm8k-sft   - Train from scratch on GSM8K (SFT + optional RL)"
    echo "  gsm8k-only  - Alias for gsm8k-sft"
    echo "  sft         - Alias for gsm8k-sft"
    echo ""
    echo "📚 Other Modes:"
    echo "  orca        - Train base model on Orca-Math dataset"
    echo "  gsm8k       - Fine-tune on GSM8K (auto-detects base checkpoint or trains from scratch)"
    echo "  gpt2        - Train GPT-2 scale model on Orca-Math (specify size as 2nd arg)"
    echo ""
    echo "💡 Examples:"
    echo "  # Train on GSM8K only (most common for your use case)"
    echo "  bash scripts/train.sh gsm8k-sft"
    echo "  bash scripts/train.sh gsm8k-sft gpt2-small"
    echo ""
    echo "  # Train with custom settings"
    echo "  EPOCHS=4 BATCH=8 bash scripts/train.sh gsm8k-sft"
    echo ""
    echo "  # Other modes (with Orca-Math)"
    echo "  bash scripts/train.sh orca                    # Train medium model"
    echo "  bash scripts/train.sh gpt2 gpt2-small         # Train GPT-2 Small"
    echo ""
    echo "🔧 Environment Variables:"
    echo "  TRANSFORMER_SIZE   - Model size (small/medium/gpt2-small, default: medium)"
    echo "  BATCH             - Batch size (default: 8, auto-adjusted per model)"
    echo "  EPOCHS            - Number of epochs (default: 8 for Orca, 4 for GSM8K)"
    echo "  LR                - Learning rate (default: 3e-4)"
    echo "  GRAD_CLIP         - Gradient clipping norm (default: 1.0)"
    echo "  WEIGHT_DECAY      - AdamW weight decay (default: 0.01)"
    echo "  DEVICE            - Device (default: cuda:0)"
    echo "  BASE_CKPT         - Base checkpoint for gsm8k mode"
    echo "  CLEAN_GSM8K_CKPT  - Remove old GSM8K checkpoints (default: 1)"
    echo ""
    echo "💻 Hardware: 2x GTX TITAN X (12GB VRAM each)"
    echo "  ✅ Supported: small, medium, gpt2-small"
    echo "  ❌ Not supported: gpt2-medium/large/xl (require >12GB VRAM)"
    echo ""
    exit 1
    ;;
esac
