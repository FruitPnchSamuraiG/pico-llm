#!/usr/bin/env bash
# Quick-start script for DPO/GRPO post-training on GSM8K
# Usage: bash scripts/train_dpo_grpo.sh [dpo|grpo] [model_size]
#
# Notes:
# - Expects a GSM8K SFT checkpoint (Stage 2 output). If missing, it will try to
#   auto-detect, otherwise it errors.
# - Stage 3 uses synthetic preferences / outcome rewards derived from GSM8K final
#   answer correctness (RLAIF-style).

set -euo pipefail

cd "$(dirname "$0")/.."
source /scratch/kk6081/ml_fall25/venv/bin/activate

# ============================================================================
# Configuration
# ============================================================================

MODE=${1:-dpo}  # dpo or grpo
MODEL_SIZE=${2:-medium}  # small, medium, gpt2-small
DEVICE=${DEVICE:-cuda:0}
OUTDIR=${OUTDIR:-/scratch/kk6081/picollm_extend}
YES=${YES:-0}  # set to 1 to skip interactive prompts
CLEAN_OLD_CHECKPOINTS=${CLEAN_OLD_CHECKPOINTS:-1}  # default: cleanup before training

# Stage-3 hyperparameters (overridable)
NUM_STEPS=${NUM_STEPS:-500}
LR=${LR:-1e-6}
WARMUP=${WARMUP:-50}
MAX_TOKENS=${MAX_TOKENS:-128}
GRAD_CLIP=${GRAD_CLIP:-1.0}
TOP_P=${TOP_P:-0.95}
TEMPERATURE=${TEMPERATURE:-1.0}
BETA=${BETA:-0.1}
KL_COEF=${KL_COEF:-0.01}
NUM_SAMPLES=${NUM_SAMPLES:-8}
ADVANTAGE_TYPE=${ADVANTAGE_TYPE:-group_relative}

# Auto-detect SFT checkpoint (GSM8K fine-tuned model from Stage 2)
# Priority: 1) Manual SFT_CKPT env var, 2) GSM8K checkpoints matching MODEL_SIZE
if [ -n "${SFT_CKPT:-}" ]; then
  BASE_CKPT="$SFT_CKPT"
  echo "✓ Using manual SFT checkpoint: $BASE_CKPT"
else
  # Helper to find checkpoint matching architecture
  find_matching_ckpt() {
    local size="$1"
    local outdir="$2"
    local want_embed want_blocks
    
    case "$size" in
      small) want_embed=384; want_blocks=3 ;;
      medium) want_embed=512; want_blocks=6 ;;
      gpt2-small) want_embed=768; want_blocks=12 ;;
      *) return 1 ;;
    esac
    
    local candidates=(
      "${outdir}/gsm8k_transformer_epoch"*.pt
      "${outdir}/finetune_gsm8k/transformer_epoch"*.pt
    )
    
    for ckpt in $(ls -t ${candidates[@]} 2>/dev/null); do
      local info
      info=$(python scripts/utils/check_checkpoint_arch.py "$ckpt" 2>/dev/null || true)
      local embed blocks
      embed=$(echo "$info" | awk '/embed_size:/ {print $2; exit}')
      blocks=$(echo "$info" | awk '/n_blocks:/ {print $2; exit}')
      
      if [[ "$embed" == "$want_embed" && "$blocks" == "$want_blocks" ]]; then
        echo "$ckpt"
        return 0
      fi
    done
    
    return 1
  }
  
  BASE_CKPT=$(find_matching_ckpt "$MODEL_SIZE" "$OUTDIR" || true)
  
  if [ -z "$BASE_CKPT" ]; then
    # Last resort: any recent checkpoint (with strong warning)
    BASE_CKPT=$(ls -t "$OUTDIR"/transformer_epoch*.pt 2>/dev/null | head -1)
    if [ -n "$BASE_CKPT" ]; then
      echo "⚠️  WARNING: No GSM8K checkpoint found matching $MODEL_SIZE!"
      echo "   Found non-matching checkpoint: $BASE_CKPT"
      echo "   This will likely cause architecture mismatch errors."
      echo ""
      echo "   Recommended: Run Stage 2 with matching model size:"
      echo "   bash scripts/train.sh gsm8k $MODEL_SIZE"
      echo ""
      if [[ "$YES" != "1" ]]; then
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
          exit 1
        fi
      else
        echo "YES=1 set; continuing (will likely fail on arch mismatch)."
      fi
    fi
  else
    echo "✓ Found GSM8K SFT checkpoint matching $MODEL_SIZE: $BASE_CKPT"
  fi
fi

if [ -z "$BASE_CKPT" ]; then
  echo "❌ No SFT checkpoint found!"
  echo ""
  echo "DPO/GRPO requires a GSM8K SFT checkpoint (Stage 2)."
  echo ""
  echo "📋 3-Stage Training Pipeline:"
  echo "  Stage 1: Orca-Math → Base Model [ASSUMED DONE]"
  echo "  Stage 2: GSM8K SFT → SFT Model  [MISSING ← YOU ARE HERE]"
  echo "  Stage 3: DPO/GRPO  → Final Model [REQUIRES STAGE 2]"
  echo ""
  echo "Run Stage 2 first:"
  echo "  bash scripts/train.sh gsm8k"
  echo ""
  echo "Or use the full pipeline script:"
  echo "  bash scripts/full_pipeline_gsm8k.sh"
  exit 1
fi

echo "=========================================="
echo "🎯 ${MODE^^} Post-Training on GSM8K (Stage 3)"
echo "=========================================="
echo "SFT checkpoint: $BASE_CKPT"
echo "Model size: $MODEL_SIZE"
echo "Device: $DEVICE"
echo "=========================================="
echo ""

# Output directory
OUT_DIR="$OUTDIR/${MODE}_gsm8k_${MODEL_SIZE}"

# Cleanup old DPO/GRPO checkpoints (default enabled)
if [[ "$CLEAN_OLD_CHECKPOINTS" == "1" ]]; then
  echo "🧹 Cleaning old ${MODE^^} checkpoints..."
  rm -rf "$OUT_DIR" 2>/dev/null || true
  rm -f "$OUTDIR"/transformer_${MODE}_*.pt 2>/dev/null || true
  echo "   ✓ Removed old ${MODE^^} outputs"
  echo ""
fi

mkdir -p "$OUT_DIR"

# Hyperparameters (conservative for GTX TITAN X 12GB)
# DPO/GRPO needs more memory due to reference model + policy model
case "$MODEL_SIZE" in
  small)
    BATCH=${BATCH:-8}   # Conservative for DPO/GRPO (uses 2 models)
    ;;
  medium)
    BATCH=${BATCH:-32}   # Increased for better GPU utilization
    ;;
  gpt2-small)
    BATCH=${BATCH:-2}   # Very conservative for largest model
    ;;
  *)
    echo "❌ Unsupported model size: $MODEL_SIZE"
    echo "   Supported: small, medium, gpt2-small"
    exit 1
    ;;
esac

echo "Stage 3 hyperparameters: steps=$NUM_STEPS lr=$LR batch=$BATCH max_new_tokens=$MAX_TOKENS top_p=$TOP_P temp=$TEMPERATURE"

# ============================================================================
# DPO Training
# ============================================================================

if [ "$MODE" == "dpo" ]; then
  echo "🚀 Starting DPO training..."
  echo ""
  
  python scripts/evaluation/dpo_grpo_training.py \
    --mode dpo \
    --init_from "$BASE_CKPT" \
    --train_data data/gsm8k_train.txt \
    --val_data data/gsm8k_val.txt \
    --out_dir "$OUT_DIR" \
    --transformer_size "$MODEL_SIZE" \
    --device "$DEVICE" \
    --num_steps "$NUM_STEPS" \
    --batch_size "$BATCH" \
    --lr "$LR" \
    --beta "$BETA" \
    --warmup_steps "$WARMUP" \
    --grad_clip "$GRAD_CLIP" \
    --max_new_tokens "$MAX_TOKENS" \
    --top_p "$TOP_P" \
    --temperature "$TEMPERATURE" \
    --save_every 100 \
    --log_every 10
  
  FINAL_MODEL="$OUT_DIR/transformer_dpo_final.pt"

# ============================================================================
# GRPO Training
# ============================================================================

elif [ "$MODE" == "grpo" ]; then
  echo "🚀 Starting GRPO training..."
  echo ""
  
  # If user didn't override NUM_SAMPLES, auto-adjust for tight memory.
  if [[ -z "${NUM_SAMPLES_OVERRIDE:-}" ]]; then
    # GRPO uses batch_size * num_samples total memory
    if [ "$BATCH" -le 2 ]; then
      NUM_SAMPLES=2  # Very tight memory
    elif [ "$BATCH" -le 4 ]; then
      NUM_SAMPLES=4  # Conservative
    else
      NUM_SAMPLES=${NUM_SAMPLES:-6}  # Reduced from 8
    fi
  fi
  
  python scripts/evaluation/dpo_grpo_training.py \
    --mode grpo \
    --init_from "$BASE_CKPT" \
    --train_data data/gsm8k_train.txt \
    --val_data data/gsm8k_val.txt \
    --out_dir "$OUT_DIR" \
    --transformer_size "$MODEL_SIZE" \
    --device "$DEVICE" \
    --num_steps "$NUM_STEPS" \
    --batch_size "$BATCH" \
    --lr "$LR" \
    --num_samples "$NUM_SAMPLES" \
    --kl_coef "$KL_COEF" \
    --advantage_type "$ADVANTAGE_TYPE" \
    --warmup_steps "$WARMUP" \
    --grad_clip "$GRAD_CLIP" \
    --max_new_tokens "$MAX_TOKENS" \
    --top_p "$TOP_P" \
    --temperature "$TEMPERATURE" \
    --save_every 100 \
    --log_every 10
  
  FINAL_MODEL="$OUT_DIR/transformer_grpo_final.pt"

else
  echo "❌ Unknown mode: $MODE"
  echo "   Use: bash scripts/train_dpo_grpo.sh [dpo|grpo] [model_size]"
  exit 1
fi

# ============================================================================
# Post-Training Evaluation
# ============================================================================

echo ""
echo "=" * 80
echo "✅ ${MODE^^} training complete!"
echo "=" * 80
echo "Final model: $FINAL_MODEL"
echo ""

if [ ! -f "$FINAL_MODEL" ]; then
  echo "⚠️  Final model not found, check training logs"
  exit 1
fi

echo "🧪 Testing model with sample prompt..."
python inference.py \
  --checkpoint "$FINAL_MODEL" \
  --prompt "Q: Janet has 3 apples and buys 2 more. How many does she have? A:" \
  --max_new_tokens 64 \
  --device "$DEVICE"

echo ""
echo "📊 To evaluate on full GSM8K test set, run:"
echo "   python scripts/evaluation/eval_reasoning.py \\"
echo "     --checkpoint $FINAL_MODEL \\"
echo "     --test_file data/gsm8k_test.txt \\"
echo "     --device $DEVICE"
echo ""
echo "🎉 Done!"
