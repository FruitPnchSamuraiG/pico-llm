#!/usr/bin/env bash
# Complete 3-Stage Training Pipeline for GSM8K Reasoning Model
# Stage 1: Orca-Math → Base Model
# Stage 2: GSM8K SFT → SFT Model
# Stage 3: DPO/GRPO (synthetic preferences via correctness; RLAIF-style) → Final Model
#
# Usage:
#   bash scripts/full_pipeline_gsm8k.sh [dpo|grpo] [model_size]
#
# Non-interactive (CI / remote):
#   YES=1 bash scripts/full_pipeline_gsm8k.sh dpo medium

set -euo pipefail

cd "$(dirname "$0")/.."
source /scratch/kk6081/ml_fall25/venv/bin/activate

# ============================================================================
# Configuration
# ============================================================================

RL_MODE=${1:-dpo}  # dpo or grpo
MODEL_SIZE=${2:-medium}  # small, medium, gpt2-small
DEVICE=${DEVICE:-cuda:0}
OUTDIR=${OUTDIR:-/scratch/kk6081/picollm_extend}

# Stage control
SKIP_STAGE1=${SKIP_STAGE1:-0}  # Set to 1 if Orca training already done
SKIP_STAGE2=${SKIP_STAGE2:-0}  # Set to 1 if GSM8K SFT already done
SKIP_STAGE3=${SKIP_STAGE3:-0}  # Set to 1 if you only want SFT
YES=${YES:-0}                 # set to 1 to skip confirmation prompt

# Stage 1 config (Orca-Math)
STAGE1_EPOCHS=${STAGE1_EPOCHS:-8}
STAGE1_BATCH=${STAGE1_BATCH:-16}
STAGE1_LR=${STAGE1_LR:-3e-4}

# Stage 2 config (GSM8K SFT)
STAGE2_EPOCHS=${STAGE2_EPOCHS:-8}
STAGE2_BATCH=${STAGE2_BATCH:-16}
STAGE2_LR=${STAGE2_LR:-2e-4}
STAGE2_RUN_RL=${STAGE2_RUN_RL:-0}  # Disable old RL (we use DPO/GRPO instead)

# Stage 3 config (DPO/GRPO)
STAGE3_STEPS=${STAGE3_STEPS:-500}
STAGE3_LR=${STAGE3_LR:-1e-6}
STAGE3_BETA=${STAGE3_BETA:-0.1}       # DPO beta
STAGE3_KL_COEF=${STAGE3_KL_COEF:-0.01} # GRPO KL coefficient
STAGE3_NUM_SAMPLES=${STAGE3_NUM_SAMPLES:-8} # GRPO group size
STAGE3_MAX_NEW_TOKENS=${STAGE3_MAX_NEW_TOKENS:-128}
STAGE3_TEMPERATURE=${STAGE3_TEMPERATURE:-1.0}
STAGE3_TOP_P=${STAGE3_TOP_P:-0.95}

# Auto-adjust batch sizes for model size
case "$MODEL_SIZE" in
  small)
    STAGE1_BATCH=${STAGE1_BATCH:-16}
    STAGE2_BATCH=${STAGE2_BATCH:-16}
    STAGE3_BATCH=16
    ;;
  medium)
    STAGE1_BATCH=${STAGE1_BATCH:-16}
    STAGE2_BATCH=${STAGE2_BATCH:-16}
    STAGE3_BATCH=8
    ;;
  gpt2-small)
    STAGE1_BATCH=${STAGE1_BATCH:-8}
    STAGE2_BATCH=${STAGE2_BATCH:-8}
    STAGE3_BATCH=4
    ;;
  *)
    echo "❌ Unsupported model size: $MODEL_SIZE"
    echo "   Supported: small, medium, gpt2-small"
    exit 1
    ;;
esac

# ============================================================================
# Pipeline Info
# ============================================================================

echo ""
echo "================================================================================"
echo "🎯 3-STAGE TRAINING PIPELINE FOR GSM8K REASONING"
echo "================================================================================"
echo ""
echo "📋 Pipeline Overview:"
echo "  Stage 1: Orca-Math SFT    → Base Model (foundational math reasoning)"
echo "  Stage 2: GSM8K SFT        → SFT Model (task-specific adaptation)"
echo "  Stage 3: ${RL_MODE^^} Post-Training → Final Model (preference optimization)"
echo ""
echo "⚙️  Configuration:"
echo "  Model Size: $MODEL_SIZE"
echo "  Device: $DEVICE"
echo "  Output Dir: $OUTDIR"
echo "  RL Algorithm: $RL_MODE"
echo ""
echo "🎛️  Hyperparameters:"
echo "  Stage 1: epochs=$STAGE1_EPOCHS, batch=$STAGE1_BATCH, lr=$STAGE1_LR"
echo "  Stage 2: epochs=$STAGE2_EPOCHS, batch=$STAGE2_BATCH, lr=$STAGE2_LR"
echo "  Stage 3: steps=$STAGE3_STEPS, batch=$STAGE3_BATCH, lr=$STAGE3_LR"
echo ""
echo "⏭️  Skip Control:"
echo "  SKIP_STAGE1=$SKIP_STAGE1 (Orca training)"
echo "  SKIP_STAGE2=$SKIP_STAGE2 (GSM8K SFT)"
echo "  SKIP_STAGE3=$SKIP_STAGE3 (DPO/GRPO)"
echo ""
echo "================================================================================"
echo ""

# Confirmation prompt
if [[ "$YES" != "1" ]]; then
  read -p "Continue with pipeline? (y/N): " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
  fi
else
  echo "YES=1 set; running non-interactively."
fi

# ============================================================================
# Stage 1: Orca-Math Base Training
# ============================================================================

STAGE1_CKPT="$OUTDIR/transformer_epoch${STAGE1_EPOCHS}.pt"

if [ "$SKIP_STAGE1" == "1" ]; then
  echo ""
  echo "⏭️  STAGE 1 SKIPPED (SKIP_STAGE1=1)"
  echo ""
  
  # Check if checkpoint exists
  if [ ! -f "$STAGE1_CKPT" ]; then
    echo "⚠️  Warning: Expected Orca checkpoint not found: $STAGE1_CKPT"
    # Try to find any checkpoint
    STAGE1_CKPT=$(ls -t "$OUTDIR"/transformer_epoch*.pt 2>/dev/null | head -1)
    if [ -z "$STAGE1_CKPT" ]; then
      echo "❌ No base checkpoint found! Cannot skip Stage 1."
      exit 1
    fi
    echo "✓ Using existing checkpoint: $STAGE1_CKPT"
  else
    echo "✓ Using Orca checkpoint: $STAGE1_CKPT"
  fi
else
  echo ""
  echo "================================================================================"
  echo "🚀 STAGE 1: Training Base Model on Orca-Math"
  echo "================================================================================"
  echo ""
  
  EPOCHS=$STAGE1_EPOCHS \
  BATCH=$STAGE1_BATCH \
  LR=$STAGE1_LR \
  TRANSFORMER_SIZE=$MODEL_SIZE \
  DEVICE=$DEVICE \
  OUTDIR=$OUTDIR \
  bash scripts/train.sh orca
  
  if [ ! -f "$STAGE1_CKPT" ]; then
    echo "❌ Stage 1 failed: Checkpoint not created"
    exit 1
  fi
  
  echo ""
  echo "✅ STAGE 1 COMPLETE"
  echo "   Base checkpoint: $STAGE1_CKPT"
  echo ""
fi

# ============================================================================
# Stage 2: GSM8K Supervised Fine-Tuning
# ============================================================================

# Expected SFT checkpoint location
STAGE2_CKPT="$OUTDIR/gsm8k_transformer_epoch${STAGE2_EPOCHS}.pt"

if [ "$SKIP_STAGE2" == "1" ]; then
  echo ""
  echo "⏭️  STAGE 2 SKIPPED (SKIP_STAGE2=1)"
  echo ""
  
  # Check if SFT checkpoint exists
  if [ ! -f "$STAGE2_CKPT" ]; then
    echo "⚠️  Warning: Expected GSM8K SFT checkpoint not found: $STAGE2_CKPT"
    # Try to find any GSM8K checkpoint
    STAGE2_CKPT=$(ls -t "$OUTDIR"/gsm8k_transformer_epoch*.pt 2>/dev/null | head -1)
    if [ -z "$STAGE2_CKPT" ]; then
      # Try finetune directory
      STAGE2_CKPT=$(ls -t "$OUTDIR"/finetune_gsm8k/transformer_epoch*.pt 2>/dev/null | head -1)
    fi
    if [ -z "$STAGE2_CKPT" ]; then
      echo "❌ No GSM8K SFT checkpoint found! Cannot skip Stage 2."
      exit 1
    fi
    echo "✓ Using existing SFT checkpoint: $STAGE2_CKPT"
  else
    echo "✓ Using GSM8K SFT checkpoint: $STAGE2_CKPT"
  fi
else
  echo ""
  echo "================================================================================"
  echo "🚀 STAGE 2: Fine-Tuning on GSM8K (Supervised Learning)"
  echo "================================================================================"
  echo "Starting from: $STAGE1_CKPT"
  echo ""
  
  BASE_CKPT=$STAGE1_CKPT \
  EPOCHS=$STAGE2_EPOCHS \
  BATCH=$STAGE2_BATCH \
  LR=$STAGE2_LR \
  TRANSFORMER_SIZE=$MODEL_SIZE \
  DEVICE=$DEVICE \
  OUTDIR=$OUTDIR \
  RUN_RL=$STAGE2_RUN_RL \
  bash scripts/train.sh gsm8k
  
  # Verify checkpoint was created
  if [ ! -f "$STAGE2_CKPT" ]; then
    echo "⚠️  Expected checkpoint not found: $STAGE2_CKPT"
    # Try to find the actual checkpoint
    STAGE2_CKPT=$(ls -t "$OUTDIR"/gsm8k_transformer_epoch*.pt 2>/dev/null | head -1)
    if [ -z "$STAGE2_CKPT" ]; then
      STAGE2_CKPT=$(ls -t "$OUTDIR"/finetune_gsm8k/transformer_epoch*.pt 2>/dev/null | head -1)
    fi
    if [ -z "$STAGE2_CKPT" ]; then
      echo "❌ Stage 2 failed: No SFT checkpoint created"
      exit 1
    fi
    echo "✓ Found SFT checkpoint: $STAGE2_CKPT"
  fi
  
  echo ""
  echo "✅ STAGE 2 COMPLETE"
  echo "   SFT checkpoint: $STAGE2_CKPT"
  echo ""
fi

# ============================================================================
# Stage 3: DPO/GRPO Post-Training
# ============================================================================

if [ "$SKIP_STAGE3" == "1" ]; then
  echo ""
  echo "⏭️  STAGE 3 SKIPPED (SKIP_STAGE3=1)"
  echo ""
  echo "================================================================================"
  echo "✅ PIPELINE COMPLETE (Stages 1-2 only)"
  echo "================================================================================"
  echo "Final model: $STAGE2_CKPT"
  echo ""
  echo "To run Stage 3 (DPO/GRPO) later:"
  echo "  SFT_CKPT=$STAGE2_CKPT bash scripts/train_dpo_grpo.sh $RL_MODE $MODEL_SIZE"
  echo ""
  exit 0
fi

echo ""
echo "================================================================================"
echo "🚀 STAGE 3: ${RL_MODE^^} Post-Training (Preference Optimization)"
echo "================================================================================"
echo "Starting from: $STAGE2_CKPT"
echo "Notes: preferences are derived from GSM8K answer correctness (synthetic / RLAIF-style)."
echo ""

echo "Stage 3 hyperparameters:"
echo "  steps=$STAGE3_STEPS lr=$STAGE3_LR batch=$STAGE3_BATCH max_new_tokens=$STAGE3_MAX_NEW_TOKENS"
if [[ "$RL_MODE" == "dpo" ]]; then
  echo "  dpo.beta=$STAGE3_BETA"
else
  echo "  grpo.kl_coef=$STAGE3_KL_COEF grpo.num_samples=$STAGE3_NUM_SAMPLES"
fi

echo "  sampling: top_p=$STAGE3_TOP_P temperature=$STAGE3_TEMPERATURE"
echo ""

# Prefer calling the trainer directly so stage-3 hyperparameters are explicit.
# (train_dpo_grpo.sh can still be used separately; this keeps the pipeline self-contained.)
python scripts/evaluation/dpo_grpo_training.py \
  --mode "$RL_MODE" \
  --init_from "$STAGE2_CKPT" \
  --train_data data/gsm8k_train.txt \
  --val_data data/gsm8k_val.txt \
  --out_dir "$OUTDIR/${RL_MODE}_gsm8k_${MODEL_SIZE}" \
  --transformer_size "$MODEL_SIZE" \
  --device "$DEVICE" \
  --num_steps "$STAGE3_STEPS" \
  --batch_size "$STAGE3_BATCH" \
  --lr "$STAGE3_LR" \
  --max_new_tokens "$STAGE3_MAX_NEW_TOKENS" \
  --top_p "$STAGE3_TOP_P" \
  --temperature "$STAGE3_TEMPERATURE" \
  --beta "$STAGE3_BETA" \
  --kl_coef "$STAGE3_KL_COEF" \
  --num_samples "$STAGE3_NUM_SAMPLES" \
  --save_every 100 \
  --log_every 10

STAGE3_CKPT="$OUTDIR/${RL_MODE}_gsm8k_${MODEL_SIZE}/transformer_${RL_MODE}_final.pt"

if [ ! -f "$STAGE3_CKPT" ]; then
  echo "❌ Stage 3 failed: Final checkpoint not created"
  exit 1
fi

echo ""
echo "✅ STAGE 3 COMPLETE"
echo "   Final checkpoint: $STAGE3_CKPT"
echo ""

# ============================================================================
# Pipeline Summary
# ============================================================================

echo ""
echo "================================================================================"
echo "🎉 FULL PIPELINE COMPLETE!"
echo "================================================================================"
echo ""
echo "📊 Training Progression:"
echo "  Stage 1 (Base):  $STAGE1_CKPT"
echo "  Stage 2 (SFT):   $STAGE2_CKPT"
echo "  Stage 3 (${RL_MODE^^}):   $STAGE3_CKPT"
echo ""
echo "================================================================================"
echo "📈 Next Steps:"
echo "================================================================================"
echo ""
echo "1️⃣  Test the final model:"
echo "   python inference.py \\"
echo "     --checkpoint $STAGE3_CKPT \\"
echo "     --prompt 'Q: Janet has 5 apples and buys 3 more. How many does she have? A:' \\"
echo "     --device $DEVICE"
echo ""
echo "2️⃣  Evaluate on GSM8K test set:"
echo "   python scripts/evaluation/eval_reasoning.py \\"
echo "     --checkpoint $STAGE3_CKPT \\"
echo "     --test_file data/gsm8k_test.txt \\"
echo "     --device $DEVICE"
echo ""
echo "3️⃣  Compare all checkpoints:"
echo "   for ckpt in '$STAGE1_CKPT' '$STAGE2_CKPT' '$STAGE3_CKPT'; do"
echo "     echo \"Evaluating: \$ckpt\""
echo "     python scripts/evaluation/eval_reasoning.py --checkpoint \$ckpt --device $DEVICE"
echo "   done"
echo ""
echo "================================================================================"
echo "🎊 Success! Your GSM8K reasoning model is ready."
echo "================================================================================"
echo ""
