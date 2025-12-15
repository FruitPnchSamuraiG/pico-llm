#!/usr/bin/env bash
# Reasoning Model Training Pipeline
# Implements <thinking> blocks for chain-of-thought reasoning

set -euo pipefail

cd "$(dirname "$0")/.."
source /scratch/kk6081/ml_fall25/venv/bin/activate

# ============================================================================
# Configuration
# ============================================================================

DEVICE=${DEVICE:-cuda:0}
OUTDIR=${OUTDIR:-/scratch/kk6081/picollm_extend}
DATA_DIR=${DATA_DIR:-data}

# Model settings
TRANSFORMER_SIZE=${TRANSFORMER_SIZE:-medium}
BLOCK_SIZE=${BLOCK_SIZE:-384}  # Longer for thinking blocks

# Reasoning settings
THINKING_STYLE=${THINKING_STYLE:-structured}  # verbose | concise | structured
REWARD_MODE=${REWARD_MODE:-orm}  # orm | prm | hybrid

# Training settings
NUM_STEPS=${NUM_STEPS:-1000}
BATCH_SIZE=${BATCH_SIZE:-4}
LR=${LR:-1e-6}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}  # Legacy parameter (backward compatibility)

# Thinking-aware generation settings (NEW!)
MAX_THINKING_TOKENS=${MAX_THINKING_TOKENS:-800}  # Generous thinking budget
MAX_ANSWER_TOKENS=${MAX_ANSWER_TOKENS:-200}      # Separate answer budget

# ============================================================================
# Functions
# ============================================================================

print_header() {
  echo ""
  echo "=========================================="
  echo "$1"
  echo "=========================================="
}

# ============================================================================
# Main
# ============================================================================

print_header "🧠 Reasoning Model Training Pipeline"

echo "Configuration:"
echo "  Model: $TRANSFORMER_SIZE"
echo "  Thinking style: $THINKING_STYLE"
echo "  Reward mode: $REWARD_MODE"
echo "  Training steps: $NUM_STEPS"
echo "  Device: $DEVICE"
echo ""

# Step 1: Find SFT checkpoint
print_header "📦 Step 1: Locating SFT Checkpoint"

SFT_CKPT=""
if [[ -n "${BASE_CKPT:-}" ]]; then
  SFT_CKPT="$BASE_CKPT"
elif [[ -f "$OUTDIR/gsm8k_transformer_epoch10.pt" ]]; then
  SFT_CKPT="$OUTDIR/gsm8k_transformer_epoch10.pt"
elif [[ -f "$OUTDIR/gsm8k_transformer_epoch8.pt" ]]; then
  SFT_CKPT="$OUTDIR/gsm8k_transformer_epoch8.pt"
else
  # Find latest gsm8k checkpoint
  SFT_CKPT=$(ls -1t "$OUTDIR"/gsm8k_transformer_epoch*.pt 2>/dev/null | head -1 || echo "")
fi

if [[ -z "$SFT_CKPT" || ! -f "$SFT_CKPT" ]]; then
  echo "❌ No SFT checkpoint found!"
  echo ""
  echo "Please train SFT model first:"
  echo "  bash scripts/train.sh gsm8k"
  echo ""
  exit 1
fi

echo "✓ Found SFT checkpoint: $SFT_CKPT"
echo ""

# Step 2: Prepare reasoning data with <thinking> blocks
print_header "📝 Step 2: Preparing Reasoning Data"

REASONING_TRAIN="$DATA_DIR/gsm8k_train_reasoning_${THINKING_STYLE}.txt"
REASONING_VAL="$DATA_DIR/gsm8k_val_reasoning_${THINKING_STYLE}.txt"

if [[ ! -f "$REASONING_TRAIN" ]]; then
  echo "Creating training data with <thinking> blocks..."
  python3 scripts/evaluation/reasoning_training.py \
    --init_from "$SFT_CKPT" \
    --train_data "$DATA_DIR/gsm8k_train.txt" \
    --out_dir "$OUTDIR/reasoning_prep" \
    --thinking_style "$THINKING_STYLE" 2>&1 | grep -E "^(✓|📝|•)"
fi

if [[ ! -f "$REASONING_VAL" && -f "$DATA_DIR/gsm8k_val.txt" ]]; then
  echo "Creating validation data..."
  python3 -c "
from pathlib import Path
import sys
sys.path.insert(0, 'scripts/evaluation')
from reasoning_training import prepare_reasoning_data
prepare_reasoning_data(
    '$DATA_DIR/gsm8k_val.txt',
    '$REASONING_VAL',
    '$THINKING_STYLE'
)
"
fi

echo "✓ Reasoning data ready"
echo "  Train: $REASONING_TRAIN"
echo "  Val:   $REASONING_VAL"
echo ""

# Step 3: Train with DPO on reasoning data
print_header "🚀 Step 3: Training Reasoning Model"

REASONING_OUT="$OUTDIR/reasoning_${THINKING_STYLE}_${REWARD_MODE}"
mkdir -p "$REASONING_OUT"

echo "Training with:"
echo "  Method: DPO with $REWARD_MODE rewards"
echo "  Thinking style: $THINKING_STYLE"
echo "  Output: $REASONING_OUT"
echo ""

# Use existing DPO trainer but with reasoning data
python3 scripts/evaluation/dpo_grpo_training.py \
  --mode dpo \
  --init_from "$SFT_CKPT" \
  --train_data "$REASONING_TRAIN" \
  --val_data "$REASONING_VAL" \
  --out_dir "$REASONING_OUT" \
  --transformer_size "$TRANSFORMER_SIZE" \
  --block_size "$BLOCK_SIZE" \
  --num_steps "$NUM_STEPS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --device "$DEVICE" \
  --log_every 10 \
  --eval_every 100 \
  --save_every 200

print_header "✅ Reasoning Model Training Complete!"

echo "📁 Checkpoints saved to: $REASONING_OUT"
echo ""
echo "🧪 Test your reasoning model:"
echo ""
echo "# Example 1: Test with thinking blocks visible"
echo "python3 scripts/inference_dpo.py \\"
echo "  --checkpoint $REASONING_OUT/transformer_dpo_final.pt \\"
echo "  --prompt 'Q: If John has 3 apples and buys 5 more, how many does he have? A:' \\"
echo "  --max_new_tokens 256"
echo ""
echo "# Example 2: Best-of-8 sampling"
echo "python3 -c \""
echo "from scripts.evaluation.reasoning_training import best_of_n_sampling, _load_inference_module
import tiktoken
import torch

inf = _load_inference_module()
enc = tiktoken.get_encoding('gpt2')
model = inf.TransformerModel(vocab_size=enc.n_vocab, block_size=$BLOCK_SIZE, d_model=512, n_heads=8, n_blocks=6, ff_mult=4)
model.load_state_dict(torch.load('$REASONING_OUT/transformer_dpo_final.pt', map_location='$DEVICE'))
model.to('$DEVICE')

prompt = 'Q: If 3 people each have 4 apples, how many apples total? A:'
gold = '12'

best, score, all_results = best_of_n_sampling(
    model, enc, inf, prompt, gold, n=8, device='$DEVICE'
)

print('Best completion:')
print(best)
print(f'Score: {score:.3f}')
print(f'All scores: {[s for _, s in all_results]}')\""
echo ""
echo "📊 Evaluate on GSM8K test set:"
echo "python3 scripts/evaluation/eval_reasoning.py \\"
echo "  --checkpoint $REASONING_OUT/transformer_dpo_final.pt \\"
echo "  --device $DEVICE \\"
echo "  --use_thinking_blocks"
echo ""
