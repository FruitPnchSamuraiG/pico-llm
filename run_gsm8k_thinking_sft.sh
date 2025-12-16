#!/bin/bash
# Train GSM8K with thinking blocks using the updated configuration

cd /home/kk6081/pico_llm_extend/pico-llm
source /scratch/kk6081/ml_fall25/venv/bin/activate

echo "=========================================="
echo "🧠 GSM8K SFT Training with Thinking Blocks"
echo "=========================================="
echo ""
echo "📋 Configuration:"
echo "  Base model: Orca epoch 8 (transformer_epoch8.pt)"
echo "  Training data: gsm8k_train_reasoning_structured.txt (with <thinking> blocks)"
echo "  Block size: 512 tokens (to fit full reasoning chains)"
echo "  Epochs: 5 (prevents overfitting)"
echo "  Learning rate: 1e-4 (preserves Orca knowledge)"
echo "  Output: /scratch/kk6081/picollm_extend/finetune_gsm8k/"
echo ""
echo "⏱️  Estimated time: ~10-15 hours (5 epochs × ~2-3 hours/epoch)"
echo ""
echo "=========================================="
echo ""

# Set environment variables
export BASE_CKPT="/scratch/kk6081/picollm_extend/transformer_epoch8.pt"
export OUTDIR="/scratch/kk6081/picollm_extend"
export DEVICE="cuda:0"
export DATA_DIR="data"

# CRITICAL FIX: Lower learning rate for SFT to prevent catastrophic forgetting
# The echo said 1e-4, but the script was defaulting to 3e-4.
# We'll set it to 2e-5 for safer fine-tuning.
export LR="2e-5"
export EPOCHS="3" # Reduce epochs further to prevent overfitting

# Run training
bash scripts/train.sh gsm8k medium 2>&1 | tee /scratch/kk6081/picollm_extend/gsm8k_reasoning_sft.log
