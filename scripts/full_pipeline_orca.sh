#!/usr/bin/env bash
set -euo pipefail

# Complete workflow: Download full Orca-Math dataset and start training
# This replaces the previous FineMath workflow with cleaner, higher-quality data

cd "$(dirname "$0")/.."
source /scratch/kk6081/ml_fall25/venv/bin/activate

echo "=========================================="
echo "🚀 Orca-Math Training Pipeline"
echo "=========================================="
echo ""
echo "This will:"
echo "  1. Download 100k Orca-Math word problems (200k available)"
echo "  2. Train transformer base model (8 epochs, ~8-10 hours)"
echo "  3. Prepare for GSM8K fine-tuning"
echo ""
echo "Expected results:"
echo "  • Base loss: < 1.5 by epoch 6"
echo "  • GSM8K accuracy: 45-60% (after fine-tuning)"
echo "  • Total time: ~20-24 hours"
echo ""
read -p "Continue? [y/N] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "Aborted."
  exit 0
fi

echo ""
echo "=========================================="
echo "📥 Step 1: Download Orca-Math Dataset"
echo "=========================================="
echo ""

# Download 100k high-quality math word problems
python3 scripts/prepare_orca_math_data.py \
  --output_dir data \
  --max_samples 100000 \
  --min_length 50 \
  --max_length 1024 \
  --val_split 0.05

echo ""
echo "=========================================="
echo "✅ Data Download Complete"
echo "=========================================="
echo ""

# Show data statistics
TRAIN_COUNT=$(wc -l < data/orca_math_train.txt)
VAL_COUNT=$(wc -l < data/orca_math_val.txt)
AVG_LEN=$(head -100 data/orca_math_train.txt | awk '{print length}' | awk '{sum+=$1; count+=1} END {print int(sum/count)}')

echo "📊 Dataset statistics:"
echo "  • Training examples: $TRAIN_COUNT"
echo "  • Validation examples: $VAL_COUNT"
echo "  • Average length: $AVG_LEN chars"
echo ""

echo "Sample examples:"
head -3 data/orca_math_train.txt | while IFS= read -r line; do
  echo "  ${line:0:150}..."
done

echo ""
echo "=========================================="
echo "🏋️  Step 2: Start Base Training"
echo "=========================================="
echo ""
echo "Training will start in 5 seconds..."
echo "Press Ctrl+C to abort"
sleep 5

# Start Orca-Math training
bash scripts/train_transformer_orca.sh

echo ""
echo "=========================================="
echo "✅ Base Training Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Verify base model quality:"
echo "   python inference.py \\"
echo "     --checkpoint /scratch/kk6081/picollm_extend/transformer_epoch8.pt \\"
echo "     --prompt 'Q: A store sells 5 apples for \$2. How much for 15 apples? A:' \\"
echo "     --device cuda:0 --max_new_tokens 100"
echo ""
echo "2. Fine-tune on GSM8K (10 epochs, ~12-14 hours):"
echo "   BASE_CKPT=/scratch/kk6081/picollm_extend/transformer_epoch8.pt \\"
echo "     EPOCHS=10 LR=5e-4 RUN_RL=1 \\"
echo "     bash scripts/train_transformer_gsm8k.sh"
echo ""
echo "3. Evaluate final accuracy:"
echo "   python inference.py \\"
echo "     --checkpoint /scratch/kk6081/picollm_extend/gsm8k_transformer_epoch10.pt \\"
echo "     --prompt 'Q: Janet has 24 eggs. She eats 3 for breakfast. How many left? A:' \\"
echo "     --device cuda:0"
echo ""
echo "=========================================="
echo "Expected GSM8K accuracy: 45-60%"
echo "=========================================="
