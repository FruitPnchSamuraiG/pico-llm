#!/usr/bin/env bash
set -euo pipefail

# Prepare GSM8K (openai/gsm8k) into line-based text files for Pico-LLM.

cd "$(dirname "$0")/.."
source /scratch/kk6081/ml_fall25/venv/bin/activate

DATA_DIR=${DATA_DIR:-data}
mkdir -p "$DATA_DIR"

# 2% train for val by default
TRAIN_SPLIT=${TRAIN_SPLIT:-"train[2%:]"}
VAL_SPLIT=${VAL_SPLIT:-"train[:2%]"}
TEST_SPLIT=${TEST_SPLIT:-"test"}

OUT_TRAIN=${OUT_TRAIN:-"$DATA_DIR/gsm8k_train.txt"}
OUT_VAL=${OUT_VAL:-"$DATA_DIR/gsm8k_val.txt"}
OUT_TEST=${OUT_TEST:-"$DATA_DIR/gsm8k_test.txt"}

python scripts/prepare_hf_reasoning_data.py \
  --dataset openai/gsm8k --config main \
  --train_split "$TRAIN_SPLIT" \
  --val_split "$VAL_SPLIT" \
  --test_split "$TEST_SPLIT" \
  --out_train "$OUT_TRAIN" \
  --out_val "$OUT_VAL" \
  --out_test "$OUT_TEST"

echo "✅ GSM8K prepared:"
echo "  $OUT_TRAIN"
echo "  $OUT_VAL"
echo "  $OUT_TEST"
