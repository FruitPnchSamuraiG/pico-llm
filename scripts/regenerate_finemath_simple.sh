#!/usr/bin/env bash
set -euo pipefail

# Regenerate FineMath data with SIMPLER examples (max 512 chars)
# This filters out complex graduate-level problems and keeps elementary examples

cd "$(dirname "$0")/.."
source /scratch/kk6081/ml_fall25/venv/bin/activate

DATA_DIR="data"
BACKUP_DIR="data/backup_complex"

echo "=========================================="
echo "🔄 Regenerating FineMath with SIMPLE examples"
echo "=========================================="

# Backup old complex data
mkdir -p "$BACKUP_DIR"
if [[ -f "$DATA_DIR/finemath_train.txt" ]]; then
  echo "📦 Backing up old data to $BACKUP_DIR/"
  mv "$DATA_DIR/finemath_train.txt" "$BACKUP_DIR/finemath_train_complex.txt"
  mv "$DATA_DIR/finemath_val.txt" "$BACKUP_DIR/finemath_val_complex.txt"
fi

# Download with stricter length filters
echo ""
echo "📥 Downloading FineMath with filters:"
echo "  - Min length: 100 chars"
echo "  - Max length: 512 chars (simple problems only)"
echo "  - Target: 100k examples"
echo ""

python3 scripts/prepare_hf_finemath_data.py \
  --output_dir "$DATA_DIR" \
  --max_samples 100000 \
  --min_length 100 \
  --max_length 512 \
  --val_split 0.05

echo ""
echo "=========================================="
echo "✅ Simple FineMath data ready!"
echo ""
echo "Next: Restart training from scratch"
echo "  bash scripts/train_transformer_finemath.sh"
echo "=========================================="
