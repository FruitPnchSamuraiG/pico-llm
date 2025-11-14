#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

# Common quick settings for verification-only runs
COMMON_ARGS=(
  --device_id cuda:1
#   --input_files 3seqs.txt
  --block_size 256
  --batch_size 16
  --num_epochs 3
  --learning_rate 0.001
  --train_subset_size 20000
  --log_interval_steps 100
  --sample_interval_seconds 5
)

# Run 1: smaller embed, k=3, fewer MLP layers
echo "[Run 1] kgram k=3, layers=1, emb=256, transformer heads=2 blocks=2, lstm_hidden=256"
$PYTHON pico-llm.py \
  "${COMMON_ARGS[@]}" \
  --embed_size 256 \
  --kgram_k 3 \
  --num_inner_mlp_layers 1 \
  --kgram_chunk_size 1 \
  --transformer_heads 2 \
  --transformer_blocks 2 \
  --lstm_hidden_size 256 \
  --run_tag r1_k3_l1_emb256_h2_b2

# Run 2: k=4, more MLP layers
echo "[Run 2] kgram k=4, layers=2, emb=512, transformer heads=4 blocks=2, lstm_hidden=512"
$PYTHON pico-llm.py \
  "${COMMON_ARGS[@]}" \
  --embed_size 512 \
  --kgram_k 4 \
  --num_inner_mlp_layers 2 \
  --kgram_chunk_size 1 \
  --transformer_heads 4 \
  --transformer_blocks 2 \
  --lstm_hidden_size 512 \
  --run_tag r2_k4_l2_emb512_h4_b2

# Run 3: increase transformer depth, chunk size for kgram
echo "[Run 3] kgram k=5, layers=2, chunk=2, emb=512, transformer heads=4 blocks=3, lstm_hidden=512"
$PYTHON pico-llm.py \
  "${COMMON_ARGS[@]}" \
  --embed_size 512 \
  --kgram_k 5 \
  --num_inner_mlp_layers 2 \
  --kgram_chunk_size 2 \
  --transformer_heads 4 \
  --transformer_blocks 3 \
  --lstm_hidden_size 512 \
  --run_tag r3_k5_l2_chunk2_emb512_h4_b3

# Run 4: larger embed, more heads
echo "[Run 4] kgram k=4, layers=3, emb=768, transformer heads=6 blocks=3, lstm_hidden=768"
$PYTHON pico-llm.py \
  "${COMMON_ARGS[@]}" \
  --embed_size 768 \
  --kgram_k 4 \
  --num_inner_mlp_layers 3 \
  --kgram_chunk_size 1 \
  --transformer_heads 6 \
  --transformer_blocks 3 \
  --lstm_hidden_size 768 \
  --run_tag r4_k4_l3_emb768_h6_b3

# Run 5: small embed, deeper transformer, different prompt
echo "[Run 5] kgram k=2, layers=1, emb=256, transformer heads=4 blocks=4, lstm_hidden=256, custom prompt"
$PYTHON pico-llm.py \
  "${COMMON_ARGS[@]}" \
  --embed_size 256 \
  --kgram_k 2 \
  --num_inner_mlp_layers 1 \
  --kgram_chunk_size 1 \
  --transformer_heads 4 \
  --transformer_blocks 4 \
  --lstm_hidden_size 256 \
  --prompt "In a tiny world," \
  --run_tag r5_k2_l1_emb256_h4_b4

echo "All runs completed. Check generated PNGs and checkpoints in $ROOT_DIR";