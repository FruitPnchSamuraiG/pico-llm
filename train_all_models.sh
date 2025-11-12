#!/bin/bash
# Train all models (K-gram, LSTM, Transformer) and generate comparison plots
# 
# Usage:
#   bash train_all_models.sh              # Quick training (CPU-friendly)
#   bash train_all_models.sh --gpu        # GPU training (larger models)

set -e  # Exit on error

echo "=========================================="
echo "Training All Models: K-gram, LSTM, Transformer"
echo "=========================================="

# Parse arguments
GPU_MODE=false
if [[ "$1" == "--gpu" ]]; then
    GPU_MODE=true
    echo "🚀 GPU mode enabled"
fi

# Set hyperparameters based on mode
if [ "$GPU_MODE" = true ]; then
    # GPU configuration (optimized for 12GB GPU)
    BATCH_SIZE=16           # Reduced from 32
    EPOCHS=3                # Reduced from 5
    BLOCK_SIZE=256          # Reduced from 512 (major memory saver!)
    EMBED_SIZE=384          # Reduced from 512
    DEVICE="cuda:0"
    MAX_STEPS=""  # No limit
    echo "Configuration: GPU (batch=$BATCH_SIZE, epochs=$EPOCHS, block=$BLOCK_SIZE, embed=$EMBED_SIZE)"
else
    # CPU configuration (smaller, faster)
    BATCH_SIZE=8
    EPOCHS=2
    BLOCK_SIZE=256
    EMBED_SIZE=256
    DEVICE="cpu"
    MAX_STEPS="--max_steps_per_epoch 50"
    echo "Configuration: CPU (batch=$BATCH_SIZE, epochs=$EPOCHS, block=$BLOCK_SIZE, embed=$EMBED_SIZE)"
fi

# Clean up old files
echo ""
echo "🧹 Cleaning up old files..."
rm -f loss_histories.pkl training_losses.png
rm -f kgram_mlp_seq_epoch*.pt lstm_seq_epoch*.pt transformer_epoch*.pt

# Run training
echo ""
echo "🏋️  Starting training..."
echo "This will train 3 models: K-gram MLP, LSTM, and Transformer"
echo ""

# Set PyTorch memory allocation config for better GPU memory management
if [ "$GPU_MODE" = true ]; then
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    echo "💾 GPU memory optimization enabled"
    echo ""
fi

python pico-llm.py \
    --enable_kgram \
    --enable_transformer \
    --batch_size $BATCH_SIZE \
    --num_epochs $EPOCHS \
    --block_size $BLOCK_SIZE \
    --embed_size $EMBED_SIZE \
    --device_id $DEVICE \
    --kgram_k 3 \
    --kgram_chunk_size 5 \
    --transformer_heads 4 \
    --transformer_blocks 3 \
    --ff_mult 2 \
    --learning_rate 3e-4 \
    --grad_clip 1.0 \
    --weight_decay 0.01 \
    --prompt "Once upon a time" \
    $MAX_STEPS

# Check if training succeeded
if [ ! -f "loss_histories.pkl" ]; then
    echo ""
    echo "❌ Error: Training did not produce loss_histories.pkl"
    exit 1
fi

# Generate plots
echo ""
echo "📊 Generating loss plots..."
python plot_losses.py --smooth 10

# Check if matplotlib is available
if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️  Warning: Could not generate plots. Install matplotlib:"
    echo "   pip install matplotlib"
    exit 1
fi

# Summary
echo ""
echo "=========================================="
echo "✅ Training Complete!"
echo "=========================================="
echo ""
echo "Generated files:"
echo "  📈 training_losses.png    - Loss comparison plot"
echo "  💾 loss_histories.pkl     - Raw loss data"
echo "  🧠 *_epoch*.pt            - Model checkpoints"
echo ""
echo "View the plot with:"
echo "  display training_losses.png    # Linux"
echo "  open training_losses.png       # Mac"
echo ""
echo "Re-plot with different settings:"
echo "  python plot_losses.py --smooth 20 --log"
echo ""
