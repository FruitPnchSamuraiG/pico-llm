#!/bin/bash
# Train all models (K-gram, LSTM, Transformer) with multiple configurations
# 
# Usage:
#   bash train_all_models.sh              # Run all 4 configurations on CPU
#   bash train_all_models.sh --gpu        # Run all 4 configurations on GPU

set -e  # Exit on error

echo "=========================================="
echo "Training All Models: Multiple Configurations"
echo "=========================================="

# Parse arguments
GPU_MODE=false
if [[ "$1" == "--gpu" ]]; then
    GPU_MODE=true
    echo "🚀 GPU mode enabled"
fi

# Set device and max steps based on mode
if [ "$GPU_MODE" = true ]; then
    DEVICE="cuda:0"
    MAX_STEPS=""  # No limit on GPU
    echo "Device: GPU (cuda:0)"
else
    DEVICE="cpu"
    MAX_STEPS="--max_steps_per_epoch 50"
    echo "Device: CPU (with limited steps for faster testing)"
fi

# Set PyTorch memory allocation config for better GPU memory management
if [ "$GPU_MODE" = true ]; then
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    echo "💾 GPU memory optimization enabled"
fi

echo ""
echo "This script will run 4 different training configurations:"
echo "  1. Baseline: Small model, k=3"
echo "  2. Larger embeddings: k=3, embed=512"
echo "  3. Wider context: k=5, embed=384"
echo "  4. Deep model: k=4, more transformer blocks"
echo ""

# Function to run a single training configuration
run_training() {
    local RUN_NAME=$1
    local BATCH_SIZE=$2
    local EPOCHS=$3
    local BLOCK_SIZE=$4
    local EMBED_SIZE=$5
    local KGRAM_K=$6
    local TRANSFORMER_HEADS=$7
    local TRANSFORMER_BLOCKS=$8
    local FF_MULT=$9
    local LR=${10}
    
    echo ""
    echo "=========================================="
    echo "🏋️  Configuration: $RUN_NAME"
    echo "=========================================="
    echo "Hyperparameters:"
    echo "  Batch size: $BATCH_SIZE"
    echo "  Epochs: $EPOCHS"
    echo "  Block size: $BLOCK_SIZE"
    echo "  Embedding size: $EMBED_SIZE"
    echo "  K-gram k: $KGRAM_K"
    echo "  Transformer heads: $TRANSFORMER_HEADS"
    echo "  Transformer blocks: $TRANSFORMER_BLOCKS"
    echo "  FF multiplier: $FF_MULT"
    echo "  Learning rate: $LR"
    echo ""
    
    # Clean up old files for this run
    rm -f loss_histories_${RUN_NAME}.pkl
    rm -f kgram_mlp_seq_epoch*.pt lstm_seq_epoch*.pt transformer_epoch*.pt
    
    # Run training
    python pico-llm.py \
        --enable_kgram \
        --enable_transformer \
        --batch_size $BATCH_SIZE \
        --num_epochs $EPOCHS \
        --block_size $BLOCK_SIZE \
        --embed_size $EMBED_SIZE \
        --device_id $DEVICE \
        --kgram_k $KGRAM_K \
        --kgram_chunk_size 5 \
        --transformer_heads $TRANSFORMER_HEADS \
        --transformer_blocks $TRANSFORMER_BLOCKS \
        --ff_mult $FF_MULT \
        --learning_rate $LR \
        --prompt "Once upon a time" \
        $MAX_STEPS
    
    # Rename loss histories file with configuration name
    if [ -f "loss_histories.pkl" ]; then
        mv loss_histories.pkl "loss_histories_${RUN_NAME}.pkl"
        echo "✅ Saved loss histories to loss_histories_${RUN_NAME}.pkl"
    else
        echo "❌ Error: Training did not produce loss_histories.pkl"
        return 1
    fi
    
    # Generate plot with configuration in filename
    local PLOT_NAME="training_losses_${RUN_NAME}.png"
    echo "📊 Generating loss plot: $PLOT_NAME..."
    python plot_losses.py \
        --input "loss_histories_${RUN_NAME}.pkl" \
        --output "$PLOT_NAME" \
        --smooth 10
    
    # Rename model checkpoints with configuration name
    for file in *_epoch*.pt; do
        if [ -f "$file" ]; then
            local new_name="${file%.pt}_${RUN_NAME}.pt"
            mv "$file" "$new_name"
            echo "💾 Saved checkpoint: $new_name"
        fi
    done
    
    echo "✅ Configuration $RUN_NAME complete!"
}

# Configuration 1: Baseline - Small model for quick training
# Good for initial testing and CPU training
run_training "baseline" \
    8 \
    2 \
    256 \
    256 \
    3 \
    4 \
    2 \
    2 \
    3e-4

# Configuration 2: Larger embeddings - Better representation capacity
# Increases model capacity while keeping context window small
run_training "large_embed" \
    8 \
    2 \
    256 \
    512 \
    3 \
    4 \
    3 \
    3 \
    3e-4

# Configuration 3: Wider context - Longer k-gram window
# Tests whether longer context helps prediction
run_training "wide_context" \
    8 \
    2 \
    256 \
    384 \
    5 \
    4 \
    3 \
    2 \
    2e-4

# Configuration 4: Deep model - More transformer layers
# Tests whether depth helps vs width
run_training "deep_model" \
    8 \
    3 \
    256 \
    384 \
    4 \
    6 \
    4 \
    4 \
    2e-4

# Summary
echo ""
echo "=========================================="
echo "✅ All Configurations Complete!"
echo "=========================================="
echo ""
echo "Generated files:"
echo "  📈 training_losses_*.png       - Loss comparison plots"
echo "  💾 loss_histories_*.pkl        - Raw loss data"
echo "  🧠 *_epoch*_*.pt              - Model checkpoints"
echo ""
echo "Configurations trained:"
echo "  1. baseline       - Small model, k=3, embed=256"
echo "  2. large_embed    - Larger embeddings, k=3, embed=512"
echo "  3. wide_context   - Wider context, k=5, embed=384"
echo "  4. deep_model     - Deeper model, k=4, 4 blocks, embed=384"
echo ""
echo "View plots with:"
echo "  display training_losses_*.png    # Linux"
echo "  open training_losses_*.png       # Mac"
echo ""
