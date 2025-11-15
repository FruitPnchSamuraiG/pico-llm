#!/usr/bin/env python3
"""
Plot training loss curves for all models.

Usage:
    python plot_losses.py                    # Use default loss_histories.pkl
    python plot_losses.py --input custom.pkl # Use custom pickle file
    python plot_losses.py --smooth 10        # Apply smoothing window
"""

import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def moving_average(data, window_size):
    """Apply moving average smoothing to data."""
    if window_size <= 1:
        return data
    
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size


def plot_losses(loss_histories, output_file="training_losses.png", smooth_window=1, log_scale=False):
    """
    Plot loss curves for all models (train and validation).
    
    Args:
        loss_histories: Dict mapping model_name -> {'train': [...], 'val': [...]}
        output_file: Where to save the plot
        smooth_window: Window size for moving average (1 = no smoothing)
        log_scale: Whether to use log scale for y-axis
    """
    plt.figure(figsize=(14, 8))
    
    # Color palette for different models
    colors = {
        'kgram_mlp_seq': '#e74c3c',      # Red
        'lstm_seq': '#3498db',            # Blue
        'transformer': '#2ecc71',         # Green
    }
    
    for model_name, histories in loss_histories.items():
        # Handle both old format (list) and new format (dict with 'train'/'val')
        if isinstance(histories, dict):
            train_history = histories.get('train', [])
            val_history = histories.get('val', [])
        else:
            # Old format: just training loss
            train_history = histories
            val_history = []
        
        if not train_history:
            print(f"Warning: No training data for {model_name}")
            continue
        
        # Extract steps and losses for training
        steps = [step for step, _ in train_history]
        losses = [loss for _, loss in train_history]
        
        # Apply smoothing if requested
        if smooth_window > 1 and len(losses) > smooth_window:
            losses_smooth = moving_average(np.array(losses), smooth_window)
            steps_smooth = steps[smooth_window-1:]
            
            # Plot smoothed training line (solid)
            color = colors.get(model_name, None)
            plt.plot(steps_smooth, losses_smooth, 
                    label=f"{model_name} (train)", 
                    linewidth=2.5, 
                    color=color,
                    alpha=0.9,
                    linestyle='-')
            
            # Plot raw training data with transparency
            plt.plot(steps, losses, 
                    color=color, 
                    alpha=0.15, 
                    linewidth=0.5)
        else:
            # Plot raw training data only
            color = colors.get(model_name, None)
            plt.plot(steps, losses, 
                    label=f"{model_name} (train)", 
                    linewidth=2.5, 
                    color=color,
                    alpha=0.8,
                    linestyle='-')
        
        # Plot validation loss if available (dashed line)
        if val_history:
            val_steps = [step for step, _ in val_history]
            val_losses = [loss for _, loss in val_history]
            
            plt.plot(val_steps, val_losses,
                    label=f"{model_name} (val)",
                    linewidth=2.5,
                    color=color,
                    alpha=0.9,
                    linestyle='--',  # Dashed for validation
                    marker='o',      # Markers since val is sparse
                    markersize=6)
    
    plt.xlabel("Training Step", fontsize=14, fontweight='bold')
    plt.ylabel("Cross-Entropy Loss", fontsize=14, fontweight='bold')
    plt.title("Training & Validation Loss: K-gram MLP vs LSTM vs Transformer", fontsize=16, pad=20, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    if log_scale:
        plt.yscale('log')
        plt.ylabel("Cross-Entropy Loss (log scale)", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved plot to {output_file}")
    
    # Also show statistics
    print("\n" + "="*60)
    print("TRAINING STATISTICS")
    print("="*60)
    for model_name, histories in loss_histories.items():
        if isinstance(histories, dict):
            train_history = histories.get('train', [])
            val_history = histories.get('val', [])
        else:
            train_history = histories
            val_history = []
            
        if not train_history:
            continue
            
        train_losses = [loss for _, loss in train_history]
        print(f"\n{model_name}:")
        print(f"  📈 TRAINING:")
        print(f"     Initial loss: {train_losses[0]:.4f}")
        print(f"     Final loss:   {train_losses[-1]:.4f}")
        print(f"     Min loss:     {min(train_losses):.4f}")
        print(f"     Improvement:  {train_losses[0] - train_losses[-1]:.4f} ({(1 - train_losses[-1]/train_losses[0])*100:.1f}% reduction)")
        print(f"     Total steps:  {len(train_history)}")
        
        if val_history:
            val_losses = [loss for _, loss in val_history]
            print(f"  📊 VALIDATION:")
            print(f"     Initial loss: {val_losses[0]:.4f}")
            print(f"     Final loss:   {val_losses[-1]:.4f}")
            print(f"     Min loss:     {min(val_losses):.4f}")
            gap = val_losses[-1] - train_losses[-1]
            print(f"     Train/Val gap: {gap:.4f} ({'overfitting' if gap > 0.5 else 'good fit'})")


def main():
    parser = argparse.ArgumentParser(description="Plot training loss curves")
    parser.add_argument("--input", type=str, default="loss_histories.pkl",
                       help="Input pickle file with loss histories")
    parser.add_argument("--output", type=str, default="training_losses.png",
                       help="Output image file")
    parser.add_argument("--smooth", type=int, default=10,
                       help="Moving average window size (1=no smoothing)")
    parser.add_argument("--log", action="store_true",
                       help="Use log scale for y-axis")
    args = parser.parse_args()
    
    # Load loss histories
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: {args.input} not found!")
        print(f"Make sure you've run training with pico-llm.py first.")
        return
    
    print(f"Loading loss histories from {args.input}...")
    with open(args.input, "rb") as f:
        loss_histories = pickle.load(f)
    
    print(f"Found {len(loss_histories)} models:")
    for model_name in loss_histories.keys():
        print(f"  - {model_name}")
    
    # Plot
    plot_losses(loss_histories, 
               output_file=args.output, 
               smooth_window=args.smooth,
               log_scale=args.log)
    
    print(f"\n🎉 Done! Open {args.output} to view the plot.")


if __name__ == "__main__":
    main()
