#!/usr/bin/env python3
"""
Interpretability visualizations for Pico-LLM Transformer.

Inspired by Anthropic's mechanistic interpretability blog posts.
Includes:
  - Attention head heatmaps
  - Logit lens (layer-wise predictions)
  - Token embedding projections (UMAP/t-SNE)
  - Feature activation patterns
  - Query-Key interaction analysis

Usage:
    python plot_interpretability.py \
      --checkpoint transformer_epoch8.pt \
      --prompt "Q: What is 2 + 2? A:" \
      --device cuda:0 \
      --output_dir interpretability_plots
"""

import argparse
import torch
import torch.nn.functional as F
import tiktoken
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys
sys.path.insert(0, str(Path(__file__).parent))

from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("pico_llm", Path(__file__).parent / "pico-llm.py")
pico_llm = module_from_spec(spec)
spec.loader.exec_module(pico_llm)
TransformerModel = pico_llm.TransformerModel
RMSNorm = pico_llm.RMSNorm

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150


class InterpretabilityHook:
    """Capture activations at each layer."""
    
    def __init__(self):
        self.activations = {}
        self.attention_weights = {}
        self.queries = {}
        self.keys = {}
        self.values = {}
    
    def register_hooks(self, model: TransformerModel):
        """Register forward hooks on all transformer blocks."""
        for block_idx, block in enumerate(model.blocks):
            # Attention input
            def attn_hook(module, input, output, idx=block_idx):
                if isinstance(output, tuple):
                    self.activations[f"block_{idx}_attn_out"] = output[0].detach()
                else:
                    self.activations[f"block_{idx}_attn_out"] = output.detach()
            
            block.register_forward_hook(attn_hook)
        
        # LM head
        def lm_head_hook(module, input, output):
            self.activations["lm_head_output"] = output.detach()
        
        model.lm_head.register_forward_hook(lm_head_hook)


def plot_attention_heads(model: TransformerModel, tokens_tensor: torch.Tensor, 
                        enc: tiktoken.Encoding, output_file: Path, 
                        device: str = "cuda:0"):
    """Visualize attention patterns for each head in each layer."""
    model.eval()
    
    # Capture attention weights during forward pass
    attention_weights = {}
    hooks = []
    
    def get_attn_hook(block_idx):
        def hook(module, input, output):
            # Store attention probabilities before multiplication with values
            if hasattr(module, 'last_attn_probs') and module.last_attn_probs is not None:
                attention_weights[f"block_{block_idx}"] = module.last_attn_probs.detach().cpu()
        return hook
    
    # Register hooks
    for block_idx, block in enumerate(model.blocks):
        block.save_attention = True
        hook = block.register_forward_hook(get_attn_hook(block_idx))
        hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        _ = model(tokens_tensor)
    
    # Decode tokens for labels
    token_ids = tokens_tensor.squeeze(1).cpu().tolist()
    token_strs = [enc.decode([tid]) for tid in token_ids]
    
    # Create subplots for each block/head combination
    n_blocks = len(model.blocks)
    n_heads = model.blocks[0].n_heads
    
    fig, axes = plt.subplots(n_blocks, min(n_heads, 4), figsize=(16, 3 * n_blocks))
    if n_blocks == 1:
        axes = axes.reshape(1, -1)
    
    for block_idx in range(n_blocks):
        if f"block_{block_idx}" not in attention_weights:
            continue
        
        attn = attention_weights[f"block_{block_idx}"]  # (batch=1, heads, seq_len, seq_len)
        attn = attn[0]  # (heads, seq_len, seq_len)
        
        for head_idx in range(min(n_heads, 4)):
            ax = axes[block_idx, head_idx]
            
            # Plot attention heatmap
            im = ax.imshow(attn[head_idx].numpy(), cmap='viridis', aspect='auto')
            ax.set_xlabel('Keys (position)')
            ax.set_ylabel('Queries (position)')
            ax.set_title(f'Block {block_idx}, Head {head_idx}')
            ax.set_xticks(range(len(token_strs)))
            ax.set_yticks(range(len(token_strs)))
            ax.set_xticklabels(token_strs, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(token_strs, fontsize=8)
            
            plt.colorbar(im, ax=ax, label='Attention weight')
    
    plt.suptitle('Attention Head Patterns by Layer', fontsize=16, weight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()
    
    # Cleanup hooks
    for hook in hooks:
        hook.remove()


def plot_logit_lens(model: TransformerModel, tokens_tensor: torch.Tensor, 
                   enc: tiktoken.Encoding, output_file: Path, 
                   top_k: int = 5, device: str = "cuda:0"):
    """Show predictions at each layer (Logit Lens / early exit analysis)."""
    model.eval()
    
    logits_by_layer = {}
    hooks = []
    
    def get_logits_hook(layer_name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            
            # hidden is in (batch, seq_len, d_model) format from TransformerBlock
            # Project to logits
            logits = model.lm_head(hidden)
            logits_by_layer[layer_name] = logits.detach().cpu()
        return hook
    
    # Register hooks on each block's output
    for block_idx, block in enumerate(model.blocks):
        hook = block.register_forward_hook(get_logits_hook(f"layer_{block_idx}"))
        hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        _ = model(tokens_tensor)
    
    # Get final position logits across layers
    # tokens_tensor is (seq_len, batch=1)
    seq_len = tokens_tensor.shape[0]
    
    # Decode true next token
    token_ids = tokens_tensor.squeeze(1).cpu().tolist()
    token_strs = [enc.decode([tid]) for tid in token_ids]
    
    # Collect top-k predictions at each layer
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    layers = sorted(logits_by_layer.keys(), key=lambda x: int(x.split("_")[1]))
    
    # Plot 1: Top-k predictions evolution
    ax = axes[0]
    layer_names = []
    predictions = {}
    
    for layer in layers:
        layer_names.append(layer)
        # logits_by_layer[layer] is (batch=1, seq_len, vocab_size)
        # Get last position (final token's prediction)
        logits = logits_by_layer[layer][0, -1, :]
        probs = F.softmax(logits, dim=-1)
        top_indices = torch.topk(probs, top_k).indices
        top_probs = torch.topk(probs, top_k).values
        
        for rank, (idx, prob) in enumerate(zip(top_indices, top_probs)):
            token_str = enc.decode([idx.item()])
            if token_str not in predictions:
                predictions[token_str] = []
            predictions[token_str].append((len(layer_names) - 1, prob.item(), rank))
    
    # Plot evolution
    for token_str, points in predictions.items():
        if len(points) > 2:  # Only plot if appears multiple times
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            ax.plot(x, y, marker='o', label=token_str, linewidth=2)
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Probability')
    ax.set_title('Token Probability Evolution Across Layers (Logit Lens)', fontweight='bold')
    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels(layer_names, rotation=45)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Entropy reduction
    ax = axes[1]
    entropies = []
    for layer in layers:
        logits = logits_by_layer[layer][0, -1, :]
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        entropies.append(entropy)
    
    ax.plot(entropies, marker='s', linewidth=2.5, markersize=8, color='#e74c3c')
    ax.fill_between(range(len(entropies)), entropies, alpha=0.3, color='#e74c3c')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Entropy')
    ax.set_title('Prediction Confidence Growth (Entropy Reduction)', fontweight='bold')
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layer_names, rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()
    
    # Cleanup hooks
    for hook in hooks:
        hook.remove()


def plot_token_embeddings(model: TransformerModel, enc: tiktoken.Encoding, 
                         output_file: Path, n_samples: int = 500, device: str = "cuda:0"):
    """Visualize token embedding space (projection to 2D)."""
    model.eval()
    
    # Get embeddings for random tokens
    sample_token_ids = torch.randint(0, enc.n_vocab, (n_samples,), device=device)
    with torch.no_grad():
        embeddings = model.embed(sample_token_ids)  # (n_samples, d_model)
    
    embeddings_np = embeddings.cpu().numpy()
    
    # PCA projection to 2D
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings_np)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Color by token frequency class
    colors = plt.cm.viridis(np.linspace(0, 1, len(sample_token_ids)))
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                        c=sample_token_ids.cpu().numpy(), cmap='tab20b', 
                        s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Label some interesting tokens
    interesting_tokens = [0, 1, 2, 13, 50, 100, 1000, 10000, 
                         enc.encode("math")[0], enc.encode("answer")[0]]
    for tid in interesting_tokens:
        if tid < len(embeddings_2d):
            idx = (sample_token_ids == tid).nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                idx = idx[0].item()
                ax.annotate(enc.decode([tid])[:10], 
                           xy=embeddings_2d[idx], 
                           fontsize=8, alpha=0.7)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontweight='bold')
    ax.set_title('Token Embedding Space (PCA Projection)', fontweight='bold', fontsize=14)
    
    cbar = plt.colorbar(scatter, ax=ax, label='Token ID')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def plot_activation_distribution(model: TransformerModel, tokens_tensor: torch.Tensor,
                                output_file: Path, device: str = "cuda:0"):
    """Show activation distributions at different layers."""
    model.eval()
    
    activations = {}
    hooks = []
    
    def get_activation_hook(layer_name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            activations[layer_name] = hidden.detach().cpu()
        return hook
    
    # Register hooks
    for block_idx, block in enumerate(model.blocks):
        hook = block.register_forward_hook(get_activation_hook(f"layer_{block_idx}"))
        hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        _ = model(tokens_tensor)
    
    # Plot activation statistics
    n_layers = len(model.blocks)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Mean activations per layer
    ax = axes[0]
    means = []
    stds = []
    for i in range(n_layers):
        if f"layer_{i}" in activations:
            act = activations[f"layer_{i}"].flatten()
            means.append(act.mean().item())
            stds.append(act.std().item())
    
    x = np.arange(len(means))
    ax.errorbar(x, means, yerr=stds, marker='o', linestyle='-', linewidth=2, 
               markersize=8, capsize=5, color='#3498db', ecolor='#e74c3c')
    ax.set_xlabel('Layer', fontweight='bold')
    ax.set_ylabel('Mean Activation (± std)', fontweight='bold')
    ax.set_title('Activation Magnitude by Layer', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{i}' for i in range(n_layers)])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Activation histograms
    ax = axes[1]
    for i in [0, n_layers // 2, n_layers - 1]:
        if f"layer_{i}" in activations:
            act = activations[f"layer_{i}"].flatten().numpy()
            ax.hist(act, bins=50, alpha=0.5, label=f'Layer {i}', density=True)
    
    ax.set_xlabel('Activation Value', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title('Activation Distribution Across Layers', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()
    
    # Cleanup
    for hook in hooks:
        hook.remove()


def plot_feature_coactivation(model: TransformerModel, tokens_tensor: torch.Tensor,
                              output_file: Path, top_k: int = 20, device: str = "cuda:0"):
    """Visualize which features tend to activate together (feature co-activation patterns)."""
    model.eval()
    
    activations = {}
    hooks = []
    
    def get_activation_hook(layer_name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            activations[layer_name] = hidden.detach().cpu()
        return hook
    
    # Register hooks
    for block_idx, block in enumerate(model.blocks):
        hook = block.register_forward_hook(get_activation_hook(f"layer_{block_idx}"))
        hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        _ = model(tokens_tensor)
    
    # Analyze co-activation patterns (using correlation)
    n_layers = len(model.blocks)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Feature activation correlation heatmap
    ax = axes[0]
    # Get activations from middle layer and average across sequence
    mid_layer = n_layers // 2
    if f"layer_{mid_layer}" in activations:
        act = activations[f"layer_{mid_layer}"]  # (batch, seq, d_model)
        act_mean = act.mean(dim=1).squeeze(0).numpy()  # Average over seq, shape (d_model,)
        
        # Sample features for correlation analysis (too many to show all)
        n_sample = min(top_k, len(act_mean))
        indices = np.argsort(np.abs(act_mean))[-n_sample:]  # Top activated features
        
        # Compute correlation between features across sequence positions
        act_seq = act[0].numpy()  # (seq, d_model)
        corr_matrix = np.corrcoef(act_seq[:, indices].T)
        
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xlabel(f'Top {n_sample} Features', fontweight='bold')
        ax.set_ylabel(f'Top {n_sample} Features', fontweight='bold')
        ax.set_title('Feature Co-activation Pattern (Correlation)', fontweight='bold')
        plt.colorbar(im, ax=ax, label='Correlation')
    
    # Plot 2: Sparsity pattern visualization
    ax = axes[1]
    sparsity_by_layer = []
    for i in range(n_layers):
        if f"layer_{i}" in activations:
            act = activations[f"layer_{i}"].flatten().numpy()
            # Compute sparsity (fraction of near-zero activations)
            threshold = 0.01
            sparsity = (np.abs(act) < threshold).mean()
            sparsity_by_layer.append(sparsity * 100)
    
    x = np.arange(len(sparsity_by_layer))
    ax.bar(x, sparsity_by_layer, color='#2ecc71', edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Layer', fontweight='bold')
    ax.set_ylabel('Sparsity (%)', fontweight='bold')
    ax.set_title('Activation Sparsity by Layer', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{i}' for i in range(len(sparsity_by_layer))])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()
    
    # Cleanup
    for hook in hooks:
        hook.remove()


def plot_feature_importance(model: TransformerModel, tokens_tensor: torch.Tensor, 
                           enc: tiktoken.Encoding, output_file: Path, 
                           top_k: int = 15, device: str = "cuda:0"):
    """Measure feature importance via attribution and ablation."""
    model.eval()
    
    activations = {}
    hooks = []
    
    def get_activation_hook(layer_name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            activations[layer_name] = hidden.detach()
        return hook
    
    # Register hooks on transformer blocks
    for block_idx, block in enumerate(model.blocks):
        hook = block.register_forward_hook(get_activation_hook(f"layer_{block_idx}"))
        hooks.append(hook)
    
    # Forward pass to get baseline output
    with torch.no_grad():
        baseline_output = model(tokens_tensor)
        baseline_logits = baseline_output[0, -1, :] if isinstance(baseline_output, torch.Tensor) else baseline_output
    
    # Calculate feature contributions (simple gradient-based attribution)
    feature_scores = {}
    for layer_name, act in activations.items():
        # Use activation magnitude as proxy for importance
        importance = act.abs().mean(dim=[0, 1]).cpu().numpy()  # Average over batch and seq
        feature_scores[layer_name] = importance
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Top features by importance across all layers
    ax = axes[0]
    all_scores = []
    all_labels = []
    for layer_name in sorted(feature_scores.keys(), key=lambda x: int(x.split('_')[1])):
        scores = feature_scores[layer_name]
        layer_idx = layer_name.split('_')[1]
        for i, score in enumerate(scores[:top_k]):
            all_scores.append(score)
            all_labels.append(f"{layer_name}_f{i}")
    
    # Show top K overall
    top_indices = np.argsort(all_scores)[-top_k:]
    top_scores = [all_scores[i] for i in top_indices]
    top_labels = [all_labels[i] for i in top_indices]
    
    y_pos = np.arange(len(top_labels))
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_labels)))
    ax.barh(y_pos, top_scores, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_labels, fontsize=8)
    ax.set_xlabel('Attribution Score', fontweight='bold')
    ax.set_title(f'Top {top_k} Most Important Features', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Layer-wise feature importance aggregation
    ax = axes[1]
    layer_importance = []
    layer_names = []
    for layer_name in sorted(feature_scores.keys(), key=lambda x: int(x.split('_')[1])):
        layer_importance.append(feature_scores[layer_name].sum())
        layer_names.append(layer_name)
    
    x = np.arange(len(layer_names))
    bars = ax.bar(x, layer_importance, color='#e74c3c', edgecolor='black', linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45)
    ax.set_xlabel('Layer', fontweight='bold')
    ax.set_ylabel('Cumulative Feature Importance', fontweight='bold')
    ax.set_title('Total Feature Importance by Layer', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()
    
    # Cleanup
    for hook in hooks:
        hook.remove()


def plot_feature_geometry(model: TransformerModel, output_file: Path, 
                         n_samples: int = 500, device: str = "cuda:0"):
    """Visualize geometric relationships between features using their embeddings."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    model.eval()
    
    # Get embedding matrix (this represents feature geometry)
    embeddings = model.embed.weight.detach().cpu().numpy()  # (vocab_size, d_model)
    
    # Sample random subset
    sample_indices = np.random.choice(len(embeddings), min(n_samples, len(embeddings)), replace=False)
    sampled_embeddings = embeddings[sample_indices]
    
    # Dimensionality reduction
    pca = PCA(n_components=50)
    embeddings_pca = pca.fit_transform(sampled_embeddings)
    
    # Further reduce to 2D with t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings_pca)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot 1: t-SNE visualization with density coloring
    ax = axes[0]
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                        c=sample_indices, cmap='tab20c', 
                        s=30, alpha=0.6, edgecolors='black', linewidth=0.3)
    ax.set_xlabel('t-SNE Dimension 1', fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontweight='bold')
    ax.set_title('Feature Geometry (t-SNE Projection)', fontweight='bold', fontsize=14)
    plt.colorbar(scatter, ax=ax, label='Token ID')
    
    # Plot 2: PCA variance explained
    ax = axes[1]
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    x = np.arange(1, len(explained_var) + 1)
    
    ax.plot(x, cumulative_var * 100, marker='o', linewidth=2.5, markersize=6, color='#3498db')
    ax.fill_between(x, cumulative_var * 100, alpha=0.3, color='#3498db')
    ax.set_xlabel('Number of Principal Components', fontweight='bold')
    ax.set_ylabel('Cumulative Explained Variance (%)', fontweight='bold')
    ax.set_title('Feature Space Dimensionality', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, min(50, len(explained_var)))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate mechanistic interpretability plots")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--prompt", type=str, default="Q: What is 2 + 2? A:",
                       help="Input prompt for analysis")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--output_dir", type=str, default="interpretability_plots",
                       help="Output directory")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_blocks", type=int, default=3)
    parser.add_argument("--block_size", type=int, default=256)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    enc = tiktoken.get_encoding("gpt2")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = TransformerModel(
        vocab_size=enc.n_vocab,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_blocks=args.n_blocks,
        block_size=args.block_size
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    # Tokenize prompt
    tokens = enc.encode(args.prompt)
    tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(1)
    
    print(f"Prompt: {args.prompt}")
    print(f"Tokens: {tokens}")
    print(f"\n📊 Generating interpretability plots...")
    
    # Generate plots
    plot_attention_heads(model, tokens_tensor, enc, 
                        output_dir / "1_attention_heads.png", str(device))
    
    plot_logit_lens(model, tokens_tensor, enc, 
                   output_dir / "2_logit_lens.png", device=str(device))
    
    plot_token_embeddings(model, enc, 
                         output_dir / "3_token_embeddings.png", device=str(device))
    
    plot_activation_distribution(model, tokens_tensor, 
                                output_dir / "4_activations.png", str(device))
    
    plot_feature_coactivation(model, tokens_tensor,
                             output_dir / "5_feature_coactivation.png", device=str(device))
    
    plot_feature_importance(model, tokens_tensor, enc,
                           output_dir / "6_feature_importance.png", device=str(device))
    
    plot_feature_geometry(model,
                         output_dir / "7_feature_geometry.png", device=str(device))
    
    print(f"\n✅ All interpretability plots saved to {output_dir}/")
    print("\n📋 Generated plots:")
    print("   1. Attention head patterns - How each head attends to tokens")
    print("   2. Logit lens - Layer-by-layer prediction evolution")
    print("   3. Token embeddings - 2D projection of embedding space")
    print("   4. Activation distribution - How hidden states evolve across layers")


if __name__ == "__main__":
    main()
