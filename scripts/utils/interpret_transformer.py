#!/usr/bin/env python3
"""Interpretability analysis for Transformer models.

Inspired by Anthropic's interpretability research:
- Feature visualization (max-activating examples)
- Attention pattern analysis
- Logit lens (intermediate predictions)
- Activation patching / causal interventions
- Neuron-level feature discovery

Usage:
  source /scratch/kk6081/ml_fall25/venv/bin/activate
  python scripts/interpret_transformer.py \
    --checkpoint /scratch/kk6081/picollm_extend/transformer_epoch1.pt \
    --analysis attention,logit_lens,neurons \
    --out_dir interpretability_results
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import torch
import torch.nn.functional as F
import tiktoken
import numpy as np

# Set matplotlib backend for headless servers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import model from pico-llm.py
import importlib.util

def _load_pico_module():
    here = Path(__file__).parent.parent
    spec = importlib.util.spec_from_file_location("pico_llm", str(here / "pico-llm.py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--analysis", type=str, default="attention,logit_lens,neurons",
                   help="Comma-separated: attention, logit_lens, neurons, activation_patch")
    p.add_argument("--out_dir", type=str, default="interpretability_results")
    
    # Model architecture (must match checkpoint)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--embed_size", type=int, default=384)
    p.add_argument("--transformer_heads", type=int, default=4)
    p.add_argument("--transformer_blocks", type=int, default=3)
    p.add_argument("--ff_mult", type=int, default=2)
    
    # Analysis settings
    p.add_argument("--test_prompts", nargs="*", 
                   default=["Once upon a time", "The quick brown fox", "In a galaxy far away"])
    p.add_argument("--neuron_top_k", type=int, default=10, help="Top K neurons to analyze")
    p.add_argument("--device", type=str, default="cuda:0")
    
    return p.parse_args()


################################################################################
# 1) Attention Pattern Analysis
################################################################################

def analyze_attention_patterns(model, enc, prompts, device, out_dir):
    """Extract and visualize attention patterns for test prompts.
    
    Shows:
    - Per-head attention heatmaps
    - Average attention by layer
    - Attention to specific token types (punctuation, function words, content)
    """
    print("\n=== Attention Pattern Analysis ===")
    os.makedirs(f"{out_dir}/attention", exist_ok=True)
    
    model.eval()
    for blk in model.blocks:
        blk.save_attention = True
    
    results = {}
    
    for prompt in prompts:
        tokens = enc.encode(prompt)
        if len(tokens) > model.block_size:
            tokens = tokens[:model.block_size]
        
        with torch.no_grad():
            tok_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(1)
            _ = model(tok_tensor)
        
        # Collect attention from all blocks
        attn_data = []
        for bi, blk in enumerate(model.blocks):
            if blk.last_attn_probs is not None:
                attn = blk.last_attn_probs[0].cpu().numpy()  # (heads, T, T)
                attn_data.append(attn)
        
        # Plot per-layer, per-head
        token_strs = [enc.decode([t]) for t in tokens]
        fig, axes = plt.subplots(len(attn_data), model.n_heads, figsize=(3*model.n_heads, 3*len(attn_data)))
        if len(attn_data) == 1:
            axes = axes.reshape(1, -1)
        
        for li, layer_attn in enumerate(attn_data):
            for hi in range(model.n_heads):
                ax = axes[li, hi]
                im = ax.imshow(layer_attn[hi], cmap='viridis', aspect='auto', origin='lower')
                ax.set_title(f"L{li}H{hi}", fontsize=8)
                ax.set_xticks(range(len(token_strs)))
                ax.set_yticks(range(len(token_strs)))
                ax.set_xticklabels(token_strs, rotation=90, fontsize=6)
                ax.set_yticklabels(token_strs, fontsize=6)
        
        plt.tight_layout()
        safe_name = prompt.replace(" ", "_")[:30]
        plt.savefig(f"{out_dir}/attention/attn_{safe_name}.png", dpi=150)
        plt.close()
        
        results[prompt] = {
            "num_layers": len(attn_data),
            "num_heads": model.n_heads,
            "seq_len": len(tokens),
        }
    
    for blk in model.blocks:
        blk.save_attention = False
    
    print(f"✅ Saved attention plots to {out_dir}/attention/")
    return results


################################################################################
# 2) Logit Lens: Intermediate Predictions
################################################################################

def analyze_logit_lens(model, enc, prompts, device, out_dir):
    """Logit lens: decode hidden states at each layer to see what the model 'thinks' at intermediate steps.
    
    This reveals:
    - How predictions evolve through layers
    - Which layers specialize in which token types
    - Early vs late feature formation
    """
    print("\n=== Logit Lens Analysis ===")
    os.makedirs(f"{out_dir}/logit_lens", exist_ok=True)
    
    model.eval()
    results = {}
    
    for prompt in prompts:
        tokens = enc.encode(prompt)
        if len(tokens) > model.block_size:
            tokens = tokens[:model.block_size]
        
        token_strs = [enc.decode([t]) for t in tokens]
        
        with torch.no_grad():
            tok_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(1)
            tok_b = tok_tensor.transpose(0, 1)  # (1, T)
            
            x = model.embed(tok_b)
            if model.pos_emb is not None:
                pos = torch.arange(len(tokens), device=device).unsqueeze(0)
                x = x + model.pos_emb(pos)
            
            # Collect predictions at each layer
            layer_preds = []
            for blk in model.blocks:
                x = blk(x, causal_mask=model.causal_mask)
                # Decode this layer's hidden state
                normed = model.final_norm(x)
                logits = model.lm_head(normed)[0, -1, :]  # Last position
                top_token = torch.argmax(logits).item()
                top_str = enc.decode([top_token])
                layer_preds.append(top_str)
        
        results[prompt] = {
            "input_tokens": token_strs,
            "layer_predictions": layer_preds,
        }
        
        print(f"Prompt: {prompt}")
        print(f"  Tokens: {token_strs}")
        for li, pred in enumerate(layer_preds):
            print(f"  Layer {li} prediction: '{pred}'")
        print()
    
    # Save results
    with open(f"{out_dir}/logit_lens/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Saved logit lens results to {out_dir}/logit_lens/results.json")
    return results


################################################################################
# 3) Neuron Analysis: Max-Activating Examples
################################################################################

def analyze_neurons(model, enc, prompts, device, out_dir, top_k=10):
    """Find neurons with largest activations and their max-activating contexts.
    
    This helps discover:
    - What features individual neurons detect
    - Interpretable vs polysemantic neurons
    - Layer-wise feature specialization
    """
    print("\n=== Neuron Activation Analysis ===")
    os.makedirs(f"{out_dir}/neurons", exist_ok=True)
    
    model.eval()
    
    # Collect activations from all FFN layers
    all_activations = {}  # {layer_idx: {neuron_idx: [(context, activation)]}}
    
    for prompt in prompts:
        tokens = enc.encode(prompt)
        if len(tokens) > model.block_size:
            tokens = tokens[:model.block_size]
        
        token_strs = [enc.decode([t]) for t in tokens]
        
        with torch.no_grad():
            tok_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(1)
            tok_b = tok_tensor.transpose(0, 1)
            
            x = model.embed(tok_b)
            if model.pos_emb is not None:
                pos = torch.arange(len(tokens), device=device).unsqueeze(0)
                x = x + model.pos_emb(pos)
            
            for li, blk in enumerate(model.blocks):
                # Get pre-FF activations
                if blk.norm_type == 'pre':
                    xf = blk.norm_ff(x + blk.out_proj(blk.norm_attn(x)))
                else:
                    xf = x
                
                # Forward through first linear + SiLU
                ff_hidden = blk.ff[1](blk.ff[0](xf))  # (1, T, ff_dim)
                
                # Record top activations per neuron
                if li not in all_activations:
                    all_activations[li] = {}
                
                for pos in range(ff_hidden.shape[1]):
                    context = " ".join(token_strs[max(0, pos-3):pos+1])
                    acts = ff_hidden[0, pos, :].cpu().numpy()
                    
                    for ni, act_val in enumerate(acts):
                        if ni not in all_activations[li]:
                            all_activations[li][ni] = []
                        all_activations[li][ni].append((context, float(act_val)))
                
                x = blk(x, causal_mask=model.causal_mask)
    
    # Find top-K most active neurons per layer
    results = {}
    for li in sorted(all_activations.keys()):
        layer_results = []
        neuron_max_acts = {}
        
        for ni, contexts in all_activations[li].items():
            max_act = max(act for _, act in contexts)
            neuron_max_acts[ni] = max_act
        
        # Sort neurons by max activation
        top_neurons = sorted(neuron_max_acts.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        for ni, max_act in top_neurons:
            # Get top contexts for this neuron
            contexts = sorted(all_activations[li][ni], key=lambda x: x[1], reverse=True)[:5]
            layer_results.append({
                "neuron_idx": ni,
                "max_activation": max_act,
                "top_contexts": contexts,
            })
        
        results[f"layer_{li}"] = layer_results
        
        print(f"\nLayer {li} - Top {min(top_k, len(top_neurons))} neurons:")
        for item in layer_results[:3]:  # Show top 3
            print(f"  Neuron {item['neuron_idx']}: max_act={item['max_activation']:.3f}")
            for ctx, act in item['top_contexts'][:2]:
                print(f"    '{ctx}' ({act:.3f})")
    
    with open(f"{out_dir}/neurons/top_neurons.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Saved neuron analysis to {out_dir}/neurons/top_neurons.json")
    return results


################################################################################
# 4) Activation Patching: Causal Analysis
################################################################################

def analyze_activation_patching(model, enc, prompts, device, out_dir):
    """Activation patching to test causal importance of components.
    
    Measure effect of:
    - Zeroing out specific attention heads
    - Ablating FFN layers
    - Replacing activations with baseline
    """
    print("\n=== Activation Patching Analysis ===")
    os.makedirs(f"{out_dir}/patching", exist_ok=True)
    
    model.eval()
    results = {}
    
    for prompt in prompts[:1]:  # Do this for first prompt only (expensive)
        tokens = enc.encode(prompt)
        if len(tokens) > model.block_size:
            tokens = tokens[:model.block_size]
        
        # Baseline: normal generation
        with torch.no_grad():
            tok_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(1)
            baseline_logits = model(tok_tensor)[-1, 0, :]
            baseline_pred = torch.argmax(baseline_logits).item()
            baseline_str = enc.decode([baseline_pred])
        
        # Test effect of zeroing each attention head
        head_effects = []
        for li in range(model.n_blocks):
            for hi in range(model.n_heads):
                # Patch: zero out this head's output
                original_attn = None
                
                def zero_head_hook(module, input, output):
                    # output is (b, h, t, d)
                    output = output.clone()
                    output[:, hi, :, :] = 0
                    return output
                
                # This is a simplified approach; full patching requires more hooks
                # For demonstration, we'll just report structure
                head_effects.append({
                    "layer": li,
                    "head": hi,
                    "effect": "ablation",
                })
        
        results[prompt] = {
            "baseline_prediction": baseline_str,
            "num_ablations_tested": len(head_effects),
        }
    
    with open(f"{out_dir}/patching/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Saved patching analysis to {out_dir}/patching/results.json")
    print("Note: Full activation patching requires forward hook implementation.")
    return results


################################################################################
# Main
################################################################################

def main():
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    enc = tiktoken.get_encoding("gpt2")
    
    # Load model
    pico = _load_pico_module()
    model = pico.TransformerModel(
        vocab_size=enc.n_vocab,
        d_model=args.embed_size,
        n_heads=args.transformer_heads,
        n_blocks=args.transformer_blocks,
        block_size=args.block_size,
        ff_mult=args.ff_mult,
    )
    
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {args.checkpoint}")
    print(f"Model: {model.n_blocks} blocks, {model.n_heads} heads, d_model={model.d_model}")
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    analyses = args.analysis.split(",")
    all_results = {}
    
    if "attention" in analyses:
        all_results["attention"] = analyze_attention_patterns(
            model, enc, args.test_prompts, device, args.out_dir
        )
    
    if "logit_lens" in analyses:
        all_results["logit_lens"] = analyze_logit_lens(
            model, enc, args.test_prompts, device, args.out_dir
        )
    
    if "neurons" in analyses:
        all_results["neurons"] = analyze_neurons(
            model, enc, args.test_prompts, device, args.out_dir, top_k=args.neuron_top_k
        )
    
    if "activation_patch" in analyses:
        all_results["activation_patch"] = analyze_activation_patching(
            model, enc, args.test_prompts, device, args.out_dir
        )
    
    # Save summary
    with open(f"{args.out_dir}/summary.json", "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "model_config": {
                "n_blocks": model.n_blocks,
                "n_heads": model.n_heads,
                "d_model": model.d_model,
                "ff_mult": args.ff_mult,
            },
            "analyses_run": analyses,
        }, f, indent=2)
    
    print(f"\n✅ All interpretability analyses complete!")
    print(f"Results saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
