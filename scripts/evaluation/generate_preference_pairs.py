#!/usr/bin/env python3
"""
Pre-generate preference pairs for DPO training.

This script generates completions offline to create a static preference dataset,
dramatically speeding up DPO training by avoiding on-the-fly generation.

Usage:
    python scripts/evaluation/generate_preference_pairs.py \
        --init_from finetune_gsm8k/transformer_epoch4.pt \
        --train_data data/gsm8k_train.txt \
        --output_file data/gsm8k_preferences.jsonl \
        --transformer_size medium \
        --num_completions 2 \
        --max_new_tokens 128
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, List, Tuple

import torch
import tiktoken

import importlib.util
import sys

# Import utilities from dpo_grpo_training
sys.path.insert(0, str(Path(__file__).parent))
from dpo_grpo_training import (
    extract_answer,
    split_qa,
    _load_inference_module,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate preference pairs for DPO training")
    
    p.add_argument("--init_from", type=str, required=True,
                   help="Base checkpoint (.pt) to generate from")
    p.add_argument("--train_data", type=str, required=True,
                   help="GSM8K training data (data/gsm8k_train.txt)")
    p.add_argument("--output_file", type=str, required=True,
                   help="Output JSONL file for preference pairs")
    
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--transformer_size", type=str, default="medium",
                   choices=["small", "medium", "gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"])
    
    p.add_argument("--num_completions", type=int, default=2,
                   help="Number of completions to generate per prompt")
    p.add_argument("--max_new_tokens", type=int, default=128,
                   help="Maximum tokens to generate per completion")
    p.add_argument("--top_p", type=float, default=0.95)
    
    p.add_argument("--reward_correct", type=float, default=1.0)
    p.add_argument("--reward_incorrect", type=float, default=0.0)
    
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=8,
                   help="Batch size for generation (not used yet, sequential for now)")
    
    return p.parse_args()


def generate_preference_pairs(
    args: argparse.Namespace,
    model: torch.nn.Module,
    train_lines: List[str],
    device: torch.device,
    enc: Any,
    inf: Any,
) -> List[dict]:
    """Generate preference pairs for all training examples."""
    
    model.eval()
    preference_pairs = []
    
    print(f"🔄 Generating {args.num_completions} completions per example...")
    
    with torch.no_grad():
        for idx, line in enumerate(train_lines):
            if (idx + 1) % 100 == 0:
                print(f"   Progress: {idx + 1}/{len(train_lines)}")
            
            prompt, gold = split_qa(line)
            if not gold:
                continue
            
            # Truncate prompt if needed
            prompt_tokens = enc.encode(prompt)
            max_prompt = max(1, args.block_size - args.max_new_tokens)
            if len(prompt_tokens) > max_prompt:
                prompt_tokens = prompt_tokens[-max_prompt:]
                prompt = enc.decode(prompt_tokens)
            
            # Generate multiple completions
            completions = []
            for _ in range(args.num_completions):
                text, _ = inf.generate_text(
                    model,
                    enc,
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    device=str(device),
                    top_p=args.top_p,
                )
                full_tokens = enc.encode(text)[: args.block_size]
                pred = extract_answer(text)
                reward = args.reward_correct if (pred == gold) else args.reward_incorrect
                completions.append({
                    "tokens": full_tokens,
                    "text": text,
                    "pred": pred,
                    "reward": reward,
                })
            
            # Sort by reward (best first)
            completions.sort(key=lambda x: x["reward"], reverse=True)
            
            # Skip if all same reward (no preference signal)
            if len(set(c["reward"] for c in completions)) <= 1:
                continue
            
            # Store preference pair
            preference_pairs.append({
                "prompt": prompt,
                "gold_answer": gold,
                "prompt_tokens": prompt_tokens,
                "chosen": completions[0],
                "rejected": completions[-1],  # Worst completion
            })
    
    print(f"✅ Generated {len(preference_pairs)} valid preference pairs")
    return preference_pairs


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device(
        args.device if (not args.device.startswith("cuda") or torch.cuda.is_available()) else "cpu"
    )
    print(f"🔧 Using device: {device}")
    
    enc = tiktoken.get_encoding("gpt2")
    inf = _load_inference_module()
    
    # Determine model architecture
    if args.transformer_size == "small":
        embed_size, heads, blocks, ff_mult = 384, 4, 3, 2
    elif args.transformer_size == "medium":
        embed_size, heads, blocks, ff_mult = 512, 8, 6, 4
    elif args.transformer_size == "gpt2-small":
        embed_size, heads, blocks, ff_mult = 768, 12, 12, 4
    elif args.transformer_size == "gpt2-medium":
        embed_size, heads, blocks, ff_mult = 1024, 16, 24, 4
    elif args.transformer_size == "gpt2-large":
        embed_size, heads, blocks, ff_mult = 1280, 20, 36, 4
    else:  # gpt2-xl
        embed_size, heads, blocks, ff_mult = 1600, 25, 48, 4
    
    print(f"📐 Model: {args.transformer_size} ({embed_size}d, {heads}h, {blocks}L)")
    
    # Load model
    model = inf.TransformerModel(
        vocab_size=enc.n_vocab,
        block_size=args.block_size,
        d_model=embed_size,
        n_heads=heads,
        n_blocks=blocks,
        ff_mult=ff_mult,
    )
    
    print(f"📦 Loading checkpoint: {args.init_from}")
    state = torch.load(args.init_from, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    
    # Load training data
    print(f"📚 Loading training data: {args.train_data}")
    train_lines = [
        ln.strip() for ln in Path(args.train_data).read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    print(f"   ✓ {len(train_lines)} training examples")
    
    # Generate preference pairs
    preference_pairs = generate_preference_pairs(
        args, model, train_lines, device, enc, inf
    )
    
    # Save to JSONL
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in preference_pairs:
            f.write(json.dumps(pair) + "\n")
    
    print(f"💾 Saved {len(preference_pairs)} preference pairs to {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
