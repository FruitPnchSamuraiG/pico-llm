#!/usr/bin/env python3
"""
Quick inference script for DPO-trained models on GSM8K-style math problems.

Usage:
    python scripts/inference_dpo.py --checkpoint /path/to/transformer_dpo_final.pt
    python scripts/inference_dpo.py --checkpoint /path/to/transformer_dpo_final.pt --prompt "Your question here"
"""

import argparse
import sys
from pathlib import Path
import torch
import tiktoken

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from inference module
import importlib.util
spec = importlib.util.spec_from_file_location("pico_inference", str(Path(__file__).parent.parent / "inference.py"))
inf_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inf_mod)

# Import model and generation
TransformerModel = inf_mod.TransformerModel
generate_text = inf_mod.generate_text


DEFAULT_PROMPTS = [
    "Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? A:",
    "Q: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take? A:",
    "Q: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make? A:",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Inference on DPO-trained model")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to DPO model checkpoint (.pt)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom prompt (default: use example GSM8K problems)")
    parser.add_argument("--max_tokens", type=int, default=128,
                        help="Maximum tokens to generate (default: 128)")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Nucleus sampling parameter (default: 0.95)")
    parser.add_argument("--repetition_penalty", type=float, default=1.2,
                        help="Repetition penalty (1.0=off, 1.2=recommended, default: 1.2)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device (default: cuda:0)")
    
    # Model architecture (for medium model)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--embed_size", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--ff_mult", type=int, default=4)
    
    return parser.parse_args()


def extract_answer(text: str) -> str:
    """Extract final answer from GSM8K-style output."""
    import re
    
    # Try #### format
    if "####" in text:
        ans = text.split("####")[-1].strip().split()[0]
        return ans
    
    # Try last number
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if nums:
        return nums[-1]
    
    return "???"


def main():
    args = parse_args()
    
    # Check device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    # Load tokenizer
    enc = tiktoken.get_encoding("gpt2")
    print(f"📚 Loaded tokenizer (vocab_size={enc.n_vocab})")
    
    # Load model
    print(f"📦 Loading model from: {args.checkpoint}")
    model = TransformerModel(
        vocab_size=enc.n_vocab,
        block_size=args.block_size,
        d_model=args.embed_size,
        n_heads=args.heads,
        n_blocks=args.blocks,
        ff_mult=args.ff_mult,
    )
    
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"✅ Model loaded ({args.embed_size}d, {args.heads}h, {args.blocks}L)")
    
    # Prepare prompts
    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS
    
    print("\n" + "=" * 80)
    print("🚀 Running Inference")
    print("=" * 80 + "\n")
    
    # Generate for each prompt
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'─' * 80}")
        print(f"Example {i}/{len(prompts)}")
        print(f"{'─' * 80}")
        print(f"\n📝 PROMPT:")
        print(prompt)
        print(f"\n🤖 GENERATED:")
        
        # Generate
        with torch.no_grad():
            output, _ = generate_text(
                model,
                enc,
                prompt,
                max_new_tokens=args.max_tokens,
                device=str(device),
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
        
        # Extract continuation (remove prompt)
        continuation = output[len(prompt):]
        print(continuation)
        
        # Extract answer
        answer = extract_answer(output)
        print(f"\n💡 EXTRACTED ANSWER: {answer}")
    
    print("\n" + "=" * 80)
    print("✅ Inference complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
