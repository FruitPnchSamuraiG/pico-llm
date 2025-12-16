#!/usr/bin/env python3
"""
Unified Inference Script for Pico-LLM.

Supports:
- Standard generation
- Thinking-aware generation (Reasoning models)
- KV-caching for speed
- Various sampling strategies (Greedy, Nucleus)
- DPO/GRPO trained models

Usage:
    python inference.py --checkpoint path/to/model.pt --prompt "Q: 2+2=? A:"
    python inference.py --checkpoint path/to/model.pt --prompt "Q: 2+2=? A: <thinking>" --thinking
"""

import argparse
import torch
import tiktoken
import sys
import os
from pathlib import Path

# Import pico-llm
# Assuming this script is in the root or scripts folder, we need to find pico-llm.py
current_dir = Path(__file__).parent.absolute()
if (current_dir / "pico-llm.py").exists():
    sys.path.append(str(current_dir))
elif (current_dir.parent / "pico-llm.py").exists():
    sys.path.append(str(current_dir.parent))
else:
    # Fallback to known location
    sys.path.append("/home/kk6081/pico_llm_extend/pico-llm")

import importlib.util
try:
    spec = importlib.util.spec_from_file_location("pico_llm", "pico-llm.py")
    if spec is None:
        # Try absolute path
        spec = importlib.util.spec_from_file_location("pico_llm", "/home/kk6081/pico_llm_extend/pico-llm/pico-llm.py")
    
    pico_llm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pico_llm)
except Exception as e:
    print(f"Error loading pico-llm.py: {e}")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Inference for Pico-LLM")
    
    # Model & Input
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Input prompt")
    parser.add_argument("--input_file", type=str, help="Optional file with prompts (one per line)")
    
    # Generation Mode
    parser.add_argument("--thinking", action="store_true", help="Enable thinking-aware generation (for reasoning models)")
    
    # Generation Params
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Max tokens to generate (standard mode)")
    parser.add_argument("--max_thinking_tokens", type=int, default=800, help="Max thinking tokens (thinking mode)")
    parser.add_argument("--max_answer_tokens", type=int, default=200, help="Max answer tokens (thinking mode)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling threshold")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device")
    
    # Model Architecture (if not inferable from checkpoint)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--embed_size", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=24)
    parser.add_argument("--ff_mult", type=int, default=4)
    parser.add_argument("--transformer_size", type=str, default=None, 
                        choices=["small", "medium", "gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"],
                        help="Preset architecture size (overrides manual dims)")

    return parser.parse_args()

def load_model(args, device):
    print(f"Loading model from {args.checkpoint}...")
    
    # Load state dict first to infer architecture
    try:
        state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
        # Handle DDP prefix if present
        if all(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

    # Auto-detect architecture from state_dict
    if 'embed.weight' in state_dict:
        detected_embed = state_dict['embed.weight'].shape[1]
        detected_vocab = state_dict['embed.weight'].shape[0]
        print(f"Auto-detected: embed_size={detected_embed}, vocab_size={detected_vocab}")
        args.embed_size = detected_embed
        args.vocab_size = detected_vocab
    
    if 'pos_emb.weight' in state_dict:
        detected_block = state_dict['pos_emb.weight'].shape[0]
        print(f"Auto-detected: block_size={detected_block}")
        args.block_size = detected_block

    # Detect blocks
    block_keys = [k for k in state_dict.keys() if k.startswith("blocks.")]
    if block_keys:
        max_block_idx = max([int(k.split('.')[1]) for k in block_keys])
        detected_blocks = max_block_idx + 1
        print(f"Auto-detected: n_blocks={detected_blocks}")
        args.blocks = detected_blocks
        
        # Detect ff_mult
        # blocks.0.ff.0.weight shape is (ff_dim, d_model)
        if 'blocks.0.ff.0.weight' in state_dict:
            ff_dim = state_dict['blocks.0.ff.0.weight'].shape[0]
            detected_ff_mult = ff_dim // args.embed_size
            print(f"Auto-detected: ff_mult={detected_ff_mult}")
            args.ff_mult = detected_ff_mult

    # Determine architecture (override if preset is used, but warn if mismatch)
    if args.transformer_size:
        if args.transformer_size == "small":
            args.embed_size, args.heads, args.blocks, args.ff_mult = 384, 4, 3, 2
        elif args.transformer_size == "medium":
            args.embed_size, args.heads, args.blocks, args.ff_mult = 512, 8, 6, 4
        elif args.transformer_size == "gpt2-small":
            args.embed_size, args.heads, args.blocks, args.ff_mult = 768, 12, 12, 4
        elif args.transformer_size == "gpt2-medium":
            args.embed_size, args.heads, args.blocks, args.ff_mult = 1024, 16, 24, 4
    
    enc = tiktoken.get_encoding("gpt2")
    
    vocab_size = getattr(args, 'vocab_size', enc.n_vocab)

    model = pico_llm.TransformerModel(
        vocab_size=vocab_size,
        block_size=args.block_size,
        d_model=args.embed_size,
        n_heads=args.heads,
        n_blocks=args.blocks,
        ff_mult=args.ff_mult
    )
    
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading state dict: {e}")
        print("Attempting to load with strict=False...")
        try:
            model.load_state_dict(state_dict, strict=False)
        except:
            print("Failed to load model.")
            sys.exit(1)
            
    model.to(device)
    model.eval()
    return model, enc

def main():
    args = parse_args()
    device = torch.device(args.device)
    
    model, enc = load_model(args, device)
    
    prompts = []
    if args.input_file:
        with open(args.input_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = [args.prompt]
        
    print(f"\nGenerating for {len(prompts)} prompts...\n" + "="*50)
    
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}: {prompt}")
        print("-" * 20)
        
        if args.thinking:
            # Ensure prompt triggers thinking if not present
            if "<thinking>" not in prompt and "A:" in prompt:
                prompt += " <thinking>"
            elif "<thinking>" not in prompt:
                 prompt += " A: <thinking>"
                 
            output, phase_info = pico_llm.generate_text_with_thinking(
                model, enc, prompt,
                max_thinking_tokens=args.max_thinking_tokens,
                max_answer_tokens=args.max_answer_tokens,
                device=args.device,
                top_p=args.top_p,
                temperature=args.temperature
            )
            print(f"Output:\n{output}")
            print(f"\n[Stats] Thinking: {phase_info['thinking_tokens']} toks, Answer: {phase_info['answer_tokens']} toks")
            
        else:
            output, _ = pico_llm.generate_text(
                model, enc, prompt,
                max_new_tokens=args.max_new_tokens,
                device=args.device,
                top_p=args.top_p,
                temperature=args.temperature
            )
            print(f"Output:\n{output}")
            
        print("="*50)

if __name__ == "__main__":
    main()
