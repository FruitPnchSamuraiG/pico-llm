#!/usr/bin/env python3
"""
Inference script for trained Pico-LLM models.

Usage:
    # Single model inference
    python inference.py --model transformer --checkpoint /scratch/kk6081/ml_fall25/checkpoints/transformer_epoch3.pt --prompt "Once upon a time"
    
    # Specify device
    python inference.py --model lstm --checkpoint lstm_epoch3.pt --prompt "Hello" --device cuda:1
"""

import argparse
import sys
import torch
import tiktoken
from pathlib import Path

# Import functions and classes from pico-llm.py
import importlib.util
spec = importlib.util.spec_from_file_location("pico_llm", "/home/kk6081/pico-llm/pico-llm.py")
pico_llm = importlib.util.module_from_spec(spec) # type: ignore
spec.loader.exec_module(pico_llm) # type: ignore

# Import model classes and generation functions
KGramMLPSeqModel = pico_llm.KGramMLPSeqModel
LSTMSeqModel = pico_llm.LSTMSeqModel
TransformerModel = pico_llm.TransformerModel
generate_text = pico_llm.generate_text
nucleus_sampling = pico_llm.nucleus_sampling


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on trained Pico-LLM models")
    
    # Model selection
    parser.add_argument("--model", type=str, choices=["kgram", "lstm", "transformer"], 
                        help="Model type to use for inference")
    parser.add_argument("--checkpoint", type=str, 
                        help="Path to model checkpoint file (.pt)")
    
    # Input
    parser.add_argument("--prompt", type=str, required=True,
                        help="Prompt to generate from")
    
    # Generation parameters
    parser.add_argument("--max_tokens", type=int, default=100,
                        help="Maximum number of tokens to generate (default: 100)")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p (nucleus) sampling (default: 0.95, 1.0=disabled)")
    
    # Model hyperparameters (must match training config)
    parser.add_argument("--block_size", type=int, default=256,
                        help="Maximum sequence length (default: 256)")
    parser.add_argument("--embed_size", type=int, default=384,
                        help="Embedding dimension (default: 384)")
    parser.add_argument("--kgram_k", type=int, default=3,
                        help="K-gram context size (default: 3)")
    parser.add_argument("--transformer_heads", type=int, default=4,
                        help="Transformer attention heads (default: 4)")
    parser.add_argument("--transformer_blocks", type=int, default=3,
                        help="Number of transformer blocks (default: 3)")
    parser.add_argument("--ff_mult", type=int, default=2,
                        help="Feedforward layer multiplier (default: 2)")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use (default: cuda:0)")
    
    # Checkpoint directory for --compare mode
    parser.add_argument("--checkpoint_dir", type=str, 
                        default="/scratch/kk6081/ml_fall25/checkpoints",
                        help="Directory containing checkpoint files")
    parser.add_argument("--epoch", type=int, default=3,
                        help="Which epoch checkpoint to use (default: 3)")
    
    return parser.parse_args()


def load_model(model_type, checkpoint_path, args, device):
    """Load a trained model from checkpoint."""
    print(f"\n Loading {model_type} model from {checkpoint_path}")
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    
    # Create model with same config as training
    if model_type == "kgram":
        model = KGramMLPSeqModel(
            vocab_size=vocab_size,
            embed_size=args.embed_size,
            k=args.kgram_k,
            num_inner_layers=1,
            chunk_size=1
        )
    elif model_type == "lstm":
        model = LSTMSeqModel(
            vocab_size=vocab_size,
            embed_size=args.embed_size,
            hidden_size=args.embed_size
        )
    elif model_type == "transformer":
        model = TransformerModel(
            vocab_size=vocab_size,
            block_size=args.block_size,
            d_model=args.embed_size,
            n_heads=args.transformer_heads,
            n_blocks=args.transformer_blocks,
            ff_mult=args.ff_mult
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"Model loaded successfully ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters)")
        return model
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)


def run_single_inference(model_type, checkpoint_path, prompt, args, device):
    """Run inference on a single prompt with one model."""
    enc = tiktoken.get_encoding("gpt2")
    model = load_model(model_type, checkpoint_path, args, device)
    
    print(f"\n{'='*70}")
    print(f"Prompt: {prompt}")
    print(f"{'='*70}")
    
    top_p = args.top_p if args.top_p < 1.0 else None
    
    final_text, _ = generate_text(
        model=model,
        enc=enc,
        init_text=prompt,
        max_new_tokens=args.max_tokens,
        device=device,
        top_p=top_p
    )
    
    print(f"\n{model_type.upper()} Output:")
    print(final_text)
    print(f"\n{'='*70}\n")


def main():
    args = parse_args()
    
    # Determine device
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"🔧 Using device: {device}")
    
    # Single model mode
    if not args.model or not args.checkpoint:
        print("Single model mode requires --model and --checkpoint")
        print("Run with --help for usage information")
        sys.exit(1)
    run_single_inference(args.model, args.checkpoint, args.prompt, args, device)


if __name__ == "__main__":
    main()
