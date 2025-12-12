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
import time
from typing import List, Tuple, Optional, Dict

# Import functions and classes from pico-llm.py
import importlib.util
import os

# Find pico-llm.py in the same directory as this script
script_dir = Path(__file__).parent.absolute()
pico_llm_path = script_dir / "pico-llm.py"

spec = importlib.util.spec_from_file_location("pico_llm", str(pico_llm_path))
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
    
    # New: decoding strategy
    parser.add_argument("--decode", type=str, default="nucleus",
                        choices=["greedy", "nucleus", "beam", "lookahead"],
                        help="Decoding strategy. 'lookahead' = test-time search (LNS)")
    parser.add_argument("--beam_width", type=int, default=4,
                        help="Beam width for --decode beam")
    parser.add_argument("--lookahead_k", type=int, default=8,
                        help="How many candidate next-tokens to consider at each step for lookahead")
    parser.add_argument("--lookahead_h", type=int, default=8,
                        help="Lookahead horizon (tokens) for lookahead scoring")
    parser.add_argument("--rep_penalty", type=float, default=0.2,
                        help="Repetition penalty weight (0 disables) used by lookahead scoring")
    
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
    
    return parser.parse_args()


# -----------------------------
# Transformer-only decoding utils
# -----------------------------

def _next_token_logits(model: torch.nn.Module, context_tokens: List[int], device: torch.device) -> torch.Tensor:
    """Return logits for next token given context (no KV cache; recomputes full forward)."""
    seq_tensor = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(1)  # (seq_len, 1)
    logits_seq = model(seq_tensor)  # (seq_len, 1, vocab)
    return logits_seq[-1, 0, :]


def _sample_from_logits(next_logits: torch.Tensor, mode: str, top_p: Optional[float]) -> int:
    if mode == "greedy" or top_p is None:
        return int(torch.argmax(next_logits).item())
    return int(nucleus_sampling(next_logits, p=float(top_p)))


def _distinct_ngram_ratio(tokens: List[int], n: int) -> float:
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / max(1, len(ngrams))


def _rep_penalty(tokens: List[int], n: int = 4) -> float:
    """Higher means more repetition; used as a penalty term."""
    # A simple repetition proxy: 1 - distinct-n
    return 1.0 - _distinct_ngram_ratio(tokens, n)


def decode_greedy_or_nucleus(model, enc, prompt: str, max_new_tokens: int, device: torch.device, top_p: Optional[float]):
    return generate_text(model=model, enc=enc, init_text=prompt, max_new_tokens=max_new_tokens, device=str(device), top_p=top_p)


def decode_beam_search(model, enc, prompt: str, max_new_tokens: int, device: torch.device, beam_width: int):
    """Length-normalized beam search using logprobs from the model."""
    model_was_training = model.training
    model.eval()

    with torch.no_grad():
        init_tokens = enc.encode(prompt)
        beams: List[Tuple[List[int], float]] = [(init_tokens[:], 0.0)]  # (tokens, logprob)

        for _ in range(max_new_tokens):
            candidates: List[Tuple[List[int], float]] = []
            for tokens, lp in beams:
                next_logits = _next_token_logits(model, tokens, device)
                logp = torch.log_softmax(next_logits, dim=-1)
                topk = torch.topk(logp, k=min(beam_width, logp.numel()))
                for j in range(topk.indices.numel()):
                    tid = int(topk.indices[j].item())
                    cand_lp = float(topk.values[j].item())
                    candidates.append((tokens + [tid], lp + cand_lp))

            # keep best beams by length-normalized score
            candidates.sort(key=lambda x: x[1] / max(1, (len(x[0]) - len(init_tokens))), reverse=True)
            beams = candidates[:beam_width]

        best_tokens, _ = beams[0]

    model.train(model_was_training)
    final_text = enc.decode(best_tokens)
    return final_text, final_text


def decode_lookahead_search(model, enc, prompt: str, max_new_tokens: int, device: torch.device,
                           top_p: Optional[float], lookahead_k: int, lookahead_h: int, rep_penalty_w: float):
    """Lookahead Nucleus Search (LNS): pick next token by scoring K candidates with H-step rollout under nucleus sampling.

    Score(candidate) = avg logprob(rollout) - rep_penalty_w * repetition(candidate+rollout)
    """
    model_was_training = model.training
    model.eval()

    with torch.no_grad():
        ctx: List[int] = enc.encode(prompt)

        for _ in range(max_new_tokens):
            next_logits = _next_token_logits(model, ctx, device)
            logp = torch.log_softmax(next_logits, dim=-1)

            # Candidate set: take top-K by logprob (cheap and stable)
            topk = torch.topk(logp, k=min(lookahead_k, logp.numel()))
            cand_ids = [int(i.item()) for i in topk.indices]

            best_id = cand_ids[0]
            best_score = -1e30

            for cand in cand_ids:
                rollout_tokens = ctx + [cand]
                total_lp = float(logp[cand].item())
                steps = 1

                # Rollout H-1 more tokens using nucleus (or greedy if top_p None)
                for _h in range(max(0, lookahead_h - 1)):
                    nl = _next_token_logits(model, rollout_tokens, device)
                    lps = torch.log_softmax(nl, dim=-1)
                    tid = _sample_from_logits(nl, mode="nucleus" if top_p is not None else "greedy", top_p=top_p)
                    total_lp += float(lps[tid].item())
                    rollout_tokens.append(tid)
                    steps += 1

                avg_lp = total_lp / max(1, steps)
                rep = _rep_penalty(rollout_tokens, n=4)
                score = avg_lp - rep_penalty_w * rep

                if score > best_score:
                    best_score = score
                    best_id = cand

            ctx.append(best_id)

    model.train(model_was_training)
    final_text = enc.decode(ctx)
    return final_text, final_text


def load_model(model_type, checkpoint_path, args, device):
    """Load a trained model from checkpoint."""
    print(f"\n🔧 Loading {model_type} model from {checkpoint_path}")
    
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

    if model_type != "transformer":
        print("This project extension is transformer-only. Re-run with --model transformer.")
        sys.exit(2)

    model = load_model(model_type, checkpoint_path, args, device)

    print(f"\n{'='*70}")
    print(f"Prompt: {prompt}")
    print(f"Decode: {args.decode}")
    print(f"{'='*70}")

    t0 = time.time()
    if args.decode == "greedy":
        final_text, _ = decode_greedy_or_nucleus(model, enc, prompt, args.max_tokens, device, top_p=None)
    elif args.decode == "nucleus":
        top_p = args.top_p if args.top_p < 1.0 else 1.0
        final_text, _ = decode_greedy_or_nucleus(model, enc, prompt, args.max_tokens, device, top_p=top_p)
    elif args.decode == "beam":
        final_text, _ = decode_beam_search(model, enc, prompt, args.max_tokens, device, beam_width=args.beam_width)
    elif args.decode == "lookahead":
        top_p = args.top_p if args.top_p < 1.0 else 1.0
        final_text, _ = decode_lookahead_search(
            model, enc, prompt, args.max_tokens, device,
            top_p=top_p,
            lookahead_k=args.lookahead_k,
            lookahead_h=args.lookahead_h,
            rep_penalty_w=args.rep_penalty
        )
    else:
        raise ValueError(f"Unknown decode: {args.decode}")

    dt = time.time() - t0

    # Basic output + quick metrics
    out_tokens = enc.encode(final_text)
    d1 = _distinct_ngram_ratio(out_tokens, 1)
    d2 = _distinct_ngram_ratio(out_tokens, 2)
    rep4 = _rep_penalty(out_tokens, 4)

    print(f"\nTRANSFORMER Output:\n{final_text}")
    print(f"\nMetrics:")
    print(f"  distinct-1: {d1:.3f}")
    print(f"  distinct-2: {d2:.3f}")
    print(f"  rep-4 (1-distinct4): {rep4:.3f}")
    print(f"  wall time: {dt:.2f}s")
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
