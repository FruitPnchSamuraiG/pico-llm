#!/usr/bin/env python3
"""
Load a trained checkpoint (SFT or PPO) and generate text from a prompt.

Features:
- Handles checkpoints saved as raw state_dict or dicts with model_state_dict/state_dict keys.
- Infers vocab_size, d_model, block_size, and presence of value head from weights.
- Allows forcing value head on/off when loading PPO-tuned models.
- Supports greedy or nucleus (top-p) sampling with temperature.
- CLI aligns with PPO guide (max_tokens alias).
"""
import argparse
import torch
import torch.nn.functional as F
import tiktoken

from pico_llm import TransformerModel


def nucleus_sample(logits, top_p: float):
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    cutoff = torch.searchsorted(cumsum, torch.tensor(top_p, device=logits.device))
    cutoff = torch.clamp(cutoff, min=1)
    kept_probs = sorted_probs[:cutoff]
    kept_idx = sorted_idx[:cutoff]
    kept_probs = kept_probs / kept_probs.sum()
    sample_pos = torch.multinomial(kept_probs, 1).item()
    return kept_idx[sample_pos].item()


def _infer_model_shapes(state_dict, default_d_model, default_block_size, default_vocab_size):
    """Infer d_model, block_size, vocab_size, and value-head presence from state_dict."""
    d_model = default_d_model
    block_size = default_block_size
    vocab_size = default_vocab_size
    if "embed.weight" in state_dict:
        vocab_size, d_model = state_dict["embed.weight"].shape
    if "pos_emb.weight" in state_dict:
        block_size = state_dict["pos_emb.weight"].shape[0]
    use_value_head = any(k.startswith("value_head") for k in state_dict.keys())
    return d_model, block_size, vocab_size, use_value_head


def load_checkpoint(checkpoint_path, device, d_model, n_heads, n_blocks, block_size, force_value_head=None):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Infer shapes and whether value head exists
    d_model_infer, block_size_infer, vocab_size_infer, has_value_head = _infer_model_shapes(
        state_dict, d_model, block_size, vocab_size
    )
    d_model = d_model_infer
    block_size = block_size_infer
    vocab_size = vocab_size_infer
    use_value_head = force_value_head if force_value_head is not None else has_value_head

    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_blocks=n_blocks,
        block_size=block_size,
        use_value_head=use_value_head,
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, enc, use_value_head, d_model, block_size, vocab_size


def generate(model, enc, prompt, max_new_tokens, device, top_p, temperature):
    tokens = enc.encode(prompt)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(1)
            outputs = model(x)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            next_logits = logits[-1, 0, :] / temperature
            if top_p is None:
                next_token = torch.argmax(next_logits).item()
            else:
                next_token = nucleus_sample(next_logits, top_p)
            tokens.append(next_token)
    return enc.decode(tokens)


def main():
    parser = argparse.ArgumentParser(description="Load a checkpoint and generate text")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--prompt", default="Once upon a time", help="Prompt to start from")
    parser.add_argument("--max_tokens", type=int, default=None, help="Number of tokens to generate (alias)")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Number of tokens to generate")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p for nucleus sampling (None = greedy)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension used in training")
    parser.add_argument("--n_heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--n_blocks", type=int, default=3, help="Transformer blocks")
    parser.add_argument("--block_size", type=int, default=1024, help="Block size used in training")
    parser.add_argument("--use_value_head", action="store_true", help="Force-enable value head on load")
    parser.add_argument("--no_value_head", action="store_true", help="Force-disable value head on load")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Resolve max tokens preference
    max_new_tokens = args.max_tokens if args.max_tokens is not None else args.max_new_tokens
    if max_new_tokens is None:
        max_new_tokens = 100

    force_value_head = None
    if args.use_value_head:
        force_value_head = True
    elif args.no_value_head:
        force_value_head = False

    model, enc, use_value_head, d_model, block_size, vocab_size = load_checkpoint(
        args.checkpoint,
        device,
        args.d_model,
        args.n_heads,
        args.n_blocks,
        args.block_size,
        force_value_head=force_value_head,
    )

    print(
        f"Loaded checkpoint with vocab_size={vocab_size}, d_model={d_model}, block_size={block_size}, value_head={use_value_head}"
    )

    text = generate(
        model,
        enc,
        args.prompt,
        max_new_tokens,
        device,
        args.top_p,
        args.temperature,
    )
    print("\n=== Generated Text ===\n")
    print(text)


if __name__ == "__main__":
    main()
