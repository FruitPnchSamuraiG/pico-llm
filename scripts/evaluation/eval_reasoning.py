#!/usr/bin/env python3
"""Evaluate a checkpoint on synthetic reasoning datasets.

This uses Pico-LLM's inference loading (TransformerModel + generation) and computes
simple accuracy by extracting an answer token from the generated text.

Supports decode modes implemented in inference.py:
  greedy / nucleus / beam / lookahead

Usage:
  source /scratch/kk6081/ml_fall25/venv/bin/activate
  python scripts/eval_reasoning.py \
    --checkpoint /scratch/kk6081/picollm_extend/transformer_reasoning_transformer_epoch1.pt \
    --data data/reasoning_arith_val.txt \
    --decode greedy
"""

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple, cast, Any

import torch
import tiktoken

# Import from inference.py to reuse model construction + decoding
import importlib.util

RE_ANSWER = re.compile(r"Answer:\s*([^\n\r]+)")
RE_GSM8K_HASH = re.compile(r"####\s*([^\n\r]+)")
RE_BOXED = re.compile(r"\\boxed\{([^}]+)\}")
RE_A_COLON_FALLBACK = re.compile(r"\bA:\s*([-+]?\d+)\b")
RE_LAST_INT = re.compile(r"([-+]?\d+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data", type=str, required=True)

    p.add_argument("--decode", type=str, default="greedy", choices=["greedy", "nucleus", "beam", "lookahead"])
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--beam_width", type=int, default=4)
    p.add_argument("--lookahead_k", type=int, default=8)
    p.add_argument("--lookahead_h", type=int, default=4)
    p.add_argument("--rep_penalty", type=float, default=0.8)

    p.add_argument("--max_new_tokens", type=int, default=64)

    # Architecture presets (recommended)
    p.add_argument(
        "--transformer_size",
        type=str,
        default="",
        choices=["", "small", "medium", "gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        help="Optional shortcut to set embed/heads/blocks/ff_mult to match training scripts.",
    )

    # Must match checkpoint (manual override)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--embed_size", type=int, default=384)
    p.add_argument("--transformer_heads", type=int, default=4)
    p.add_argument("--transformer_blocks", type=int, default=3)
    p.add_argument("--ff_mult", type=int, default=2)

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--limit", type=int, default=200)
    return p.parse_args()


def _load_inference_module() -> Any:
    here = Path(__file__).parent.parent
    inf_path = here / "inference.py"
    spec = importlib.util.spec_from_file_location("pico_inference", str(inf_path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    cast(Any, spec.loader).exec_module(mod)  # type: ignore[call-arg]
    return mod


def _extract_int_like(text: str) -> Optional[str]:
    """Extract an answer token from text.

    Priority:
      1) '#### X' (GSM8K)
      2) '\\boxed{X}'
      3) 'Answer: X'
      4) 'A: X'
      5) last integer anywhere
    """
    m = RE_GSM8K_HASH.search(text)
    if m:
        return m.group(1).strip().split()[0]

    mbox = RE_BOXED.search(text)
    if mbox:
        return mbox.group(1).strip().split()[0]

    m2 = RE_ANSWER.search(text)
    if m2:
        return m2.group(1).strip().split()[0]

    m3 = RE_A_COLON_FALLBACK.search(text)
    if m3:
        return m3.group(1).strip()

    ms = RE_LAST_INT.findall(text)
    if ms:
        return ms[-1].strip()

    return None


def split_qa(line: str) -> Tuple[str, str]:
    """Try to split both legacy 'Answer:' datasets and GSM8K '####' datasets."""
    if "####" in line:
        q_part, ans_part = line.split("####", 1)
        gold = _extract_int_like("#### " + ans_part.strip()) or ""
        prompt = q_part.strip()
        if not prompt.endswith(" A:"):
            prompt += " A:"
        return prompt, gold

    if " A: " not in line:
        return line.strip(), ""

    q, rest = line.split(" A: ", 1)
    gold = _extract_int_like(rest) or ""
    return (q.strip() + " A:"), gold


def main() -> None:
    args = parse_args()

    # Apply preset sizes if requested
    if args.transformer_size:
        if args.transformer_size == "small":
            args.embed_size, args.transformer_heads, args.transformer_blocks, args.ff_mult = 384, 4, 3, 2
        elif args.transformer_size == "medium":
            args.embed_size, args.transformer_heads, args.transformer_blocks, args.ff_mult = 512, 8, 6, 4
        elif args.transformer_size == "gpt2-small":
            args.embed_size, args.transformer_heads, args.transformer_blocks, args.ff_mult = 768, 12, 12, 4
        elif args.transformer_size == "gpt2-medium":
            args.embed_size, args.transformer_heads, args.transformer_blocks, args.ff_mult = 1024, 16, 24, 4
        elif args.transformer_size == "gpt2-large":
            args.embed_size, args.transformer_heads, args.transformer_blocks, args.ff_mult = 1280, 20, 36, 4
        elif args.transformer_size == "gpt2-xl":
            args.embed_size, args.transformer_heads, args.transformer_blocks, args.ff_mult = 1600, 25, 48, 4

    device = torch.device(args.device if (not args.device.startswith("cuda") or torch.cuda.is_available()) else "cpu")
    enc = tiktoken.get_encoding("gpt2")

    inf = _load_inference_module()

    model = inf.TransformerModel(
        vocab_size=enc.n_vocab,
        block_size=args.block_size,
        d_model=args.embed_size,
        n_heads=args.transformer_heads,
        n_blocks=args.transformer_blocks,
        ff_mult=args.ff_mult,
    )

    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    data_path = Path(args.data)
    lines = [ln.strip() for ln in data_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if args.limit > 0:
        lines = lines[: args.limit]

    correct = 0
    total = 0

    for ln in lines:
        prompt, gold = split_qa(ln)

        # Truncate overly-long prompts to fit within the model context window.
        # Need headroom for generation: prompt_len + max_new_tokens <= block_size
        max_prompt_tokens = max(1, args.block_size - args.max_new_tokens)
        p_tokens = enc.encode(prompt)
        if len(p_tokens) > max_prompt_tokens:
            p_tokens = p_tokens[-max_prompt_tokens:]
            prompt = enc.decode(p_tokens)

        if args.decode == "greedy":
            text, _ = inf.generate_text(model, enc, prompt, max_new_tokens=args.max_new_tokens, device=str(device), top_p=None)
        elif args.decode == "nucleus":
            text, _ = inf.generate_text(model, enc, prompt, max_new_tokens=args.max_new_tokens, device=str(device), top_p=args.top_p)
        elif args.decode == "beam":
            text, _ = inf.decode_beam_search(model, enc, prompt, args.max_new_tokens, device, beam_width=args.beam_width)
        else:
            text, _ = inf.decode_lookahead_search(
                model,
                enc,
                prompt,
                args.max_new_tokens,
                device,
                top_p=args.top_p,
                lookahead_k=args.lookahead_k,
                lookahead_h=args.lookahead_h,
                rep_penalty_w=args.rep_penalty,
            )

        pred = _extract_int_like(text)
        total += 1
        if pred is not None and gold != "" and pred == gold:
            correct += 1

    acc = correct / max(1, total)
    print(f"checkpoint: {args.checkpoint}")
    print(f"data: {args.data} (n={total})")
    print(f"decode: {args.decode}")
    print(f"accuracy: {acc:.3f} ({correct}/{total})")


if __name__ == "__main__":
    main()
