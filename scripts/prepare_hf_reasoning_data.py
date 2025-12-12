#!/usr/bin/env python3
"""Prepare a Hugging Face reasoning dataset for Pico-LLM.

Pico-LLM's trainer expects plain text files where:
  - each line is one training example
  - examples are truncated to `block_size` tokens by the training pipeline

This script downloads a dataset from the Hugging Face Hub and writes two files:
  --out_train and --out_val

It is intentionally defensive about schemas: it tries to extract a reasonable
single text field from each row.

Default target dataset (recommended by user):
  open-thoughts/OpenThoughts-114k

Usage:
  source /scratch/kk6081/ml_fall25/venv/bin/activate
  python scripts/prepare_hf_reasoning_data.py \
    --dataset open-thoughts/OpenThoughts-114k \
    --out_train data/open_thoughts_train.txt \
    --out_val data/open_thoughts_val.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from datasets import load_dataset


PREFERRED_TEXT_KEYS = [
    # common
    "text",
    "content",
    "completion",
    "output",
    "response",
    "assistant",
    "chosen",
    # instruction-style
    "prompt",
    "instruction",
    "input",
    "question",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True, help="HF dataset id, e.g. open-thoughts/OpenThoughts-114k")
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--val_split", type=str, default="validation")

    p.add_argument("--out_train", type=str, required=True)
    p.add_argument("--out_val", type=str, required=True)

    p.add_argument("--limit_train", type=int, default=0, help="0 = no limit")
    p.add_argument("--limit_val", type=int, default=0, help="0 = no limit")

    p.add_argument(
        "--min_chars",
        type=int,
        default=20,
        help="Skip examples shorter than this many characters after normalization.",
    )

    return p.parse_args()


def _stringify(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    # Some datasets store messages as dicts/lists.
    # Use a simple, deterministic stringification.
    if isinstance(x, dict):
        parts = []
        for k in sorted(x.keys()):
            parts.append(f"{k}: { _stringify(x[k]) }")
        return "\n".join(parts)
    if isinstance(x, list):
        return "\n".join(_stringify(v) for v in x)
    return str(x)


def _extract_text(row: Dict[str, Any]) -> Optional[str]:
    # 1) if there's an explicit 'text'-like field, use it.
    for k in PREFERRED_TEXT_KEYS:
        if k in row and row[k] is not None:
            s = _stringify(row[k]).strip()
            if s:
                return s

    # 2) If there's a messages/conversation style field, stringify it.
    for k in ["messages", "conversation", "conversations", "dialogue", "chat"]:
        if k in row and row[k] is not None:
            s = _stringify(row[k]).strip()
            if s:
                return s

    # 3) As a last resort, join all string-ish fields.
    chunks: List[str] = []
    for k, v in row.items():
        s = _stringify(v).strip()
        if s:
            chunks.append(f"{k}: {s}")
    if chunks:
        return "\n".join(chunks)

    return None


def _normalize(s: str) -> str:
    # single-line training format
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = " ".join(s.split())
    return s.strip()


def _iter_text(ds: Iterable[Dict[str, Any]], limit: int, min_chars: int) -> Iterable[str]:
    n = 0
    for row in ds:
        if limit and n >= limit:
            break
        t = _extract_text(row)
        if not t:
            continue
        t = _normalize(t)
        if len(t) < min_chars:
            continue
        yield t
        n += 1


def _write_lines(path: Path, lines: Iterable[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln)
            f.write("\n")
            n += 1
    return n


def main() -> None:
    args = parse_args()

    try:
        ds_train = load_dataset(args.dataset, split=args.train_split)
    except Exception as e:
        raise SystemExit(f"Failed to load dataset split '{args.train_split}' for {args.dataset}: {e}")

    try:
        ds_val = load_dataset(args.dataset, split=args.val_split)
    except Exception as e:
        raise SystemExit(
            f"Failed to load dataset split '{args.val_split}' for {args.dataset}: {e}\n"
            "Tip: many datasets only have 'train'. You can use slice syntax like 'train[:1%]' for val."
        )

    out_train = Path(args.out_train)
    out_val = Path(args.out_val)

    n_train = _write_lines(out_train, _iter_text(ds_train, args.limit_train, args.min_chars))
    n_val = _write_lines(out_val, _iter_text(ds_val, args.limit_val, args.min_chars))

    print("✅ Wrote:")
    print(f"  train: {out_train} (n={n_train})")
    print(f"  val:   {out_val} (n={n_val})")


if __name__ == "__main__":
    main()
