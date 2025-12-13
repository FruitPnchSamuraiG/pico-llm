#!/usr/bin/env python3
"""Prepare a Hugging Face reasoning dataset for Pico-LLM.

Pico-LLM's trainer expects plain text files where:
  - each line is one training example
  - examples are truncated to `block_size` tokens by the training pipeline

This script downloads a dataset from the Hugging Face Hub and writes text files.
It is intentionally defensive about schemas: it tries to extract a reasonable
single text field from each row.

Supported special-case formatting:
  - openai/gsm8k (config: main): converts to a stable single-line format:
      "Q: ... A: ... Answer: <final_number>"
    This makes automatic evaluation (numeric final answer extraction) reliable.

Usage:
  source /scratch/kk6081/ml_fall25/venv/bin/activate

  # OpenThoughts
  python scripts/prepare_hf_reasoning_data.py \
    --dataset open-thoughts/OpenThoughts-114k \
    --out_train data/open_thoughts_train.txt \
    --out_val data/open_thoughts_val.txt

  # GSM8K
  python scripts/prepare_hf_reasoning_data.py \
    --dataset openai/gsm8k --config main \
    --out_train data/gsm8k_train.txt \
    --out_val data/gsm8k_val.txt \
    --out_test data/gsm8k_test.txt \
    --train_split "train[2%:]" \
    --val_split "train[:2%]" \
    --test_split test
"""

from __future__ import annotations

import argparse
import re
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

_RE_LAST_INT = re.compile(r"([-+]?\d+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True, help="HF dataset id, e.g. open-thoughts/OpenThoughts-114k")
    p.add_argument("--config", type=str, default="", help="Optional HF dataset config name (e.g., 'main' for openai/gsm8k)")

    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--val_split", type=str, default="validation")
    p.add_argument("--test_split", type=str, default="", help="Optional test split name or slice (e.g., 'test')")

    p.add_argument("--out_train", type=str, required=True)
    p.add_argument("--out_val", type=str, required=True)
    p.add_argument("--out_test", type=str, default="", help="Optional path for test export")

    p.add_argument("--limit_train", type=int, default=0, help="0 = no limit")
    p.add_argument("--limit_val", type=int, default=0, help="0 = no limit")
    p.add_argument("--limit_test", type=int, default=0, help="0 = no limit")

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


def _normalize(s: str) -> str:
    # single-line training format
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = " ".join(s.split())
    return s.strip()


def _gsm8k_extract_final_int(answer: str) -> Optional[str]:
    """Extract the final integer answer from GSM8K's answer field.

    GSM8K answers often contain a rationale and end with something like:
      "#### 42"
    We fall back to the last integer anywhere.
    """
    if not answer:
        return None
    a = answer.strip()

    # Most common marker
    if "####" in a:
        tail = a.split("####")[-1].strip()
        m = _RE_LAST_INT.search(tail)
        if m:
            return m.group(1)

    ms = _RE_LAST_INT.findall(a)
    if ms:
        return ms[-1]
    return None


def _format_gsm8k_row(row: Dict[str, Any]) -> Optional[str]:
    # Expected keys: question, answer
    q = _stringify(row.get("question", "")).strip()
    a = _stringify(row.get("answer", "")).strip()
    if not q or not a:
        return None

    final = _gsm8k_extract_final_int(a)
    if final is None:
        return None

    # Keep the worked solution as training signal, but enforce a stable final answer tag.
    # This aligns with eval_reasoning.py's extraction logic.
    line = f"Q: {q} A: {a} Answer: {final}"
    return _normalize(line)


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


def _iter_text(
    ds: Iterable[Dict[str, Any]],
    limit: int,
    min_chars: int,
    dataset_id: str,
) -> Iterable[str]:
    n = 0
    for row in ds:
        if limit and n >= limit:
            break

        t: Optional[str] = None

        # Dataset-specific adapter(s)
        if dataset_id == "openai/gsm8k":
            t = _format_gsm8k_row(row)
        else:
            t = _extract_text(row)
            if t:
                t = _normalize(t)

        if not t:
            continue
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


def _load_split(dataset: str, config: str, split: str):
    try:
        if config:
            return load_dataset(dataset, config, split=split)
        return load_dataset(dataset, split=split)
    except Exception as e:
        raise SystemExit(f"Failed to load dataset split '{split}' for {dataset}{(':'+config) if config else ''}: {e}")


def main() -> None:
    args = parse_args()

    ds_train = _load_split(args.dataset, args.config, args.train_split)

    try:
        ds_val = _load_split(args.dataset, args.config, args.val_split)
    except SystemExit as e:
        raise SystemExit(
            f"{e}\n"
            "Tip: many datasets only have 'train'. You can use slice syntax like 'train[:1%]' for val."
        )

    ds_test = None
    if args.test_split:
        ds_test = _load_split(args.dataset, args.config, args.test_split)

    out_train = Path(args.out_train)
    out_val = Path(args.out_val)
    out_test = Path(args.out_test) if args.out_test else None

    n_train = _write_lines(out_train, _iter_text(ds_train, args.limit_train, args.min_chars, args.dataset))  # type: ignore
    n_val = _write_lines(out_val, _iter_text(ds_val, args.limit_val, args.min_chars, args.dataset))  # type: ignore

    n_test = 0
    if ds_test is not None and out_test is not None:
        n_test = _write_lines(out_test, _iter_text(ds_test, args.limit_test, args.min_chars, args.dataset))  # type: ignore

    print("✅ Wrote:")
    print(f"  train: {out_train} (n={n_train})")
    print(f"  val:   {out_val} (n={n_val})")
    if out_test is not None:
        print(f"  test:  {out_test} (n={n_test})")


if __name__ == "__main__":
    main()
