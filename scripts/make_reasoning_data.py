#!/usr/bin/env python3
"""Generate synthetic reasoning datasets as plain-text lines.

Each line is a single training example (fits Pico-LLM's `--input_files` format).
We generate two task families:

1) Arithmetic (2-step):
   - Add/sub/mul with small integers.
   - Output includes a short chain-of-thought and a final `Answer: <int>`.

2) Logic (comparisons / transitivity):
   - Simple relational facts (older/taller) over 3 entities.
   - Output includes reasoning trace and final `Answer: <name>`.

Outputs:
  data/reasoning_arith_train.txt
  data/reasoning_arith_val.txt
  data/reasoning_logic_train.txt
  data/reasoning_logic_val.txt

Usage:
  python scripts/make_reasoning_data.py --out_dir data --n_train 5000 --n_val 500
"""

import argparse
import os
import random
from typing import List, Tuple

NAMES = [
    "Tom", "Anna", "Ben", "Lily", "Emma", "Noah", "Mia", "Ava", "Leo", "Zoe",
    "Max", "Ivy", "Sam", "Nina", "Omar", "Kai", "Eli", "Uma", "Ian", "Ada",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_train", type=int, default=5000)
    p.add_argument("--n_val", type=int, default=500)
    p.add_argument("--max_int", type=int, default=20)
    return p.parse_args()


def _arith_example(max_int: int) -> Tuple[str, str]:
    a = random.randint(0, max_int)
    b = random.randint(0, max_int)
    c = random.randint(0, max_int)

    # pick a 2-step expression: (a + b) - c or (a + b) + c or (a * b) + c
    template = random.choice(["add_sub", "add_add", "mul_add"])

    if template == "add_sub":
        x = a + b
        ans = x - c
        q = f"Q: Compute ( {a} + {b} ) - {c}. Let's think step by step."
        cot = f"A: First {a} + {b} = {x}. Then {x} - {c} = {ans}. Answer: {ans}"
    elif template == "add_add":
        x = a + b
        ans = x + c
        q = f"Q: Compute ( {a} + {b} ) + {c}. Let's think step by step."
        cot = f"A: First {a} + {b} = {x}. Then {x} + {c} = {ans}. Answer: {ans}"
    else:
        x = a * b
        ans = x + c
        q = f"Q: Compute ( {a} * {b} ) + {c}. Let's think step by step."
        cot = f"A: First {a} * {b} = {x}. Then {x} + {c} = {ans}. Answer: {ans}"

    return q, cot


def _logic_example() -> Tuple[str, str]:
    # Choose 3 unique names
    x, y, z = random.sample(NAMES, 3)
    attr = random.choice(["older", "taller"])

    # We enforce x > y > z
    q = (
        f"Q: {x} is {attr} than {y}. {y} is {attr} than {z}. "
        f"Who is the most {attr}? Let's think step by step."
    )
    cot = (
        f"A: If {x} is {attr} than {y} and {y} is {attr} than {z}, "
        f"then {x} is {attr} than {z}. So {x} is the most {attr}. Answer: {x}"
    )
    return q, cot


def _write_lines(path: str, lines: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln.rstrip("\n") + "\n")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    # Arithmetic
    ar_train = []
    ar_val = []
    for _ in range(args.n_train):
        q, a = _arith_example(args.max_int)
        ar_train.append(q + " " + a)
    for _ in range(args.n_val):
        q, a = _arith_example(args.max_int)
        ar_val.append(q + " " + a)

    # Logic
    lg_train = []
    lg_val = []
    for _ in range(args.n_train):
        q, a = _logic_example()
        lg_train.append(q + " " + a)
    for _ in range(args.n_val):
        q, a = _logic_example()
        lg_val.append(q + " " + a)

    _write_lines(os.path.join(args.out_dir, "reasoning_arith_train.txt"), ar_train)
    _write_lines(os.path.join(args.out_dir, "reasoning_arith_val.txt"), ar_val)
    _write_lines(os.path.join(args.out_dir, "reasoning_logic_train.txt"), lg_train)
    _write_lines(os.path.join(args.out_dir, "reasoning_logic_val.txt"), lg_val)

    print("✅ Wrote synthetic reasoning datasets to:")
    print(f"  {os.path.join(args.out_dir, 'reasoning_arith_train.txt')}")
    print(f"  {os.path.join(args.out_dir, 'reasoning_arith_val.txt')}")
    print(f"  {os.path.join(args.out_dir, 'reasoning_logic_train.txt')}")
    print(f"  {os.path.join(args.out_dir, 'reasoning_logic_val.txt')}")


if __name__ == "__main__":
    main()
