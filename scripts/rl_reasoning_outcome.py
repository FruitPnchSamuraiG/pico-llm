#!/usr/bin/env python3
"""RL-style post-training for reasoning (outcome reward, deepseek-RO style).

This is a lightweight, educational implementation that fits Pico-LLM constraints.

Goal
----
Given prompts of the form:
  Q: ... A:
We sample multiple candidate completions, score them with an *outcome reward*
(e.g., exact-match on a final answer), and then update the policy to increase
likelihood of the best-scoring completion(s).

This approximates outcome-based RL ("RLAIF"/"deepseek-RO" style) without PPO.
Concretely, we use a *weighted SFT* objective:
  maximize  sum_i w_i * log p_θ(y_i | x)
where w_i is derived from reward (0/1 by default).

This is NOT a full RLHF/PPO implementation.
It is intended as:
  - a post-training phase that is closer to RL than plain SFT
  - a way to connect lecture concepts to runnable code

Data format
-----------
Use a dataset with explicit gold answers so reward is well-defined.
Recommended here: the repo's synthetic arithmetic/logic data:
  data/reasoning_arith_{train,val}.txt
  data/reasoning_logic_{train,val}.txt
Each line is assumed to include "Q:" and a gold answer somewhere after, e.g.:
  Q: ... A: ... Answer: 7

Usage
-----
source /scratch/kk6081/ml_fall25/venv/bin/activate
python scripts/rl_reasoning_outcome.py \
  --init_from /scratch/kk6081/picollm_extend/transformer_epoch1.pt \
  --train_data data/reasoning_arith_train.txt \
  --val_data data/reasoning_arith_val.txt \
  --out_dir /scratch/kk6081/picollm_extend/rl_reasoning \
  --device cuda:0

Notes
-----
- For OpenThoughts-114k there is no single canonical answer field, so outcome reward
  is ill-defined. Use synthetic reasoning or GSM8K-style data instead.
- This script uses multiple-sample "best-of-n" generation; compute cost scales with
  (num_samples * max_new_tokens).
"""

from __future__ import annotations

import argparse
import math
import random
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple, cast

import torch
import torch.nn.functional as F
import tiktoken

import importlib.util

RE_ANSWER = re.compile(r"Answer:\s*([^\n\r]+)")
RE_A_COLON_FALLBACK = re.compile(r"\bA:\s*([-+]?\d+)\b")
RE_LAST_INT = re.compile(r"([-+]?\d+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--init_from", type=str, required=True, help="Base checkpoint (.pt state_dict)")
    p.add_argument("--train_data", type=str, required=True)
    p.add_argument("--val_data", type=str, default="")
    p.add_argument("--out_dir", type=str, required=True)

    # model arch (must match init_from)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--embed_size", type=int, default=384)
    p.add_argument("--transformer_heads", type=int, default=4)
    p.add_argument("--transformer_blocks", type=int, default=3)
    p.add_argument("--ff_mult", type=int, default=2)

    # RL-ish knobs
    p.add_argument("--num_steps", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_samples", type=int, default=4, help="Candidates per prompt")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--temperature", type=float, default=1.0)

    # optimization
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # reward shaping
    p.add_argument("--reward_correct", type=float, default=1.0)
    p.add_argument("--reward_incorrect", type=float, default=0.0)

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=20)

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
    m = RE_ANSWER.search(text)
    if m:
        return m.group(1).strip().split()[0]
    m2 = RE_A_COLON_FALLBACK.search(text)
    if m2:
        return m2.group(1).strip()
    ms = RE_LAST_INT.findall(text)
    if ms:
        return ms[-1].strip()
    return None


def split_qa(line: str) -> Tuple[str, str]:
    # same convention as scripts/eval_reasoning.py
    if " A: " not in line:
        return line.strip(), ""
    q, rest = line.split(" A: ", 1)
    gold = _extract_int_like(rest)
    return (q.strip() + " A:"), (gold or "")


def _logprob_of_continuation(
    model: torch.nn.Module,
    prompt_tokens: List[int],
    full_tokens: List[int],
    device: torch.device,
) -> torch.Tensor:
    """Compute log p(y | x) for continuation tokens y in full_tokens.

    full_tokens = prompt_tokens + continuation_tokens
    Returns a scalar tensor of summed log-probs over continuation tokens.
    """
    # time-first (T, B)
    tok = torch.tensor(full_tokens, dtype=torch.long, device=device).unsqueeze(1)
    with torch.no_grad():
        logits = model(tok)  # (T, 1, V)

    # predict token at position t from logits[t-1]
    cont_start = len(prompt_tokens)
    lp = torch.tensor(0.0, device=device)
    for i in range(cont_start, len(full_tokens)):
        prev_logits = logits[i - 1, 0, :]
        logp = F.log_softmax(prev_logits, dim=-1)[full_tokens[i]]
        lp = lp + logp
    return lp


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

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

    state = torch.load(args.init_from, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.train()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_lines = [ln.strip() for ln in Path(args.train_data).read_text(encoding="utf-8").splitlines() if ln.strip()]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def sample_batch() -> List[str]:
        return random.sample(train_lines, k=min(args.batch_size, len(train_lines)))

    running_reward = 0.0
    running_acc = 0.0

    for step in range(1, args.num_steps + 1):
        batch_lines = sample_batch()

        # build candidates
        losses: List[torch.Tensor] = []
        batch_rewards: List[float] = []
        batch_acc: List[float] = []

        optimizer.zero_grad(set_to_none=True)

        for ln in batch_lines:
            prompt, gold = split_qa(ln)
            if gold == "":
                continue

            prompt_tokens = enc.encode(prompt)
            # keep room for generation
            max_prompt = max(1, args.block_size - args.max_new_tokens)
            if len(prompt_tokens) > max_prompt:
                prompt_tokens = prompt_tokens[-max_prompt:]
                prompt = enc.decode(prompt_tokens)

            candidates: List[Tuple[List[int], str]] = []
            for _ in range(args.num_samples):
                # nucleus sampling generation from inference.py helper
                text, _ = inf.generate_text(
                    model,
                    enc,
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    device=str(device),
                    top_p=args.top_p,
                )
                full_tokens = enc.encode(text)
                # truncate to block_size
                full_tokens = full_tokens[: args.block_size]
                candidates.append((full_tokens, text))

            # reward each candidate and pick best
            scored: List[Tuple[float, List[int], str]] = []
            for full_tokens, text in candidates:
                pred = _extract_int_like(text)
                r = args.reward_correct if (pred is not None and pred == gold) else args.reward_incorrect
                scored.append((r, full_tokens, text))

            scored.sort(key=lambda x: x[0], reverse=True)
            best_r, best_tokens, best_text = scored[0]

            # weighted SFT: -w * log p(y|x)
            # compute logprob of continuation and backprop through model (needs grads)
            tok = torch.tensor(best_tokens, dtype=torch.long, device=device).unsqueeze(1)
            logits = model(tok)

            cont_start = len(prompt_tokens)
            # loss over continuation token predictions
            # logits at i-1 predicts token i
            lp = torch.tensor(0.0, device=device)
            for i in range(max(1, cont_start), len(best_tokens)):
                prev_logits = logits[i - 1, 0, :]
                lp = lp + F.log_softmax(prev_logits, dim=-1)[best_tokens[i]]

            # normalize by length to stabilize
            denom = max(1, len(best_tokens) - cont_start)
            lp = lp / denom

            w = float(best_r)
            loss = -w * lp
            losses.append(loss)

            batch_rewards.append(float(best_r))
            batch_acc.append(1.0 if best_r > 0 else 0.0)

        if not losses:
            continue

        total_loss = torch.stack(losses).mean()
        total_loss.backward()
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        running_reward = 0.95 * running_reward + 0.05 * (sum(batch_rewards) / max(1, len(batch_rewards)))
        running_acc = 0.95 * running_acc + 0.05 * (sum(batch_acc) / max(1, len(batch_acc)))

        if step % args.log_every == 0 or step == 1:
            print(
                f"step={step:04d} loss={total_loss.item():.4f} "
                f"avg_reward={running_reward:.3f} pass@{args.num_samples}~{running_acc:.3f}"
            )

    # save
    ckpt_path = out_dir / "transformer_rl_reasoning.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"✅ Saved RL-post-trained checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
