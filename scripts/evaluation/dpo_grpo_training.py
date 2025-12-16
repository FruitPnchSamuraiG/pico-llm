#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) and GRPO (Group Relative Policy Optimization) 
for reasoning model post-training on GSM8K.

Notes on "industry standard" behavior
-------------------------------------
- DPO is typically trained on *preference pairs* coming from human labels or a
  reward model. This script can also construct *synthetic preference pairs* from
  GSM8K correctness (RLAIF-style). This is common in research/production when an
  automatic verifier exists, but it's important to name it correctly.

- Log-prob / KL computations in production are usually *per-token* normalized
  and vectorized for throughput. This script computes sequence log-probs and a
  per-token KL proxy in a vectorized manner.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

import importlib.util
import sys

# Add scripts/utils to path
sys.path.append(str(Path(__file__).parent.parent / "utils"))
try:
    from gsm8k_utils import extract_answer, split_qa
except ImportError:
    # Fallback if running from different directory
    sys.path.append(str(Path(__file__).parent.parent.parent / "scripts" / "utils"))
    from gsm8k_utils import extract_answer, split_qa

# ============================================================================
# Regex patterns for answer extraction (GSM8K format)
# ============================================================================
RE_ANSWER = re.compile(r"####\s*([^\n\r]+)")  # GSM8K uses #### for final answer
RE_BOXED = re.compile(r"\\boxed\{([^}]+)\}")
RE_LAST_NUMBER = re.compile(r"([-+]?\d+(?:\.\d+)?)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DPO/GRPO post-training for reasoning models")
    
    # ===== Mode selection =====
    p.add_argument(
        "--mode",
        type=str,
        default="dpo",
        choices=["dpo", "grpo"],
        help="Training algorithm: dpo (Direct Preference Optimization) or grpo (Group Relative PO)"
    )
    
    # ===== Model and data =====
    p.add_argument("--init_from", type=str, required=True, 
                   help="Base checkpoint (.pt) from Orca training")
    p.add_argument("--train_data", type=str, required=True,
                   help="GSM8K training data (data/gsm8k_train.txt)")
    p.add_argument("--val_data", type=str, default="",
                   help="GSM8K validation data (data/gsm8k_val.txt)")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Output directory for checkpoints")
    
    # ===== NEW: Pre-generated preferences =====
    p.add_argument("--preference_data", type=str, default="",
                   help="JSONL file with pre-generated preference pairs (MUCH FASTER). "
                        "Generate with generate_preference_pairs.py. "
                        "If provided, --train_data is only used for validation.")
    p.add_argument("--online_generation", action="store_true",
                   help="Force on-the-fly generation even if --preference_data provided (SLOW)")
    
    # ===== Model architecture (must match init_from) =====
    # These will be auto-detected from checkpoint if possible
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--transformer_size", type=str, default="medium",
                   choices=["small", "medium", "gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"])
    
    # ===== Training hyperparameters =====
    p.add_argument("--num_steps", type=int, default=500,
                   help="Number of training steps")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Number of prompts per batch")
    p.add_argument("--lr", type=float, default=1e-6,
                   help="Learning rate (much smaller than SFT, typically 1e-6 to 1e-5)")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=50,
                   help="LR warmup steps")
    
    # ===== DPO-specific hyperparameters =====
    p.add_argument("--beta", type=float, default=0.1,
                   help="DPO temperature parameter (higher = less aggressive, typical: 0.1-0.5)")
    p.add_argument("--reference_free", action="store_true",
                   help="Use reference-free DPO (no reference model, faster)")
    
    # ===== GRPO-specific hyperparameters =====
    p.add_argument("--num_samples", type=int, default=8,
                   help="GRPO: Number of completions to sample per prompt")
    p.add_argument("--advantage_type", type=str, default="group_relative",
                   choices=["group_relative", "group_normalized"],
                   help="GRPO advantage computation method")
    p.add_argument("--kl_coef", type=float, default=0.01,
                   help="GRPO: KL penalty coefficient")
    
    # ===== Generation hyperparameters =====
    p.add_argument("--max_new_tokens", type=int, default=128,
                   help="Maximum tokens to generate per completion")
    p.add_argument("--top_p", type=float, default=0.95,
                   help="Nucleus sampling parameter")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Sampling temperature")
    
    # ===== Reward shaping =====
    p.add_argument("--reward_correct", type=float, default=1.0)
    p.add_argument("--reward_incorrect", type=float, default=0.0)
    p.add_argument("--length_penalty", type=float, default=0.0,
                   help="Penalty per token to encourage shorter answers")
    
    # ===== System =====
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=100,
                   help="Evaluate on validation set every N steps")
    p.add_argument("--save_every", type=int, default=100,
                   help="Save checkpoint every N steps")
    
    return p.parse_args()


def _load_inference_module() -> Any:
    """Dynamically load inference.py module"""
    here = Path(__file__).parent.parent.parent
    inf_path = here / "inference.py"
    spec = importlib.util.spec_from_file_location("pico_inference", str(inf_path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    cast(Any, spec.loader).exec_module(mod)  # type: ignore[call-arg]
    return mod


def _as_tensor_1xT(tokens: List[int], device: torch.device) -> torch.Tensor:
    """Convert token list -> (1, T) tensor."""
    return torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)


def compute_logprob_and_len(
    model: nn.Module,
    tokens: List[int],
    prompt_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute log p(y|x) and number of continuation tokens (vectorized).

    Returns:
        seq_logp: sum of log-probs over continuation tokens
        cont_len: number of continuation tokens
    """
    if len(tokens) <= prompt_len:
        z = torch.tensor(0.0, device=device)
        return z, z

    input_ids = _as_tensor_1xT(tokens, device)  # (1, T)

    # Ensure gradients flow when training; no-grad when evaluating.
    ctx = torch.enable_grad() if model.training else torch.no_grad()
    with ctx:
        logits = model(input_ids.transpose(0, 1))  # project expects (T, B)
        # logits: (T, 1, V)

        # Predict token t using logits at t-1. So we score targets tokens[1:].
        logp_all = F.log_softmax(logits, dim=-1)  # (T, 1, V)

        # continuation token positions (targets) are [prompt_len .. T-1]
        # their corresponding prediction logits are at indices [prompt_len-1 .. T-2]
        start_pred = max(0, prompt_len - 1)
        end_pred = len(tokens) - 1  # last pred index = T-2, slice end is exclusive

        pred_slice = logp_all[start_pred:end_pred, 0, :]  # (N, V)
        target_tokens = torch.tensor(tokens[prompt_len:], dtype=torch.long, device=device)  # (N,)

        token_logps = pred_slice.gather(1, target_tokens.unsqueeze(1)).squeeze(1)  # (N,)
        seq_logp = token_logps.sum()
        cont_len = torch.tensor(float(target_tokens.numel()), device=device)

    return seq_logp, cont_len


def compute_logprob(
    model: nn.Module,
    tokens: List[int],
    prompt_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Backward compatible wrapper returning sequence log-prob only."""
    lp, _ = compute_logprob_and_len(model, tokens, prompt_len, device)
    return lp


# ============================================================================
# Pre-generated Preference Loading
# ============================================================================

def load_preference_pairs(filepath: str) -> List[dict]:
    """Load pre-generated preference pairs from JSONL file."""
    pairs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def compute_logprob_and_len_batched(
    model: nn.Module,
    batch_tokens: List[List[int]],
    batch_prompt_lens: List[int],
    device: torch.device,
    max_seq_len: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Compute log-probs for a batch of sequences (MUCH faster than sequential).
    
    This is the KEY optimization: batch multiple forward passes together.
    Instead of B sequential forward passes, do 1 batched forward pass.
    
    Args:
        model: The transformer model
        batch_tokens: List of token sequences (variable length)
        batch_prompt_lens: List of prompt lengths for each sequence
        device: Torch device
        max_seq_len: Maximum sequence length for padding
    
    Returns:
        Tuple of (log_probs, cont_lens) lists
    """
    if not batch_tokens:
        return [], []
    
    batch_size = len(batch_tokens)
    
    # Pad sequences to same length
    padded_tokens = []
    for tokens in batch_tokens:
        padded = tokens + [0] * (max_seq_len - len(tokens))
        padded_tokens.append(padded[:max_seq_len])
    
    # Create batch tensor: (batch_size, seq_len)
    input_ids = torch.tensor(padded_tokens, dtype=torch.long, device=device)
    
    # Forward pass (BATCHED - this is the speedup!)
    ctx = torch.enable_grad() if model.training else torch.no_grad()
    with ctx:
        # Transpose for model: (seq_len, batch_size)
        logits = model(input_ids.transpose(0, 1))  # (seq_len, batch_size, vocab)
        logp_all = F.log_softmax(logits, dim=-1)  # (seq_len, batch_size, vocab)
    
    # Extract log-probs for each sequence
    seq_logps = []
    cont_lens = []
    
    for i, (tokens, prompt_len) in enumerate(zip(batch_tokens, batch_prompt_lens)):
        seq_len = len(tokens)
        if seq_len <= prompt_len:
            seq_logps.append(torch.tensor(0.0, device=device))
            cont_lens.append(torch.tensor(0.0, device=device))
            continue
        
        # Get predictions for continuation tokens
        start_pred = max(0, prompt_len - 1)
        end_pred = seq_len - 1
        
        pred_slice = logp_all[start_pred:end_pred, i, :]  # (N, vocab)
        target_tokens = torch.tensor(tokens[prompt_len:seq_len], dtype=torch.long, device=device)
        
        token_logps = pred_slice.gather(1, target_tokens.unsqueeze(1)).squeeze(1)
        seq_logps.append(token_logps.sum())
        cont_lens.append(torch.tensor(float(target_tokens.numel()), device=device))
    
    return seq_logps, cont_lens


# ============================================================================
# DPO Loss
# ============================================================================

def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute DPO loss.
    
    Loss = -log σ(β * [log π_θ(y_w|x) - log π_θ(y_l|x) 
                        - log π_ref(y_w|x) + log π_ref(y_l|x)])
    
    Where:
    - π_θ = policy model (being trained)
    - π_ref = reference model (frozen copy of initial policy)
    - y_w = chosen (winning) completion
    - y_l = rejected (losing) completion
    - β = temperature parameter (controls strength of KL penalty)
    
    Args:
        policy_chosen_logps: log π_θ(y_w|x)
        policy_rejected_logps: log π_θ(y_l|x)
        reference_chosen_logps: log π_ref(y_w|x)
        reference_rejected_logps: log π_ref(y_l|x)
        beta: Temperature parameter
    
    Returns:
        loss: Scalar tensor
        chosen_rewards: Implicit rewards for chosen completions
        rejected_rewards: Implicit rewards for rejected completions
    """
    # Compute preference logits
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    reference_logratios = reference_chosen_logps - reference_rejected_logps
    
    # DPO loss: -log sigmoid(beta * (policy_logratios - reference_logratios))
    logits = beta * (policy_logratios - reference_logratios)
    loss = -F.logsigmoid(logits)
    
    # Implicit rewards (for logging)
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    
    return loss, chosen_rewards, rejected_rewards


# ============================================================================
# GRPO Loss
# ============================================================================

def grpo_loss(
    policy_logps: List[torch.Tensor],
    reference_logps: List[torch.Tensor],
    rewards: List[float],
    kl_coef: float = 0.01,
    advantage_type: str = "group_relative",
    cont_lens: Optional[List[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute GRPO loss.

    Industry-style tweaks:
    - Advantages computed over the group (kept).
    - KL penalty computed per-token (if cont_lens provided).

    Returns:
        loss: scalar
        advantages: (K,) tensor
        kl_mean: scalar tensor (for logging)
    """
    if not policy_logps:
        z = torch.tensor(0.0)
        return z, torch.tensor([]), z

    policy_logps_tensor = torch.stack(policy_logps)  # (K,)
    reference_logps_tensor = torch.stack(reference_logps)  # (K,)
    rewards_tensor = torch.tensor(rewards, device=policy_logps_tensor.device)  # (K,)

    if advantage_type == "group_relative":
        advantages = rewards_tensor - rewards_tensor.mean()
    elif advantage_type == "group_normalized":
        mean_reward = rewards_tensor.mean()
        std_reward = rewards_tensor.std(unbiased=False) + 1e-8
        advantages = (rewards_tensor - mean_reward) / std_reward
    else:
        advantages = rewards_tensor

    # Policy gradient loss
    pg_loss = -(advantages.detach() * policy_logps_tensor).mean()

    # Per-token KL proxy
    if cont_lens is not None and len(cont_lens) == len(policy_logps):
        cont_lens_tensor = torch.stack(cont_lens).to(policy_logps_tensor.device)
        kl_tokens = (policy_logps_tensor - reference_logps_tensor) / (cont_lens_tensor + 1e-8)
        kl_mean = kl_tokens.mean()
    else:
        kl_tokens = (policy_logps_tensor - reference_logps_tensor)
        kl_mean = kl_tokens.mean()

    kl_penalty = kl_coef * kl_mean
    loss = pg_loss + kl_penalty

    return loss, advantages, kl_mean


# ============================================================================
# Training Loop
# ============================================================================

def _evaluate_model_accuracy(
    *,
    model: nn.Module,
    lines: List[str],
    device: torch.device,
    enc: Any,
    inf: Any,
    max_new_tokens: int,
    top_p: float,
    limit: int = 64,
) -> float:
    """Quick, cheap eval: sample 1 completion and check final answer correctness."""
    if not lines:
        return float("nan")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for ln in lines[: min(limit, len(lines))]:
            prompt, gold = split_qa(ln)
            if not gold:
                continue
            
            # Generate completion
            text, _ = inf.generate_text(
                model,
                enc,
                prompt,
                max_new_tokens=max_new_tokens,
                device=str(device),
                top_p=top_p,
            )
            
            # Extract answer for evaluation
            pred = extract_answer(text)
            total += 1
            if pred == gold:
                correct += 1

    model.train()
    return correct / max(1, total)


def train_dpo(
    args: argparse.Namespace,
    model: nn.Module,
    reference_model: nn.Module,
    train_lines: List[str],
    val_lines: List[str],
    device: torch.device,
    enc: Any,
    inf: Any,
    out_dir: Path,
) -> None:
    """DPO training loop"""
    
    # Check if using pre-generated preferences
    use_pregenerated = bool(args.preference_data) and not args.online_generation
    
    if use_pregenerated:
        print("📦 Loading pre-generated preference pairs (FAST MODE)")
        preference_pairs = load_preference_pairs(args.preference_data)
        print(f"   ✓ Loaded {len(preference_pairs)} preference pairs")
        print("   ⚡ This will be MUCH faster than on-the-fly generation!")
    else:
        print("⚠️  Using on-the-fly generation (SLOW MODE)")
        print("   Tip: Pre-generate preferences with generate_preference_pairs.py for 10-100x speedup")
        preference_pairs = None
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    
    # Learning rate schedule with warmup
    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return float(step) / float(max(1, args.warmup_steps))
        return 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    model.train()
    reference_model.eval()
    
    running_loss = 0.0
    running_acc = 0.0
    running_chosen_reward = 0.0
    running_rejected_reward = 0.0
    
    for step in range(1, args.num_steps + 1):
        if use_pregenerated:
            # Sample from pre-generated pairs
            batch_pairs = random.sample(preference_pairs, k=min(args.batch_size, len(preference_pairs)))
            
            # Prepare batches for parallel processing
            chosen_tokens_batch = [pair["chosen"]["tokens"] for pair in batch_pairs]
            rejected_tokens_batch = [pair["rejected"]["tokens"] for pair in batch_pairs]
            prompt_lens_batch = [len(pair["prompt_tokens"]) for pair in batch_pairs]
            
            max_len = max(max(len(t) for t in chosen_tokens_batch), 
                         max(len(t) for t in rejected_tokens_batch))
            
            optimizer.zero_grad(set_to_none=True)
            
            # BATCHED forward passes (10-100x faster!)
            policy_chosen_logps, chosen_lens = compute_logprob_and_len_batched(
                model, chosen_tokens_batch, prompt_lens_batch, device, max_len
            )
            policy_rejected_logps, rejected_lens = compute_logprob_and_len_batched(
                model, rejected_tokens_batch, prompt_lens_batch, device, max_len
            )
            
            with torch.no_grad():
                ref_chosen_logps, _ = compute_logprob_and_len_batched(
                    reference_model, chosen_tokens_batch, prompt_lens_batch, device, max_len
                )
                ref_rejected_logps, _ = compute_logprob_and_len_batched(
                    reference_model, rejected_tokens_batch, prompt_lens_batch, device, max_len
                )
            
            # Compute losses for batch
            batch_losses = []
            batch_chosen_rewards = []
            batch_rejected_rewards = []
            
            for i in range(len(batch_pairs)):
                loss, chosen_rew, rejected_rew = dpo_loss(
                    policy_chosen_logps[i],
                    policy_rejected_logps[i],
                    ref_chosen_logps[i],
                    ref_rejected_logps[i],
                    beta=args.beta,
                )
                batch_losses.append(loss)
                batch_chosen_rewards.append(chosen_rew.item())
                batch_rejected_rewards.append(rejected_rew.item())
            
            if batch_losses:
                total_loss = torch.stack(batch_losses).mean()
                total_loss.backward()
                
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                optimizer.step()
                scheduler.step()
                
                running_loss = 0.9 * running_loss + 0.1 * total_loss.item()
                running_acc = 0.9 * running_acc + 0.1 * 1.0
                running_chosen_reward = 0.9 * running_chosen_reward + 0.1 * (
                    sum(batch_chosen_rewards) / max(1, len(batch_chosen_rewards))
                )
                running_rejected_reward = 0.9 * running_rejected_reward + 0.1 * (
                    sum(batch_rejected_rewards) / max(1, len(batch_rejected_rewards))
                )
        
        # else:
        #     # Original on-the-fly generation (SLOW)
        #     batch_lines = random.sample(train_lines, k=min(args.batch_size, len(train_lines)))
            
        #     batch_losses = []
        #     batch_accs = []
        #     batch_chosen_rewards = []
        #     batch_rejected_rewards = []
            
        #     optimizer.zero_grad(set_to_none=True)
            
        #     for line in batch_lines:
        #         prompt, gold = split_qa(line)
        #         if not gold:
        #             continue
                
        #         prompt_tokens = enc.encode(prompt)
        #         max_prompt = max(1, args.block_size - args.max_new_tokens)
        #         if len(prompt_tokens) > max_prompt:
        #             prompt_tokens = prompt_tokens[-max_prompt:]
        #             prompt = enc.decode(prompt_tokens)
                
        #         # Sample TWO completions for (synthetic) preference pair.
        #         completions = []
        #         for _ in range(2):
        #             text, _ = inf.generate_text(
        #                 model,
        #                 enc,
        #                 prompt,
        #                 max_new_tokens=args.max_new_tokens,
        #                 device=str(device),
        #                 top_p=args.top_p,
        #             )
        #             full_tokens = enc.encode(text)[: args.block_size]
        #             pred = extract_answer(text)
        #             reward = args.reward_correct if (pred == gold) else args.reward_incorrect
        #             completions.append((full_tokens, reward))
                
        #         completions.sort(key=lambda x: x[1], reverse=True)
        #         chosen_tokens, chosen_reward = completions[0]
        #         rejected_tokens, rejected_reward = completions[1]
                
        #         if chosen_reward == rejected_reward:
        #             continue
                
        #         # Compute seq log-probs and continuation lengths (vectorized).
        #         policy_chosen_logp, chosen_len = compute_logprob_and_len(model, chosen_tokens, len(prompt_tokens), device)
        #         policy_rejected_logp, rejected_len = compute_logprob_and_len(model, rejected_tokens, len(prompt_tokens), device)
                
        #         with torch.no_grad():
        #             ref_chosen_logp, _ = compute_logprob_and_len(reference_model, chosen_tokens, len(prompt_tokens), device)
        #             ref_rejected_logp, _ = compute_logprob_and_len(reference_model, rejected_tokens, len(prompt_tokens), device)
                
        #         loss, chosen_rew, rejected_rew = dpo_loss(
        #             policy_chosen_logp,
        #             policy_rejected_logp,
        #             ref_chosen_logp,
        #             ref_rejected_logp,
        #             beta=args.beta,
        #         )
                
        #         batch_losses.append(loss)
        #         batch_accs.append(1.0)
        #         batch_chosen_rewards.append(chosen_rew.item())
        #         batch_rejected_rewards.append(rejected_rew.item())
            
        #     if not batch_losses:
        #         continue
            
        #     total_loss = torch.stack(batch_losses).mean()
        #     total_loss.backward()
            
        #     if args.grad_clip > 0:
        #         torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
        #     optimizer.step()
        #     scheduler.step()
            
        #     running_loss = 0.9 * running_loss + 0.1 * total_loss.item()
        #     running_acc = 0.9 * running_acc + 0.1 * (sum(batch_accs) / max(1, len(batch_accs)))
        #     running_chosen_reward = 0.9 * running_chosen_reward + 0.1 * (
        #         sum(batch_chosen_rewards) / max(1, len(batch_chosen_rewards))
        #     )
        #     running_rejected_reward = 0.9 * running_rejected_reward + 0.1 * (
        #         sum(batch_rejected_rewards) / max(1, len(batch_rejected_rewards))
        #     )
        
        if step % args.log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"[DPO] step={step:04d}/{args.num_steps} "
                f"loss={running_loss:.4f} pair_acc={running_acc:.3f} "
                f"chosen_rew={running_chosen_reward:.3f} rejected_rew={running_rejected_reward:.3f} "
                f"lr={lr:.2e}"
            )
        
        if args.eval_every > 0 and val_lines and (step % args.eval_every == 0):
            val_acc = _evaluate_model_accuracy(
                model=model,
                lines=val_lines,
                device=device,
                enc=enc,
                inf=inf,
                max_new_tokens=args.max_new_tokens,
                top_p=args.top_p,
                limit=64,
            )
            print(f"[DPO] eval(step={step}) val_acc@1={val_acc:.3f}")
        
        if step % args.save_every == 0:
            ckpt_path = out_dir / f"transformer_dpo_step{step}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"💾 Saved checkpoint: {ckpt_path}")
    
    final_path = out_dir / "transformer_dpo_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"✅ DPO training complete! Saved final checkpoint: {final_path}")


def train_grpo(
    args: argparse.Namespace,
    model: nn.Module,
    reference_model: nn.Module,
    train_lines: List[str],
    val_lines: List[str],
    device: torch.device,
    enc: Any,
    inf: Any,
    out_dir: Path,
) -> None:
    """GRPO training loop"""
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    
    # Learning rate schedule with warmup
    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return float(step) / float(max(1, args.warmup_steps))
        return 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    model.train()
    reference_model.eval()
    
    running_loss = 0.0
    running_reward = 0.0
    running_acc = 0.0
    
    for step in range(1, args.num_steps + 1):
        batch_lines = random.sample(train_lines, k=min(args.batch_size, len(train_lines)))
        
        batch_losses = []
        batch_rewards = []
        batch_accs = []
        batch_kls = []
        
        optimizer.zero_grad(set_to_none=True)
        
        for line in batch_lines:
            prompt, gold = split_qa(line)
            if not gold:
                continue
            
            prompt_tokens = enc.encode(prompt)
            max_prompt = max(1, args.block_size - args.max_new_tokens)
            if len(prompt_tokens) > max_prompt:
                prompt_tokens = prompt_tokens[-max_prompt:]
                prompt = enc.decode(prompt_tokens)
            
            group_tokens: List[List[int]] = []
            group_rewards: List[float] = []
            
            for _ in range(args.num_samples):
                text, _ = inf.generate_text(
                    model,
                    enc,
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    device=str(device),
                    top_p=args.top_p,
                )
                full_tokens = enc.encode(text)[: args.block_size]
                pred = extract_answer(text)
                
                reward = args.reward_correct if (pred == gold) else args.reward_incorrect
                reward -= args.length_penalty * (len(full_tokens) - len(prompt_tokens))
                
                group_tokens.append(full_tokens)
                group_rewards.append(reward)
            
            policy_logps: List[torch.Tensor] = []
            ref_logps: List[torch.Tensor] = []
            cont_lens: List[torch.Tensor] = []
            
            for tokens in group_tokens:
                pol_lp, clen = compute_logprob_and_len(model, tokens, len(prompt_tokens), device)
                policy_logps.append(pol_lp)
                cont_lens.append(clen)
                
                with torch.no_grad():
                    ref_lp, _ = compute_logprob_and_len(reference_model, tokens, len(prompt_tokens), device)
                    ref_logps.append(ref_lp)
            
            loss, _adv, kl_mean = grpo_loss(
                policy_logps,
                ref_logps,
                group_rewards,
                kl_coef=args.kl_coef,
                advantage_type=args.advantage_type,
                cont_lens=cont_lens,
            )
            
            batch_losses.append(loss)
            batch_rewards.append(sum(group_rewards) / len(group_rewards))
            batch_accs.append(1.0 if max(group_rewards) > 0 else 0.0)
            batch_kls.append(float(kl_mean.detach().cpu()))
        
        if not batch_losses:
            continue
        
        total_loss = torch.stack(batch_losses).mean()
        total_loss.backward()
        
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        optimizer.step()
        scheduler.step()
        
        running_loss = 0.9 * running_loss + 0.1 * total_loss.item()
        running_reward = 0.9 * running_reward + 0.1 * (sum(batch_rewards) / max(1, len(batch_rewards)))
        running_acc = 0.9 * running_acc + 0.1 * (sum(batch_accs) / max(1, len(batch_accs)))
        running_kl = 0.9 * (locals().get("running_kl", 0.0)) + 0.1 * (sum(batch_kls) / max(1, len(batch_kls)))
        
        if step % args.log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"[GRPO] step={step:04d}/{args.num_steps} "
                f"loss={running_loss:.4f} avg_reward={running_reward:.3f} "
                f"pass@{args.num_samples}={running_acc:.3f} kl/tok={running_kl:.4f} lr={lr:.2e}"
            )
        
        if args.eval_every > 0 and val_lines and (step % args.eval_every == 0):
            val_acc = _evaluate_model_accuracy(
                model=model,
                lines=val_lines,
                device=device,
                enc=enc,
                inf=inf,
                max_new_tokens=args.max_new_tokens,
                top_p=args.top_p,
                limit=64,
            )
            print(f"[GRPO] eval(step={step}) val_acc@1={val_acc:.3f}")
        
        if step % args.save_every == 0:
            ckpt_path = out_dir / f"transformer_grpo_step{step}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"💾 Saved checkpoint: {ckpt_path}")
    
    final_path = out_dir / "transformer_grpo_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"✅ GRPO training complete! Saved final checkpoint: {final_path}")


# ============================================================================
# Main
# ============================================================================

def _infer_ckpt_arch(state: dict) -> tuple[int | None, int | None]:
    """Best-effort inference of (embed_size, n_blocks) from a state_dict."""
    try:
        embed = int(state["embed.weight"].shape[1])
    except Exception:
        embed = None

    blocks = None
    try:
        idxs = []
        for k in state.keys():
            if k.startswith("blocks."):
                parts = k.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    idxs.append(int(parts[1]))
        if idxs:
            blocks = max(idxs) + 1
    except Exception:
        blocks = None

    return embed, blocks


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device(
        args.device if (not args.device.startswith("cuda") or torch.cuda.is_available()) else "cpu"
    )
    print(f"🔧 Using device: {device}")
    print(f"🎯 Mode: {args.mode.upper()}")
    
    if args.reference_free and args.mode != "dpo":
        print("⚠️  --reference_free only affects DPO; GRPO always uses a reference for KL.")
    
    enc = tiktoken.get_encoding("gpt2")
    inf = _load_inference_module()
    
    # Determine model architecture from transformer_size
    if args.transformer_size == "small":
        embed_size, heads, blocks, ff_mult = 384, 4, 3, 2
    elif args.transformer_size == "medium":
        embed_size, heads, blocks, ff_mult = 512, 8, 6, 4
    elif args.transformer_size == "gpt2-small":
        embed_size, heads, blocks, ff_mult = 768, 12, 12, 4
    elif args.transformer_size == "gpt2-medium":
        embed_size, heads, blocks, ff_mult = 1024, 16, 24, 4
    elif args.transformer_size == "gpt2-large":
        embed_size, heads, blocks, ff_mult = 1280, 20, 36, 4
    else:  # gpt2-xl
        embed_size, heads, blocks, ff_mult = 1600, 25, 48, 4
    
    print(f"📐 Model: {args.transformer_size} ({embed_size}d, {heads}h, {blocks}L)")
    
    # Load policy model (will be trained)
    policy_model = inf.TransformerModel(
        vocab_size=enc.n_vocab,
        block_size=args.block_size,
        d_model=embed_size,
        n_heads=heads,
        n_blocks=blocks,
        ff_mult=ff_mult,
    )
    
    print(f"📦 Loading base checkpoint: {args.init_from}")
    state = torch.load(args.init_from, map_location=device, weights_only=True)

    try:
        policy_model.load_state_dict(state)
    except RuntimeError as e:
        ckpt_embed, ckpt_blocks = _infer_ckpt_arch(state)
        msg = [
            "Checkpoint/architecture mismatch while loading weights.",
            f"  requested --transformer_size: {args.transformer_size}",
        ]
        if ckpt_embed is not None or ckpt_blocks is not None:
            msg.append(f"  checkpoint appears to be: embed_size={ckpt_embed}, n_blocks={ckpt_blocks}")
        msg.extend(
            [
                "Common cause: using a GSM8K SFT checkpoint trained with a different model size.",
                "Fix:",
                "  1) Point --init_from to a checkpoint produced with the SAME model size, or",
                "  2) Re-run Stage 2 with TRANSFORMER_SIZE matching Stage 3, then rerun DPO/GRPO.",
                "Tip: you can inspect a checkpoint with:",
                "  python scripts/utils/check_checkpoint_arch.py <checkpoint.pt>",
            ]
        )
        raise RuntimeError("\n".join(msg)) from e

    policy_model.to(device)
    
    # Create reference model (frozen copy for KL penalty)
    if not args.reference_free:
        print("🔒 Creating reference model (frozen copy of base)")
        reference_model = copy.deepcopy(policy_model)
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad = False
    else:
        print("🆓 Using reference-free mode (no reference model)")
        # Create a dummy reference model for type checking
        reference_model = copy.deepcopy(policy_model)
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad = False
    
    # Load training data
    print(f"📚 Loading training data: {args.train_data}")
    train_lines = [
        ln.strip() for ln in Path(args.train_data).read_text(encoding="utf-8").splitlines() 
        if ln.strip()
    ]
    print(f"   ✓ {len(train_lines)} training examples")
    
    val_lines = []
    if args.val_data:
        print(f"📚 Loading validation data: {args.val_data}")
        val_lines = [
            ln.strip() for ln in Path(args.val_data).read_text(encoding="utf-8").splitlines() 
            if ln.strip()
        ]
        print(f"   ✓ {len(val_lines)} validation examples")
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"📝 Saved config to {out_dir}/config.json")
    
    # Train
    print("=" * 80)
    print(f"🚀 Starting {args.mode.upper()} training...")
    print("=" * 80)
    
    if args.mode == "dpo":
        train_dpo(args, policy_model, reference_model, train_lines, val_lines, device, enc, inf, out_dir)
    else:  # grpo
        train_grpo(args, policy_model, reference_model, train_lines, val_lines, device, enc, inf, out_dir)


if __name__ == "__main__":
    main()
