#!/usr/bin/env python3
"""
Advanced Reasoning Model Training with <thinking> blocks.

Implements modern reasoning model approaches:
1. Chain-of-Thought with explicit thinking tokens
2. Process Reward Models (PRM) for intermediate steps
3. Outcome Reward Models (ORM) for final answers
4. Inference-time search (Best-of-N, self-consistency)

Based on DeepSeek R1 and o1-style reasoning models.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple, Dict, cast
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

import importlib.util
import sys

# Add scripts/utils to path
sys.path.append(str(Path(__file__).parent.parent / "utils"))
try:
    from gsm8k_utils import extract_answer, extract_thinking_and_answer, split_qa, SPECIAL_TOKENS, THINKING_START, THINKING_END, ANSWER_START
except ImportError:
    # Fallback if running from different directory
    sys.path.append(str(Path(__file__).parent.parent.parent / "scripts" / "utils"))
    from gsm8k_utils import extract_answer, extract_thinking_and_answer, split_qa, SPECIAL_TOKENS, THINKING_START, THINKING_END, ANSWER_START

# ============================================================================
# Process Reward Model (PRM)
# ============================================================================

class ProcessRewardModel:
    """
    Process Reward Model for scoring intermediate reasoning steps.
    
    In a full implementation, this would be a separate neural network trained
    to predict P(correct final answer | reasoning prefix). For now, we use
    heuristics based on mathematical structure.
    """
    
    def __init__(self, gold_answer: str):
        self.gold_answer = gold_answer
    
    def score_step(self, step: str, step_idx: int, total_steps: int) -> float:
        """
        Score a single reasoning step.
        
        Returns a reward in [0, 1] indicating step quality.
        """
        score = 0.5  # Base score
        
        # Positive signals
        if any(op in step for op in ['+', '-', '*', '/', '=', '<', '>']):
            score += 0.1  # Contains mathematical operations
        
        if any(keyword in step.lower() for keyword in ['so', 'therefore', 'thus', 'hence']):
            score += 0.05  # Contains logical connectives
        
        if re.search(r'\d+', step):
            score += 0.05  # Contains numbers
        
        if step.strip().endswith('.') or step.strip().endswith(','):
            score += 0.05  # Proper sentence structure
        
        # Negative signals
        if len(step.strip()) < 10:
            score -= 0.1  # Too short
        
        if step.count('...') > 2:
            score -= 0.1  # Vague/incomplete
        
        # Position-based rewards
        if step_idx == 0:
            # First step should set up the problem
            if any(word in step.lower() for word in ['have', 'given', 'know', 'start']):
                score += 0.1
        
        if step_idx == total_steps - 1:
            # Last step should contain the answer
            if self.gold_answer and self.gold_answer in step:
                score += 0.3
        
        return max(0.0, min(1.0, score))
    
    def score_thinking(self, thinking: str) -> Tuple[float, List[float]]:
        """
        Score entire thinking block.
        
        Returns:
            (average_score, step_scores)
        """
        if not thinking:
            return 0.0, []
        
        # Split into steps (sentences or lines)
        steps = [s.strip() for s in thinking.split('\n') if s.strip()]
        if not steps:
            steps = [s.strip() for s in thinking.split('.') if s.strip()]
        
        step_scores = [
            self.score_step(step, i, len(steps))
            for i, step in enumerate(steps)
        ]
        
        avg_score = sum(step_scores) / len(step_scores) if step_scores else 0.0
        return avg_score, step_scores


# ============================================================================
# Reasoning Data Augmentation
# ============================================================================

def augment_with_thinking(
    prompt: str,
    solution: str,
    gold_answer: str,
    thinking_style: str = "verbose"
) -> str:
    """
    Augment a GSM8K example with explicit <thinking> blocks.
    
    Args:
        prompt: Question prompt (e.g., "Q: ... A:")
        solution: Original solution text
        gold_answer: Gold answer
        thinking_style: "verbose" | "concise" | "structured"
    
    Returns:
        Augmented text with <thinking> and <answer> blocks
    """
    if thinking_style == "structured":
        # Structured thinking format
        thinking = f"{THINKING_START}\n"
        thinking += "Let me break this down step by step:\n"
        thinking += solution.strip() + "\n"
        thinking += f"{THINKING_END}\n"
        answer = f"{ANSWER_START} {gold_answer} #### {gold_answer}"
    
    elif thinking_style == "concise":
        # Concise thinking
        thinking = f"{THINKING_START} {solution.strip()} {THINKING_END}\n"
        answer = f"{ANSWER_START} {gold_answer} #### {gold_answer}"
    
    else:  # verbose (default)
        # Verbose thinking with explicit reasoning
        thinking = f"{THINKING_START}\n"
        thinking += "I need to solve this problem carefully.\n"
        thinking += solution.strip() + "\n"
        thinking += f"So the answer is {gold_answer}.\n"
        thinking += f"{THINKING_END}\n"
        answer = f"{ANSWER_START} The answer is {gold_answer}. #### {gold_answer}"
    
    return prompt + " " + thinking + answer


def prepare_reasoning_data(
    data_file: str,
    output_file: str,
    thinking_style: str = "structured",
    max_examples: int = 0
) -> None:
    """
    Convert standard GSM8K data to reasoning format with <thinking> blocks.
    """
    lines = Path(data_file).read_text(encoding="utf-8").splitlines()
    augmented_lines = []
    
    for line in lines[:max_examples] if max_examples > 0 else lines:
        if not line.strip() or "####" not in line:
            continue
        
        # Parse original format: "Q: ... A: solution #### answer"
        parts = line.split(" A: ", 1)
        if len(parts) != 2:
            continue
        
        question = parts[0].strip()
        solution_and_answer = parts[1]
        
        if "####" in solution_and_answer:
            solution, answer = solution_and_answer.split("####", 1)
            gold = extract_answer("#### " + answer.strip())
            
            if gold:
                prompt = question + " A:"
                augmented = augment_with_thinking(prompt, solution, gold, thinking_style)
                augmented_lines.append(augmented)
    
    Path(output_file).write_text("\n".join(augmented_lines), encoding="utf-8")
    print(f"✓ Created {len(augmented_lines)} reasoning examples in {output_file}")


# ============================================================================
# Inference-Time Search
# ============================================================================

def best_of_n_sampling(
    model: nn.Module,
    enc: Any,
    inf: Any,
    prompt: str,
    gold_answer: str,
    n: int = 8,
    max_new_tokens: int = 256,
    max_thinking_tokens: int = 800,
    max_answer_tokens: int = 200,
    temperature: float = 1.0,
    top_p: float = 0.95,
    device: str = "cuda:0",
    scoring_method: str = "orm",  # "orm" | "prm" | "logprob"
    use_thinking_mode: bool = True
) -> Tuple[str, float, List[Tuple[str, float]]]:
    """
    Best-of-N sampling: generate N completions and select the best.
    
    Args:
        max_new_tokens: Legacy parameter, used if use_thinking_mode=False
        max_thinking_tokens: Token budget for thinking phase (default: 800)
        max_answer_tokens: Token budget for answer phase (default: 200)
        use_thinking_mode: Use thinking-aware generation (recommended)
        scoring_method:
            - "orm": Use Outcome Reward Model (binary correct/incorrect)
            - "prm": Use Process Reward Model (score thinking quality)
            - "logprob": Use sequence log-probability
    
    Returns:
        (best_completion, best_score, all_results)
    """
    model.eval()
    results = []
    
    with torch.no_grad():
        for i in range(n):
            # Generate completion with thinking-aware mode
            if use_thinking_mode and hasattr(inf, 'generate_text_with_thinking'):
                text, phase_info = inf.generate_text_with_thinking(
                    model,
                    enc,
                    prompt,
                    max_thinking_tokens=max_thinking_tokens,
                    max_answer_tokens=max_answer_tokens,
                    device=device,
                    top_p=top_p,
                    temperature=temperature
                )
            else:
                # Fallback to standard generation
                text, _ = inf.generate_text(
                    model,
                    enc,
                    prompt,
                    max_new_tokens=max_new_tokens,
                    device=device,
                    top_p=top_p,
                    temperature=temperature
                )
            
            # Extract answer
            thinking, answer_text = extract_thinking_and_answer(text)
            pred_answer = extract_answer(answer_text or text)
            
            # Score based on method
            if scoring_method == "orm":
                # Outcome-based: binary reward
                score = 1.0 if (pred_answer == gold_answer) else 0.0
            
            elif scoring_method == "prm":
                # Process-based: score thinking quality + correctness
                if thinking:
                    prm = ProcessRewardModel(gold_answer)
                    thinking_score, _ = prm.score_thinking(thinking)
                    correctness = 1.0 if (pred_answer == gold_answer) else 0.0
                    score = 0.5 * thinking_score + 0.5 * correctness
                else:
                    score = 1.0 if (pred_answer == gold_answer) else 0.0
            
            else:  # logprob
                # Use model confidence (log-probability)
                tokens = enc.encode(text)
                prompt_tokens = enc.encode(prompt)
                # For simplicity, use length-normalized score
                score = 1.0 / (len(tokens) - len(prompt_tokens) + 1)
            
            results.append((text, score))
    
    # Select best
    results.sort(key=lambda x: x[1], reverse=True)
    best_completion, best_score = results[0]
    
    model.train()
    return best_completion, best_score, results


def self_consistency_decoding(
    model: nn.Module,
    enc: Any,
    inf: Any,
    prompt: str,
    n: int = 8,
    max_new_tokens: int = 256,
    max_thinking_tokens: int = 800,
    max_answer_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.95,
    device: str = "cuda:0",
    use_thinking_mode: bool = True
) -> Tuple[str, str, Dict[str, int]]:
    """
    Self-consistency: sample multiple CoTs and take majority vote on answer.
    
    Returns:
        (best_completion, majority_answer, answer_counts)
    """
    model.eval()
    completions = []
    answers = []
    
    with torch.no_grad():
        for i in range(n):
            # Generate with thinking-aware mode
            if use_thinking_mode and hasattr(inf, 'generate_text_with_thinking'):
                text, phase_info = inf.generate_text_with_thinking(
                    model,
                    enc,
                    prompt,
                    max_thinking_tokens=max_thinking_tokens,
                    max_answer_tokens=max_answer_tokens,
                    device=device,
                    top_p=top_p,
                    temperature=temperature
                )
            else:
                # Fallback to standard generation
                text, _ = inf.generate_text(
                    model,
                    enc,
                    prompt,
                    max_new_tokens=max_new_tokens,
                    device=device,
                    top_p=top_p,
                    temperature=temperature
                )
            completions.append(text)
            
            # Extract answer
            _, answer_text = extract_thinking_and_answer(text)
            pred = extract_answer(answer_text or text)
            if pred:
                answers.append(pred)
    
    # Majority vote
    if not answers:
        model.train()
        return completions[0] if completions else "", "", {}
    
    answer_counts = Counter(answers)
    majority_answer = answer_counts.most_common(1)[0][0]
    
    # Find best completion with majority answer
    best_completion = ""
    for comp in completions:
        if majority_answer in comp:
            best_completion = comp
            break
    
    model.train()
    return best_completion, majority_answer, dict(answer_counts)


# ============================================================================
# Training Loop with PRM/ORM
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Advanced Reasoning Model Training")
    
    # Model and data
    p.add_argument("--init_from", type=str, required=True)
    p.add_argument("--train_data", type=str, required=True)
    p.add_argument("--val_data", type=str, default="")
    p.add_argument("--out_dir", type=str, required=True)
    
    # Reasoning-specific
    p.add_argument("--thinking_style", type=str, default="structured",
                   choices=["verbose", "concise", "structured"])
    p.add_argument("--reward_mode", type=str, default="orm",
                   choices=["orm", "prm", "hybrid"],
                   help="Reward model type: ORM (outcome), PRM (process), or hybrid")
    p.add_argument("--prm_weight", type=float, default=0.3,
                   help="Weight for PRM rewards in hybrid mode")
    
    # Training
    p.add_argument("--num_steps", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--grad_clip", type=float, default=1.0)
    
    # Generation
    p.add_argument("--max_new_tokens", type=int, default=256,
                   help="Max tokens (includes thinking + answer)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    
    # Model architecture
    p.add_argument("--block_size", type=int, default=384)
    p.add_argument("--transformer_size", type=str, default="medium")
    
    # System
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=200)
    
    # Inference-time search
    p.add_argument("--use_best_of_n", action="store_true",
                   help="Use Best-of-N sampling during evaluation")
    p.add_argument("--n_samples", type=int, default=8,
                   help="Number of samples for Best-of-N or self-consistency")
    
    return p.parse_args()


def _load_inference_module() -> Any:
    """Dynamically load inference.py module"""
    here = Path(__file__).parent.parent.parent
    inf_path = here / "inference.py"
    spec = importlib.util.spec_from_file_location("pico_inference", str(inf_path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    cast(Any, spec.loader).exec_module(mod)
    return mod


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")
    print(f"🧠 Reward mode: {args.reward_mode.upper()}")
    print(f"💭 Thinking style: {args.thinking_style}")
    
    # Check if reasoning data exists, if not create it
    reasoning_train = args.train_data.replace(".txt", f"_reasoning_{args.thinking_style}.txt")
    if not Path(reasoning_train).exists():
        print(f"\n📝 Creating reasoning data with <thinking> blocks...")
        prepare_reasoning_data(args.train_data, reasoning_train, args.thinking_style)
    
    print(f"\n✅ Ready to train reasoning model!")
    print(f"   Training data: {reasoning_train}")
    print(f"   Output: {args.out_dir}")
    print(f"\n💡 Reasoning features:")
    print(f"   • Explicit <thinking> blocks before answers")
    print(f"   • {args.reward_mode.upper()} reward model")
    if args.use_best_of_n:
        print(f"   • Best-of-{args.n_samples} inference-time search")
    print()


if __name__ == "__main__":
    main()
