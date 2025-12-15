#!/usr/bin/env python3
"""
PPO-based Reasoning Model Training
==================================
Post-training script using Proximal Policy Optimization (PPO) to develop
reasoning capabilities in the transformer model.

Usage:
    # Step 1: Train base model with pico-llm.py first
    python pico-llm.py --enable_transformer --block_size 512 --embed_size 512
    
    # Step 2: Run PPO fine-tuning on the checkpoint
    python ppo_train.py --checkpoint transformer_checkpoint.pt --prompts reasoning_prompts.txt
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tiktoken
import time
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np

# Import model classes from pico-llm
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from pico_llm import TransformerModel, RMSNorm


@dataclass
class PPOConfig:
    """PPO hyperparameters for reasoning fine-tuning"""
    # Core PPO settings
    clip_range: float = 0.2           # Clipping parameter for PPO objective
    value_coef: float = 0.5           # Value loss coefficient
    entropy_coef: float = 0.01        # Entropy bonus coefficient
    kl_coef: float = 0.1              # KL penalty coefficient (vs reference model)
    
    # Training settings
    ppo_epochs: int = 4               # PPO epochs per batch
    batch_size: int = 8               # Number of prompts per rollout batch
    mini_batch_size: int = 2          # Mini-batch size for PPO updates
    max_new_tokens: int = 128         # Max tokens to generate per prompt
    learning_rate: float = 1e-5       # Learning rate for PPO
    max_grad_norm: float = 1.0        # Gradient clipping
    
    # Sampling settings
    temperature: float = 1.0          # Sampling temperature
    top_p: float = 0.95              # Nucleus sampling threshold
    
    # Training loop settings
    num_iterations: int = 100         # Total PPO iterations
    save_every: int = 10              # Save checkpoint every N iterations
    log_every: int = 1                # Log metrics every N iterations


@dataclass
class RolloutBatch:
    """Container for a batch of rollout trajectories"""
    prompts: List[str]
    prompt_tokens: List[List[int]]
    response_tokens: List[List[int]]
    all_tokens: List[List[int]]
    logprobs: List[torch.Tensor]
    values: List[torch.Tensor]
    rewards: List[torch.Tensor]
    advantages: List[torch.Tensor]
    returns: List[torch.Tensor]


class ReasoningRewardFunction:
    """
    Reward function for reasoning tasks.
    
    Implements multiple reward signals:
    1. Format rewards: Does the response follow chain-of-thought structure?
    2. Length rewards: Encourage detailed reasoning steps
    3. Correctness rewards: Verify final answers (when ground truth available)
    4. Coherence rewards: Penalize repetition and incoherence
    """
    
    def __init__(self, enc):
        self.enc = enc
        
    def compute_reward(self, prompt: str, response: str, metadata: Optional[Dict] = None) -> float:
        """
        Compute reward for a generated response.
        
        Args:
            prompt: Input prompt
            response: Generated response
            metadata: Optional dict with ground truth answer, expected format, etc.
            
        Returns:
            Scalar reward value
        """
        reward = 0.0
        
        # 1. Format reward: Check for reasoning structure
        if self._has_reasoning_structure(response):
            reward += 1.0
        
        # 2. Length reward: Encourage detailed but not excessive responses
        reward += self._length_reward(response)
        
        # 3. Correctness reward: Check answer if ground truth provided
        if metadata and 'answer' in metadata:
            if self._check_correctness(response, metadata['answer']):
                reward += 5.0  # Big bonus for correct answer
            else:
                reward -= 1.0  # Small penalty for wrong answer
        
        # 4. Coherence penalty: Penalize repetition
        reward += self._coherence_penalty(response)
        
        return reward
    
    def _has_reasoning_structure(self, response: str) -> bool:
        """Check if response contains reasoning markers"""
        markers = [
            "step", "first", "then", "therefore", "because",
            "let's", "we can", "so", "thus", "hence"
        ]
        response_lower = response.lower()
        return any(marker in response_lower for marker in markers)
    
    def _length_reward(self, response: str) -> float:
        """Reward appropriate length responses"""
        words = len(response.split())
        if words < 10:
            return -0.5  # Too short
        elif words < 50:
            return 0.5   # Good length
        elif words < 100:
            return 0.3   # Still reasonable
        else:
            return -0.2  # Too verbose
    
    def _check_correctness(self, response: str, answer: str) -> bool:
        """Check if response contains correct answer"""
        # Simple substring check (can be made more sophisticated)
        return answer.lower() in response.lower()
    
    def _coherence_penalty(self, response: str) -> float:
        """Penalize excessive repetition"""
        words = response.split()
        if len(words) < 5:
            return 0.0
        
        # Check for repeated phrases
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        unique_ratio = len(set(bigrams)) / len(bigrams) if bigrams else 1.0
        
        if unique_ratio < 0.5:
            return -1.0  # Heavy repetition
        elif unique_ratio < 0.7:
            return -0.3  # Some repetition
        return 0.0


class PPOTrainer:
    """
    PPO trainer for reasoning model development.
    """
    
    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        reward_fn: ReasoningRewardFunction,
        config: PPOConfig,
        device: str = "cuda"
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.reward_fn = reward_fn
        self.config = config
        self.device = device
        self.enc = tiktoken.get_encoding("gpt2")
        
        # Optimizer
        self.optimizer = optim.AdamW(
            policy_model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.01
        )
        
        # Metrics tracking
        self.metrics_history = []
        
    def generate_rollouts(self, prompts: List[str]) -> RolloutBatch:
        """
        Generate rollouts for a batch of prompts.
        
        Returns:
            RolloutBatch containing trajectories with logprobs and values
        """
        self.policy_model.eval()
        
        batch_prompts = []
        batch_prompt_tokens = []
        batch_response_tokens = []
        batch_all_tokens = []
        batch_logprobs = []
        batch_values = []
        
        with torch.no_grad():
            for prompt in prompts:
                prompt_tokens = self.enc.encode(prompt)
                all_tokens = prompt_tokens.copy()
                response_tokens = []
                step_logprobs = []
                step_values = []
                
                # Generate response token by token
                for _ in range(self.config.max_new_tokens):
                    # Prepare input
                    tokens_tensor = torch.tensor(
                        all_tokens, dtype=torch.long, device=self.device
                    ).unsqueeze(1)  # (seq_len, 1)
                    
                    # Get logits and values
                    logits, values = self.policy_model(tokens_tensor, return_value=True)
                    
                    # Get last step
                    last_logits = logits[-1, 0, :]  # (vocab_size,)
                    last_value = values[-1, 0, 0]    # scalar
                    
                    # Sample token
                    probs = F.softmax(last_logits / self.config.temperature, dim=-1)
                    
                    # Top-p sampling
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumsum_probs <= self.config.top_p
                    mask = torch.cat([mask.new_ones(1), mask[:-1]])  # Keep at least 1 token
                    
                    filtered_probs = sorted_probs * mask.float()
                    filtered_probs = filtered_probs / filtered_probs.sum()
                    
                    # Sample from filtered distribution
                    sample_idx = torch.multinomial(filtered_probs, 1).item()
                    token = sorted_indices[sample_idx].item()
                    
                    # Get log probability
                    logprob = torch.log(probs[token] + 1e-10)
                    
                    # Store
                    all_tokens.append(token)
                    response_tokens.append(token)
                    step_logprobs.append(logprob)
                    step_values.append(last_value)
                    
                    # Stop at EOS or newline (simple stopping criterion)
                    if token == self.enc.encode("\n")[0] and len(response_tokens) > 10:
                        break
                
                batch_prompts.append(prompt)
                batch_prompt_tokens.append(prompt_tokens)
                batch_response_tokens.append(response_tokens)
                batch_all_tokens.append(all_tokens)
                batch_logprobs.append(torch.stack(step_logprobs))
                batch_values.append(torch.stack(step_values))
        
        # Compute rewards
        batch_rewards = []
        for i, prompt in enumerate(prompts):
            response = self.enc.decode(batch_response_tokens[i])
            reward = self.reward_fn.compute_reward(prompt, response)
            # Distribute reward across time steps (terminal reward)
            reward_tensor = torch.zeros(len(batch_response_tokens[i]), device=self.device)
            reward_tensor[-1] = reward  # Only terminal reward
            batch_rewards.append(reward_tensor)
        
        # Compute advantages and returns (GAE)
        batch_advantages = []
        batch_returns = []
        gamma = 0.99
        gae_lambda = 0.95
        
        for values, rewards in zip(batch_values, batch_rewards):
            advantages = []
            returns = []
            gae = 0
            next_value = 0
            
            # Compute GAE from last to first
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + gamma * next_value - values[t]
                gae = delta + gamma * gae_lambda * gae
                advantages.insert(0, gae)
                returns.insert(0, gae + values[t])
                next_value = values[t]
            
            batch_advantages.append(torch.tensor(advantages, device=self.device))
            batch_returns.append(torch.tensor(returns, device=self.device))
        
        return RolloutBatch(
            prompts=batch_prompts,
            prompt_tokens=batch_prompt_tokens,
            response_tokens=batch_response_tokens,
            all_tokens=batch_all_tokens,
            logprobs=batch_logprobs,
            values=batch_values,
            rewards=batch_rewards,
            advantages=batch_advantages,
            returns=batch_returns
        )
    
    def compute_ppo_loss(self, rollout: RolloutBatch) -> Dict[str, float]:
        """
        Compute PPO loss for a batch of rollouts.
        
        Returns:
            Dictionary of loss components
        """
        self.policy_model.train()
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        clip_frac = 0.0
        num_samples = 0
        
        # Process each trajectory
        for idx in range(len(rollout.prompts)):
            # Get trajectory data
            all_tokens = rollout.all_tokens[idx]
            old_logprobs = rollout.logprobs[idx]
            old_values = rollout.values[idx]
            advantages = rollout.advantages[idx]
            returns = rollout.returns[idx]
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Forward pass through policy
            tokens_tensor = torch.tensor(
                all_tokens, dtype=torch.long, device=self.device
            ).unsqueeze(1)
            
            logits, values = self.policy_model(tokens_tensor, return_value=True)
            
            # Get response logits/values (skip prompt tokens)
            prompt_len = len(rollout.prompt_tokens[idx])
            response_logits = logits[prompt_len:, 0, :]  # (response_len, vocab_size)
            response_values = values[prompt_len:, 0, 0]   # (response_len,)
            
            # Get reference logits for KL
            with torch.no_grad():
                ref_logits = self.ref_model(tokens_tensor)
                ref_response_logits = ref_logits[prompt_len:, 0, :]
            
            # Compute new log probs
            response_tokens_tensor = torch.tensor(
                rollout.response_tokens[idx], dtype=torch.long, device=self.device
            )
            log_probs_all = F.log_softmax(response_logits, dim=-1)
            new_logprobs = log_probs_all.gather(1, response_tokens_tensor.unsqueeze(1)).squeeze(1)
            
            # Compute reference log probs for KL
            ref_log_probs_all = F.log_softmax(ref_response_logits, dim=-1)
            ref_logprobs = ref_log_probs_all.gather(1, response_tokens_tensor.unsqueeze(1)).squeeze(1)
            
            # Policy loss (PPO clipped objective)
            ratio = torch.exp(new_logprobs - old_logprobs.detach())
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(response_values, returns)
            
            # Entropy bonus
            probs = F.softmax(response_logits, dim=-1)
            entropy = -(probs * log_probs_all).sum(dim=-1).mean()
            
            # KL divergence vs reference model
            kl = (new_logprobs - ref_logprobs).mean()
            
            # Aggregate losses
            total_policy_loss += policy_loss
            total_value_loss += value_loss
            total_entropy += entropy
            total_kl += kl
            clip_frac += ((ratio - 1.0).abs() > self.config.clip_range).float().mean().item()
            num_samples += 1
        
        # Average over batch
        avg_policy_loss = total_policy_loss / num_samples
        avg_value_loss = total_value_loss / num_samples
        avg_entropy = total_entropy / num_samples
        avg_kl = total_kl / num_samples
        avg_clip_frac = clip_frac / num_samples
        
        # Combined loss
        loss = (
            avg_policy_loss +
            self.config.value_coef * avg_value_loss -
            self.config.entropy_coef * avg_entropy +
            self.config.kl_coef * avg_kl
        )
        
        return {
            "loss": loss,
            "policy_loss": avg_policy_loss.item(),
            "value_loss": avg_value_loss.item(),
            "entropy": avg_entropy.item(),
            "kl": avg_kl.item(),
            "clip_frac": avg_clip_frac
        }
    
    def train_step(self, prompts: List[str]) -> Dict[str, float]:
        """
        Single PPO training step.
        
        1. Generate rollouts
        2. Compute advantages
        3. Update policy for multiple epochs
        
        Returns:
            Dictionary of metrics
        """
        # Generate rollouts
        rollout = self.generate_rollouts(prompts)
        
        # Compute mean reward
        mean_reward = torch.stack([r.sum() for r in rollout.rewards]).mean().item()
        
        # PPO updates
        metrics_list = []
        for epoch in range(self.config.ppo_epochs):
            metrics = self.compute_ppo_loss(rollout)
            
            # Backward and optimize
            self.optimizer.zero_grad()
            metrics["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(),
                self.config.max_grad_norm
            )
            self.optimizer.step()
            
            metrics_list.append({k: v for k, v in metrics.items() if k != "loss"})
        
        # Average metrics over PPO epochs
        avg_metrics = {}
        for key in metrics_list[0].keys():
            avg_metrics[key] = sum(m[key] for m in metrics_list) / len(metrics_list)
        avg_metrics["mean_reward"] = mean_reward
        
        return avg_metrics
    
    def train(self, prompts: List[str], save_dir: str = "ppo_checkpoints"):
        """
        Main training loop.
        
        Args:
            prompts: List of reasoning prompts
            save_dir: Directory to save checkpoints
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Starting PPO training for {self.config.num_iterations} iterations")
        print(f"Batch size: {self.config.batch_size}, PPO epochs: {self.config.ppo_epochs}")
        print(f"Checkpoint directory: {save_dir}\n")
        
        for iteration in range(1, self.config.num_iterations + 1):
            start_time = time.time()
            
            # Sample prompts for this iteration
            batch_prompts = np.random.choice(
                prompts,
                size=min(self.config.batch_size, len(prompts)),
                replace=False
            ).tolist()
            
            # Training step
            metrics = self.train_step(batch_prompts)
            self.metrics_history.append(metrics)
            
            elapsed = time.time() - start_time
            
            # Logging
            if iteration % self.config.log_every == 0:
                print(f"Iteration {iteration}/{self.config.num_iterations} ({elapsed:.1f}s)")
                print(f"  Reward: {metrics['mean_reward']:.3f}")
                print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
                print(f"  Value Loss: {metrics['value_loss']:.4f}")
                print(f"  KL: {metrics['kl']:.4f}")
                print(f"  Entropy: {metrics['entropy']:.4f}")
                print(f"  Clip Frac: {metrics['clip_frac']:.3f}")
                print()
            
            # Save checkpoint
            if iteration % self.config.save_every == 0:
                checkpoint_path = os.path.join(save_dir, f"ppo_iter_{iteration}.pt")
                self.save_checkpoint(checkpoint_path, iteration)
                print(f"Saved checkpoint: {checkpoint_path}\n")
        
        # Save final checkpoint
        final_path = os.path.join(save_dir, "ppo_final.pt")
        self.save_checkpoint(final_path, self.config.num_iterations)
        print(f"Training complete! Final model saved to: {final_path}")
        
        # Save metrics
        metrics_path = os.path.join(save_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
        print(f"Metrics saved to: {metrics_path}")
    
    def save_checkpoint(self, path: str, iteration: int):
        """Save training checkpoint"""
        torch.save({
            "iteration": iteration,
            "model_state_dict": self.policy_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.__dict__,
            "metrics_history": self.metrics_history
        }, path)


def load_prompts(prompt_file: str) -> List[str]:
    """Load prompts from file (one per line)"""
    with open(prompt_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def main():
    parser = argparse.ArgumentParser(description="PPO-based reasoning model training")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to base model checkpoint (from pico-llm.py)")
    parser.add_argument("--prompts", type=str, required=True,
                        help="Path to reasoning prompts file (one per line)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--save_dir", type=str, default="/scratch/sm12779/ppo_checkpoints",
                        help="Directory to save PPO checkpoints")
    
    # PPO hyperparameters
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Number of prompts per batch")
    parser.add_argument("--ppo_epochs", type=int, default=4,
                        help="PPO update epochs per batch")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--num_iterations", type=int, default=100,
                        help="Total training iterations")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Max tokens to generate per prompt")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load base checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab

    # Handle checkpoints saved as full dict or raw state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        d_model = checkpoint.get("embed_size", 512)
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        d_model = checkpoint.get("embed_size", 512)
    else:
        # Assume raw state_dict saved via torch.save(model.state_dict(), ...)
        state_dict = checkpoint
        d_model = 512
    
    # Create policy model (trainable) with value head
    policy_model = TransformerModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=4,
        n_blocks=3,
        block_size=256,
        use_value_head=True  # Enable value head for PPO
    ).to(device)
    
    # Load pretrained weights
    policy_model.load_state_dict(state_dict, strict=False)
    print("Loaded policy model with value head")
    
    # Create reference model (frozen) without value head
    ref_model = TransformerModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=4,
        n_blocks=3,
        block_size=256,
        use_value_head=False
    ).to(device)
    ref_model.load_state_dict(state_dict, strict=False)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    print("Loaded frozen reference model\n")
    
    # Load prompts
    prompts = load_prompts(args.prompts)
    print(f"Loaded {len(prompts)} reasoning prompts\n")
    
    # Create reward function
    enc = tiktoken.get_encoding("gpt2")
    reward_fn = ReasoningRewardFunction(enc)
    
    # Create PPO config
    config = PPOConfig(
        batch_size=args.batch_size,
        ppo_epochs=args.ppo_epochs,
        learning_rate=args.learning_rate,
        num_iterations=args.num_iterations,
        max_new_tokens=args.max_new_tokens
    )
    
    # Create trainer
    trainer = PPOTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        reward_fn=reward_fn,
        config=config,
        device=device
    )
    
    # Train
    trainer.train(prompts, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
