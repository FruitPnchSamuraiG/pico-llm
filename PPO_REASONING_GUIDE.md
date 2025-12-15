# PPO-Based Reasoning Model Training Guide

This guide explains how to use Proximal Policy Optimization (PPO) to develop reasoning capabilities in your transformer model.

## Overview

The PPO training pipeline adds reasoning capabilities to a pre-trained language model by:
1. **Generating responses** to reasoning prompts
2. **Scoring responses** with a reward function that evaluates reasoning quality
3. **Updating the model** using PPO to maximize rewards while staying close to the original model

## Quick Start

### Step 1: Train Base Model

First, train a base transformer model using the standard supervised training:

```bash
python pico-llm.py \
  --enable_transformer \
  --block_size 512 \
  --embed_size 512 \
  --transformer_heads 8 \
  --transformer_blocks 6 \
  --device_id cuda:0
```

This will create a checkpoint file: `transformer_checkpoint.pt`

### Step 2: Run PPO Reasoning Training

Once you have a base checkpoint, run PPO fine-tuning:

```bash
python ppo_train.py \
  --checkpoint transformer_checkpoint.pt \
  --prompts reasoning_prompts.txt \
  --device cuda \
  --batch_size 8 \
  --ppo_epochs 4 \
  --num_iterations 100 \
  --learning_rate 1e-5
```

This will:
- Load your base model and add a value head for RL
- Generate reasoning responses for prompts
- Compute rewards based on reasoning quality
- Update the model using PPO
- Save checkpoints to `ppo_checkpoints/`

### Step 3: Use the Reasoning Model

After training, use the PPO-tuned model for generation:

```bash
python load_model.py \
  --checkpoint ppo_checkpoints/ppo_final.pt \
  --prompt "Solve step by step: What is 25 + 37?" \
  --max_tokens 100 \
  --device cuda
```

Or load it programmatically:

```python
import torch
from pico_llm import TransformerModel
import tiktoken

# Load checkpoint
checkpoint = torch.load("ppo_checkpoints/ppo_final.pt")

# Create model with value head
model = TransformerModel(
    vocab_size=50257,
    d_model=512,
    n_heads=8,
    n_blocks=6,
    block_size=512,
    use_value_head=True
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Generate reasoning
enc = tiktoken.get_encoding("gpt2")
prompt = "Solve step by step: What is 15 + 27?"
tokens = enc.encode(prompt)

with torch.no_grad():
    for _ in range(50):
        x = torch.tensor(tokens).unsqueeze(1)
        logits, _ = model(x, return_value=True)
        next_token = torch.argmax(logits[-1, 0, :]).item()
        tokens.append(next_token)

print(enc.decode(tokens))
```

## How PPO Works for Reasoning

### 1. Reward Function

The reward function (`ReasoningRewardFunction`) evaluates generated responses on multiple criteria:

- **Format rewards** (+1.0): Does the response show reasoning structure (steps, "because", "therefore", etc.)?
- **Length rewards** (+0.5 to -0.5): Encourages detailed but not excessive responses
- **Correctness rewards** (+5.0 / -1.0): Big bonus for correct answers (when ground truth available)
- **Coherence penalty** (-1.0 to 0): Penalizes repetitive or incoherent text

You can customize rewards in `ReasoningRewardFunction.compute_reward()`.

### 2. PPO Algorithm

PPO updates the policy (transformer) while:
- **Clipping policy changes** to prevent large destructive updates
- **Using a value function** to estimate future rewards (helps with credit assignment)
- **KL penalty vs reference model** to prevent the model from drifting too far from original behavior
- **Entropy bonus** to encourage exploration

### 3. Training Loop

Each iteration:
1. Sample a batch of reasoning prompts
2. Generate responses using the current policy
3. Compute rewards for each response
4. Calculate advantages (how much better than expected)
5. Update policy for multiple PPO epochs
6. Log metrics and save checkpoints

## Configuration

### PPO Hyperparameters (in `ppo_train.py`)

```python
PPOConfig(
    clip_range=0.2,          # How much to clip policy updates
    value_coef=0.5,          # Weight of value loss
    entropy_coef=0.01,       # Exploration bonus
    kl_coef=0.1,            # Penalty for diverging from reference
    ppo_epochs=4,            # Update steps per batch
    batch_size=8,            # Prompts per iteration
    max_new_tokens=128,      # Max response length
    learning_rate=1e-5,      # Lower than pretraining
)
```

### Key Parameters to Tune

| Parameter | Effect | Recommendation |
|-----------|--------|----------------|
| `batch_size` | More prompts = more stable but slower | 4-16 depending on GPU |
| `ppo_epochs` | More updates per batch = better optimization | 2-6 |
| `learning_rate` | How fast to update policy | 1e-6 to 3e-5 |
| `kl_coef` | How much to constrain updates | 0.01-0.2 |
| `max_new_tokens` | Response length | 64-256 |
| `clip_range` | Stability of updates | 0.1-0.3 |

## Creating Custom Reasoning Prompts

Format: one prompt per line in a text file

```text
Solve this step by step: What is 15 + 27?
Think carefully: Is 17 a prime number?
Explain your reasoning: Which is larger, 1/2 or 1/3?
```

For better results:
- Include explicit reasoning cues ("step by step", "think carefully", "explain")
- Mix difficulty levels
- Include problems where correctness can be verified
- Add metadata for ground truth answers (optional, requires code modification)

## Customizing Rewards

Edit `ReasoningRewardFunction.compute_reward()` to implement custom logic:

```python
def compute_reward(self, prompt: str, response: str, metadata: Optional[Dict] = None) -> float:
    reward = 0.0
    
    # Your custom reward logic
    if self._contains_mathematical_notation(response):
        reward += 2.0
    
    if self._follows_chain_of_thought(response):
        reward += 3.0
    
    # Verify correctness with external tool
    if metadata and 'answer' in metadata:
        if self._verify_with_calculator(response, metadata['answer']):
            reward += 10.0
    
    return reward
```

## Monitoring Training

### Real-time Metrics

During training, watch these metrics:

- **Reward**: Should increase over time (target depends on your reward scale)
- **KL divergence**: Should stay < 0.5 (if too high, increase `kl_coef`)
- **Clip fraction**: 0.1-0.3 is good (if too high, reduce `learning_rate`)
- **Entropy**: Should slowly decrease but not collapse
- **Value loss**: Should decrease over time

### Saved Outputs

- `ppo_checkpoints/ppo_iter_N.pt`: Model checkpoints every N iterations
- `ppo_checkpoints/ppo_final.pt`: Final trained model
- `ppo_checkpoints/metrics.json`: Full training metrics history

### Visualize Training

```python
import json
import matplotlib.pyplot as plt

with open("ppo_checkpoints/metrics.json") as f:
    metrics = json.load(f)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot([m["mean_reward"] for m in metrics])
plt.title("Mean Reward")
plt.xlabel("Iteration")

plt.subplot(1, 3, 2)
plt.plot([m["kl"] for m in metrics])
plt.title("KL Divergence")
plt.xlabel("Iteration")

plt.subplot(1, 3, 3)
plt.plot([m["policy_loss"] for m in metrics])
plt.title("Policy Loss")
plt.xlabel("Iteration")

plt.tight_layout()
plt.savefig("ppo_training_curves.png")
```

## Troubleshooting

### Issue: Rewards not increasing

**Solutions:**
- Check reward function - are rewards too sparse or always negative?
- Increase `learning_rate` (try 3e-5)
- Increase `ppo_epochs` (try 6-8)
- Reduce `kl_coef` to allow more exploration

### Issue: Model generates nonsense

**Solutions:**
- Reduce `learning_rate` (try 5e-6)
- Increase `kl_coef` to stay closer to reference model
- Check reward function - is it rewarding bad behavior?
- Reduce `clip_range` for more stable updates

### Issue: Training is too slow

**Solutions:**
- Reduce `batch_size` and `max_new_tokens`
- Reduce `ppo_epochs` (2-3 is often sufficient)
- Use smaller base model (reduce `n_blocks` or `d_model`)
- Use mixed precision training (requires code modification)

### Issue: Out of memory

**Solutions:**
- Reduce `batch_size` (try 2-4)
- Reduce `max_new_tokens` (try 64)
- Use gradient accumulation (requires code modification)
- Use a smaller base model

## Advanced Usage

### Multi-GPU Training

Modify `ppo_train.py` to wrap models with `DataParallel`:

```python
policy_model = nn.DataParallel(policy_model)
ref_model = nn.DataParallel(ref_model)
```

### Custom Reward Model

Instead of heuristic rewards, train a reward model:

```python
class LearnedRewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.reward_head = nn.Linear(base_model.d_model, 1)
    
    def forward(self, tokens):
        x = self.base(tokens)
        return self.reward_head(x[-1])  # Reward for final token

# Use in reward function
def compute_reward(self, prompt, response, metadata=None):
    tokens = self.enc.encode(prompt + response)
    reward = self.reward_model(torch.tensor(tokens).unsqueeze(1))
    return reward.item()
```

### Resume Training

```bash
python ppo_train.py \
  --checkpoint ppo_checkpoints/ppo_iter_50.pt \
  --prompts reasoning_prompts.txt \
  --num_iterations 100  # Will continue from iteration 50
```

## References

- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347)
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [Constitutional AI](https://arxiv.org/abs/2212.08073)

## Support

For issues or questions:
1. Check this README first
2. Review the example prompts in `reasoning_prompts.txt`
3. Inspect reward function logic in `ppo_train.py`
4. Monitor training metrics for anomalies
