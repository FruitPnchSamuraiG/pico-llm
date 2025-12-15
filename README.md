# Pico-LLM: Reasoning Model Training System

A complete 3-stage training pipeline for mathematical reasoning models with explicit chain-of-thought capabilities.

## Quick Start

```bash
# Stage 1: Train base model on Orca-Math (~8-12 hours)
bash scripts/train.sh orca

# Stage 2: Fine-tune on GSM8K (~2-4 hours)
bash scripts/train.sh gsm8k

# Stage 3: Choose one:
# 3a. Reasoning model with <thinking> blocks (~1-2 hours)
bash scripts/train_reasoning.sh

# 3b. Fast optimized DPO (~30 min - 1 hour)
bash scripts/fast_dpo_train.sh dpo medium
```

## Features

### Training Pipeline
- **Stage 1 (Orca)**: 200k math problems, 8 epochs → `transformer_epoch8.pt`
- **Stage 2 (GSM8K)**: 7.5k problems, 10 epochs → `gsm8k_transformer_epoch10.pt`
- **Stage 3 (Reasoning/DPO)**: Preference optimization → final model

### Reasoning Capabilities
Models trained with `train_reasoning.sh` generate explicit thinking:
```
Q: If John has 5 apples and buys 3 more, how many does he have? A:
<thinking>
1. John starts with 5 apples
2. He buys 3 more apples
3. Total = 5 + 3 = 8 apples
</thinking>
<answer> The answer is 8. #### 8
```

### Advanced Features
- **Explicit `<thinking>` blocks** for interpretable reasoning
- **🆕 Thinking-aware generation**: Separate token budgets for thinking (800 tokens) and answer (200 tokens)
  - No more truncated reasoning chains
  - Automatically detects `</thinking>` to switch phases
  - Configurable via `MAX_THINKING_TOKENS` and `MAX_ANSWER_TOKENS`
- **Process Reward Model (PRM)**: Scores intermediate steps
- **Outcome Reward Model (ORM)**: Binary correct/incorrect (DeepSeek R1 style)
- **Best-of-N sampling**: Generate N solutions, pick best (+10-20% accuracy)
- **Self-consistency**: Majority vote over multiple reasoning paths
- **Optimized DPO**: Pre-generated preferences, batched processing (10-100x faster)

## Configuration

### Model Sizes

| Size | Params | VRAM | Notes |
|------|--------|------|-------|
| `small` | 10M | ~2GB | Fast prototyping |
| `medium` | 40M | ~4GB | **Recommended** |
| `gpt2-small` | 117M | ~8GB | Higher quality |

### Environment Variables

```bash
# Model
TRANSFORMER_SIZE=medium         # small | medium | gpt2-small

# Training
EPOCHS=10                       # SFT epochs
NUM_STEPS=1000                  # DPO/GRPO steps
BATCH_SIZE=8                    # Batch size
LR=3e-4                         # Learning rate
DEVICE=cuda:0                   # GPU device

# Reasoning-specific
THINKING_STYLE=structured       # structured | verbose | concise
REWARD_MODE=orm                 # orm | prm | hybrid
```

## Training Scripts

### `scripts/train.sh`
Base training and GSM8K SFT:
```bash
bash scripts/train.sh orca                    # Stage 1
bash scripts/train.sh gsm8k                   # Stage 2
bash scripts/train.sh gpt2 gpt2-small         # GPT-2 scale models
```

**Custom settings:**
```bash
# Conservative SFT
LR_OVERRIDE=1e-4 EPOCHS_OVERRIDE=15 bash scripts/train.sh gsm8k

# Different model size
TRANSFORMER_SIZE=gpt2-small bash scripts/train.sh orca
```

### `scripts/train_reasoning.sh`
Reasoning models with `<thinking>` blocks:
```bash
# Default: structured thinking + ORM
bash scripts/train_reasoning.sh

# Verbose thinking + PRM rewards
THINKING_STYLE=verbose REWARD_MODE=prm bash scripts/train_reasoning.sh

# Hybrid rewards (30% process + 70% outcome)
REWARD_MODE=hybrid bash scripts/train_reasoning.sh
```

### `scripts/fast_dpo_train.sh`
Optimized DPO/GRPO training:
```bash
# DPO (pairwise preferences)
bash scripts/fast_dpo_train.sh dpo medium

# GRPO (group-based optimization)
bash scripts/fast_dpo_train.sh grpo medium

# Custom hyperparameters
NUM_STEPS=2000 BATCH_SIZE=4 bash scripts/fast_dpo_train.sh dpo medium
```

## Inference & Evaluation

### Basic Inference
```bash
python scripts/inference_dpo.py \
  --checkpoint /scratch/kk6081/picollm_extend/gsm8k_transformer_epoch10.pt \
  --prompt "Q: If 2+2=? A:"
```

### Reasoning Demo
```bash
python scripts/evaluation/demo_reasoning.py
```

### GSM8K Evaluation
```bash
python scripts/evaluation/eval_reasoning.py \
  --checkpoint <path> \
  --device cuda:0
```

### Best-of-N Sampling (Python)
```python
from scripts.evaluation.reasoning_training import best_of_n_sampling
import tiktoken, torch

# Load model (example)
model = ...
enc = tiktoken.get_encoding('gpt2')

# Generate 8 solutions with thinking-aware generation, pick best
best, score, all = best_of_n_sampling(
    model, enc, inf,
    prompt="Q: ... A: <thinking>",
    gold_answer="42",
    n=8,
    max_thinking_tokens=800,     # 🆕 Generous thinking budget
    max_answer_tokens=200,       # 🆕 Separate answer budget
    scoring_method="orm",        # or "prm" or "logprob"
    use_thinking_mode=True       # 🆕 Enable thinking-aware generation
)
```

### Thinking-Aware Generation (Python)
```python
import sys
sys.path.append('.')
import inference as inf
from pico_llm import TransformerModel

# Standard generation (combined 256 token limit)
text, _ = inf.generate_text(model, enc, prompt, max_new_tokens=256)

# 🆕 Thinking-aware generation (separate budgets)
text, phase_info = inf.generate_text_with_thinking(
    model, enc, prompt,
    max_thinking_tokens=800,   # Up to 800 tokens for reasoning
    max_answer_tokens=200,     # Up to 200 tokens for answer
    device="cuda:0",
    top_p=0.95
)

# phase_info = {
#     "thinking_tokens": 156,
#     "answer_tokens": 23,
#     "phase_switched": True  # Model generated </thinking>
# }
```

## Project Structure

```
pico-llm.py                           # Core training (Stages 1-2)
inference.py                          # Basic inference
scripts/
  train.sh                            # Orca + GSM8K SFT
  train_reasoning.sh                  # Reasoning training
  fast_dpo_train.sh                   # Optimized DPO/GRPO
  inference_dpo.py                    # DPO inference
  evaluation/
    dpo_grpo_training.py              # DPO/GRPO implementation
    reasoning_training.py             # Reasoning logic
    generate_preference_pairs.py      # Pre-generate preferences
    demo_reasoning.py                 # Interactive demo
    eval_reasoning.py                 # Evaluation
  data_prep/
    prepare_orca_math_data.py         # Download Orca-Math
    prepare_hf_gsm8k_data.sh          # Download GSM8K
data/
  orca_math_{train,val}.txt           # 200k math problems
  gsm8k_{train,val,test}.txt          # 7.5k grade-school math
```

## Key Improvements

1. ✅ **Fixed SFT**: 10 epochs (was 4), better LR (3e-4 vs 2e-4)
2. ✅ **Optimized DPO**: Pre-generated preferences, batched processing
3. ✅ **Reasoning**: Explicit `<thinking>` blocks like DeepSeek R1/o1
4. ✅ **🆕 Thinking-aware generation**: Separate token budgets for thinking vs answer
   - Old: 256 tokens total (thinking + answer compete for budget)
   - New: 800 thinking + 200 answer = no truncated reasoning
5. ✅ **Process rewards**: Fine-grained step feedback
6. ✅ **Inference search**: Best-of-N, self-consistency
7. ✅ **Orca checkpoint**: Properly loads `/scratch/.../transformer_epoch8.pt`

## Common Issues

**Model generates nonsense?**  
→ Verify correct checkpoint:
```bash
BASE_CKPT=/scratch/kk6081/picollm_extend/transformer_epoch8.pt bash scripts/train.sh gsm8k
```

**OOM errors?**  
→ Reduce batch size:
```bash
BATCH_SIZE=4 bash scripts/train.sh gsm8k
```

**DPO training slow?**  
→ Use pre-generated preferences (automatic in `fast_dpo_train.sh`)

**Thinking blocks cut off?**  
→ ✅ **SOLVED: Thinking-aware generation** now uses separate token budgets by default

## Hardware Requirements

- **GPU**: 12GB+ VRAM (GTX TITAN X or better)
- **Python**: 3.9+
- **CUDA**: 11.0+

```bash
pip install torch tiktoken
```

## Examples

### Full Pipeline
```bash
# 12-18 hours total
bash scripts/train.sh orca                      # 8-12h
bash scripts/train.sh gsm8k                     # 2-4h
bash scripts/train_reasoning.sh                 # 1-2h
```

### Fast Prototype
```bash
# ~3-4 hours with pre-trained Orca
bash scripts/train.sh gsm8k                     # 2-4h
bash scripts/fast_dpo_train.sh dpo medium      # 30min-1h
```

### Reasoning Experiments
```bash
# Test different thinking styles
for style in structured verbose concise; do
  THINKING_STYLE=$style bash scripts/train_reasoning.sh
done

# Test different reward models
for reward in orm prm hybrid; do
  REWARD_MODE=$reward bash scripts/train_reasoning.sh
done
```

## How Thinking-Aware Generation Works

The system now has **two-phase generation**:

1. **Thinking Phase** (up to 800 tokens by default)
   - Generates reasoning steps inside `<thinking>` block
   - Continues until model produces `</thinking>` token
   - Can be configured via `max_thinking_tokens`

2. **Answer Phase** (up to 200 tokens by default)
   - Starts after `</thinking>` is detected
   - Generates final answer in `<answer>` block
   - Stops after `max_answer_tokens` or natural completion

**Example flow:**
```
Prompt: "Q: If 2+2=? A: <thinking>"
↓
Model generates: "Let me add the numbers: 2+2=4</thinking><answer>4</answer>"
                 └─────────── 8 tokens ────────────┘└──── 5 tokens ────┘
                 (from thinking budget)              (from answer budget)
```

**Why this matters:**
- Complex problems can use full 800 tokens for reasoning
- Answer quality isn't compromised by long reasoning chains
- Models learn to naturally close `</thinking>` blocks

## Citation

Implements modern reasoning model techniques from:
- Wei et al. (2022): Chain-of-Thought Prompting
- Rafailov et al. (2023): Direct Preference Optimization
- DeepSeek-AI (2025): DeepSeek-R1

## License

MIT

