# Pico-LLM: Unified Inference & Training

Pico-LLM is a lightweight, research-friendly codebase for training and evaluating small language models (LLMs) on reasoning and instruction-following tasks. It supports:

- **Supervised Fine-Tuning (SFT)** for instruction-following and reasoning datasets
- **DPO/GRPO** (Direct Preference Optimization / Generalized Rank Preference Optimization) for preference-based RLHF-style training
- **Unified Inference** with auto-architecture detection, KV-caching, and robust sampling
- **Flexible model architectures**: Transformer, LSTM, and K-gram MLP
- **Efficient batch generation and evaluation scripts**
- **Robust token handling**: Prevents CUDA errors from vocab mismatches (auto-detects vocab size from checkpoint)

## Quick Start

### 1. Inference

Run inference on a trained checkpoint (auto-detects architecture and vocab size):

```bash
python inference.py --checkpoint path/to/model.pt --prompt "Q: 2+2=? A: <thinking>"
```

- Use `--thinking` for reasoning models (multi-phase generation)
- Supports batch prompts via `--input_file`
- Handles all model sizes and vocabularies automatically
- Automatically clamps prompt tokens to model vocab size to avoid CUDA errors

### 2. Supervised Fine-Tuning (SFT)

Train a model on reasoning or instruction-following data:

```bash
python pico-llm.py --enable_transformer --input_files data/gsm8k_train.txt --num_epochs 3 --checkpoint_dir checkpoints/
```

- Supports TinyStories and custom datasets
- Model size and architecture are configurable via flags (see below)
- Checkpoints and loss histories are saved for each run

### 3. DPO/GRPO Training (Preference Optimization)

For preference-based RLHF-style training:

```bash
python scripts/evaluation/dpo_grpo_training.py --checkpoint_dir checkpoints/ --input_file data/gsm8k_train_prompts_only.txt
```

- Uses SFT-trained checkpoints as initialization
- Supports multi-GPU and efficient batch generation

### 4. Batch Generation & Evaluation

Generate outputs for a set of prompts (with KV-caching):

```bash
python scripts/evaluation/generate_grpo_groups_multi_gpu.py --checkpoint path/to/model.pt --input_file data/gsm8k_test.txt --output_file results.jsonl
```

- Fast, memory-efficient batch generation
- Designed for large-scale evaluation

### 5. Mechanistic Interpretability Visualizations

Analyze model internals with 7 publication-quality interpretability plots inspired by Anthropic's mechanistic interpretability work:

```bash
python plot_interpretability.py --checkpoint path/to/model.pt --prompt "Q: What is 2 + 2? A:" --device cuda:0 --output_dir interpretability_plots
```

**Generated plots:**
1. **Attention Head Patterns** - Visualize attention weights across all heads and layers (how tokens attend to each other)
2. **Logit Lens** - Layer-by-layer prediction evolution showing when the model "makes up its mind" 
3. **Token Embeddings** - 2D PCA projection of embedding space structure (semantic/syntactic clustering)
4. **Activation Distribution** - Mean activations and histograms per layer (detect scaling issues)
5. **Feature Co-activation** - Correlation heatmap of which features fire together + sparsity patterns
6. **Feature Importance** - Attribution-based ranking of most important features by layer
7. **Feature Geometry** - t-SNE projection + PCA variance of embedding space dimensionality

Use these plots to:
- Debug model behavior and understand learned representations
- Identify bottleneck layers or dead neurons
- Analyze feature interactions and compositionality
- Assess embedding space structure and dimensionality

## Key Features

- **Auto-architecture and vocab detection**: Inference and evaluation scripts automatically detect model size, block count, and vocabulary from checkpoints.
- **Robust token handling**: Prevents CUDA errors from vocab mismatches (prompt tokens are clamped to model vocab size).
- **KV-caching**: Fast Transformer generation for both single and batch inference.
- **Unified codebase**: All training and inference logic in `pico-llm.py` and `inference.py`.
- **Flexible data loading**: Supports HuggingFace TinyStories, GSM8K, OrcaMath, and custom text files.

## Model Architectures

- **Transformer** (default): GPT-2 style, configurable size
- **LSTM**: Baseline sequence model
- **K-gram MLP**: Fixed-window feedforward baseline

## Example Training Commands

**Train a small Transformer on TinyStories:**
```bash
python pico-llm.py --enable_transformer --transformer_size small --num_epochs 2
```

**Train on custom data:**
```bash
python pico-llm.py --enable_transformer --input_files data/gsm8k_train.txt --num_epochs 3
```

**Resume or finetune from checkpoint:**
```bash
python pico-llm.py --enable_transformer --init_from checkpoints/transformer_epoch2.pt
```

## Advanced: DPO/GRPO Training

- See `scripts/evaluation/dpo_grpo_training.py` for preference-based optimization.
- Use `generate_grpo_groups_multi_gpu.py` for efficient batch generation and evaluation.

## File Overview

- `pico-llm.py`: All model definitions, training, and generation logic
- `inference.py`: Unified inference script (auto-detects model, robust to vocab mismatches)
- `scripts/evaluation/`: DPO/GRPO training and batch generation scripts
- `data/`: Example datasets (GSM8K, TinyStories, OrcaMath)

## Tips

- For reasoning models, always use prompts with `<thinking>` and `--thinking` flag for best results.
- All scripts are robust to model size and vocabulary mismatches.
- Checkpoints are saved after every epoch for easy resumption.
- Batch generation scripts use KV-caching for speed and memory efficiency.

