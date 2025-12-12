# Test-Time Search (TTS) Extension: Lookahead Nucleus Search (LNS)

This document describes the **new additions** made on top of the original Pico-LLM project.

## Summary

This extension adds a **Transformer-only** experimental pipeline for **test-time search** during generation.

You can now compare:
- **Nucleus sampling** (top-p)
- **Beam search**
- **Lookahead Nucleus Search (LNS)**: a lightweight test-time search method that scores next-token candidates by rolling out short continuations and choosing the best candidate under a simple objective.

---

## What was added/changed

### 1) New decoding modes in `inference.py`

`inference.py` now supports a new `--decode` flag:

- `--decode greedy`  
- `--decode nucleus`  
- `--decode beam`  
- `--decode lookahead` (LNS / test-time search)

Additional flags:
- `--beam_width` (beam search)
- `--lookahead_k` (candidate next-token count)
- `--lookahead_h` (lookahead horizon in tokens)
- `--rep_penalty` (weight for repetition penalty used in LNS scoring)

During inference it prints quick, easy-to-graph metrics:
- `distinct-1` and `distinct-2`
- `rep-4` (defined as `1 - distinct-4`)
- wall-clock time

---

### 2) New script: `scripts/train_transformer_fast.sh`

This is a **fast dev mode** training script to quickly produce a Transformer checkpoint suitable for iterating on decoding/search.

Defaults (override via env vars):
- `DEVICE=cuda:0`
- `EPOCHS=1`
- `MAX_STEPS=200` (passed as `--max_steps_per_epoch`)
- `BLOCK_SIZE=256`, `EMBED=384`, `HEADS=4`, `BLOCKS=3`, `FF_MULT=2`

It uses the required virtual environment:
- `/scratch/kk6081/ml_fall25/venv/`

---

### 3) Updated script: `scripts/run_tts_grid.sh`

This script runs a small decoding grid for a given checkpoint and prompt.

- Defaults to `DEVICE=cuda:0`
- Adds `--fast` mode that reduces runtime by using smaller lookahead settings

---

## Checkpoint output location

Training now supports an explicit output directory:

- `--checkpoint_dir /scratch/kk6081/picollm_extend` (default)

This directory will contain:
- `transformer_epoch*.pt` checkpoints
- `loss_histories.pkl`

---

## Training: fast vs full

### Fast dev training (recommended while iterating)

```bash
source /scratch/kk6081/ml_fall25/venv/bin/activate
bash scripts/train_transformer_fast.sh
```

This uses `--max_steps_per_epoch` to stop early.

### Full training

```bash
source /scratch/kk6081/ml_fall25/venv/bin/activate
bash scripts/train_transformer_full.sh
```

This runs each epoch over the full dataset (no `--max_steps_per_epoch`).

---

## How to run (end-to-end)

All runs should start by activating your venv:

```bash
source /scratch/kk6081/ml_fall25/venv/bin/activate
```

### Step 1: Train a quick Transformer checkpoint (fast dev)

From the repo root:

```bash
bash scripts/train_transformer_fast.sh
```

Outputs:
- `transformer_epoch1.pt` (and more if you increase epochs)


### Step 2: Compare decoding methods (fast grid)

```bash
bash scripts/run_tts_grid.sh transformer_epoch1.pt "Once upon a time" --fast
```

To run the full grid (slower):

```bash
bash scripts/run_tts_grid.sh transformer_epoch1.pt "Once upon a time"
```

---

## What is Lookahead Nucleus Search (LNS)?

At each generation step:
1. Compute the model distribution for the next token.
2. Take the top-`K` candidate tokens by log-probability.
3. For each candidate, do a short rollout of length `H` (sampling with nucleus/top-p).
4. Score each candidate using:

**Score = average logprob(rollout) − rep_penalty * repetition(rollout)**

Then pick the candidate with the best score.

This is a simple, training-free way to trade **more test-time compute** for better generations.

---

## Notes / limitations

- No KV cache is used yet; each token step recomputes a full forward pass. This is fine for small models and short generations but makes lookahead slower.
- Metrics are simple; for a paper you may want to add perplexity on a held-out prompt set, or task-specific accuracy.

---

## Suggested citations (for a paper write-up)

- Holtzman et al. (2019). *The Curious Case of Neural Text Degeneration* (nucleus sampling)
- Wang et al. (2022). *Self-Consistency Improves Chain of Thought Reasoning* (search via sampling)
- Yao et al. (2023). *Tree of Thoughts* (search over reasoning steps)

---

## Files touched

- Modified: `inference.py`
- Added: `scripts/train_transformer_fast.sh`
- Modified: `scripts/run_tts_grid.sh`
- Added: `README_TTS.md` (this file)
