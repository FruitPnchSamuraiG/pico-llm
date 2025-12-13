# pico-llm (Transformer-only extension)

Educational Transformer-only project:
- **Base training** on TinyStories subsets
- **Decoding / test-time search**: greedy / nucleus / beam / **Lookahead Nucleus Search (LNS)**
- **Reasoning**: HF dataset export -> SFT finetune, plus **optional RL-style outcome post-training**
- **Interpretability** tooling inspired by Anthropic / Transformer Circuits

## Environment / constraints
- GPU: 2x GTX TITAN X (12GB). Default device: **`cuda:0`**
- Venv: `/scratch/kk6081/ml_fall25/venv/`
- Artifacts/checkpoints: **`/scratch/kk6081/picollm_extend/`**

## TL;DR (minimal commands)

```bash
source /scratch/kk6081/ml_fall25/venv/bin/activate
cd /home/kk6081/pico_llm_extend/pico-llm

# 1) Train base transformer
bash scripts/train_transformer_fast.sh

# 2) Decode (includes LNS)
python inference.py --model transformer --checkpoint /scratch/kk6081/picollm_extend/transformer_epoch1.pt \
  --prompt "Once upon a time" --decode lookahead --lookahead_k 8 --lookahead_h 6 --device cuda:0

# 3) Reasoning finetune (OpenThoughts SFT)
bash scripts/train_transformer_reasoning.sh

# 4) GSM8K SFT + RL-outcome (default RUN_RL=1)
bash scripts/train_transformer_gsm8k.sh

# 5) Interpretability
python scripts/interpret_transformer.py --checkpoint /scratch/kk6081/picollm_extend/transformer_epoch1.pt \
  --analysis attention,logit_lens,neurons --out_dir /scratch/kk6081/picollm_extend/interpretability_base \
  --embed_size 384 --transformer_heads 4 --transformer_blocks 3 --ff_mult 2 --test_prompts "Once upon a time" --device cuda:0
```

## Training

### Base Transformer (TinyStories)
- Fast dev: `bash scripts/train_transformer_fast.sh`
- Full run: `bash scripts/train_transformer_full.sh`

Scaling knobs (see `pico-llm.py`):
- `--transformer_size {small,medium}`
- `--tinystories_train_subset_size N`
- Faster training knobs: `--sample_interval_seconds`, `--sample_every_steps`, `--lr_schedule`, `--lr_warmup_steps`, `--lr_min_ratio`

If you hit OOM on 12GB:
- `BATCH=8` (or `4`) and/or `TRANSFORMER_SIZE=small`

## Decoding / test-time search

`inference.py` supports:
- `--decode greedy`
- `--decode nucleus`
- `--decode beam`
- `--decode lookahead` (**Lookahead Nucleus Search / LNS**)

LNS summary: pick the next token by scoring top-K candidates using a short H-step rollout:

**score = avg_logprob(rollout) - rep_penalty * repetition(rollout)**

(Implementation: `inference.py: decode_lookahead_search`.)

## Reasoning: datasets + training

### A) OpenThoughts SFT (no RL)
Default: `open-thoughts/OpenThoughts-114k` -> exported to line-based text.

```bash
bash scripts/train_transformer_reasoning.sh
```

Outputs copied back as:
- `/scratch/kk6081/picollm_extend/transformer_reasoning_transformer_epoch*.pt`

### B) GSM8K: SFT + RL-outcome (best-of-n)
Dataset:
- https://huggingface.co/datasets/openai/gsm8k (config `main`)

Prepare text files (optional; script auto-runs if missing):
- `bash scripts/prepare_hf_gsm8k_data.sh` -> writes `data/gsm8k_{train,val,test}.txt`

Train:
```bash
bash scripts/train_transformer_gsm8k.sh
```

Notes:
- Stage 1: SFT finetune -> checkpoints copied back as `transformer_gsm8k_transformer_epoch*.pt`
- Stage 2: RL-style outcome post-training (default **enabled**; disable via `RUN_RL=0`)
  - Output copied back as `/scratch/kk6081/picollm_extend/transformer_gsm8k_rl.pt`

RL reading pointers:
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Dr. Tulu draft: https://www.datocms-assets.com/64837/1763496622-dr_tulu_draft.pdf

### Reasoning evaluation
Heuristic numeric accuracy (works best with GSM8K / synthetic arithmetic):

```bash
python scripts/eval_reasoning.py \
  --checkpoint /scratch/kk6081/picollm_extend/transformer_gsm8k_transformer_epoch1.pt \
  --data data/gsm8k_test.txt \
  --decode greedy
```

## Interpretability & analysis

Tool: `scripts/interpret_transformer.py` (attention heatmaps, logit lens, neuron max-activation contexts, patching stub).

Transformer Circuits hub:
- https://transformer-circuits.pub/

Referenced posts:
- Framework: https://transformer-circuits.pub/2021/framework/index.html
- Induction heads: https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
- Monosemantic features: https://transformer-circuits.pub/2023/monosemantic-features/index.html
- Toy models of superposition: https://transformer-circuits.pub/2022/toy_model/index.html

Example:

```bash
python scripts/interpret_transformer.py \
  --checkpoint /scratch/kk6081/picollm_extend/transformer_epoch1.pt \
  --analysis attention,logit_lens,neurons \
  --out_dir /scratch/kk6081/picollm_extend/interpretability_test \
  --embed_size 384 --transformer_heads 4 --transformer_blocks 3 --ff_mult 2 \
  --test_prompts "Once upon a time" "2 + 2 =" \
  --device cuda:0
```

Smoke-test outputs (already verified previously):
- `/scratch/kk6081/picollm_extend/interpretability_test/attention/attn_*.png`
- `/scratch/kk6081/picollm_extend/interpretability_test/logit_lens/results.json`
- `/scratch/kk6081/picollm_extend/interpretability_test/neurons/top_neurons.json`
- `/scratch/kk6081/picollm_extend/interpretability_test/summary.json`

