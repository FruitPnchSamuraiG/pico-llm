# pico-llm (Transformer-only extension)

Educational Transformer-only project:
- **Base training** on TinyStories subsets
- **Reasoning**: HF dataset export -> SFT finetune, plus **optional RL-style outcome post-training**
- **Interpretability** tooling inspired by Anthropic / Transformer Circuits

## Comprehensive flowchart

```mermaid
flowchart TD
  VENV[Activate venv<br/>/scratch/kk6081/ml_fall25/venv/] --> BASE

  subgraph BASE[Stage 1: Base Transformer training (TinyStories)]
    TFAST[bash scripts/train_transformer_fast.sh] --> CKPT1[/scratch/.../transformer_epoch*.pt/]
    TFULL[bash scripts/train_transformer_full.sh] --> CKPT1
  end

  CKPT1 -->|init_from| OT
  CKPT1 -->|init_from| GSM

  subgraph OT[Stage 2A: OpenThoughts SFT (no RL)]
    OTDATA[python scripts/prepare_hf_reasoning_data.py<br/>dataset: open-thoughts/OpenThoughts-114k] --> OTFILES[data/open_thoughts_{train,val}.txt]
    OTSFT[bash scripts/train_transformer_reasoning.sh] --> OTCKPT[/scratch/.../transformer_reasoning_transformer_epoch*.pt/]
    OTFILES --> OTSFT
  end

  subgraph GSM[Stage 2B: GSM8K SFT + RL-outcome]
    GSM_PREP[bash scripts/prepare_hf_gsm8k_data.sh<br/>dataset: openai/gsm8k (main)] --> GSMFILES[data/gsm8k_{train,val,test}.txt]
    GSMSFT[bash scripts/train_transformer_gsm8k.sh<br/>Stage 1: SFT] --> GSMCKPT[/scratch/.../transformer_gsm8k_transformer_epoch*.pt/]
    GSMFILES --> GSMSFT

    GSMSFT -->|RUN_RL=1 (default)| RLOUT
    subgraph RLOUT[Stage 2: RL-style outcome post-training (best-of-n)]
      RL[python scripts/rl_reasoning_outcome.py] --> RLCKPT[/scratch/.../transformer_gsm8k_rl.pt/]
    end
  end

  subgraph EVAL[Evaluation]
    E1[python scripts/eval_reasoning.py<br/>(heuristic numeric accuracy)]
  end

  OTCKPT --> E1
  GSMCKPT --> E1
  RLCKPT --> E1

  subgraph INTERP[Interpretability]
    IT[python scripts/interpret_transformer.py] --> IOUT[/scratch/.../interpretability_*/]
    IOUT --> VIEW[python scripts/interpretability_viewer.py<br/>Web UI]
  end

  CKPT1 --> IT
  OTCKPT --> IT
  GSMCKPT --> IT
  RLCKPT --> IT
```

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

# 2) Reasoning finetune (OpenThoughts SFT)
bash scripts/train_transformer_reasoning.sh

# 3) GSM8K SFT + RL-outcome (default RUN_RL=1)
bash scripts/train_transformer_gsm8k.sh

# 4) Interpretability
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
  --data data/gsm8k_test.txt
```

## Interpretability & analysis

Tool: `scripts/interpret_transformer.py` (attention heatmaps, logit lens, neuron max-activation contexts, patching stub).

### Web UI (browse saved results)

After you generate results with `interpret_transformer.py`, you can browse them with a lightweight local web UI:

```bash
python scripts/interpretability_viewer.py \
  --root /scratch/kk6081/picollm_extend/interpretability_test \
  --host 127.0.0.1 --port 8000
```

Then open: `http://127.0.0.1:8000/`

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

