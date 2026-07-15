# phi3-text-to-sql

Fine-tune **Microsoft Phi-3-mini** with **LoRA (QLoRA)** using **[Unsloth](https://github.com/unslothai/unsloth)** to convert **natural language → SQL**.

> Looking for a **PyTorch-only** (Transformers + PEFT, no Unsloth) version?  
> See [`phi3-text-to-sql-pytorch`](https://github.com/shrinidhi-mahishi/phi3-text-to-sql-pytorch).

```
NL question + schema
        ↓
  Phi-3 + LoRA adapters (Unsloth)
        ↓
      SQL query
```

## Why this project

| Piece | Choice |
|---|---|
| Base model | `unsloth/Phi-3-mini-4k-instruct` (3.8B, strong SLM) |
| Method | LoRA / 4-bit QLoRA via Unsloth |
| Task | Text-to-SQL over a small analytics warehouse schema |
| Eval | Exact + normalized SQL match |

Fits a Principal DS portfolio story: **SLM specialization** (cheaper/faster than giant LLMs for constrained SQL generation).

## Setup

### 1) GPU environment

Unsloth needs NVIDIA CUDA (local GPU or Google Colab).

```bash
cd phi3-text-to-sql
python3.12 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# Install Unsloth (pick one)
# Colab:
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# Or follow: https://github.com/unslothai/unsloth#installation
```

### 2) Build dataset

```bash
python scripts/prepare_dataset.py
# → data/train.jsonl, data/eval.jsonl
```

Seed data covers `products`, `price_list`, `accounts`, `entitlements` (quote/analytics style). Extend `scripts/schema_and_seed.py` for more coverage.

### 3) Train LoRA

```bash
python scripts/train_lora.py \
  --model unsloth/Phi-3-mini-4k-instruct \
  --epochs 3 \
  --lora-r 16 \
  --lora-alpha 16 \
  --batch-size 2 \
  --grad-accum 4 \
  --lr 2e-4
```

Adapters save to `outputs/phi3-sql-lora/lora_adapters/`.

### 4) Inference

```bash
python scripts/infer.py \
  --question "What is the Enterprise list price for Cloud Shield (CS-ENT)?"
```

### 5) Evaluate

```bash
python scripts/evaluate.py
```

## Project layout

```
phi3-text-to-sql/
├── configs/default.yaml
├── data/                      # generated train/eval jsonl
├── notebooks/phi3_text_to_sql_unsloth.ipynb
├── scripts/
│   ├── schema_and_seed.py     # schema + NL/SQL pairs
│   ├── prepare_dataset.py     # JSONL builder
│   ├── train_lora.py          # Unsloth LoRA SFT
│   ├── infer.py
│   └── evaluate.py
├── requirements.txt
└── README.md
```

## Prompt format

Each example is a Phi-3 chat:

1. **System** — Text-to-SQL rules (SQL only)  
2. **User** — schema + natural language question  
3. **Assistant** — SQL

## Colab

Open `notebooks/phi3_text_to_sql_unsloth.ipynb`, enable GPU, run cells.

## Resume one-liner

> Fine-tuned Phi-3-mini with Unsloth LoRA for Text-to-SQL over a structured analytics schema, with offline exact/normalized SQL evaluation.

## Notes

- Demo dataset is small by design — swap in Spider / WikiSQL / your warehouse traces for production quality.
- For serving, merge adapters or keep PEFT load; export GGUF via Unsloth if you need llama.cpp.
