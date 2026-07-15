#!/usr/bin/env python3
"""Run NL→SQL inference with a Unsloth Phi-3 LoRA adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

from schema_and_seed import SCHEMA

ROOT = Path(__file__).resolve().parents[1]
SYSTEM = (
    "You are an expert Text-to-SQL assistant. "
    "Given a database schema and a natural language question, "
    "write a single valid SQL query. "
    "Return ONLY the SQL — no markdown fences, no commentary."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--adapter",
        type=Path,
        default=ROOT / "outputs" / "phi3-sql-lora" / "lora_adapters",
    )
    p.add_argument("--base-model", default="unsloth/Phi-3-mini-4k-instruct")
    p.add_argument("--question", type=str, required=True)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-seq-length", type=int, default=2048)
    return p.parse_args()


def build_messages(question: str) -> list[dict]:
    user = f"Schema:\n{SCHEMA}\n\nQuestion:\n{question}\n\nSQL:"
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]


def main() -> None:
    args = parse_args()
    from unsloth import FastLanguageModel

    # Load base + attach adapters (Unsloth accepts adapter path as model_name too)
    model_name = str(args.adapter if args.adapter.exists() else args.base_model)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    messages = build_messages(args.question)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=0.1,
        do_sample=False,
        use_cache=True,
    )
    decoded = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    )
    sql = decoded.strip()
    if sql.startswith("```"):
        sql = sql.strip("`")
        sql = sql.replace("sql\n", "", 1).strip()
    print(sql)


if __name__ == "__main__":
    main()
