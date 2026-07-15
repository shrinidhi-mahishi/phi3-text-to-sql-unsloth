#!/usr/bin/env python3
"""Simple offline eval: exact-match + normalized SQL match on eval.jsonl."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def normalize_sql(sql: str) -> str:
    s = sql.strip().rstrip(";").lower()
    s = re.sub(r"\s+", " ", s)
    return s


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-file", type=Path, default=ROOT / "data" / "eval.jsonl")
    p.add_argument(
        "--adapter",
        type=Path,
        default=ROOT / "outputs" / "phi3-sql-lora" / "lora_adapters",
    )
    p.add_argument("--base-model", default="unsloth/Phi-3-mini-4k-instruct")
    p.add_argument("--limit", type=int, default=0, help="0 = all")
    p.add_argument("--max-new-tokens", type=int, default=256)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from unsloth import FastLanguageModel
    from schema_and_seed import SCHEMA

    system = (
        "You are an expert Text-to-SQL assistant. "
        "Given a database schema and a natural language question, "
        "write a single valid SQL query. "
        "Return ONLY the SQL — no markdown fences, no commentary."
    )

    model_name = str(args.adapter if args.adapter.exists() else args.base_model)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    rows = []
    with args.eval_file.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if args.limit > 0:
        rows = rows[: args.limit]

    exact = 0
    norm = 0
    for i, row in enumerate(rows, 1):
        q = row["question"]
        gold = row["sql"]
        user = f"Schema:\n{SCHEMA}\n\nQuestion:\n{q}\n\nSQL:"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        pred = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        ).strip()
        if pred.startswith("```"):
            pred = pred.strip("`").replace("sql\n", "", 1).strip()

        e = pred.strip() == gold.strip()
        n = normalize_sql(pred) == normalize_sql(gold)
        exact += int(e)
        norm += int(n)
        print(f"[{i}/{len(rows)}] exact={e} norm={n}")
        print(f"  Q: {q}")
        print(f"  gold: {gold}")
        print(f"  pred: {pred}")

    print("---")
    print(f"exact_match: {exact}/{len(rows)} = {exact / len(rows):.3f}")
    print(f"normalized_match: {norm}/{len(rows)} = {norm / len(rows):.3f}")


if __name__ == "__main__":
    main()
