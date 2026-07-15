#!/usr/bin/env python3
"""Build train/eval JSONL for Phi-3 NL→SQL fine-tuning."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from schema_and_seed import EXAMPLES, SCHEMA

ROOT = Path(__file__).resolve().parents[1]
SYSTEM = (
    "You are an expert Text-to-SQL assistant. "
    "Given a database schema and a natural language question, "
    "write a single valid SQL query. "
    "Return ONLY the SQL — no markdown fences, no commentary."
)


def build_record(question: str, sql: str) -> dict:
    user = (
        f"Schema:\n{SCHEMA}\n\n"
        f"Question:\n{question}\n\n"
        "SQL:"
    )
    # Phi-3 chat style messages (Unsloth / tokenizer apply_chat_template)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": sql.strip()},
        ],
        "question": question,
        "sql": sql.strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-ratio", type=float, default=0.15)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = [build_record(q, s) for q, s in EXAMPLES]
    rng.shuffle(rows)

    n_eval = max(1, int(len(rows) * args.eval_ratio))
    eval_rows = rows[:n_eval]
    train_rows = rows[n_eval:]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train.jsonl"
    eval_path = args.out_dir / "eval.jsonl"

    for path, subset in ((train_path, train_rows), (eval_path, eval_rows)):
        with path.open("w", encoding="utf-8") as f:
            for row in subset:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {
        "n_train": len(train_rows),
        "n_eval": len(eval_rows),
        "schema_chars": len(SCHEMA),
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {train_path} ({len(train_rows)}) and {eval_path} ({len(eval_rows)})")


if __name__ == "__main__":
    main()
