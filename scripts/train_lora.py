#!/usr/bin/env python3
"""Fine-tune Phi-3-mini with LoRA via Unsloth for Text-to-SQL.

Requires a CUDA GPU (local or Colab). Install Unsloth first — see README.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phi-3 LoRA Text-to-SQL (Unsloth)")
    p.add_argument("--model", default="unsloth/Phi-3-mini-4k-instruct")
    p.add_argument("--train-file", type=Path, default=ROOT / "data" / "train.jsonl")
    p.add_argument("--eval-file", type=Path, default=ROOT / "data" / "eval.jsonl")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "phi3-sql-lora")
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--load-in-4bit", action="store_true", default=True)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--max-steps", type=int, default=-1, help="Override epochs if > 0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.train_file.exists():
        raise SystemExit(
            f"Missing {args.train_file}. Run: python scripts/prepare_dataset.py"
        )

    # Unsloth must be imported before transformers in many setups
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    import torch

    print(f"Loading base model: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )
    model.print_trainable_parameters()

    data_files = {"train": str(args.train_file)}
    if args.eval_file.exists():
        data_files["eval"] = str(args.eval_file)
    ds = load_dataset("json", data_files=data_files)

    def formatting_prompts_func(examples):
        """Convert chat messages → Phi-3 text via tokenizer chat template."""
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return {"text": texts}

    ds = ds.map(formatting_prompts_func, batched=True, remove_columns=ds["train"].column_names)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sft_args = SFTConfig(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=5,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps and args.max_steps > 0 else -1,
        learning_rate=args.lr,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=args.seed,
        report_to="none",
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
    )

    eval_dataset = ds["eval"] if "eval" in ds else None
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=eval_dataset,
        args=sft_args,
    )

    stats = trainer.train()
    adapter_dir = args.output_dir / "lora_adapters"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    meta = {
        "base_model": args.model,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "train_file": str(args.train_file),
        "output": str(adapter_dir),
        "train_runtime": getattr(stats.metrics, "get", lambda k, d=None: d)(
            "train_runtime", None
        )
        if hasattr(stats, "metrics")
        else None,
        "metrics": dict(stats.metrics) if hasattr(stats, "metrics") else {},
    }
    (args.output_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"Saved LoRA adapters → {adapter_dir}")


if __name__ == "__main__":
    main()
