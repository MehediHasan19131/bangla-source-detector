"""Fine-tune a Bangla transformer (BanglaBERT by default) for synthetic-vs-translated detection.

    python -m bsd.train                                   # full run
    python -m bsd.train --train-limit 512 --epochs 1      # quick smoke test
    python -m bsd.train --model xlm-roberta-base --run-name xlmr
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from . import LABELS, metrics
from .data import load_split


def softmax_translated(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (exp / exp.sum(axis=1, keepdims=True))[:, 1]


def to_dataset(df, tokenizer, max_length: int) -> Dataset:
    ds = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))
    return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=max_length),
                  batched=True, remove_columns=["text"])


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    return metrics.compute(labels, logits.argmax(axis=1), softmax_translated(logits))


def evaluate(trainer: Trainer, ds: Dataset, run_name: str, split: str) -> dict:
    out = trainer.predict(ds)
    result = compute_metrics((out.predictions, out.label_ids))
    metrics.save_scores(run_name, split, out.label_ids, softmax_translated(out.predictions))
    result["samples_per_second"] = round(out.metrics["test_samples_per_second"], 1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="csebuetnlp/banglabert")
    parser.add_argument("--run-name", default="banglabert")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train-limit", type=int, default=None, help="use only N training rows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", default="none", help="'wandb' to log the run to Weights & Biases")
    parser.add_argument("--push-to-hub", default=None, help="HF repo id, e.g. user/banglabert-source-detector")
    parser.add_argument("--eval-only", action="store_true", help="re-evaluate models/<run-name> without training")
    args = parser.parse_args()
    set_seed(args.seed)

    train_df, val_df = load_split("train", args.data), load_split("val", args.data)
    if args.train_limit:
        train_df = train_df.sample(n=min(args.train_limit, len(train_df)), random_state=args.seed)
        val_df = val_df.sample(n=min(args.train_limit // 4 or 1, len(val_df)), random_state=args.seed)

    out_dir = Path("models") / args.run_name
    source = str(out_dir) if args.eval_only else args.model
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(
        source, num_labels=2, id2label=LABELS, label2id={v: k for k, v in LABELS.items()}
    )
    train_ds = to_dataset(train_df, tokenizer, args.max_length)
    val_ds = to_dataset(val_df, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to=args.report_to,
        run_name=args.run_name,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    results_path = Path(f"results/{args.run_name}.json")
    if args.eval_only:
        results = json.loads(results_path.read_text())
    else:
        start = time.time()
        trainer.train()
        results = {
            "model": args.model,
            "config": {k: v for k, v in vars(args).items() if k not in ("data", "eval_only")},
            "train_rows": len(train_df),
            "train_minutes": round((time.time() - start) / 60, 1),
            "device": str(trainer.args.device),
        }
        trainer.save_model(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))

    for split in ("test", "ood"):
        ds = to_dataset(load_split(split, args.data), tokenizer, args.max_length)
        results[split] = evaluate(trainer, ds, args.run_name, split)
        print(f"{split}: acc={results[split]['accuracy']:.3f} f1={results[split]['f1']:.3f} "
              f"synthetic recall={results[split]['recall_synthetic']:.3f}")

    results_path.parent.mkdir(exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))
    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    main()
