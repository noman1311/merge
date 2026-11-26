"""Fine-tune a code encoder for clone detection.

Trains CodeBERT, GraphCodeBERT, or UniXcoder on a single clone benchmark
(BigCloneBench or CLCDSA) to produce a domain specialist.

Example:
    python finetune.py \\
        --model_name microsoft/codebert-base \\
        --dataset_type bcb \\
        --output_dir checkpoints/codebert_bcb \\
        --epochs 5 --batch_size 32 --lr 2e-5 --seed 42
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bigclonebench_loader import (
    DEFAULT_DATASET_DIR as BCB_DEFAULT_DATASET_DIR,
    get_dataloader as get_bcb_dataloader,
)
from data.clcdsa_torch_loader import (
    DEFAULT_DATASET_DIR as CLCDSA_DEFAULT_DATASET_DIR,
    get_dataloader as get_clcdsa_dataloader,
)
from training.model import CloneDetectionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune CodeBERT-like models on clone detection datasets")
    parser.add_argument("--model_name", type=str, default="microsoft/codebert-base")
    parser.add_argument("--dataset_type", type=str, default="bcb", choices=["bcb", "clcdsa"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def evaluate(
    model: CloneDetectionModel,
    dataloader,
    dataset_size: int,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> Dict[str, float]:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs["loss"]

            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=-1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="binary",
        zero_division=0,
    )
    accuracy = accuracy_score(all_labels, all_preds)

    return {
        "loss": total_loss / max(dataset_size, 1),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def main() -> None:
    args = parse_args()

    if args.dataset_type == "bcb":
        get_dataloader = get_bcb_dataloader
        default_dataset_dir = BCB_DEFAULT_DATASET_DIR
    else:
        get_dataloader = get_clcdsa_dataloader
        default_dataset_dir = CLCDSA_DEFAULT_DATASET_DIR

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else default_dataset_dir

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    scaler = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"Using device: {device}")
    if amp_enabled:
        print(f"Mixed precision enabled with dtype: {amp_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_dataset, train_loader = get_dataloader(
        split="train",
        tokenizer=tokenizer,
        dataset_dir=dataset_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=4,
        pin_memory=True,
        use_weighted_sampling=True,
    )
    valid_dataset, valid_loader = get_dataloader(
        split="valid",
        tokenizer=tokenizer,
        dataset_dir=dataset_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=4,
        pin_memory=True,
        use_weighted_sampling=False,
    )

    model = CloneDetectionModel(model_name_or_path=args.model_name).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    global_step = 0

    print(
        f"Dataset: {args.dataset_type} | Dataset dir: {dataset_dir} | "
        f"Train size: {len(train_dataset)} | Valid size: {len(valid_dataset)} | "
        f"Total steps: {total_steps} | Warmup steps: {warmup_steps}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss_sum = 0.0

        log_window_loss = 0.0
        log_window_steps = 0
        log_window_samples = 0
        log_window_start = time.time()

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in progress:
            global_step += 1

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs["loss"]

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            scheduler.step()

            batch_size = labels.size(0)
            loss_value = loss.item()
            epoch_loss_sum += loss_value * batch_size

            log_window_loss += loss_value
            log_window_steps += 1
            log_window_samples += batch_size

            if global_step % args.log_every == 0:
                elapsed = time.time() - log_window_start
                throughput = log_window_samples / max(elapsed, 1e-8)
                mean_loss = log_window_loss / max(log_window_steps, 1)
                current_lr = scheduler.get_last_lr()[0]

                print(
                    f"step={global_step} loss={mean_loss:.4f} lr={current_lr:.7f} "
                    f"throughput={throughput:.2f} samples/s"
                )

                log_window_loss = 0.0
                log_window_steps = 0
                log_window_samples = 0
                log_window_start = time.time()

        train_loss = epoch_loss_sum / max(len(train_dataset), 1)
        valid_metrics = evaluate(
            model=model,
            dataloader=valid_loader,
            dataset_size=len(valid_dataset),
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )

        print(
            f"Epoch {epoch} done | train_loss={train_loss:.4f} | "
            f"valid_loss={valid_metrics['loss']:.4f} | valid_f1={valid_metrics['f1']:.4f} | "
            f"precision={valid_metrics['precision']:.4f} | recall={valid_metrics['recall']:.4f} | "
            f"accuracy={valid_metrics['accuracy']:.4f}"
        )

        if valid_metrics["f1"] > best_f1:
            best_f1 = valid_metrics["f1"]
            best_epoch = epoch
            epochs_without_improvement = 0

            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))

            metrics_path = output_dir / "best_metrics.json"
            with metrics_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "best_epoch": best_epoch,
                        "best_f1": best_f1,
                        "valid_metrics": valid_metrics,
                        "model_name": args.model_name,
                        "dataset_type": args.dataset_type,
                        "dataset_dir": str(dataset_dir),
                        "batch_size": args.batch_size,
                        "lr": args.lr,
                        "max_length": args.max_length,
                    },
                    handle,
                    indent=2,
                )

            print(f"New best checkpoint saved to: {output_dir}")
        else:
            epochs_without_improvement += 1
            print(f"No F1 improvement for {epochs_without_improvement} epoch(s)")

            if epochs_without_improvement >= args.patience:
                print(f"Early stopping triggered (patience={args.patience})")
                break

    print(f"Training complete. Best epoch: {best_epoch}, best valid F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
