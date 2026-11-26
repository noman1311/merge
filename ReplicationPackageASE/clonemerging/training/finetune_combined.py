#!/usr/bin/env python3
"""
Multi-task baseline: Fine-tune UniXcoder on BCB + CLCDSA combined.
This serves as the "train on all data" baseline to compare against
data-free model merging.

Usage:
    CUDA_VISIBLE_DEVICES=1 python training/finetune_combined.py \
        --output_dir checkpoints/unixcoder_combined \
        --base_model microsoft/unixcoder-base \
        --epochs 10 \
        --batch_size 32 \
        --lr 2e-5
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bigclonebench_loader import get_dataloader as get_bcb_dataloader
from data.clcdsa_torch_loader import get_dataloader as get_clcdsa_dataloader
from training.model import CloneDetectionModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="microsoft/unixcoder-base")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def evaluate(model, dataloader, dataset_size, device, amp_enabled, amp_dtype):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs["loss"]
            preds = torch.argmax(outputs["logits"], dim=-1)

            if loss is not None:
                total_loss += loss.item() * labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    accuracy = accuracy_score(all_labels, all_preds)
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "loss": total_loss / max(dataset_size, 1),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16

    print(f"Device: {device}, AMP dtype: {amp_dtype}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # Load BCB train
    print("Loading BCB training data...")
    bcb_train_dataset, _ = get_bcb_dataloader(
        split="train", tokenizer=tokenizer,
        batch_size=args.batch_size, max_length=args.max_length,
        num_workers=args.num_workers, pin_memory=True,
        use_weighted_sampling=False,
    )

    # Load CLCDSA train
    print("Loading CLCDSA training data...")
    clcdsa_train_dataset, _ = get_clcdsa_dataloader(
        split="train", tokenizer=tokenizer,
        batch_size=args.batch_size, max_length=args.max_length,
        num_workers=args.num_workers, pin_memory=True,
        use_weighted_sampling=False,
    )

    # Combine datasets
    combined_dataset = ConcatDataset([bcb_train_dataset, clcdsa_train_dataset])
    print(f"Combined training set: {len(combined_dataset)} pairs "
          f"(BCB: {len(bcb_train_dataset)}, CLCDSA: {len(clcdsa_train_dataset)})")

    # Weighted sampler for combined dataset
    # BCB is ~14% positive, CLCDSA is ~50% positive
    # Use simple shuffle for combined - both datasets are large enough
    combined_loader = DataLoader(
        combined_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Load validation sets separately
    print("Loading validation sets...")
    bcb_val_dataset, bcb_val_loader = get_bcb_dataloader(
        split="valid", tokenizer=tokenizer,
        batch_size=args.batch_size * 2, max_length=args.max_length,
        num_workers=args.num_workers, pin_memory=True,
        use_weighted_sampling=False,
    )

    clcdsa_val_dataset, clcdsa_val_loader = get_clcdsa_dataloader(
        split="valid", tokenizer=tokenizer,
        batch_size=args.batch_size * 2, max_length=args.max_length,
        num_workers=args.num_workers, pin_memory=True,
        use_weighted_sampling=False,
    )

    # Create model
    model = CloneDetectionModel(args.base_model).to(device)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(combined_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    scaler = GradScaler("cuda", enabled=amp_enabled)

    print(f"\nTraining config:")
    print(f"  Total samples: {len(combined_dataset)}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Steps/epoch: {len(combined_loader)}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Patience: {args.patience}")

    # Training loop
    best_combined_f1 = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(combined_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1

            if num_batches % 500 == 0:
                pbar.set_postfix({"loss": f"{epoch_loss/num_batches:.4f}"})

        avg_loss = epoch_loss / num_batches
        print(f"\nEpoch {epoch+1} train loss: {avg_loss:.4f}")

        # Evaluate on both validation sets
        print("Evaluating on BCB validation...")
        bcb_metrics = evaluate(model, bcb_val_loader, len(bcb_val_dataset),
                               device, amp_enabled, amp_dtype)
        print(f"  BCB val F1: {bcb_metrics['f1']:.4f}")

        print("Evaluating on CLCDSA validation...")
        clcdsa_metrics = evaluate(model, clcdsa_val_loader, len(clcdsa_val_dataset),
                                   device, amp_enabled, amp_dtype)
        print(f"  CLCDSA val F1: {clcdsa_metrics['f1']:.4f}")

        combined_f1 = (bcb_metrics["f1"] + clcdsa_metrics["f1"]) / 2
        print(f"  Combined val F1: {combined_f1:.4f}")

        # Save best model
        if combined_f1 > best_combined_f1:
            best_combined_f1 = combined_f1
            patience_counter = 0
            print(f"  New best! Saving to {output_dir}")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(str(output_dir))

            best_metrics = {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "bcb_val_f1": bcb_metrics["f1"],
                "clcdsa_val_f1": clcdsa_metrics["f1"],
                "combined_val_f1": combined_f1,
                "model_name": args.base_model,
            }
            with open(output_dir / "best_metrics.json", "w") as f:
                json.dump(best_metrics, f, indent=2)
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nTraining complete. Best combined val F1: {best_combined_f1:.4f}")
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()