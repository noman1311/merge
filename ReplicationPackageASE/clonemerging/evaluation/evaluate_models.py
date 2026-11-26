"""Evaluate a clone detection checkpoint on a benchmark test set.

Loads a fine-tuned or merged checkpoint and reports accuracy, precision, recall,
and F1 on one of the supported benchmarks (BigCloneBench, CLCDSA, or
GPTCloneBench), writing the metrics to a JSON file.

Example:
    python evaluate_models.py \\
        --checkpoint_dir checkpoints/merged_codebert_wudi \\
        --dataset_type bcb \\
        --output_file results/merged_codebert_wudi_bcb.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.amp import autocast
from tqdm import tqdm
from transformers import AutoTokenizer

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
from data.gptclonebench_loader import (
    DEFAULT_DATASET_DIR as GPT_DEFAULT_DATASET_DIR,
    get_dataloader as get_gpt_dataloader,
)
from training.model import CloneDetectionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate clone detection checkpoints on test sets")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--dataset_type", type=str, required=True, choices=["bcb", "clcdsa", "gptclonebench"])
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_pairs", type=int, default=10000,
                        help="GPTCloneBench only: total pairs cap (half true, half false)")
    parser.add_argument("--language", type=str, default="java",
                        help="GPTCloneBench only: source language (java/py/c/cs)")
    return parser.parse_args()


def _resolve_loader(dataset_type: str):
    if dataset_type == "bcb":
        return get_bcb_dataloader, BCB_DEFAULT_DATASET_DIR
    if dataset_type == "gptclonebench":
        return get_gpt_dataloader, GPT_DEFAULT_DATASET_DIR
    return get_clcdsa_dataloader, CLCDSA_DEFAULT_DATASET_DIR


def _resolve_base_model_name(checkpoint_dir: Path) -> str | None:
    clone_cfg_path = checkpoint_dir / "clone_detection_config.json"
    if clone_cfg_path.exists():
        with clone_cfg_path.open("r", encoding="utf-8") as handle:
            cfg = json.load(handle)
        maybe_name = cfg.get("model_name")
        if isinstance(maybe_name, str) and maybe_name:
            return maybe_name

    metrics_path = checkpoint_dir / "best_metrics.json"
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        maybe_name = metrics.get("model_name")
        if isinstance(maybe_name, str) and maybe_name:
            return maybe_name

    config_path = checkpoint_dir / "config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            hf_config = json.load(handle)
        maybe_name = hf_config.get("_name_or_path")
        if isinstance(maybe_name, str) and maybe_name:
            return maybe_name

    return None


def _load_tokenizer_with_fallback(checkpoint_dir: Path):
    try:
        return AutoTokenizer.from_pretrained(str(checkpoint_dir))
    except Exception as exc:
        base_model_name = _resolve_base_model_name(checkpoint_dir)
        if not base_model_name:
            raise RuntimeError(
                "Failed to load tokenizer from checkpoint and no fallback model name was found."
            ) from exc
        print(
            f"Tokenizer load from checkpoint failed, falling back to base model tokenizer: {base_model_name}"
        )
        return AutoTokenizer.from_pretrained(base_model_name)


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
            if loss is not None:
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
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "loss": total_loss / max(dataset_size, 1),
    }


def print_summary_table(rows: List[Dict[str, str]]) -> None:
    headers = ["model_name", "dataset", "f1", "precision", "recall", "accuracy"]
    widths = {header: len(header) for header in headers}

    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row[header])))

    def _fmt(values: Dict[str, str]) -> str:
        return " | ".join(str(values[h]).ljust(widths[h]) for h in headers)

    separator = "-+-".join("-" * widths[h] for h in headers)
    print("\nEvaluation Summary")
    print(_fmt({h: h for h in headers}))
    print(separator)
    for row in rows:
        print(_fmt(row))


def main() -> None:
    args = parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    get_dataloader, default_dataset_dir = _resolve_loader(args.dataset_type)
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else default_dataset_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16

    print(f"Using device: {device}")
    if amp_enabled:
        print(f"Mixed precision inference enabled with dtype: {amp_dtype}")

    tokenizer = _load_tokenizer_with_fallback(checkpoint_dir)
    model = CloneDetectionModel.load_pretrained(checkpoint_dir).to(device)

    loader_kwargs = dict(
        split="test",
        tokenizer=tokenizer,
        dataset_dir=dataset_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        pin_memory=amp_enabled,
    )
    if args.dataset_type == "gptclonebench":
        loader_kwargs["max_pairs"] = args.max_pairs
        loader_kwargs["language"] = args.language
    else:
        loader_kwargs["use_weighted_sampling"] = False
    test_dataset, test_loader = get_dataloader(**loader_kwargs)

    print(f"Evaluating dataset={args.dataset_type} | test_size={len(test_dataset)}")
    if args.dataset_type == "bcb":
        print("Per-type clone breakdown unavailable for CodeXGLUE BCB split; reporting overall metrics.")
    metrics = evaluate(
        model=model,
        dataloader=test_loader,
        dataset_size=len(test_dataset),
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )

    model_name = checkpoint_dir.name
    result = {
        "model_name": model_name,
        "dataset": args.dataset_type,
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "accuracy": metrics["accuracy"],
    }

    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(f"Saved results to: {output_file}")
    print_summary_table(
        [
            {
                "model_name": result["model_name"],
                "dataset": result["dataset"],
                "f1": f"{result['f1']:.4f}",
                "precision": f"{result['precision']:.4f}",
                "recall": f"{result['recall']:.4f}",
                "accuracy": f"{result['accuracy']:.4f}",
            }
        ]
    )


if __name__ == "__main__":
    main()
