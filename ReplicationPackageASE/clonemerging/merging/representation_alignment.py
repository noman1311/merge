"""Linear layer-wise representation aligners between fine-tuned code encoders.

Trains a small linear projection per transformer layer that maps one model's
hidden states into another model's representation space. These aligners are used
when attempting architecture merging across models with different tokenizers.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bigclonebench_loader import DEFAULT_DATASET_DIR as BCB_DEFAULT_DATASET_DIR
from data.bigclonebench_loader import load_code_map, parse_split_file
from training.model import CloneDetectionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train linear layer-wise representation aligners between fine-tuned code models"
    )
    parser.add_argument("--source_checkpoint", type=str, required=True)
    parser.add_argument("--source_base", type=str, required=True)
    parser.add_argument("--target_checkpoint", type=str, required=True)
    parser.add_argument("--target_base", type=str, required=True)
    parser.add_argument("--data_source", type=str, default="bcb_train", choices=["bcb_train"])
    parser.add_argument("--bcb_dataset_dir", type=str, default=str(BCB_DEFAULT_DATASET_DIR))
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--coral_weight", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_checkpoint_model_name(checkpoint_dir: Path, fallback: str) -> str:
    metrics_path = checkpoint_dir / "best_metrics.json"
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        model_name = metrics.get("model_name")
        if isinstance(model_name, str) and model_name:
            return model_name

    config_path = checkpoint_dir / "config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        model_name = config.get("_name_or_path")
        if isinstance(model_name, str) and model_name:
            return model_name

    return fallback


def _load_tokenizer(checkpoint_dir: Path, fallback_model_name: str):
    try:
        return AutoTokenizer.from_pretrained(str(checkpoint_dir))
    except Exception:
        return AutoTokenizer.from_pretrained(
            _resolve_checkpoint_model_name(checkpoint_dir, fallback_model_name)
        )


def _mean_pool(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
    pooled = (hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return pooled / denom


def _get_bcb_train_code_samples(dataset_dir: Path, num_samples: int, seed: int) -> List[str]:
    code_map = load_code_map(dataset_dir)
    train_samples = parse_split_file(dataset_dir / "train.txt")

    unique_ids: List[str] = []
    seen = set()
    for _, id1, id2, _, _, _ in train_samples:
        for function_id in (id1, id2):
            if function_id in code_map and function_id not in seen:
                seen.add(function_id)
                unique_ids.append(function_id)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    shuffled_indices = torch.randperm(len(unique_ids), generator=generator).tolist()
    selected_ids = [unique_ids[idx] for idx in shuffled_indices[: min(num_samples, len(unique_ids))]]
    return [code_map[function_id] for function_id in selected_ids]


class AlignmentDataCollector:
    def __init__(
        self,
        source_model: CloneDetectionModel,
        source_tokenizer,
        target_model: CloneDetectionModel,
        target_tokenizer,
        max_length: int = 512,
        device: Optional[torch.device] = None,
    ) -> None:
        self.source_model = source_model.eval()
        self.source_tokenizer = source_tokenizer
        self.target_model = target_model.eval()
        self.target_tokenizer = target_tokenizer
        self.max_length = max_length
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.source_model.to(self.device)
        self.target_model.to(self.device)

    def collect(
        self,
        code_samples: Sequence[str],
        batch_size: int,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        source_layers: List[List[torch.Tensor]] = [[] for _ in range(12)]
        target_layers: List[List[torch.Tensor]] = [[] for _ in range(12)]

        with torch.no_grad():
            for start in tqdm(range(0, len(code_samples), batch_size), desc="Collecting hidden states"):
                batch_codes = list(code_samples[start : start + batch_size])

                source_batch = self.source_tokenizer(
                    batch_codes,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                target_batch = self.target_tokenizer(
                    batch_codes,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_attention_mask=True,
                    return_tensors="pt",
                )

                source_batch = {key: value.to(self.device) for key, value in source_batch.items()}
                target_batch = {key: value.to(self.device) for key, value in target_batch.items()}

                source_outputs = self.source_model.encoder(
                    input_ids=source_batch["input_ids"],
                    attention_mask=source_batch["attention_mask"],
                    output_hidden_states=True,
                    return_dict=True,
                )
                target_outputs = self.target_model.encoder(
                    input_ids=target_batch["input_ids"],
                    attention_mask=target_batch["attention_mask"],
                    output_hidden_states=True,
                    return_dict=True,
                )

                for layer_idx in range(12):
                    source_repr = _mean_pool(
                        source_outputs.hidden_states[layer_idx + 1],
                        source_batch["attention_mask"],
                    )
                    target_repr = _mean_pool(
                        target_outputs.hidden_states[layer_idx + 1],
                        target_batch["attention_mask"],
                    )
                    source_layers[layer_idx].append(source_repr.detach().cpu())
                    target_layers[layer_idx].append(target_repr.detach().cpu())

        return [
            (torch.cat(source_layers[layer_idx], dim=0), torch.cat(target_layers[layer_idx], dim=0))
            for layer_idx in range(12)
        ]


class LayerAligner(nn.Module):
    def __init__(self, hidden_size: int = 768) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size)
        nn.init.eye_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.projection(hidden_states)


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if source.size(0) < 2 or target.size(0) < 2:
        return torch.tensor(0.0, device=source.device, dtype=source.dtype)

    dim = source.size(1)
    source_centered = source - source.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)

    cov_source = source_centered.T @ source_centered / max(source.size(0) - 1, 1)
    cov_target = target_centered.T @ target_centered / max(target.size(0) - 1, 1)
    return torch.norm(cov_source - cov_target, p="fro").pow(2) / (4 * dim * dim)


def train_layer_aligner(
    layer_idx: int,
    source_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    hidden_size: int,
    batch_size: int,
    epochs: int,
    lr: float,
    coral_weight: float,
    device: torch.device,
    num_workers: int,
) -> Tuple[LayerAligner, Dict[str, float]]:
    aligner = LayerAligner(hidden_size=hidden_size).to(device)
    optimizer = torch.optim.AdamW(aligner.parameters(), lr=lr)
    dataset = TensorDataset(source_tensor, target_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    final_metrics = {"mse_loss": 0.0, "coral_loss": 0.0, "total_loss": 0.0}

    for epoch in range(epochs):
        aligner.train()
        loss_sum = 0.0
        mse_sum = 0.0
        coral_sum = 0.0
        sample_count = 0

        for batch_source, batch_target in loader:
            batch_source = batch_source.to(device)
            batch_target = batch_target.to(device)

            aligned = aligner(batch_source)
            mse = F.mse_loss(aligned, batch_target)
            coral = coral_loss(aligned, batch_target) if coral_weight > 0 else torch.tensor(0.0, device=device)
            loss = mse + coral_weight * coral

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_n = batch_source.size(0)
            sample_count += batch_n
            loss_sum += loss.item() * batch_n
            mse_sum += mse.item() * batch_n
            coral_sum += coral.item() * batch_n

        final_metrics = {
            "mse_loss": mse_sum / max(sample_count, 1),
            "coral_loss": coral_sum / max(sample_count, 1),
            "total_loss": loss_sum / max(sample_count, 1),
        }
        print(
            f"layer={layer_idx:02d} epoch={epoch + 1}/{epochs} "
            f"mse={final_metrics['mse_loss']:.6f} coral={final_metrics['coral_loss']:.6f} "
            f"total={final_metrics['total_loss']:.6f}"
        )

    return aligner.cpu(), final_metrics


def _resolve_encoder_stack(encoder_model):
    if hasattr(encoder_model, "embeddings") and hasattr(encoder_model, "encoder"):
        return encoder_model
    if hasattr(encoder_model, "roberta") and hasattr(encoder_model.roberta, "encoder"):
        return encoder_model.roberta
    raise ValueError("Unsupported encoder model structure for aligned wrapper")


class AlignedModelWrapper(nn.Module):
    def __init__(
        self,
        source_model: CloneDetectionModel,
        aligners: Sequence[LayerAligner],
        apply_layers: Optional[Iterable[int]] = None,
    ) -> None:
        super().__init__()
        self.source_model = source_model
        self.aligners = nn.ModuleList(aligners)
        self.apply_layers = set(apply_layers if apply_layers is not None else range(len(aligners)))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        encoder_model = self.source_model.encoder
        encoder_stack = _resolve_encoder_stack(encoder_model)
        embedding_output = encoder_stack.embeddings(input_ids=input_ids)

        extended_attention_mask = encoder_model.get_extended_attention_mask(
            attention_mask,
            input_ids.shape,
            input_ids.device,
        )

        hidden_states = embedding_output
        for layer_idx, layer_module in enumerate(encoder_stack.encoder.layer):
            layer_outputs = layer_module(hidden_states, attention_mask=extended_attention_mask)
            hidden_states = layer_outputs[0]
            if layer_idx in self.apply_layers:
                hidden_states = self.aligners[layer_idx](hidden_states)

        cls_representation = hidden_states[:, 0, :]
        logits = self.source_model.classifier(self.source_model.dropout(cls_representation))
        result: Dict[str, torch.Tensor] = {"loss": None, "logits": logits}
        if labels is not None:
            result["loss"] = self.source_model.loss_fn(logits, labels)
        return result

    @classmethod
    def from_pretrained(
        cls,
        source_checkpoint: str | Path,
        alignment_dir: str | Path,
        device: Optional[torch.device] = None,
    ) -> "AlignedModelWrapper":
        source_model = CloneDetectionModel.load_pretrained(source_checkpoint)
        alignment_path = Path(alignment_dir) / "alignment_layers.pt"
        if not alignment_path.exists():
            raise FileNotFoundError(f"Missing alignment weights: {alignment_path}")

        payload = torch.load(alignment_path, map_location="cpu", weights_only=True)
        aligners = [LayerAligner(hidden_size=payload["hidden_size"]) for _ in range(payload["num_layers"])]
        for layer_idx, aligner in enumerate(aligners):
            aligner.load_state_dict(payload["layer_state_dicts"][str(layer_idx)])

        wrapper = cls(source_model=source_model, aligners=aligners)
        if device is not None:
            wrapper = wrapper.to(device)
        return wrapper


def save_alignment_bundle(
    output_dir: Path,
    aligners: Sequence[LayerAligner],
    hidden_size: int,
    metadata: Dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "num_layers": len(aligners),
        "hidden_size": hidden_size,
        "layer_state_dicts": {str(idx): aligner.state_dict() for idx, aligner in enumerate(aligners)},
    }
    torch.save(payload, output_dir / "alignment_layers.pt")
    with (output_dir / "alignment_config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    source_checkpoint = Path(args.source_checkpoint)
    target_checkpoint = Path(args.target_checkpoint)

    source_model = CloneDetectionModel.load_pretrained(source_checkpoint)
    target_model = CloneDetectionModel.load_pretrained(target_checkpoint)
    hidden_size = int(source_model.config.hidden_size)
    if hidden_size != int(target_model.config.hidden_size):
        raise ValueError(
            f"Hidden size mismatch: source={source_model.config.hidden_size}, target={target_model.config.hidden_size}"
        )

    source_tokenizer = _load_tokenizer(source_checkpoint, args.source_base)
    target_tokenizer = _load_tokenizer(target_checkpoint, args.target_base)

    if args.data_source == "bcb_train":
        code_samples = _get_bcb_train_code_samples(
            dataset_dir=Path(args.bcb_dataset_dir),
            num_samples=args.num_samples,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unsupported data_source: {args.data_source}")

    print(f"Collected {len(code_samples)} raw code samples from {args.data_source}")

    collector = AlignmentDataCollector(
        source_model=source_model,
        source_tokenizer=source_tokenizer,
        target_model=target_model,
        target_tokenizer=target_tokenizer,
        max_length=args.max_length,
        device=device,
    )
    paired_hidden_states = collector.collect(code_samples=code_samples, batch_size=args.batch_size)

    aligners: List[LayerAligner] = []
    layer_metrics: Dict[str, Dict[str, float]] = {}
    for layer_idx, (source_tensor, target_tensor) in enumerate(paired_hidden_states):
        print(
            f"Training aligner for layer {layer_idx:02d} with {source_tensor.size(0)} samples and hidden_size={source_tensor.size(1)}"
        )
        aligner, metrics = train_layer_aligner(
            layer_idx=layer_idx,
            source_tensor=source_tensor,
            target_tensor=target_tensor,
            hidden_size=hidden_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            coral_weight=args.coral_weight,
            device=device,
            num_workers=args.num_workers,
        )
        aligners.append(aligner)
        layer_metrics[str(layer_idx)] = metrics

    output_dir = Path(args.output_dir)
    metadata = {
        "source_checkpoint": str(source_checkpoint),
        "source_base": args.source_base,
        "target_checkpoint": str(target_checkpoint),
        "target_base": args.target_base,
        "data_source": args.data_source,
        "num_samples": len(code_samples),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "coral_weight": args.coral_weight,
        "hidden_size": hidden_size,
        "num_layers": len(aligners),
        "layer_metrics": layer_metrics,
    }
    save_alignment_bundle(output_dir=output_dir, aligners=aligners, hidden_size=hidden_size, metadata=metadata)

    print(f"Saved alignment bundle to: {output_dir}")
    print("Layer losses:")
    for layer_idx in range(len(aligners)):
        metrics = layer_metrics[str(layer_idx)]
        print(
            f"  layer_{layer_idx:02d}: mse={metrics['mse_loss']:.6f}, "
            f"coral={metrics['coral_loss']:.6f}, total={metrics['total_loss']:.6f}"
        )


if __name__ == "__main__":
    main()