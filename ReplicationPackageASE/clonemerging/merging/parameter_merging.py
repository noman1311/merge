"""Parameter-level merging of fine-tuned clone detection encoders.

Combines two or more specialists that share a pre-trained base by operating on
their task vectors (fine-tuned weights minus base weights). Supports weighted
averaging, TIES, and DARE-TIES (drop-and-rescale followed by TIES). No training
data is used at the merging step.

Example:
    python parameter_merging.py \\
        --base_model microsoft/codebert-base \\
        --checkpoints codebert_bcb codebert_clcdsa \\
        --checkpoint_root checkpoints \\
        --base_models microsoft/codebert-base microsoft/codebert-base \\
        --weights 1.0 1.0 --output_dir checkpoints/merged_dare_ties
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CloneMerging.RealMerge")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DARE-TIES parameter merging on real fine-tuned checkpoints"
    )
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--checkpoint_root", type=str, required=True)
    parser.add_argument(
        "--base_models",
        nargs="+",
        required=True,
        help="One base model per checkpoint, same order as --checkpoints",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        required=True,
        help="Merge weights (lambda), same order as --checkpoints",
    )
    parser.add_argument("--drop_rate", type=float, default=0.3)
    parser.add_argument("--trim_percentile", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def _normalize_weights(weights: List[float]) -> List[float]:
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Sum of --weights must be > 0")
    return [w / total for w in weights]


def _load_encoder_state(model_ref: str) -> Dict[str, torch.Tensor]:
    logger.info(f"Loading encoder: {model_ref}")
    model = AutoModel.from_pretrained(model_ref)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    del model
    return state


def _load_classifier_state(checkpoint_dir: Path) -> Tuple[Dict[str, torch.Tensor], float, int]:
    path = checkpoint_dir / "classifier_head.bin"
    if not path.exists():
        raise FileNotFoundError(f"Missing classifier head: {path}")

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    if "classifier" not in payload:
        raise ValueError(f"Invalid classifier payload in {path}")

    classifier = {
        name: tensor.detach().cpu().clone()
        for name, tensor in payload["classifier"].items()
    }
    dropout = float(payload.get("dropout", 0.1))
    num_labels = int(payload.get("num_labels", classifier.get("bias", torch.zeros(2)).numel()))
    return classifier, dropout, num_labels


def _compute_task_vector(
    finetuned_state: Dict[str, torch.Tensor],
    source_base_state: Dict[str, torch.Tensor],
    target_base_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    delta: Dict[str, torch.Tensor] = {}
    matched = 0
    skipped = 0

    for name, target_tensor in target_base_state.items():
        if name not in finetuned_state or name not in source_base_state:
            skipped += 1
            continue
        ft = finetuned_state[name]
        src_base = source_base_state[name]
        if ft.shape != src_base.shape or ft.shape != target_tensor.shape:
            skipped += 1
            continue

        delta[name] = ft - src_base
        matched += 1

    logger.info(f"Task vector: matched={matched}, skipped={skipped}")
    return delta


def _apply_dare(tensor: torch.Tensor, drop_rate: float, generator: torch.Generator) -> torch.Tensor:
    if drop_rate <= 0:
        return tensor
    if drop_rate >= 1:
        return torch.zeros_like(tensor)

    keep_prob = 1.0 - drop_rate
    mask = (torch.rand(tensor.shape, generator=generator) < keep_prob).to(tensor.dtype)
    return (tensor * mask) / keep_prob


def _trim_tensor(tensor: torch.Tensor, percentile: float) -> torch.Tensor:
    if tensor.numel() == 0:
        return tensor
    if percentile <= 0:
        return tensor

    abs_np = tensor.abs().numpy()
    threshold = float(np.percentile(abs_np, percentile))
    if threshold <= 0:
        return tensor

    mask = (tensor.abs() >= threshold).to(tensor.dtype)
    return tensor * mask


def _ties_merge_tensors(
    tensors: List[torch.Tensor],
    weights: List[float],
    trim_percentile: float,
) -> torch.Tensor:
    trimmed = [_trim_tensor(t, trim_percentile) for t in tensors]

    sign_score = torch.zeros_like(trimmed[0])
    for weight, tensor in zip(weights, trimmed):
        sign_score += weight * tensor.sign()
    elected_sign = sign_score.sign()

    merged = torch.zeros_like(trimmed[0])
    for weight, tensor in zip(weights, trimmed):
        consistent = ((tensor.sign() == elected_sign) & (elected_sign != 0)).to(tensor.dtype)
        merged += weight * tensor * consistent

    return merged


def _merge_task_vectors(
    base_state: Dict[str, torch.Tensor],
    task_vectors: List[Dict[str, torch.Tensor]],
    weights: List[float],
    drop_rate: float,
    trim_percentile: float,
    seed: int,
) -> Dict[str, torch.Tensor]:
    merged_state: Dict[str, torch.Tensor] = {
        name: tensor.detach().cpu().clone() for name, tensor in base_state.items()
    }

    all_names = sorted({name for task in task_vectors for name in task.keys()})
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    logger.info(f"Merging {len(all_names)} parameter tensors with DARE-TIES")

    for idx, name in enumerate(all_names):
        tensors: List[torch.Tensor] = []
        for task in task_vectors:
            if name in task:
                tensors.append(_apply_dare(task[name], drop_rate=drop_rate, generator=generator))
            else:
                tensors.append(torch.zeros_like(merged_state[name]))

        merged_delta = _ties_merge_tensors(
            tensors=tensors,
            weights=weights,
            trim_percentile=trim_percentile,
        )
        merged_state[name] = merged_state[name] + merged_delta

        if (idx + 1) % 50 == 0:
            logger.info(f"Merged {idx + 1}/{len(all_names)} tensors")

    return merged_state


def _merge_classifier_heads(
    classifier_states: List[Dict[str, torch.Tensor]],
    weights: List[float],
    drop_rate: float,
    trim_percentile: float,
    seed: int,
) -> Dict[str, torch.Tensor]:
    names = sorted(classifier_states[0].keys())
    for state in classifier_states[1:]:
        if sorted(state.keys()) != names:
            raise ValueError("Classifier heads have mismatched parameter names")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1337)

    merged: Dict[str, torch.Tensor] = {}
    for name in names:
        ref_shape = classifier_states[0][name].shape
        for state in classifier_states[1:]:
            if state[name].shape != ref_shape:
                raise ValueError(f"Classifier tensor shape mismatch for '{name}'")

        dare_tensors = [
            _apply_dare(state[name], drop_rate=drop_rate, generator=generator)
            for state in classifier_states
        ]
        merged[name] = _ties_merge_tensors(
            tensors=dare_tensors,
            weights=weights,
            trim_percentile=trim_percentile,
        )

    return merged


def _save_outputs(
    output_dir: Path,
    base_model: str,
    merged_encoder_state: Dict[str, torch.Tensor],
    merged_classifier_state: Dict[str, torch.Tensor],
    num_labels: int,
    dropout: float,
    metadata: Dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_model = AutoModel.from_pretrained(base_model)
    load_result = merged_model.load_state_dict(merged_encoder_state, strict=False)
    if load_result.unexpected_keys:
        logger.warning(f"Unexpected keys while loading merged encoder: {len(load_result.unexpected_keys)}")
    merged_model.save_pretrained(str(output_dir))

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(str(output_dir))

    classifier_payload = {
        "classifier": merged_classifier_state,
        "dropout": dropout,
        "num_labels": num_labels,
    }
    torch.save(classifier_payload, output_dir / "classifier_head.bin")

    with (output_dir / "clone_detection_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "num_labels": num_labels,
                "dropout": dropout,
                "model_name": base_model,
            },
            handle,
            indent=2,
        )

    with (output_dir / "merge_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    logger.info(f"Merged checkpoint saved to: {output_dir}")


def main() -> None:
    args = parse_args()

    if not (0.0 <= args.drop_rate < 1.0):
        raise ValueError("--drop_rate must be in [0, 1)")

    checkpoints = args.checkpoints
    base_models = args.base_models
    if len(checkpoints) != len(base_models):
        raise ValueError("--checkpoints and --base_models must have equal lengths")
    if len(args.weights) != len(checkpoints):
        raise ValueError("--weights length must match number of checkpoints")

    weights = _normalize_weights(args.weights)

    checkpoint_root = Path(args.checkpoint_root)
    checkpoint_dirs = [checkpoint_root / name for name in checkpoints]
    for checkpoint_dir in checkpoint_dirs:
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

    logger.info("=" * 72)
    logger.info("Real DARE-TIES Parameter Merging")
    logger.info("=" * 72)
    logger.info(f"Target base model: {args.base_model}")
    logger.info(f"Checkpoints: {checkpoints}")
    logger.info(f"Source base models: {base_models}")
    logger.info(f"Normalized weights: {[round(w, 4) for w in weights]}")
    logger.info(f"Drop rate: {args.drop_rate}, trim percentile: {args.trim_percentile}")

    target_base_state = _load_encoder_state(args.base_model)

    task_vectors: List[Dict[str, torch.Tensor]] = []
    classifier_states: List[Dict[str, torch.Tensor]] = []
    dropouts: List[float] = []
    num_labels_list: List[int] = []

    for checkpoint_dir, source_base in zip(checkpoint_dirs, base_models):
        finetuned_state = _load_encoder_state(str(checkpoint_dir))
        source_base_state = _load_encoder_state(source_base)

        logger.info(f"Computing task vector for checkpoint={checkpoint_dir.name}")
        delta = _compute_task_vector(
            finetuned_state=finetuned_state,
            source_base_state=source_base_state,
            target_base_state=target_base_state,
        )
        task_vectors.append(delta)

        cls_state, dropout, num_labels = _load_classifier_state(checkpoint_dir)
        classifier_states.append(cls_state)
        dropouts.append(dropout)
        num_labels_list.append(num_labels)

        del finetuned_state
        del source_base_state

    if len(set(num_labels_list)) != 1:
        raise ValueError(f"Classifier num_labels mismatch across checkpoints: {num_labels_list}")

    merged_encoder_state = _merge_task_vectors(
        base_state=target_base_state,
        task_vectors=task_vectors,
        weights=weights,
        drop_rate=args.drop_rate,
        trim_percentile=args.trim_percentile,
        seed=args.seed,
    )

    merged_classifier_state = _merge_classifier_heads(
        classifier_states=classifier_states,
        weights=weights,
        drop_rate=args.drop_rate,
        trim_percentile=args.trim_percentile,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    metadata = {
        "base_model": args.base_model,
        "checkpoints": checkpoints,
        "source_base_models": base_models,
        "weights": weights,
        "drop_rate": args.drop_rate,
        "trim_percentile": args.trim_percentile,
        "seed": args.seed,
    }

    _save_outputs(
        output_dir=output_dir,
        base_model=args.base_model,
        merged_encoder_state=merged_encoder_state,
        merged_classifier_state=merged_classifier_state,
        num_labels=num_labels_list[0],
        dropout=float(sum(dropouts) / max(len(dropouts), 1)),
        metadata=metadata,
    )


if __name__ == "__main__":
    main()
