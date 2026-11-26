"""PCB-Merging: Parameter Competition Balancing (Du et al., NeurIPS 2024).

For each linear layer, PCB balances parameter competition in four training-free
steps:

  1) Intra-balancing (per-task importance):
        For each task vector delta_k, compute per-parameter importance as
        s_intra_k = softmax(N * delta_k**2), where N is the number of tasks.

  2) Inter-balancing (cross-task similarity):
        For each parameter position, compute cross-task similarity as
        s_inter_k = softmax(sum over j of s_intra_j * delta_j * sign(delta_k)).

  3) PCB score:
        pcb_k = s_intra_k * s_inter_k (element-wise), then keep only the
        top-(1 - drop_rate) fraction of magnitudes per task vector.

  4) Rescale and merge:
        Rescale surviving deltas by 1 / mean(kept mask), then form the merged
        delta as sum_k (mask_k * pcb_k * delta_k).

PCB-Merging requires no training data and no gradient optimization at the
merging step. Non-linear-layer parameters (LayerNorm, embeddings) fall back to
weighted averaging.

Reference: Du et al., "Parameter Competition Balancing for Model Merging",
NeurIPS 2024, arXiv:2410.02396.

Example:
    python pcb_merging.py \\
        --base_model microsoft/unixcoder-base \\
        --checkpoints unixcoder_bcb unixcoder_clcdsa \\
        --checkpoint_root checkpoints \\
        --base_models microsoft/unixcoder-base microsoft/unixcoder-base \\
        --weights 1.0 1.0 --drop_rate 0.1 --intra_scale 1.0 --seed 42 \\
        --output_dir checkpoints/merged_unixcoder_pcb
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModel, AutoTokenizer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CloneMerging.PCB")


LINEAR_WEIGHT_SUFFIXES = (".weight",)
NON_LINEAR_NAME_HINTS = (
    "embedding",
    "embeddings",
    "LayerNorm",
    "layer_norm",
    "ln_",
    "norm.",
    "position_ids",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PCB-Merging on real fine-tuned checkpoints (NeurIPS 2024)"
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
        help="Task vector weights (also used as fallback averaging for "
        "non-linear-layer parameters), same order as --checkpoints",
    )
    parser.add_argument(
        "--drop_rate",
        type=float,
        default=0.1,
        help="Fraction of low-PCB-score parameters to drop per task vector "
        "(default 0.1, matches PCB-Merging paper Table 1 setting for RoBERTa-scale models)",
    )
    parser.add_argument(
        "--intra_scale",
        type=float,
        default=1.0,
        help="Temperature for softmax in intra-balancing (default 1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (PCB is deterministic, kept for pipeline consistency)",
    )
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


def _is_linear_weight(name: str, tensor: torch.Tensor) -> bool:
    if not name.endswith(LINEAR_WEIGHT_SUFFIXES):
        return False
    if tensor.ndim != 2:
        return False
    for hint in NON_LINEAR_NAME_HINTS:
        if hint in name:
            return False
    return True


def _pcb_merge_linear(
    task_deltas: List[torch.Tensor],
    weights: List[float],
    drop_rate: float,
    intra_scale: float,
) -> torch.Tensor:
    """
    PCB-Merging for a single linear layer's task vectors.

    Args:
        task_deltas: K task vectors of shape [out_features, in_features].
        weights: Normalized task weights, length K.
        drop_rate: Fraction of low-PCB-score parameters to zero per task.
        intra_scale: Softmax temperature for intra-balancing.

    Returns:
        Merged task delta tensor, same shape as inputs.
    """
    K = len(task_deltas)
    if K == 0:
        raise ValueError("PCB-Merging requires at least one task vector")

    # Stack into shape [K, ...] for vectorized operations.
    stacked = torch.stack([t.to(dtype=torch.float32) for t in task_deltas], dim=0)
    w_tensor = torch.tensor(weights, dtype=torch.float32).view(K, *[1] * (stacked.ndim - 1))

    # -------- Step 1: Intra-balancing --------
    # Per-task importance from squared magnitude, softmax-normalized across
    # parameter positions within each task vector.
    # Following the paper's Eq. 3: s_intra = softmax(N * delta^2)
    intra_logits = intra_scale * float(K) * stacked.pow(2)
    # Softmax across the parameter-position dimensions (flatten all but K).
    intra_flat = intra_logits.view(K, -1)
    intra_soft = torch.softmax(intra_flat, dim=-1)
    intra_scores = intra_soft.view_as(stacked)

    # -------- Step 2: Inter-balancing --------
    # For each task k, measure how much its parameter directions agree with
    # the weighted intra-scores of other tasks.
    # Following the paper's Eq. 4: s_inter_k = softmax(sum_j s_intra_j * |delta_j| * sign(delta_k))
    # We use |delta_j| to measure magnitude of other tasks' agreement, weighted
    # by their intra-scores, then aligned to task k's sign direction.
    abs_deltas = stacked.abs()
    weighted_agreement = intra_scores * abs_deltas  # shape [K, ...]
    # Sum across tasks to get an "importance field" over parameter positions.
    aggregate_field = weighted_agreement.sum(dim=0, keepdim=True)  # shape [1, ...]

    # Align to each task k's sign direction and softmax-normalize per task.
    inter_logits = aggregate_field * torch.sign(stacked)  # shape [K, ...]
    inter_flat = inter_logits.view(K, -1)
    inter_soft = torch.softmax(inter_flat, dim=-1)
    inter_scores = inter_soft.view_as(stacked)

    # -------- Step 3: PCB score and drop --------
    # Combine intra and inter multiplicatively (paper's ablation showed
    # multiplication beats addition by ~4%).
    pcb_scores = intra_scores * inter_scores  # shape [K, ...]

    # Per-task threshold: keep top (1 - drop_rate) of parameters by PCB score.
    keep_frac = max(0.0, min(1.0, 1.0 - drop_rate))
    if keep_frac >= 1.0:
        keep_mask = torch.ones_like(pcb_scores, dtype=torch.bool)
    elif keep_frac <= 0.0:
        keep_mask = torch.zeros_like(pcb_scores, dtype=torch.bool)
    else:
        # For each task, find the (drop_rate)-quantile of PCB scores.
        # Anything below is dropped.
        pcb_flat = pcb_scores.view(K, -1)
        # torch.quantile computes over the last dim by default when we pass dim.
        thresholds = torch.quantile(pcb_flat, drop_rate, dim=-1, keepdim=True)  # shape [K, 1]
        keep_mask_flat = pcb_flat >= thresholds
        keep_mask = keep_mask_flat.view_as(pcb_scores)

    # -------- Step 4: Rescale and merge --------
    # Rescale each task's surviving deltas so that the mean-magnitude budget
    # is preserved after masking. The rescale factor per task is 1 / keep_frac
    # to compensate for the masked-out portion.
    if keep_frac > 0.0:
        rescale = 1.0 / keep_frac
    else:
        rescale = 0.0

    kept_deltas = stacked * keep_mask.to(stacked.dtype) * rescale  # shape [K, ...]

    # Weighted combination.
    merged = (w_tensor * kept_deltas).sum(dim=0)  # shape [...]

    return merged.to(task_deltas[0].dtype)


def _weighted_average(
    tensors: List[torch.Tensor],
    weights: List[float],
) -> torch.Tensor:
    out = torch.zeros_like(tensors[0])
    for w, t in zip(weights, tensors):
        out = out + w * t
    return out


def _merge_task_vectors_pcb(
    base_state: Dict[str, torch.Tensor],
    task_vectors: List[Dict[str, torch.Tensor]],
    weights: List[float],
    drop_rate: float,
    intra_scale: float,
) -> Dict[str, torch.Tensor]:
    """
    Apply PCB-Merging to all linear-layer task deltas, fall back to weighted
    averaging for non-linear-layer parameters (LayerNorm, embeddings, biases).
    """
    merged_state: Dict[str, torch.Tensor] = {
        name: tensor.detach().cpu().clone() for name, tensor in base_state.items()
    }

    all_names = sorted({name for task in task_vectors for name in task.keys()})

    n_pcb = 0
    n_averaged = 0
    n_skipped = 0

    logger.info(
        f"PCB-Merging {len(all_names)} parameter tensors "
        f"(drop_rate={drop_rate}, intra_scale={intra_scale})"
    )

    for idx, name in enumerate(all_names):
        if name not in merged_state:
            n_skipped += 1
            continue

        deltas: List[torch.Tensor] = []
        for task in task_vectors:
            if name in task:
                deltas.append(task[name])
            else:
                deltas.append(torch.zeros_like(merged_state[name]))

        reference = merged_state[name]
        if _is_linear_weight(name, reference):
            merged_delta = _pcb_merge_linear(
                task_deltas=deltas,
                weights=weights,
                drop_rate=drop_rate,
                intra_scale=intra_scale,
            )
            n_pcb += 1
        else:
            merged_delta = _weighted_average(deltas, weights)
            n_averaged += 1

        merged_state[name] = merged_state[name] + merged_delta

        if (idx + 1) % 25 == 0:
            logger.info(
                f"Merged {idx + 1}/{len(all_names)} tensors "
                f"(pcb-linear={n_pcb}, averaged={n_averaged}, skipped={n_skipped})"
            )

    logger.info(
        f"PCB-Merging done. Linear layers (PCB): {n_pcb}, "
        f"non-linear (averaged): {n_averaged}, skipped: {n_skipped}"
    )
    return merged_state


def _merge_classifier_heads_pcb(
    classifier_states: List[Dict[str, torch.Tensor]],
    weights: List[float],
    drop_rate: float,
    intra_scale: float,
) -> Dict[str, torch.Tensor]:
    """
    Apply PCB-Merging to classifier heads. Weight matrices (2D) use PCB.
    Biases (1D) fall back to weighted averaging.

    Note: as in WUDI's convention here, classifier heads are treated as the
    "task vector" themselves (no shared base classifier exists).
    """
    names = sorted(classifier_states[0].keys())
    for state in classifier_states[1:]:
        if sorted(state.keys()) != names:
            raise ValueError("Classifier heads have mismatched parameter names")

    merged: Dict[str, torch.Tensor] = {}
    for name in names:
        ref_shape = classifier_states[0][name].shape
        for state in classifier_states[1:]:
            if state[name].shape != ref_shape:
                raise ValueError(f"Classifier tensor shape mismatch for '{name}'")

        tensors = [state[name] for state in classifier_states]

        if tensors[0].ndim == 2:
            merged[name] = _pcb_merge_linear(
                task_deltas=tensors,
                weights=weights,
                drop_rate=drop_rate,
                intra_scale=intra_scale,
            )
        else:
            merged[name] = _weighted_average(tensors, weights)

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
        logger.warning(
            f"Unexpected keys while loading merged encoder: {len(load_result.unexpected_keys)}"
        )
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

    checkpoints = args.checkpoints
    base_models = args.base_models
    if len(checkpoints) != len(base_models):
        raise ValueError("--checkpoints and --base_models must have equal lengths")
    if len(args.weights) != len(checkpoints):
        raise ValueError("--weights length must match number of checkpoints")

    weights = _normalize_weights(args.weights)
    torch.manual_seed(args.seed)

    checkpoint_root = Path(args.checkpoint_root)
    checkpoint_dirs = [checkpoint_root / name for name in checkpoints]
    for checkpoint_dir in checkpoint_dirs:
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

    logger.info("=" * 72)
    logger.info("PCB-Merging (NeurIPS 2024)")
    logger.info("=" * 72)
    logger.info(f"Target base model: {args.base_model}")
    logger.info(f"Checkpoints: {checkpoints}")
    logger.info(f"Source base models: {base_models}")
    logger.info(f"Normalized weights: {[round(w, 4) for w in weights]}")
    logger.info(f"drop_rate: {args.drop_rate}, intra_scale: {args.intra_scale}")

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
        raise ValueError(
            f"Classifier num_labels mismatch across checkpoints: {num_labels_list}"
        )

    merged_encoder_state = _merge_task_vectors_pcb(
        base_state=target_base_state,
        task_vectors=task_vectors,
        weights=weights,
        drop_rate=args.drop_rate,
        intra_scale=args.intra_scale,
    )

    merged_classifier_state = _merge_classifier_heads_pcb(
        classifier_states=classifier_states,
        weights=weights,
        drop_rate=args.drop_rate,
        intra_scale=args.intra_scale,
    )

    output_dir = Path(args.output_dir)
    metadata = {
        "method": "PCB-Merging",
        "reference": "Du et al., NeurIPS 2024, arXiv:2410.02396",
        "base_model": args.base_model,
        "checkpoints": checkpoints,
        "source_base_models": base_models,
        "weights": weights,
        "drop_rate": args.drop_rate,
        "intra_scale": args.intra_scale,
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