"""WUDI-Merging: interference-free parameter merging (Cheng et al., ICML 2025).

For each linear layer, the task vectors approximately span a linear subspace of
the input. WUDI optimizes the merged task vector to minimize cross-task
interference within that subspace, without any training data or rescaling
coefficients. Non-linear-layer parameters (LayerNorm, embeddings) fall back to
weighted averaging, since the linear-subspace argument does not apply to them.

Example:
    python wudi_merging.py \\
        --base_model microsoft/unixcoder-base \\
        --checkpoints unixcoder_bcb unixcoder_clcdsa \\
        --checkpoint_root checkpoints \\
        --base_models microsoft/unixcoder-base microsoft/unixcoder-base \\
        --weights 1.0 1.0 --num_steps 300 --lr 1e-5 --seed 42 \\
        --output_dir checkpoints/merged_unixcoder_wudi
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModel, AutoTokenizer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CloneMerging.WUDI")


# Suffixes that identify linear-layer weight matrices. These are the parameters
# WUDI applies its interference-minimization to. Everything else (LayerNorm,
# embeddings, biases) is handled by weighted averaging.
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
        description="WUDI-Merging on real fine-tuned checkpoints (ICML 2025)"
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
        help="Initialization weights for WUDI (also used as fallback averaging "
        "for non-linear-layer parameters), same order as --checkpoints",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=300,
        help="Adam steps per linear layer (default 300, matches WUDI RoBERTa impl)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Adam learning rate (default 1e-5, matches WUDI RoBERTa impl)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda or cpu (default: cuda if available, else cpu)",
    )
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


def _is_linear_weight(name: str, tensor: torch.Tensor) -> bool:
    """Identify linear-layer weight matrices that WUDI should optimize."""
    if not name.endswith(LINEAR_WEIGHT_SUFFIXES):
        return False
    # 2D tensor of the form [out_features, in_features].
    if tensor.ndim != 2:
        return False
    # Exclude embeddings, layer norms, and other non-linear-layer params.
    for hint in NON_LINEAR_NAME_HINTS:
        if hint in name:
            return False
    return True


def _wudi_optimize_layer(
    task_deltas: List[torch.Tensor],
    weights: List[float],
    num_steps: int,
    lr: float,
    device: torch.device,
) -> torch.Tensor:
    """
    WUDI optimization for a single linear layer.

    Args:
        task_deltas: List of K task vectors for this layer, each of shape
            [out_features, in_features]. Sitting on CPU.
        weights: Normalized initialization weights, length K.
        num_steps: Adam optimization steps.
        lr: Adam learning rate.
        device: torch device.

    Returns:
        Merged task delta tensor, same shape as inputs, on CPU.

    The objective follows Eq. 7 of Cheng et al. 2025: for each task vector
    tau_i, minimize the projection of (tau_merged - tau_i) onto tau_i's
    subspace, normalized by ||tau_i||_F^2 to balance across tasks of different
    magnitudes.
    """
    if len(task_deltas) == 0:
        raise ValueError("WUDI optimization requires at least one task vector")

    # Move task deltas to device. These are the "ground truth" tau_i tensors.
    tau_list = [t.to(device=device, dtype=torch.float32) for t in task_deltas]
    w_tensor = torch.tensor(weights, device=device, dtype=torch.float32)

    # Initialize merged task vector as the weighted sum of task vectors. This is
    # the same starting point as plain task arithmetic. WUDI then refines it.
    merged = torch.zeros_like(tau_list[0])
    for w, tau in zip(weights, tau_list):
        merged = merged + w * tau
    merged = merged.detach().clone().requires_grad_(True)

    optimizer = torch.optim.Adam([merged], lr=lr)

    # Precompute per-task normalization constants ||tau_i||_F^2.
    tau_norms_sq = [tau.pow(2).sum().clamp(min=1e-12) for tau in tau_list]

    for step in range(num_steps):
        optimizer.zero_grad()
        loss = torch.zeros((), device=device, dtype=torch.float32)
        for tau, tau_norm_sq, w in zip(tau_list, tau_norms_sq, w_tensor):
            # Residual: what merged carries beyond tau_i.
            residual = merged - tau
            # Project residual onto tau_i's row space (linear-subspace
            # interference, per WUDI Eq. 7).
            # residual @ tau^T gives interference matrix; Frobenius norm
            # squared, normalized by ||tau||_F^2.
            projection = residual @ tau.transpose(-1, -2)
            layer_loss = projection.pow(2).sum() / tau_norm_sq
            loss = loss + w * layer_loss

        loss.backward()
        optimizer.step()

    merged_cpu = merged.detach().to(device="cpu", dtype=task_deltas[0].dtype)
    return merged_cpu


def _weighted_average(
    tensors: List[torch.Tensor],
    weights: List[float],
) -> torch.Tensor:
    out = torch.zeros_like(tensors[0])
    for w, t in zip(weights, tensors):
        out = out + w * t
    return out


def _merge_task_vectors_wudi(
    base_state: Dict[str, torch.Tensor],
    task_vectors: List[Dict[str, torch.Tensor]],
    weights: List[float],
    num_steps: int,
    lr: float,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Apply WUDI to all linear-layer task deltas, fall back to weighted
    averaging for non-linear-layer parameters (LayerNorm, embeddings, biases).
    """
    merged_state: Dict[str, torch.Tensor] = {
        name: tensor.detach().cpu().clone() for name, tensor in base_state.items()
    }

    all_names = sorted({name for task in task_vectors for name in task.keys()})

    n_linear = 0
    n_averaged = 0
    n_skipped = 0

    logger.info(
        f"WUDI merging {len(all_names)} parameter tensors "
        f"(num_steps={num_steps}, lr={lr}, device={device})"
    )

    for idx, name in enumerate(all_names):
        if name not in merged_state:
            n_skipped += 1
            continue

        # Collect this tensor across all task vectors, zero-fill if missing.
        deltas: List[torch.Tensor] = []
        for task in task_vectors:
            if name in task:
                deltas.append(task[name])
            else:
                deltas.append(torch.zeros_like(merged_state[name]))

        reference = merged_state[name]
        if _is_linear_weight(name, reference):
            merged_delta = _wudi_optimize_layer(
                task_deltas=deltas,
                weights=weights,
                num_steps=num_steps,
                lr=lr,
                device=device,
            )
            n_linear += 1
        else:
            merged_delta = _weighted_average(deltas, weights)
            n_averaged += 1

        merged_state[name] = merged_state[name] + merged_delta

        if (idx + 1) % 25 == 0:
            logger.info(
                f"Merged {idx + 1}/{len(all_names)} tensors "
                f"(wudi-linear={n_linear}, averaged={n_averaged}, skipped={n_skipped})"
            )

    logger.info(
        f"WUDI merging done. Linear layers (WUDI): {n_linear}, "
        f"non-linear (averaged): {n_averaged}, skipped: {n_skipped}"
    )
    return merged_state


def _merge_classifier_heads_wudi(
    classifier_states: List[Dict[str, torch.Tensor]],
    weights: List[float],
    num_steps: int,
    lr: float,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Apply WUDI to classifier heads. The classifier weight matrices are linear,
    so we apply WUDI optimization directly. Biases (1D) fall back to weighted
    averaging.

    Note: unlike the encoder, classifier heads are NOT computed as deltas from
    a base since there is no shared base classifier. We treat each classifier
    state itself as the "task vector" and optimize the merged classifier to
    minimize cross-task interference between them.
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
            # Linear weight: apply WUDI.
            merged[name] = _wudi_optimize_layer(
                task_deltas=tensors,
                weights=weights,
                num_steps=num_steps,
                lr=lr,
                device=device,
            )
        else:
            # Bias or other 1D parameter: weighted average.
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


def _resolve_device(requested: str = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()

    checkpoints = args.checkpoints
    base_models = args.base_models
    if len(checkpoints) != len(base_models):
        raise ValueError("--checkpoints and --base_models must have equal lengths")
    if len(args.weights) != len(checkpoints):
        raise ValueError("--weights length must match number of checkpoints")

    weights = _normalize_weights(args.weights)
    device = _resolve_device(args.device)

    # Seed for any incidental randomness (Adam itself is deterministic given
    # fixed initialization, but we set the seed for consistency with the rest
    # of the pipeline).
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    checkpoint_root = Path(args.checkpoint_root)
    checkpoint_dirs = [checkpoint_root / name for name in checkpoints]
    for checkpoint_dir in checkpoint_dirs:
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

    logger.info("=" * 72)
    logger.info("WUDI-Merging (ICML 2025)")
    logger.info("=" * 72)
    logger.info(f"Target base model: {args.base_model}")
    logger.info(f"Checkpoints: {checkpoints}")
    logger.info(f"Source base models: {base_models}")
    logger.info(f"Normalized weights: {[round(w, 4) for w in weights]}")
    logger.info(f"Steps: {args.num_steps}, lr: {args.lr}, device: {device}, seed: {args.seed}")

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

    merged_encoder_state = _merge_task_vectors_wudi(
        base_state=target_base_state,
        task_vectors=task_vectors,
        weights=weights,
        num_steps=args.num_steps,
        lr=args.lr,
        device=device,
    )

    merged_classifier_state = _merge_classifier_heads_wudi(
        classifier_states=classifier_states,
        weights=weights,
        num_steps=args.num_steps,
        lr=args.lr,
        device=device,
    )

    output_dir = Path(args.output_dir)
    metadata = {
        "method": "WUDI-Merging",
        "reference": "Cheng et al., ICML 2025, arXiv:2503.08099",
        "base_model": args.base_model,
        "checkpoints": checkpoints,
        "source_base_models": base_models,
        "weights": weights,
        "num_steps": args.num_steps,
        "lr": args.lr,
        "device": str(device),
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