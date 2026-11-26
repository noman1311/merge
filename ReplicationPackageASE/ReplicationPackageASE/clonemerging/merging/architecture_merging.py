"""Architecture-level merging via greedy layer selection.

Builds a composite encoder by choosing, at each transformer layer position, the
layer from whichever specialist maximizes validation combined F1. This is used
to merge models that share a tokenizer but not a pre-trained base (for example
CodeBERT and GraphCodeBERT), where parameter merging is less effective.

Example:
    python architecture_merging.py \\
        --checkpoints codebert_bcb graphcodebert_bcb \\
        --base_models microsoft/codebert-base microsoft/graphcodebert-base \\
        --checkpoint_root checkpoints --output_dir checkpoints/franken_greedy
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Subset
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
from evaluation.evaluate_models import evaluate
from merging.representation_alignment import AlignedModelWrapper
from training.model import CloneDetectionModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CloneMerging.ArchGreedy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Greedy layer-level architecture merging of fine-tuned code encoders"
    )
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument(
        "--aligned_checkpoints",
        nargs="+",
        default=None,
        help="Alignment bundle dirs matching --checkpoints; use 'none' where alignment is not needed",
    )
    parser.add_argument("--checkpoint_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="microsoft/codebert-base")
    parser.add_argument("--eval_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bcb_dataset_dir", type=str, default=str(BCB_DEFAULT_DATASET_DIR))
    parser.add_argument("--clcdsa_dataset_dir", type=str, default=str(CLCDSA_DEFAULT_DATASET_DIR))
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_subset_dataloader(
    dataset,
    sample_count: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> Tuple[Subset, DataLoader]:
    total = len(dataset)
    if total == 0:
        raise ValueError("Dataset is empty; cannot build evaluation subset")

    n = min(sample_count, total)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randperm(total, generator=generator)[:n].tolist()

    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return subset, loader


def _get_encoder_layers(encoder_module) -> Sequence[torch.nn.Module]:
    if hasattr(encoder_module, "encoder") and hasattr(encoder_module.encoder, "layer"):
        return encoder_module.encoder.layer
    if (
        hasattr(encoder_module, "roberta")
        and hasattr(encoder_module.roberta, "encoder")
        and hasattr(encoder_module.roberta.encoder, "layer")
    ):
        return encoder_module.roberta.encoder.layer
    raise ValueError("Unsupported encoder structure; expected encoder.layer")


def _extract_layer_pool(
    models: List[CloneDetectionModel],
    aligned_layer_states_by_model: List[Dict[int, Dict[str, torch.Tensor]]],
) -> List[List[Dict[str, Optional[Dict[str, torch.Tensor]]]]]:
    pool: List[List[Dict[str, Optional[Dict[str, torch.Tensor]]]]] = []
    for model_idx, model in enumerate(models):
        layers = _get_encoder_layers(model.encoder)
        model_layers: List[Dict[str, Optional[Dict[str, torch.Tensor]]]] = []
        for layer_idx, layer in enumerate(layers):
            layer_state = {k: v.detach().cpu().clone() for k, v in layer.state_dict().items()}
            aligner_state = aligned_layer_states_by_model[model_idx].get(layer_idx)
            model_layers.append({"layer_state": layer_state, "aligner_state": aligner_state})
        pool.append(model_layers)
    return pool


def _register_layer_aligner_hooks(
    model: CloneDetectionModel,
    layer_aligner_states: Dict[int, Dict[str, torch.Tensor]],
    device: torch.device,
) -> List[torch.utils.hooks.RemovableHandle]:
    if not layer_aligner_states:
        return []

    layers = _get_encoder_layers(model.encoder)
    handles: List[torch.utils.hooks.RemovableHandle] = []

    for layer_idx, aligner_state in layer_aligner_states.items():
        aligner = torch.nn.Linear(model.config.hidden_size, model.config.hidden_size)
        if "weight" in aligner_state and "bias" in aligner_state:
            linear_state = aligner_state
        elif "projection.weight" in aligner_state and "projection.bias" in aligner_state:
            linear_state = {
                "weight": aligner_state["projection.weight"],
                "bias": aligner_state["projection.bias"],
            }
        else:
            raise ValueError(
                f"Unsupported aligner state keys for layer {layer_idx}: {sorted(aligner_state.keys())}"
            )
        aligner.load_state_dict(linear_state)
        aligner = aligner.to(device)
        aligner.eval()

        def _hook(_module, _inputs, output, aligner_module=aligner):
            if isinstance(output, tuple):
                aligned_hidden = aligner_module(output[0])
                return (aligned_hidden, *output[1:])
            return aligner_module(output)

        handles.append(layers[layer_idx].register_forward_hook(_hook))

    return handles


def _evaluate_combined(
    model: CloneDetectionModel,
    bcb_loader,
    bcb_size: int,
    clcdsa_loader,
    clcdsa_size: int,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    layer_aligner_states: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
) -> Dict[str, float]:
    active_aligner_states = {
        idx: state for idx, state in (layer_aligner_states or {}).items() if state is not None
    }
    handles = _register_layer_aligner_hooks(model, active_aligner_states, device=device)
    try:
        bcb_metrics = evaluate(
            model=model,
            dataloader=bcb_loader,
            dataset_size=bcb_size,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        clcdsa_metrics = evaluate(
            model=model,
            dataloader=clcdsa_loader,
            dataset_size=clcdsa_size,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
    finally:
        for handle in handles:
            handle.remove()

    combined_f1 = 0.5 * (bcb_metrics["f1"] + clcdsa_metrics["f1"])
    return {
        "combined_f1": float(combined_f1),
        "bcb_f1": float(bcb_metrics["f1"]),
        "clcdsa_f1": float(clcdsa_metrics["f1"]),
        "bcb_precision": float(bcb_metrics["precision"]),
        "bcb_recall": float(bcb_metrics["recall"]),
        "bcb_accuracy": float(bcb_metrics["accuracy"]),
        "clcdsa_precision": float(clcdsa_metrics["precision"]),
        "clcdsa_recall": float(clcdsa_metrics["recall"]),
        "clcdsa_accuracy": float(clcdsa_metrics["accuracy"]),
    }


def _copy_classifier_from_source(target: CloneDetectionModel, source: CloneDetectionModel) -> None:
    target.classifier.load_state_dict(source.classifier.state_dict())
    target.dropout_prob = source.dropout_prob
    target.dropout = torch.nn.Dropout(source.dropout_prob)


def _snapshot_classifier(model: CloneDetectionModel) -> Tuple[Dict[str, torch.Tensor], float]:
    return (
        {k: v.detach().cpu().clone() for k, v in model.classifier.state_dict().items()},
        model.dropout_prob,
    )


def _restore_classifier(
    model: CloneDetectionModel,
    classifier_state: Dict[str, torch.Tensor],
    dropout_prob: float,
) -> None:
    model.classifier.load_state_dict(classifier_state)
    model.dropout_prob = dropout_prob
    model.dropout = torch.nn.Dropout(dropout_prob)


def _save_and_validate_checkpoint(
    model: CloneDetectionModel,
    tokenizer,
    output_dir: Path,
    bcb_loader,
    bcb_size: int,
    clcdsa_loader,
    clcdsa_size: int,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    layer_aligner_states: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
) -> Dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    active_aligner_states = {
        idx: state for idx, state in (layer_aligner_states or {}).items() if state is not None
    }
    if active_aligner_states:
        torch.save({str(idx): state for idx, state in active_aligner_states.items()}, output_dir / "selected_layer_aligners.pt")

    reloaded = CloneDetectionModel.load_pretrained(output_dir).to(device)
    metrics = _evaluate_combined(
        model=reloaded,
        bcb_loader=bcb_loader,
        bcb_size=bcb_size,
        clcdsa_loader=clcdsa_loader,
        clcdsa_size=clcdsa_size,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        layer_aligner_states=active_aligner_states,
    )
    del reloaded
    return metrics


def _print_source_summary(rows: List[Dict[str, str]]) -> None:
    headers = ["source", "combined_f1", "bcb_f1", "clcdsa_f1"]
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row[header])))

    def _fmt(values: Dict[str, str]) -> str:
        return " | ".join(str(values[h]).ljust(widths[h]) for h in headers)

    separator = "-+-".join("-" * widths[h] for h in headers)
    print("\nSource Model Summary")
    print(_fmt({h: h for h in headers}))
    print(separator)
    for row in rows:
        print(_fmt(row))


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    checkpoint_root = Path(args.checkpoint_root)
    checkpoint_dirs = [checkpoint_root / name for name in args.checkpoints]
    if args.aligned_checkpoints is None:
        aligned_checkpoint_args = ["none"] * len(args.checkpoints)
    else:
        aligned_checkpoint_args = args.aligned_checkpoints
        if len(aligned_checkpoint_args) != len(args.checkpoints):
            raise ValueError("--aligned_checkpoints must have the same length as --checkpoints")

    aligned_checkpoint_dirs: List[Optional[Path]] = []
    for aligned_arg in aligned_checkpoint_args:
        if aligned_arg.lower() == "none":
            aligned_checkpoint_dirs.append(None)
        else:
            aligned_path = Path(aligned_arg)
            if not aligned_path.exists():
                raise FileNotFoundError(f"Aligned checkpoint bundle not found: {aligned_path}")
            aligned_checkpoint_dirs.append(aligned_path)

    for checkpoint_dir in checkpoint_dirs:
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16

    logger.info(f"Using device: {device}")
    if amp_enabled:
        logger.info(f"Mixed precision enabled with dtype: {amp_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    bcb_dataset, _ = get_bcb_dataloader(
        split="valid",
        tokenizer=tokenizer,
        dataset_dir=Path(args.bcb_dataset_dir),
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        pin_memory=amp_enabled,
        use_weighted_sampling=False,
    )
    clcdsa_dataset, _ = get_clcdsa_dataloader(
        split="valid",
        tokenizer=tokenizer,
        dataset_dir=Path(args.clcdsa_dataset_dir),
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        pin_memory=amp_enabled,
        use_weighted_sampling=False,
    )

    bcb_subset, bcb_loader = _build_subset_dataloader(
        dataset=bcb_dataset,
        sample_count=args.eval_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=amp_enabled,
        seed=args.seed,
    )
    clcdsa_subset, clcdsa_loader = _build_subset_dataloader(
        dataset=clcdsa_dataset,
        sample_count=args.eval_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=amp_enabled,
        seed=args.seed + 1,
    )

    logger.info(
        f"Validation subsets: BCB={len(bcb_subset)} samples, CLCDSA={len(clcdsa_subset)} samples"
    )

    source_models: List[CloneDetectionModel] = []
    aligned_layer_states_by_model: List[Dict[int, Dict[str, torch.Tensor]]] = []

    for checkpoint_dir, aligned_dir in zip(checkpoint_dirs, aligned_checkpoint_dirs):
        if aligned_dir is None:
            source_models.append(CloneDetectionModel.load_pretrained(checkpoint_dir).to(device))
            aligned_layer_states_by_model.append({})
        else:
            aligned_wrapper = AlignedModelWrapper.from_pretrained(
                source_checkpoint=checkpoint_dir,
                alignment_dir=aligned_dir,
                device=device,
            )
            source_models.append(aligned_wrapper.source_model.to(device))
            aligned_layer_states_by_model.append(
                {
                    layer_idx: {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in aligned_wrapper.aligners[layer_idx].state_dict().items()
                    }
                    for layer_idx in range(len(aligned_wrapper.aligners))
                }
            )

    source_rows: List[Dict[str, str]] = []
    best_source_idx = 0
    best_source_score = float("-inf")

    for idx, model in enumerate(source_models):
        is_aligned_source = aligned_checkpoint_dirs[idx] is not None
        if is_aligned_source:
            metrics = {
                "combined_f1": float("-inf"),
                "bcb_f1": 0.0,
                "clcdsa_f1": 0.0,
                "bcb_precision": 0.0,
                "bcb_recall": 0.0,
                "bcb_accuracy": 0.0,
                "clcdsa_precision": 0.0,
                "clcdsa_recall": 0.0,
                "clcdsa_accuracy": 0.0,
            }
            logger.info(
                "Skipping standalone source eval for aligned model '%s' (tokenizer mismatch by design)",
                args.checkpoints[idx],
            )
        else:
            metrics = _evaluate_combined(
                model=model,
                bcb_loader=bcb_loader,
                bcb_size=len(bcb_subset),
                clcdsa_loader=clcdsa_loader,
                clcdsa_size=len(clcdsa_subset),
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                layer_aligner_states=aligned_layer_states_by_model[idx],
            )

        if metrics["combined_f1"] > best_source_score:
            best_source_score = metrics["combined_f1"]
            best_source_idx = idx

        source_rows.append(
            {
                "source": args.checkpoints[idx],
                "combined_f1": (
                    "-inf" if metrics["combined_f1"] == float("-inf") else f"{metrics['combined_f1']:.4f}"
                ),
                "bcb_f1": f"{metrics['bcb_f1']:.4f}",
                "clcdsa_f1": f"{metrics['clcdsa_f1']:.4f}",
            }
        )

    _print_source_summary(source_rows)
    logger.info(
        f"Best source model: {args.checkpoints[best_source_idx]} (combined_f1={best_source_score:.4f})"
    )

    franken_model = CloneDetectionModel.load_pretrained(checkpoint_dirs[best_source_idx]).to(device)
    logger.info(f"Initialized franken model from: {args.checkpoints[best_source_idx]}")

    franken_layers = _get_encoder_layers(franken_model.encoder)
    layer_pool = _extract_layer_pool(source_models, aligned_layer_states_by_model)
    num_layers = len(franken_layers)

    for model_layers in layer_pool:
        if len(model_layers) != num_layers:
            raise ValueError("Source models do not have matching layer counts")

    selected_path: List[int] = []
    layer_scores: List[float] = []
    selected_layer_aligners: Dict[int, Optional[Dict[str, torch.Tensor]]] = {
        layer_idx: aligned_layer_states_by_model[best_source_idx].get(layer_idx)
        for layer_idx in range(num_layers)
    }

    logger.info(f"Starting greedy layer refinement over {num_layers} layers")

    baseline_metrics = _evaluate_combined(
        model=franken_model,
        bcb_loader=bcb_loader,
        bcb_size=len(bcb_subset),
        clcdsa_loader=clcdsa_loader,
        clcdsa_size=len(clcdsa_subset),
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        layer_aligner_states=selected_layer_aligners,
    )
    logger.info(
        f"Baseline (best source model) F1: {baseline_metrics['combined_f1']:.4f} "
        f"(BCB={baseline_metrics['bcb_f1']:.4f}, CLCDSA={baseline_metrics['clcdsa_f1']:.4f})"
    )

    for layer_idx in range(num_layers):
        current_model_idx = best_source_idx
        best_metrics = baseline_metrics
        best_model_idx = best_source_idx
        original_layer_state = {
            k: v.detach().cpu().clone() for k, v in franken_layers[layer_idx].state_dict().items()
        }
        best_layer_state = original_layer_state
        best_aligner_state = selected_layer_aligners.get(layer_idx)

        other_models = [idx for idx in range(len(source_models)) if idx != best_source_idx]
        if layer_idx == 0:
            other_models = [idx for idx in other_models if aligned_checkpoint_dirs[idx] is None]

        for try_model_idx in other_models:
            franken_layers[layer_idx].load_state_dict(
                layer_pool[try_model_idx][layer_idx]["layer_state"],
                strict=True,
            )
            trial_aligners = dict(selected_layer_aligners)
            trial_aligners[layer_idx] = layer_pool[try_model_idx][layer_idx]["aligner_state"]

            metrics = _evaluate_combined(
                model=franken_model,
                bcb_loader=bcb_loader,
                bcb_size=len(bcb_subset),
                clcdsa_loader=clcdsa_loader,
                clcdsa_size=len(clcdsa_subset),
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                layer_aligner_states=trial_aligners,
            )

            if metrics["combined_f1"] > best_metrics["combined_f1"]:
                best_metrics = metrics
                best_model_idx = try_model_idx
                best_layer_state = {
                    k: v.detach().cpu().clone() for k, v in franken_layers[layer_idx].state_dict().items()
                }
                best_aligner_state = layer_pool[try_model_idx][layer_idx]["aligner_state"]
            else:
                franken_layers[layer_idx].load_state_dict(best_layer_state, strict=True)

        franken_layers[layer_idx].load_state_dict(best_layer_state, strict=True)
        selected_layer_aligners[layer_idx] = best_aligner_state

        if best_model_idx != current_model_idx:
            logger.info(
                "Layer %d: swapped from %s to %s | combined_f1=%.4f (bcb=%.4f, clcdsa=%.4f)",
                layer_idx,
                args.checkpoints[current_model_idx],
                args.checkpoints[best_model_idx],
                best_metrics["combined_f1"],
                best_metrics["bcb_f1"],
                best_metrics["clcdsa_f1"],
            )
        else:
            logger.info(
                "Layer %d: kept %s | combined_f1=%.4f (bcb=%.4f, clcdsa=%.4f)",
                layer_idx,
                args.checkpoints[current_model_idx],
                best_metrics["combined_f1"],
                best_metrics["bcb_f1"],
                best_metrics["clcdsa_f1"],
            )
        
        selected_path.append(best_model_idx)
        layer_scores.append(float(best_metrics["combined_f1"]))
        baseline_metrics = best_metrics

    output_dir = Path(args.output_dir)
    pre_classifier_metrics = _save_and_validate_checkpoint(
        model=franken_model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        bcb_loader=bcb_loader,
        bcb_size=len(bcb_subset),
        clcdsa_loader=clcdsa_loader,
        clcdsa_size=len(clcdsa_subset),
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        layer_aligner_states=selected_layer_aligners,
    )
    logger.info(
        "Saved post-layer-search checkpoint and validated it | combined_f1=%.4f (bcb=%.4f, clcdsa=%.4f)",
        pre_classifier_metrics["combined_f1"],
        pre_classifier_metrics["bcb_f1"],
        pre_classifier_metrics["clcdsa_f1"],
    )

    used_model_indices = sorted(set(selected_path))
    logger.info(
        "Refining classifier head: trying only models used in final path: %s",
        [args.checkpoints[idx] for idx in used_model_indices],
    )
    best_classifier_idx = best_source_idx
    best_classifier_metrics = pre_classifier_metrics
    original_classifier_state, original_dropout_prob = _snapshot_classifier(franken_model)

    for try_classifier_idx in used_model_indices:
        _restore_classifier(franken_model, original_classifier_state, original_dropout_prob)
        _copy_classifier_from_source(franken_model, source_models[try_classifier_idx])

        metrics = _evaluate_combined(
            model=franken_model,
            bcb_loader=bcb_loader,
            bcb_size=len(bcb_subset),
            clcdsa_loader=clcdsa_loader,
            clcdsa_size=len(clcdsa_subset),
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            layer_aligner_states=selected_layer_aligners,
        )

        if metrics["combined_f1"] > best_classifier_metrics["combined_f1"]:
            best_classifier_metrics = metrics
            best_classifier_idx = try_classifier_idx
            original_classifier_state, original_dropout_prob = _snapshot_classifier(franken_model)
            logger.info(
                "Classifier candidate improved: %s | combined_f1=%.4f",
                args.checkpoints[try_classifier_idx],
                metrics["combined_f1"],
            )
        else:
            _restore_classifier(franken_model, original_classifier_state, original_dropout_prob)

    _restore_classifier(franken_model, original_classifier_state, original_dropout_prob)
    logger.info(
        f"Selected classifier head from: {args.checkpoints[best_classifier_idx]} "
        f"(combined_f1={best_classifier_metrics['combined_f1']:.4f})"
    )

    final_metrics = _save_and_validate_checkpoint(
        model=franken_model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        bcb_loader=bcb_loader,
        bcb_size=len(bcb_subset),
        clcdsa_loader=clcdsa_loader,
        clcdsa_size=len(clcdsa_subset),
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        layer_aligner_states=selected_layer_aligners,
    )
    logger.info(
        "Final saved checkpoint validated | combined_f1=%.4f (bcb=%.4f, clcdsa=%.4f)",
        final_metrics["combined_f1"],
        final_metrics["bcb_f1"],
        final_metrics["clcdsa_f1"],
    )

    merge_metadata = {
        "method": "greedy_layer_refinement",
        "base_model": args.base_model,
        "checkpoints": args.checkpoints,
        "aligned_checkpoints": [str(path) if path is not None else "none" for path in aligned_checkpoint_dirs],
        "initial_model_index": best_source_idx,
        "initial_model_name": args.checkpoints[best_source_idx],
        "selected_path_indices": selected_path,
        "selected_path_names": [args.checkpoints[idx] for idx in selected_path],
        "selected_path_has_alignment": [selected_layer_aligners[idx] is not None for idx in range(num_layers)],
        "layer_scores_combined_f1": layer_scores,
        "classifier_source_index": best_classifier_idx,
        "classifier_source_name": args.checkpoints[best_classifier_idx],
        "eval_samples_per_dataset": args.eval_samples,
        "valid_subset_sizes": {
            "bcb": len(bcb_subset),
            "clcdsa": len(clcdsa_subset),
        },
        "final_subset_metrics": final_metrics,
        "seed": args.seed,
    }

    with (output_dir / "architecture_merge_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(merge_metadata, handle, indent=2)

    print("\nGreedy Layer Refinement Summary")
    print(f"Output: {output_dir}")
    print(f"Initial model: {args.checkpoints[best_source_idx]}")
    print(f"Classifier head from: {args.checkpoints[best_classifier_idx]}")
    print("Layer selections (swaps from initial model):")
    for layer_idx, model_idx in enumerate(selected_path):
        if model_idx != best_source_idx:
            print(f"  layer_{layer_idx:02d}: {args.checkpoints[model_idx]} (swapped)")
        else:
            print(f"  layer_{layer_idx:02d}: {args.checkpoints[model_idx]} (kept)")
    print(
        "Final subset combined F1: "
        f"{final_metrics['combined_f1']:.4f} "
        f"(BCB={final_metrics['bcb_f1']:.4f}, CLCDSA={final_metrics['clcdsa_f1']:.4f})"
    )


if __name__ == "__main__":
    main()
