"""Zero-shot evaluation of Qwen2.5-Coder-7B-Instruct on GPTCloneBench.

Runs 4 model replicas in parallel via torch.multiprocessing (one per GPU),
shards the dataset, and merges the per-GPU predictions into a single file.

Usage examples:
    # Pilot (single GPU, 5 pairs, verbose)
    python eval_qwen_gptclonebench.py --pilot-only

    # Full run (4 GPUs, 10K pairs)
    python eval_qwen_gptclonebench.py

    # Custom subset for testing
    python eval_qwen_gptclonebench.py --num-pairs 100 --num-gpus 2 --batch-size 8
"""

import argparse
import datetime as _dt
import json
import logging
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Tuple

# Make the project-local GPTCloneBench loader importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "clonemerging"))

import numpy as np
import torch
import torch.multiprocessing as mp
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from data.gptclonebench_loader import build_pairs  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_DATA_DIR = _PROJECT_ROOT / "data/GPTCloneBench/GPTCloneBench/standalone"
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "llm_baseline/qwen_output"
CONDA_ENV = "clonemerge"

SYSTEM_PROMPT = (
    "You are an expert software engineer specializing in code clone detection."
)
USER_PROMPT_TEMPLATE = (
    "Determine if the following two code snippets are SEMANTIC CLONES - that is, "
    "whether they implement the same functionality, even if they look syntactically "
    "different (different variable names, control structures, or implementation "
    "strategies).\n\n"
    "Code Snippet 1:\n{code1}\n\n"
    "Code Snippet 2:\n{code2}\n\n"
    "Are these two snippets semantic clones (functionally equivalent)? \n"
    'Answer with exactly one word: "Yes" or "No".\n\n'
    "Answer:"
)

CHECKPOINT_INTERVAL = 200
PROGRESS_INTERVAL = 100
MAX_NEW_TOKENS = 10
PROMPT_OVERHEAD_TOKENS = 200  # reserved for system+template+answer


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _configure_logging(log_path: Path, rank: int | None = None) -> logging.Logger:
    prefix = f"[GPU {rank}] " if rank is not None else ""

    class _RankFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            record.msg = f"{prefix}{record.msg}"
            return super().format(record)

    fmt = _RankFormatter("%(asctime)s %(levelname)s %(message)s",
                         datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger() if rank is None else logging.getLogger(f"worker.{rank}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Prompt building & parsing
# ---------------------------------------------------------------------------


def _build_messages(code1: str, code2: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(code1=code1, code2=code2)},
    ]


def _truncate_pair(
    tokenizer,
    code1: str,
    code2: str,
    model_max_length: int,
) -> Tuple[str, str, bool]:
    """Truncate code1 and code2 proportionally from the END if combined token
    count would exceed (model_max_length - PROMPT_OVERHEAD_TOKENS).

    Returns (code1, code2, was_truncated).
    """
    budget = model_max_length - PROMPT_OVERHEAD_TOKENS
    if budget <= 0:
        return code1, code2, False

    ids1 = tokenizer.encode(code1, add_special_tokens=False)
    ids2 = tokenizer.encode(code2, add_special_tokens=False)
    n1, n2 = len(ids1), len(ids2)
    if n1 + n2 <= budget:
        return code1, code2, False

    # Proportional truncation
    total = n1 + n2
    keep1 = max(1, int(budget * (n1 / total)))
    keep2 = max(1, budget - keep1)
    new_code1 = tokenizer.decode(ids1[:keep1], skip_special_tokens=True)
    new_code2 = tokenizer.decode(ids2[:keep2], skip_special_tokens=True)
    return new_code1, new_code2, True


def _parse_answer(text: str) -> int:
    """Return 1 if the response starts with 'yes', else 0."""
    if not text:
        return 0
    stripped = text.strip().lower()
    # Strip any markdown / quotes / punctuation prefix
    while stripped and stripped[0] in '*"\'`(:.- ':
        stripped = stripped[1:]
    return 1 if stripped.startswith("yes") else 0


def _is_parseable(text: str) -> bool:
    if not text:
        return False
    stripped = text.strip().lower()
    while stripped and stripped[0] in '*"\'`(:.- ':
        stripped = stripped[1:]
    return stripped.startswith("yes") or stripped.startswith("no")


# ---------------------------------------------------------------------------
# Inference (single batch)
# ---------------------------------------------------------------------------


def _generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    device: torch.device,
) -> Tuple[list[str], list[int]]:
    """Run batched greedy generation and return (decoded_texts, num_new_tokens)."""
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    ).to(device)

    prompt_lens = enc["attention_mask"].sum(dim=1).tolist()

    with torch.no_grad():
        out = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # With left-padding, the prompt length per row = input_ids.shape[1].
    prompt_seq_len = enc["input_ids"].shape[1]
    new_tokens = out[:, prompt_seq_len:]
    texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    n_new = [int((row != tokenizer.pad_token_id).sum().item()) for row in new_tokens]
    return texts, n_new


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _load_model(model_name: str, device: torch.device, logger: logging.Logger):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading tokenizer for {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    logger.info(f"Loading model {model_name} to {device}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    mem_alloc = torch.cuda.memory_allocated(device) / 1024**3
    mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
    logger.info(
        f"Model loaded. GPU mem allocated={mem_alloc:.2f} GiB, reserved={mem_reserved:.2f} GiB"
    )
    return model, tokenizer


def _load_existing_indices(out_path: Path) -> set[int]:
    if not out_path.exists():
        return set()
    seen = set()
    with out_path.open("r") as f:
        for line in f:
            try:
                obj = json.loads(line)
                seen.add(int(obj["index"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return seen


def worker_main(
    rank: int,
    num_gpus: int,
    model_name: str,
    shard: List[Tuple[int, str, str, int]],
    out_path: str,
    log_path: str,
    batch_size: int,
    resume: bool,
    sentinel_dir: str,
) -> None:
    """Each worker processes its shard on GPU `rank`."""
    logger = _configure_logging(Path(log_path), rank=rank)
    sentinel_dir_p = Path(sentinel_dir)
    out_path_p = Path(out_path)

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    try:
        model, tokenizer = _load_model(model_name, device, logger)
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"OOM loading model on GPU {rank}: {e}")
        (sentinel_dir_p / f"gpu{rank}_crashed.txt").write_text(f"OOM: {e}\n")
        return
    except Exception as e:
        logger.exception(f"Failed to load model on GPU {rank}: {e}")
        (sentinel_dir_p / f"gpu{rank}_crashed.txt").write_text(f"load failed: {e}\n")
        return

    model_max = getattr(model.config, "max_position_embeddings", 32768)
    logger.info(f"model_max_length (positions) = {model_max}")

    already_done: set[int] = _load_existing_indices(out_path_p) if resume else set()
    if resume and already_done:
        logger.info(f"Resume: skipping {len(already_done)} already-predicted pairs")

    pending = [t for t in shard if t[0] not in already_done]
    logger.info(f"Shard size = {len(shard)}, pending = {len(pending)}, batch_size = {batch_size}")

    out_f = out_path_p.open("a")
    written_since_flush = 0
    shard_tp = shard_fp = shard_tn = shard_fn = 0
    truncated_count = 0
    unparseable_count = 0
    errored_count = 0
    per_pair_times: List[float] = []
    new_token_counts: List[int] = []

    t_start = time.time()
    try:
        for batch_start in range(0, len(pending), batch_size):
            batch = pending[batch_start: batch_start + batch_size]
            prompts: list[str] = []
            labels: list[int] = []
            indices: list[int] = []
            trunc_flags: list[bool] = []

            for idx, code1, code2, label in batch:
                c1, c2, was_trunc = _truncate_pair(tokenizer, code1, code2, model_max)
                if was_trunc:
                    truncated_count += 1
                messages = _build_messages(c1, c2)
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompts.append(prompt)
                labels.append(label)
                indices.append(idx)
                trunc_flags.append(was_trunc)

            t0 = time.time()
            try:
                texts, n_new = _generate_batch(model, tokenizer, prompts, device)
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"OOM during generation at batch_start={batch_start}: {e}. Falling back to per-pair.")
                torch.cuda.empty_cache()
                texts = []
                n_new = []
                for p in prompts:
                    try:
                        t_one, n_one = _generate_batch(model, tokenizer, [p], device)
                        texts.append(t_one[0])
                        n_new.append(n_one[0])
                    except Exception as e2:
                        logger.exception(f"per-pair generation failed: {e2}")
                        texts.append("")
                        n_new.append(0)
                        errored_count += 1
            except Exception as e:
                logger.exception(f"generation failed at batch_start={batch_start}: {e}")
                texts = [""] * len(prompts)
                n_new = [0] * len(prompts)
                errored_count += len(prompts)

            elapsed = time.time() - t0
            per_call_share = elapsed / max(len(prompts), 1)

            for idx, label, text, n_tok, was_trunc in zip(indices, labels, texts, n_new, trunc_flags):
                pred = _parse_answer(text)
                if not _is_parseable(text):
                    unparseable_count += 1
                if pred == 1 and label == 1:
                    shard_tp += 1
                elif pred == 1 and label == 0:
                    shard_fp += 1
                elif pred == 0 and label == 0:
                    shard_tn += 1
                else:
                    shard_fn += 1
                per_pair_times.append(per_call_share)
                new_token_counts.append(int(n_tok))

                rec = {
                    "index": int(idx),
                    "label": int(label),
                    "prediction": int(pred),
                    "raw_output": text,
                    "truncated": bool(was_trunc),
                    "new_tokens": int(n_tok),
                }
                out_f.write(json.dumps(rec) + "\n")
                written_since_flush += 1

            if written_since_flush >= CHECKPOINT_INTERVAL:
                out_f.flush()
                os.fsync(out_f.fileno())
                written_since_flush = 0

            done = batch_start + len(batch)
            if (done % PROGRESS_INTERVAL) < batch_size or done == len(pending):
                tp, fp, fn = shard_tp, shard_fp, shard_fn
                prec = tp / (tp + fp) if (tp + fp) else 0.0
                rec_ = tp / (tp + fn) if (tp + fn) else 0.0
                f1 = (2 * prec * rec_) / (prec + rec_) if (prec + rec_) else 0.0
                el = time.time() - t_start
                eta = el * (len(pending) - done) / done if done else 0.0
                logger.info(
                    f"progress {done}/{len(pending)}  f1_so_far={f1:.3f}  "
                    f"prec={prec:.3f}  rec={rec_:.3f}  "
                    f"elapsed={el:.1f}s  ETA={eta:.1f}s"
                )

        out_f.flush()
        os.fsync(out_f.fileno())
        out_f.close()

        elapsed_total = time.time() - t_start
        logger.info(
            f"DONE shard. pairs_processed={len(pending)}  truncated={truncated_count}  "
            f"unparseable={unparseable_count}  errored={errored_count}  "
            f"elapsed={elapsed_total:.1f}s"
        )

        # Per-shard summary marker
        summary = {
            "rank": rank,
            "shard_size": len(shard),
            "processed": len(pending),
            "truncated": truncated_count,
            "unparseable": unparseable_count,
            "errored": errored_count,
            "shard_elapsed_sec": elapsed_total,
            "mean_inference_sec_per_pair": (
                float(np.mean(per_pair_times)) if per_pair_times else 0.0
            ),
            "mean_new_tokens": (
                float(np.mean(new_token_counts)) if new_token_counts else 0.0
            ),
            "shard_confusion": {
                "tp": shard_tp, "fp": shard_fp, "tn": shard_tn, "fn": shard_fn,
            },
        }
        (sentinel_dir_p / f"gpu{rank}_done.json").write_text(json.dumps(summary, indent=2))

    except Exception as e:
        logger.exception(f"worker crashed: {e}")
        (sentinel_dir_p / f"gpu{rank}_crashed.txt").write_text(f"runtime error: {e}\n")
        try:
            out_f.flush()
            out_f.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Pilot mode
# ---------------------------------------------------------------------------


def _run_pilot(
    pairs: List[Tuple[str, str, int]],
    indices: List[int],
    model_name: str,
    output_dir: Path,
) -> None:
    """Single-GPU pilot: print full prompt, raw output, parsed prediction for 5 pairs."""
    print("\n" + "=" * 60)
    print("PILOT MODE: single-GPU, 5 pairs, verbose")
    print("=" * 60)

    logger = _configure_logging(output_dir / "qwen_run.log", rank=None)

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    model, tokenizer = _load_model(model_name, device, logger)
    model_max = getattr(model.config, "max_position_embeddings", 32768)

    # Pick 5 pairs with a mix of labels
    selected: List[Tuple[int, str, str, int]] = []
    want_pos = 3
    want_neg = 2
    for idx in indices:
        code1, code2, label = pairs[idx]
        if label == 1 and want_pos > 0:
            selected.append((idx, code1, code2, label))
            want_pos -= 1
        elif label == 0 and want_neg > 0:
            selected.append((idx, code1, code2, label))
            want_neg -= 1
        if want_pos == 0 and want_neg == 0:
            break

    for i, (idx, code1, code2, label) in enumerate(selected, 1):
        c1, c2, was_trunc = _truncate_pair(tokenizer, code1, code2, model_max)
        messages = _build_messages(c1, c2)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        print(f"\n--- Pilot pair {i}/5 (orig index={idx}, label={label}, truncated={was_trunc}) ---")
        print(">>> PROMPT (first 800 chars):")
        print(prompt[:800])
        if len(prompt) > 800:
            print(f"... [+{len(prompt) - 800} more chars]")

        t0 = time.time()
        texts, n_new = _generate_batch(model, tokenizer, [prompt], device)
        dt = time.time() - t0
        pred = _parse_answer(texts[0])

        print(f">>> RAW OUTPUT: {texts[0]!r}  (new_tokens={n_new[0]}, {dt:.2f}s)")
        print(f">>> PARSED PREDICTION: {pred}  (label was {label}, "
              f"{'CORRECT' if pred == label else 'WRONG'})")

    print("\n" + "=" * 60)
    print("PILOT COMPLETE - review the above and confirm before launching the 4-GPU run.")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Merge & metrics
# ---------------------------------------------------------------------------


def _merge_predictions(
    output_dir: Path,
    num_gpus: int,
    total_pairs: int,
) -> Tuple[List[dict], List[int]]:
    """Merge per-GPU jsonl files in original index order. Returns (records, missing_indices)."""
    by_index: dict[int, dict] = {}
    for rank in range(num_gpus):
        p = output_dir / f"predictions_gpu{rank}.jsonl"
        if not p.exists():
            continue
        with p.open("r") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                idx = int(obj["index"])
                # Keep the last write for each index (post-resume)
                by_index[idx] = obj

    ordered = [by_index[i] for i in range(total_pairs) if i in by_index]
    missing = [i for i in range(total_pairs) if i not in by_index]

    out_path = output_dir / "predictions.jsonl"
    with out_path.open("w") as f:
        for rec in ordered:
            f.write(json.dumps(rec) + "\n")
    return ordered, missing


def _compute_metrics(
    records: List[dict],
    wall_clock_sec: float,
    num_gpus: int,
    batch_size: int,
    total_pairs: int,
    model_name: str,
) -> dict:
    if not records:
        return {"error": "no predictions"}

    y_true = np.array([r["label"] for r in records])
    y_pred = np.array([r["prediction"] for r in records])
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    # cm rows: true 1, true 0; cols: pred 1, pred 0
    tp = int(cm[0, 0])
    fn = int(cm[0, 1])
    fp = int(cm[1, 0])
    tn = int(cm[1, 1])

    truncated = sum(1 for r in records if r.get("truncated"))
    unparseable = sum(1 for r in records if not _is_parseable(r.get("raw_output", "")))
    new_tokens = [r.get("new_tokens", 0) for r in records]

    throughput = len(records) / wall_clock_sec if wall_clock_sec > 0 else 0.0

    # We don't have true per-pair latencies in the merged file (we have one share
    # per batched call), so use raw outputs' new-token mean as a proxy for compute
    # cost; for per-pair time, derive single-GPU-equivalent = wall_clock * num_gpus / N.
    if len(records) > 0:
        single_gpu_per_pair = (wall_clock_sec * num_gpus) / len(records)
    else:
        single_gpu_per_pair = 0.0

    return {
        "model": model_name,
        "num_gpus": num_gpus,
        "batch_size": batch_size,
        "total_pairs_dataset": total_pairs,
        "pairs_evaluated": len(records),
        "truncated": truncated,
        "unparseable": unparseable,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        },
        "wall_clock_sec": float(wall_clock_sec),
        "throughput_pairs_per_sec": float(throughput),
        "mean_inference_sec_per_pair_single_gpu_equiv": float(single_gpu_per_pair),
        "mean_new_tokens_per_pair": float(np.mean(new_tokens)) if new_tokens else 0.0,
    }


# ---------------------------------------------------------------------------
# CLI / orchestration
# ---------------------------------------------------------------------------


def _print_startup_banner(args: argparse.Namespace, num_pairs: int, label_dist: dict) -> None:
    print("=" * 60)
    print(f"Qwen2.5-Coder-7B-Instruct -> GPTCloneBench  (zero-shot)")
    print("=" * 60)
    print(f"Timestamp:     {_dt.datetime.now().isoformat()}")
    print(f"Project root:  {_PROJECT_ROOT}")
    print(f"Conda env:     {CONDA_ENV}")
    print(f"Model:         {args.model}")
    print(f"Data path:     {args.data_path}")
    print(f"Output dir:    {args.output_dir}")
    print(f"Dataset size:  {num_pairs}")
    print(f"Label dist.:   {label_dist}")
    print(f"Num GPUs:      {args.num_gpus}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Pilot mode:    {args.pilot_only}")
    print(f"Resume mode:   {args.resume}")
    print(f"torch:         {torch.__version__}  CUDA={torch.cuda.is_available()}")
    print("nvidia-smi:")
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.free,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        for line in smi.splitlines():
            print("  " + line)
    except Exception as e:
        print(f"  (nvidia-smi failed: {e})")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--num-pairs", type=int, default=10_000)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "qwen_run.log"

    logger = _configure_logging(log_path, rank=None)

    # Load pairs
    logger.info(f"Loading GPTCloneBench from {args.data_path}")
    pairs = build_pairs(
        Path(args.data_path),
        language="java",
        max_pairs=args.num_pairs,
        seed=args.seed,
    )
    label_dist = dict(Counter(p[2] for p in pairs))

    _print_startup_banner(args, len(pairs), label_dist)

    # ---------------- Pilot mode ----------------
    if args.pilot_only:
        all_indices = list(range(len(pairs)))
        _run_pilot(pairs, all_indices, args.model, output_dir)
        return

    # ---------------- Multi-GPU full run ----------------
    sentinel_dir = output_dir / "_sentinels"
    sentinel_dir.mkdir(exist_ok=True)
    for f in sentinel_dir.glob("*.txt"):
        f.unlink()
    for f in sentinel_dir.glob("*.json"):
        f.unlink()

    # Clear stale per-GPU outputs unless resuming
    if not args.resume:
        for rank in range(args.num_gpus):
            p = output_dir / f"predictions_gpu{rank}.jsonl"
            if p.exists():
                p.unlink()

    # Build shards (contiguous index ranges)
    N = len(pairs)
    shards: List[List[Tuple[int, str, str, int]]] = [[] for _ in range(args.num_gpus)]
    edges = np.linspace(0, N, args.num_gpus + 1, dtype=int)
    for rank in range(args.num_gpus):
        s, e = int(edges[rank]), int(edges[rank + 1])
        shards[rank] = [(i, pairs[i][0], pairs[i][1], pairs[i][2]) for i in range(s, e)]
        logger.info(f"Shard {rank}: indices {s}..{e - 1} (size {e - s})")

    # Launch workers
    mp.set_start_method("spawn", force=True)
    procs = []
    t_start = time.time()
    for rank in range(args.num_gpus):
        out_path = output_dir / f"predictions_gpu{rank}.jsonl"
        p = mp.Process(
            target=worker_main,
            args=(
                rank,
                args.num_gpus,
                args.model,
                shards[rank],
                str(out_path),
                str(log_path),
                args.batch_size,
                args.resume,
                str(sentinel_dir),
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
    wall_clock_sec = time.time() - t_start

    # Detect crashes
    crashed = list(sentinel_dir.glob("gpu*_crashed.txt"))
    if crashed:
        logger.warning(f"{len(crashed)} workers crashed: {[c.name for c in crashed]}")

    # Merge predictions
    records, missing = _merge_predictions(output_dir, args.num_gpus, N)
    if missing:
        logger.warning(f"{len(missing)} pairs missing from merged output "
                       f"(first 10: {missing[:10]})")

    metrics = _compute_metrics(
        records, wall_clock_sec, args.num_gpus, args.batch_size, N, args.model,
    )
    metrics["missing_pairs"] = len(missing)
    metrics["crashed_workers"] = [c.name for c in crashed]
    metrics["timestamp"] = _dt.datetime.now().isoformat()
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Final summary
    cm = metrics.get("confusion_matrix", {})
    hh = int(wall_clock_sec // 3600)
    mm = int((wall_clock_sec % 3600) // 60)
    ss = int(wall_clock_sec % 60)
    print("\n" + "=" * 60)
    print("Qwen2.5-Coder-7B-Instruct on GPTCloneBench")
    print("=" * 60)
    print(f"Project root: {_PROJECT_ROOT}")
    print(f"Conda env:    {CONDA_ENV}")
    print(f"GPUs used:    {args.num_gpus} x A100")
    print(f"Batch size:   {args.batch_size} per GPU")
    print()
    print(f"Pairs evaluated: {metrics['pairs_evaluated']} / {N}")
    print(f"Truncated:       {metrics['truncated']}")
    print(f"Unparseable:     {metrics['unparseable']}")
    print(f"Missing:         {metrics['missing_pairs']}")
    print()
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1:        {metrics['f1']:.3f}")
    print()
    print("Confusion matrix:")
    print("                Predicted Clone   Predicted Non-Clone")
    print(f"True Clone      {cm.get('tp', 0):<17} {cm.get('fn', 0)}")
    print(f"True Non-Clone  {cm.get('fp', 0):<17} {cm.get('tn', 0)}")
    print()
    print(f"Throughput:                  {metrics['throughput_pairs_per_sec']:.2f} pairs/sec "
          f"(across {args.num_gpus} GPUs)")
    print(f"Mean inference time / pair:  "
          f"{metrics['mean_inference_sec_per_pair_single_gpu_equiv']:.3f} sec "
          f"(single-GPU equivalent)")
    print(f"Total wall-clock time:       {hh:02d}:{mm:02d}:{ss:02d}")
    print(f"Mean tokens generated/pair:  {metrics['mean_new_tokens_per_pair']:.2f}")
    print("=" * 60)
    logger.info(f"Wrote {output_dir / 'predictions.jsonl'} "
                f"and {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
