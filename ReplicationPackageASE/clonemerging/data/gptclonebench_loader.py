"""GPTCloneBench standalone semantic clone loader for code clone detection evaluation.

Data layout (standalone/)::

    true_semantic_clones/{language}/prompt_N/{MT3,T4}/Clone_XYZ.{ext}
    false_semantic_clones/{language}/Gpt_false_pair_XYZ.{ext}

Pair construction (Java-focused, but language-agnostic):

  True  pairs (label=1): two files from the same prompt_N directory (any subdir).
                         All files under the same prompt implement the same
                         functionality -> semantic clones.
  False pairs (label=0): one file from true_semantic_clones (any prompt) paired
                         with one file from false_semantic_clones.  The files
                         implement different functionalities -> not clones.

Pairs are balanced (equal true/false) and capped at *max_pairs* total.
"""

import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

DEFAULT_DATASET_DIR = Path(
    "data/GPTCloneBench/GPTCloneBench/standalone"
)

_LANG_EXT: Dict[str, str] = {
    "java": "java",
    "py": "py",
    "c": "c",
    "cs": "cs",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_prompt_files(standalone_dir: Path, language: str) -> Dict[str, List[Path]]:
    """Return ``{prompt_name: [sorted file paths]}`` for all prompt dirs."""
    ext = _LANG_EXT[language]
    true_dir = standalone_dir / "true_semantic_clones" / language
    result: Dict[str, List[Path]] = {}
    for prompt_dir in sorted(true_dir.iterdir()):
        if not prompt_dir.is_dir():
            continue
        files: List[Path] = []
        for entry in sorted(prompt_dir.iterdir()):
            if entry.is_dir():
                files.extend(sorted(entry.glob(f"*.{ext}")))
            elif entry.suffix == f".{ext}":
                files.append(entry)
        if len(files) >= 2:
            result[prompt_dir.name] = files
    return result


def _collect_false_files(standalone_dir: Path, language: str) -> List[Path]:
    """Return sorted list of false-clone file paths for *language*."""
    ext = _LANG_EXT[language]
    false_dir = standalone_dir / "false_semantic_clones" / language
    return sorted(false_dir.glob(f"*.{ext}"))


def _sample_true_pairs(
    prompt_files: Dict[str, List[Path]],
    n: int,
    rng: random.Random,
) -> List[Tuple[Path, Path]]:
    """Sample *n* unordered intra-prompt file pairs, distributed proportionally
    by C(prompt_size, 2) across all prompts."""
    prompt_names = sorted(prompt_files)
    sizes = [len(prompt_files[p]) for p in prompt_names]
    max_avail = [s * (s - 1) // 2 for s in sizes]
    total_avail = sum(max_avail)
    n = min(n, total_avail)

    # Proportional budget per prompt (C(n,2)-weighted)
    budgets = [int(n * m / total_avail) for m in max_avail]
    budgets[-1] += n - sum(budgets)  # assign remainder to last prompt

    pairs: List[Tuple[Path, Path]] = []
    for prompt_name, budget, total in zip(prompt_names, budgets, max_avail):
        if budget <= 0:
            continue
        files = prompt_files[prompt_name]
        nf = len(files)
        budget = min(budget, total)

        if total <= budget * 3:
            # Enumerate all pairs (small prompt) and sample without replacement
            all_pairs = [
                (files[i], files[j])
                for i in range(nf)
                for j in range(i + 1, nf)
            ]
            pairs.extend(rng.sample(all_pairs, budget))
        else:
            # Random index sampling with de-duplication via seen-set
            # Collision rate ~ budget/total is negligible when total >> budget
            seen: set = set()
            while len(seen) < budget:
                i, j = sorted(rng.sample(range(nf), 2))
                seen.add((i, j))
            pairs.extend((files[i], files[j]) for i, j in seen)

    rng.shuffle(pairs)
    return pairs[:n]


def _sample_false_pairs(
    all_true_files: List[Path],
    false_files: List[Path],
    n: int,
    rng: random.Random,
) -> List[Tuple[Path, Path]]:
    """Sample *n* false pairs: each is (true_clone_file, false_clone_file).

    Sampling is without replacement on each side when the pool is large enough;
    otherwise falls back to sampling with replacement.
    """
    if len(all_true_files) >= n:
        true_sample = rng.sample(all_true_files, n)
    else:
        true_sample = rng.choices(all_true_files, k=n)

    if len(false_files) >= n:
        false_sample = rng.sample(false_files, n)
    else:
        false_sample = rng.choices(false_files, k=n)

    return list(zip(true_sample, false_sample))


# ---------------------------------------------------------------------------
# Public pair-builder
# ---------------------------------------------------------------------------


def build_pairs(
    standalone_dir: Path,
    language: str = "java",
    max_pairs: int = 10_000,
    seed: int = 42,
) -> List[Tuple[str, str, int]]:
    """Return a shuffled list of ``(code1, code2, label)`` triples.

    *label* is 1 for semantic clones and 0 for non-clones.
    The list is balanced (equal number of true and false pairs) and contains
    at most *max_pairs* entries total.
    """
    rng = random.Random(seed)

    prompt_files = _collect_prompt_files(standalone_dir, language)
    if not prompt_files:
        raise FileNotFoundError(
            f"No prompt directories found under {standalone_dir / 'true_semantic_clones' / language}"
        )

    false_files = _collect_false_files(standalone_dir, language)
    if not false_files:
        raise FileNotFoundError(
            f"No false clone files found under {standalone_dir / 'false_semantic_clones' / language}"
        )

    n_true = max_pairs // 2
    n_false = max_pairs - n_true

    true_file_pairs = _sample_true_pairs(prompt_files, n_true, rng)

    all_true_files: List[Path] = [f for files in prompt_files.values() for f in files]
    false_file_pairs = _sample_false_pairs(all_true_files, false_files, n_false, rng)

    samples: List[Tuple[str, str, int]] = []
    for f1, f2 in true_file_pairs:
        samples.append(
            (
                f1.read_text(encoding="utf-8", errors="replace"),
                f2.read_text(encoding="utf-8", errors="replace"),
                1,
            )
        )
    for ft, ff in false_file_pairs:
        samples.append(
            (
                ft.read_text(encoding="utf-8", errors="replace"),
                ff.read_text(encoding="utf-8", errors="replace"),
                0,
            )
        )

    rng.shuffle(samples)
    return samples


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class GPTCloneBenchDataset(Dataset):
    """PyTorch Dataset for GPTCloneBench standalone semantic clones.

    Only supports ``split="test"`` - this benchmark is evaluation-only.
    """

    def __init__(
        self,
        split: str,
        tokenizer,
        dataset_dir: Path = DEFAULT_DATASET_DIR,
        max_length: int = 512,
        max_pairs: int = 10_000,
        seed: int = 42,
        language: str = "java",
    ) -> None:
        if split != "test":
            raise ValueError(
                "GPTCloneBench is evaluation-only: split must be 'test'"
            )

        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.samples = build_pairs(
            Path(dataset_dir),
            language=language,
            max_pairs=max_pairs,
            seed=seed,
        )
        self.labels: List[int] = [s[2] for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        code1, code2, label = self.samples[index]

        encoding = self.tokenizer(
            code1,
            code2,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def get_dataloader(
    split: str,
    tokenizer,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    batch_size: int = 32,
    max_length: int = 512,
    num_workers: int = 4,
    pin_memory: bool = True,
    max_pairs: int = 10_000,
    seed: int = 42,
    language: str = "java",
) -> Tuple[GPTCloneBenchDataset, DataLoader]:
    """Build and return ``(dataset, dataloader)`` for GPTCloneBench evaluation.

    Args:
        split:       Must be ``"test"`` (evaluation-only benchmark).
        tokenizer:   HuggingFace tokenizer (e.g. CodeBERT / GraphCodeBERT).
        dataset_dir: Path to the ``standalone/`` directory.
        batch_size:  Mini-batch size for the DataLoader.
        max_length:  Token sequence length cap (default 512).
        num_workers: DataLoader worker processes.
        pin_memory:  Pin host memory for faster GPU transfer.
        max_pairs:   Total pair cap.  Half will be true clones, half false.
        seed:        Random seed for reproducible pair sampling.
        language:    Source language to use (default ``"java"``).

    Returns:
        ``(GPTCloneBenchDataset, DataLoader)``
    """
    if split != "test":
        raise ValueError("GPTCloneBench is evaluation-only: split must be 'test'")

    dataset = GPTCloneBenchDataset(
        split=split,
        tokenizer=tokenizer,
        dataset_dir=Path(dataset_dir),
        max_length=max_length,
        max_pairs=max_pairs,
        seed=seed,
        language=language,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return dataset, dataloader


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    dataset, loader = get_dataloader(
        split="test",
        tokenizer=tokenizer,
        batch_size=32,
        max_pairs=100,
        seed=42,
        num_workers=0,
        pin_memory=False,
    )

    batch = next(iter(loader))
    label_dist = dict(Counter(dataset.labels))

    print(f"Dataset size:       {len(dataset)}")
    print(f"Label distribution: {label_dist}")
    print(f"input_ids shape:    {batch['input_ids'].shape}")
    print(f"attention_mask:     {batch['attention_mask'].shape}")
    print(f"label shape:        {batch['label'].shape}")
    print("OK")
