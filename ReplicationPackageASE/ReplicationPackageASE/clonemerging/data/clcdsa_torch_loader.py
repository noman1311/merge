"""CLCDSA data loader.

Builds balanced cross-language clone pairs from the CLCDSA source solutions and
yields tokenized pairs. A processed cache is written on first use to speed up
subsequent runs.
"""

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoTokenizer

from data.clcdsa_loader import CLCDSADataset, ClonePair

DEFAULT_DATASET_DIR = Path("data/CLCDSA/Source Codes/")
DEFAULT_CACHE_PATH = Path("data/CLCDSA/clcdsa_cache.pkl")


class CLCDSATorchDataset(Dataset):
    def __init__(
        self,
        split: str,
        tokenizer,
        dataset_dir: Path = DEFAULT_DATASET_DIR,
        cache_path: Path = DEFAULT_CACHE_PATH,
        max_length: int = 512,
    ) -> None:
        if split not in {"train", "valid", "test"}:
            raise ValueError("split must be one of: train, valid, test")

        self.split = split
        self.tokenizer = tokenizer
        self.dataset_dir = Path(dataset_dir)
        self.cache_path = Path(cache_path)
        self.max_length = max_length

        dataset = CLCDSADataset.load(
            data_dir=str(self.dataset_dir),
            cache_path=str(self.cache_path),
        )

        if split == "train":
            self.samples: List[ClonePair] = dataset.train
        elif split == "valid":
            self.samples = dataset.valid
        else:
            self.samples = dataset.test

        self.labels: List[int] = [sample.label for sample in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]

        encoding = self.tokenizer(
            sample.code_a,
            sample.code_b,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(sample.label, dtype=torch.long),
        }
        return item


def get_dataloader(
    split: str,
    tokenizer,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    batch_size: int = 32,
    max_length: int = 512,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_weighted_sampling: Optional[bool] = None,
    cache_path: Path = DEFAULT_CACHE_PATH,
) -> Tuple[CLCDSATorchDataset, DataLoader]:
    dataset = CLCDSATorchDataset(
        split=split,
        tokenizer=tokenizer,
        dataset_dir=dataset_dir,
        cache_path=cache_path,
        max_length=max_length,
    )

    if use_weighted_sampling is None:
        use_weighted_sampling = split == "train"

    sampler = None
    shuffle = split == "train"

    if use_weighted_sampling and len(dataset) > 0:
        label_tensor = torch.tensor(dataset.labels, dtype=torch.long)
        class_counts = torch.bincount(label_tensor, minlength=2).float()
        class_weights = torch.zeros_like(class_counts)
        nonzero_mask = class_counts > 0
        class_weights[nonzero_mask] = 1.0 / class_counts[nonzero_mask]

        sample_weights = class_weights[label_tensor]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return dataset, dataloader


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    train_dataset, train_loader = get_dataloader(
        split="train",
        tokenizer=tokenizer,
        batch_size=32,
    )

    label_distribution = Counter(train_dataset.labels)
    one_batch = next(iter(train_loader))

    print(f"Dataset size (train): {len(train_dataset)}")
    print(f"Label distribution: {dict(label_distribution)}")
    print(f"input_ids shape: {one_batch['input_ids'].shape}")
    print(f"attention_mask shape: {one_batch['attention_mask'].shape}")
    print(f"label shape: {one_batch['label'].shape}")