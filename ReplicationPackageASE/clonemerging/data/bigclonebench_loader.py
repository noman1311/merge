"""BigCloneBench data loader.

Reads the CodeXGLUE BigCloneBench clone-detection split (index files plus the
function source) and yields tokenized code pairs for training and evaluation.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoTokenizer

DEFAULT_DATASET_DIR = Path(
    "data/CodeXGLUE/Code-Code/Clone-detection-BigCloneBench/dataset"
)


def load_code_map(dataset_dir: Path) -> Dict[str, str]:
    data_path = dataset_dir / "data.jsonl"
    code_map: Dict[str, str] = {}

    with data_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            function_id = str(row["idx"])
            code_map[function_id] = row["func"]

    return code_map


def parse_split_file(split_path: Path) -> List[Tuple[str, str, int, Optional[str], str]]:
    samples: List[Tuple[str, str, int, Optional[str], str]] = []

    with split_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) == 4:
                idx, id1, id2, label = parts
                clone_type = None
            elif len(parts) == 3:
                idx = str(line_number - 1)
                id1, id2, label = parts
                clone_type = None
            elif len(parts) >= 5:
                idx, id1, id2, label = parts[:4]
                clone_type = parts[4]
            else:
                raise ValueError(f"Unexpected line format in {split_path}: {line}")

            samples.append((idx, id1, id2, int(label), clone_type, line))

    return samples


class BigCloneBenchDataset(Dataset):
    def __init__(
        self,
        split: str,
        tokenizer,
        dataset_dir: Path = DEFAULT_DATASET_DIR,
        max_length: int = 512,
    ) -> None:
        if split not in {"train", "valid", "test"}:
            raise ValueError("split must be one of: train, valid, test")

        self.split = split
        self.tokenizer = tokenizer
        self.dataset_dir = Path(dataset_dir)
        self.max_length = max_length

        self.code_map = load_code_map(self.dataset_dir)
        split_path = self.dataset_dir / f"{split}.txt"
        self.samples = parse_split_file(split_path)

        self.labels: List[int] = []
        self.clone_types: List[Optional[str]] = []

        for idx, id1, id2, label, clone_type, raw_line in self.samples:
            if id1 not in self.code_map or id2 not in self.code_map:
                raise KeyError(
                    f"Missing function id in data.jsonl for split '{split}' at idx={idx}: {raw_line}"
                )
            self.labels.append(label)
            self.clone_types.append(clone_type)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        _, id1, id2, label, _, _ = self.samples[index]
        code1 = self.code_map[id1]
        code2 = self.code_map[id2]

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

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
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
) -> Tuple[BigCloneBenchDataset, DataLoader]:
    dataset = BigCloneBenchDataset(
        split=split,
        tokenizer=tokenizer,
        dataset_dir=dataset_dir,
        max_length=max_length,
    )

    if use_weighted_sampling is None:
        use_weighted_sampling = split == "train"

    sampler = None
    shuffle = split == "train"

    if use_weighted_sampling:
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
