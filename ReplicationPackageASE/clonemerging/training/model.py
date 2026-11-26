"""Clone detection model: a code encoder with a binary classification head.

Wraps a pre-trained encoder (CodeBERT, GraphCodeBERT, or UniXcoder) and adds a
lightweight classification head over the pooled representation of a code pair.
Used for both fine-tuning and evaluation.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class CloneDetectionModel(nn.Module):
    def __init__(self, model_name_or_path: str, num_labels: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.dropout_prob = dropout

        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.encoder = AutoModel.from_pretrained(model_name_or_path, config=self.config)

        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_representation = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls_representation))

        loss = None
        result: Dict[str, torch.Tensor] = {"loss": loss, "logits": logits}
        if labels is not None:
            result["loss"] = self.loss_fn(logits, labels)
        return result

    def save_pretrained(self, save_directory: str) -> None:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        self.encoder.save_pretrained(str(save_path))

        classifier_state = {
            "classifier": self.classifier.state_dict(),
            "dropout": self.dropout_prob,
            "num_labels": self.num_labels,
        }
        torch.save(classifier_state, save_path / "classifier_head.bin")

        with (save_path / "clone_detection_config.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "num_labels": self.num_labels,
                    "dropout": self.dropout_prob,
                },
                handle,
                indent=2,
            )

    @classmethod
    def load_pretrained(cls, checkpoint_dir: str | Path) -> "CloneDetectionModel":
        checkpoint_path = Path(checkpoint_dir)

        classifier_head_path = checkpoint_path / "classifier_head.bin"
        if not classifier_head_path.exists():
            raise FileNotFoundError(f"Missing classifier head file: {classifier_head_path}")

        try:
            classifier_state = torch.load(classifier_head_path, map_location="cpu", weights_only=True)
        except TypeError:
            classifier_state = torch.load(classifier_head_path, map_location="cpu")

        config_path = checkpoint_path / "clone_detection_config.json"
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                clone_config = json.load(handle)
            num_labels = int(clone_config.get("num_labels", classifier_state.get("num_labels", 2)))
            dropout = float(clone_config.get("dropout", classifier_state.get("dropout", 0.1)))
        else:
            num_labels = int(classifier_state.get("num_labels", 2))
            dropout = float(classifier_state.get("dropout", 0.1))

        model = cls(
            model_name_or_path=str(checkpoint_path),
            num_labels=num_labels,
            dropout=dropout,
        )
        model.classifier.load_state_dict(classifier_state["classifier"])
        return model
