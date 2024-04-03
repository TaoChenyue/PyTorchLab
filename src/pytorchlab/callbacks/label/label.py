from pathlib import Path
from typing import Any, Mapping

import yaml
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor

from pytorchlab.callbacks.utils.save_path import get_batch_save_path
from pytorchlab.typehints import OutputsDict

__all__ = ["LabelCallback"]


class LabelCallback(Callback):
    def __init__(
        self,
        batch_idx: int = 0,
        input_names: list[str] = [],
        output_names: list[str] = [],
    ) -> None:
        super().__init__()
        self.batch_idx = batch_idx
        self.input_names = input_names
        self.output_names = output_names

    def get_labels(self, outputs: OutputsDict):
        labels = {}
        labels.update(
            {
                f"input_{k}": v
                for k, v in outputs.get("inputs", {}).get("labels", {}).items()
                if k in self.input_names
            }
        )
        labels.update(
            {
                f"output_{k}": v
                for k, v in outputs.get("outputs", {}).get("labels", {}).items()
                if k in self.output_names
            }
        )
        return labels

    def save_labels(self, labels: dict[str, Tensor], save_path: Path):
        for k, v in labels.items():
            if v.ndim > 1:
                v = v.argmax(dim=-1)
            yaml.dump(
                v.tolist(),
                open(save_path / f"{k}.yaml", "w", encoding="utf-8"),
            )

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: OutputsDict,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx != self.batch_idx:
            return

        save_path = get_batch_save_path(
            trainer,
            pl_module,
            batch_idx,
            dataloader_idx,
        )

        labels = self.get_labels(outputs)
        self.save_labels(labels, save_path)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        save_path = get_batch_save_path(
            trainer,
            pl_module,
            batch_idx,
            dataloader_idx,
        )
        labels = self.get_labels(outputs)
        self.save_labels(labels, save_path)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        save_path = get_batch_save_path(
            trainer,
            pl_module,
            batch_idx,
            dataloader_idx,
        )
        labels = self.get_labels(outputs)
        self.save_labels(labels, save_path)
