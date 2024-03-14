import json
from pathlib import Path
from typing import Any, Mapping

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor

from pytorchlab.callbacks.utils import get_batch_save_path
from pytorchlab.typehints import OutputsDict

__all__ = ["LabelNameCallback"]


class LabelNameCallback(Callback):
    def __init__(
        self,
        batch_idx: int = 0,
        on_epoch: bool = False,
        label_nums: int = 1,
        default_dir: str = "output",
        name_list: list[str] = [],
        **kwargs,
    ) -> None:
        super().__init__()
        self.batch_idx: int = batch_idx
        self.on_epoch = on_epoch
        self.label_nums = label_nums
        self.default_dir = default_dir
        self.name_list = name_list
        self.kwargs = kwargs

    def get_labels(self, outputs: OutputsDict):
        labels = {}
        labels.update(
            {
                f"input_{k}": [self.name_list[int(v)] for v in v][: self.label_nums]
                for k, v in outputs.get("inputs", {}).get("labels", {}).items()
            }
        )
        labels.update(
            {
                f"output_{k}": [self.name_list[int(v)] for v in v][: self.label_nums]
                for k, v in outputs.get("inputs", {}).get("labels", {}).items()
            }
        )
        return labels

    def save_labels(self, labels: dict[str, Tensor], save_path: Path):
        json.dump(labels, open(save_path / "labels.json", "w", encoding="utf-8"))

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

        save_paths = [
            get_batch_save_path(
                trainer,
                pl_module,
                batch_idx,
                dataloader_idx,
                on_epoch=False,
                default=self.default_dir,
            )
        ]
        if self.on_epoch:
            save_paths.append(
                get_batch_save_path(
                    trainer,
                    pl_module,
                    batch_idx,
                    dataloader_idx,
                    on_epoch=True,
                    default=self.default_dir,
                )
            )
        labels = self.get_labels(outputs)
        for save_path in save_paths:
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
            default=self.default_dir,
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
            default=self.default_dir,
        )
        labels = self.get_labels(outputs)
        self.save_labels(labels, save_path)
