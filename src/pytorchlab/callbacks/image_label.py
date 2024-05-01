from pathlib import Path
from typing import Any, Mapping

import yaml
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor
from torchvision.utils import make_grid, save_image

from pytorchlab.typehints import OutputsDict
from pytorchlab.utils.state import get_batch_save_path

__all__ = [
    "ImageLabelCallback",
]


class ImageLabelCallback(Callback):
    def __init__(
        self,
        image_names: tuple[list[str], list[str]] = ([], []),
        label_names: tuple[list[str], list[str]] = ([], []),
        batch_idx: list[int] = [0],
        show_epoch: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.image_names = image_names
        self.label_names = label_names
        self.batch_idx = batch_idx
        self.show_epoch = show_epoch
        self.kwargs = kwargs

    def get_images(self, outputs: OutputsDict):
        images = {}
        images.update(
            {
                f"input_{k}": v
                for k, v in outputs.get("inputs", {}).items()
                if k in self.image_names[0]
            }
        )
        images.update(
            {
                f"output_{k}": v
                for k, v in outputs.get("outputs", {}).items()
                if k in self.image_names[1]
            }
        )
        return images

    def save_images(self, images: dict[str, Tensor], save_path: Path):
        for k, v in images.items():
            save_image(make_grid(v, **self.kwargs), save_path / f"{k}.png")

    def get_labels(self, outputs: OutputsDict):
        labels = {}
        labels.update(
            {
                f"input_{k}": v
                for k, v in outputs.get("inputs", {}).items()
                if k in self.label_names[0]
            }
        )
        labels.update(
            {
                f"output_{k}": v
                for k, v in outputs.get("outputs", {}).items()
                if k in self.label_names[1]
            }
        )
        return labels

    def save_labels(self, labels: dict[str, Tensor], save_path: Path):
        for k, v in labels.items():
            yaml.dump(
                v.tolist(),
                open(save_path / f"{k}.yaml", "w", encoding="utf-8"),
            )

    def _batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: OutputsDict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        save_path = get_batch_save_path(
            trainer,
            pl_module,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
            show_epoch=self.show_epoch,
        )
        if save_path is None:
            return
        images = self.get_images(outputs)
        self.save_images(images, save_path)
        labels = self.get_labels(outputs)
        self.save_labels(labels, save_path)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: OutputsDict,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx not in self.batch_idx:
            return
        self._batch_end(trainer, pl_module, outputs, batch_idx, dataloader_idx)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._batch_end(trainer, pl_module, outputs, batch_idx, dataloader_idx)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._batch_end(trainer, pl_module, outputs, batch_idx, dataloader_idx)
