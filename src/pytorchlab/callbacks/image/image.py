from pathlib import Path
from typing import Any, Mapping

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor
from torchvision.utils import make_grid, save_image

from pytorchlab.callbacks.utils.save_path import get_batch_save_path
from pytorchlab.typehints import OutputsDict

__all__ = ["ImageCallback"]


class ImageCallback(Callback):
    def __init__(
        self,
        batch_idx: int = 0,
        input_names: list[str] = [],
        output_names: list[str] = [],
        **kwargs,
    ) -> None:
        super().__init__()
        self.batch_idx: int = batch_idx
        self.input_names = input_names
        self.output_names = output_names
        self.kwargs = kwargs

    def get_images(self, outputs: OutputsDict):
        images = {}
        images.update(
            {
                f"input_{k}": v
                for k, v in outputs.get("inputs", {}).get("images", {}).items()
                if k in self.input_names
            }
        )
        images.update(
            {
                f"output_{k}": v
                for k, v in outputs.get("outputs", {}).get("images", {}).items()
                if k in self.output_names
            }
        )
        return images

    def save_images(self, images: dict[str, Tensor], save_path: Path):
        for k, v in images.items():
            save_image(make_grid(v, **self.kwargs), save_path / f"{k}.png")

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
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )
        if save_path is None:
            return
        images = self.get_images(outputs)
        self.save_images(images, save_path)

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
        images = self.get_images(outputs)
        self.save_images(images, save_path)

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
        images = self.get_images(outputs)
        self.save_images(images, save_path)
