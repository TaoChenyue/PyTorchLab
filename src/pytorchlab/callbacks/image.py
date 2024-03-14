from pathlib import Path
from typing import Any, Mapping

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor
from torchvision.utils import make_grid, save_image

from pytorchlab.callbacks.utils import get_batch_save_path
from pytorchlab.typehints import OutputsDict

__all__ = ["ImageCallback"]


class ImageCallback(Callback):
    def __init__(
        self,
        batch_idx: int = 0,
        on_epoch: bool = False,
        image_nums: int = 1,
        default_dir: str = "output",
        **kwargs,
    ) -> None:
        super().__init__()
        self.batch_idx: int = batch_idx
        self.on_epoch = on_epoch
        self.image_nums = image_nums
        self.default_dir = default_dir
        self.kwargs = kwargs

    def get_images(self, outputs: OutputsDict):
        images = {}
        images.update(
            {
                f"input_{k}": v[: self.image_nums]
                for k, v in outputs.get("inputs", {}).get("images", {}).items()
            }
        )
        images.update(
            {
                f"output_{k}": v[: self.image_nums]
                for k, v in outputs.get("outputs", {}).get("images", {}).items()
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
        images = self.get_images(outputs)
        for save_path in save_paths:
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
            default=self.default_dir,
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
            default=self.default_dir,
        )
        images = self.get_images(outputs)
        self.save_images(images, save_path)
