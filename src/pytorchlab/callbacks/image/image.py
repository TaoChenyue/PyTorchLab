import random
from typing import Any, Literal, Mapping, Sequence

from lightning import LightningModule, Trainer
from torch import Tensor
from torchvision.utils import make_grid, save_image

from pytorchlab.callbacks.image.base import BaseImageCallback


class ImageCallback(BaseImageCallback):
    def __init__(
        self,
        batch_range: tuple[int, int] = (0, 0),
        batch_indexes: list[int] = [],
        image_slice: tuple[int | None, int | None, int | None] = (None, None, None),
        on_epoch: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(on_epoch=on_epoch)
        self.batch_range = batch_range
        self.batch_now: int | None = None
        self.batch_indexes = batch_indexes
        self.image_slice = slice(*image_slice)
        self.on_epoch = on_epoch
        self.kwargs = kwargs

    def _epoch_start(self, trainer: Trainer, pl_module: LightningModule):
        self.batch_now = random.randint(*self.batch_range)

    def _epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        self.batch_now = None

    def _get_images(
        self, outputs: Tensor | Mapping[str, Any] | None
    ) -> list[Tensor] | None:
        if not isinstance(outputs, Mapping):
            return None
        image = outputs.get("outputs", {}).get("images", [])
        return image

    def _batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Sequence[Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
        mode: Literal["val", "test", "predict"] = "val",
    ) -> None:
        if mode == "val" and batch_idx != self.batch_now:
            return

        images = self._get_images(outputs)
        if images is None:
            return

        save_dir = self._get_save_dir(
            mode,
            trainer,
            pl_module,
            batch_idx,
            dataloader_idx,
        )
        if save_dir is None:
            return

        images_dict = {
            f"input_{index}.png": batch[index][self.image_slice]
            for index in self.batch_indexes
        }
        images_dict.update(
            {
                f"output_{index}.png": images[index][self.image_slice]
                for index in range(len(images))
            }
        )

        for name, image in images_dict.items():
            save_image(make_grid(image, **self.kwargs), save_dir / name)

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        return self._epoch_start(trainer, pl_module)

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        return self._epoch_end(trainer, pl_module)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        return self._batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
            mode="val",
        )

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        return self._epoch_start(trainer, pl_module)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        return self._epoch_end(trainer, pl_module)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        return self._batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
            mode="test",
        )

    def on_predict_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        return self._epoch_start(trainer, pl_module)

    def on_predict_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        return self._epoch_end(trainer, pl_module)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        return self._batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
            mode="predict",
        )
