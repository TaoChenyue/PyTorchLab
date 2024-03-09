import random
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor
from torchvision.utils import make_grid, save_image


class ImageCallback(Callback):
    def __init__(
        self,
        batch_range: tuple[int, int] = (0, 0),
        batch_indexes: list[int] = [],
        image_slice: tuple[int | None, int | None, int | None] = (None, None, None),
        on_epoch: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.batch_range = batch_range
        self.batch_now: int | None = None
        self.batch_indexes = batch_indexes
        self.image_slice = slice(*image_slice)
        self.on_epoch = on_epoch
        self.kwargs = kwargs

    def _epoch_start(self):
        self.batch_now = random.randint(*self.batch_range)

    def _epoch_end(self):
        self.batch_now = None

    def _get_image(self, outputs: Tensor | Mapping[str, Any] | None):
        if outputs is None:
            return None
        if isinstance(outputs, Tensor):
            image = outputs
        if isinstance(outputs, Mapping):
            image = outputs.get("output", None)
        return image

    def _get_log_dir(self, pl_module: LightningModule):
        log_path = pl_module.logger.log_dir
        if log_path is None:
            return None
        return Path(log_path)

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
        image = self._get_image(outputs)
        if image is None:
            return
        log_dir = self._get_log_dir(pl_module)
        if log_dir is None:
            return
        image = image[self.image_slice]
        if mode == "val":
            save_dir = [log_dir / "val_images" / "_latest"]
            if self.on_epoch:
                save_dir = [log_dir / "val_images" / f"epoch={trainer.current_epoch}"]
        else:
            save_dir = [log_dir / f"{mode}_images" / f"batch={batch_idx}"]

        for save_dir in save_dir:
            save_dir.mkdir(exist_ok=True, parents=True)

            images = {
                f"index={index}.png": batch[index][self.image_slice]
                for index in self.batch_indexes
            }
            images["output.png"] = image
            for name, image in images.items():
                save_image(make_grid(image, **self.kwargs), save_dir / name)

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        return self._epoch_start()

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        return self._epoch_end()

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
        return self._epoch_start()

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        return self._epoch_end()

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
        return self._epoch_start()

    def on_predict_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        return self._epoch_end()

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
