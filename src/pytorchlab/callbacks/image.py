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

    def _get_log_dir(self, pl_module: LightningModule):
        log_path = pl_module.logger.log_dir
        if log_path is None:
            return None
        return Path(log_path)

    def _get_save_dir(
        self,
        trainer: Trainer,
        log_dir: Path,
        mode: Literal["val", "test", "predict"],
        batch_idx: int = 0,
        dataloader_idx: int = 0,
    ) -> Path:
        if mode == "val":
            save_dir = (
                (
                    log_dir
                    / "val_images"
                    / f"dataloader={dataloader_idx}"
                    / f"epoch={trainer.current_epoch}"
                )
                if self.on_epoch
                else (
                    log_dir / "val_images" / f"dataloader={dataloader_idx}" / "_latest"
                )
            )

        else:
            save_dir = (
                log_dir
                / f"{mode}_images"
                / f"dataloader={dataloader_idx}"
                / f"batch={batch_idx}"
            )
        return save_dir

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

        log_dir = self._get_log_dir(pl_module)
        if log_dir is None:
            return
        save_dir = self._get_save_dir(trainer, log_dir, mode, batch_idx, dataloader_idx)

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

        save_dir.mkdir(exist_ok=True, parents=True)
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
