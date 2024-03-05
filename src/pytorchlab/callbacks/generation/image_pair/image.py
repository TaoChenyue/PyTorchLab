import random
from pathlib import Path
from typing import Any, Mapping

from lightning import LightningModule, Trainer
from lightning.pytorch import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from torchvision.utils import make_grid, save_image


class GenerationImagePairCallback(Callback):
    def __init__(
        self,
        batch: int = 0,
        slice_num: int = 1,
        **kwargs,
    ) -> None:
        self.save_path = None
        self.batch = batch
        self.batch_now = batch
        self.slice_num = slice_num
        self.kwargs = kwargs

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        log_path = trainer.logger.log_dir
        if log_path is None:
            return
        name = f"{pl_module.__class__.__name__}_images"
        self.save_path = Path(log_path) / name / "val"
        self.save_path.mkdir(exist_ok=True, parents=True)
        self.batch_now = random.randint(0, self.batch)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.save_path is None:
            return
        if batch_idx == self.batch_now:
            x, y = batch[0:2]
            out_name = f"epoch_{pl_module.current_epoch}"
            save_image(
                make_grid(x[: self.slice_num], **self.kwargs),
                self.save_path / f"{out_name}_A.png",
            )
            save_image(
                make_grid(y[: self.slice_num], **self.kwargs),
                self.save_path / f"{out_name}_B.png",
            )
            save_image(
                make_grid(outputs[: self.slice_num], **self.kwargs),
                self.save_path / f"{out_name}_C.png",
            )

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self.save_path = None

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        log_path = trainer.logger.log_dir
        if log_path is None:
            return
        name = f"{pl_module.__class__.__name__}_images"
        self.save_path = Path(log_path) / name / "test"
        self.save_path.mkdir(exist_ok=True, parents=True)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.save_path is None:
            return
        x, y = batch[0:2]
        out_name = f"batch_{batch_idx}_dataloader_{dataloader_idx}"
        save_image(
            make_grid(x[: self.slice_num], **self.kwargs),
            self.save_path / f"{out_name}_A.png",
        )
        save_image(
            make_grid(y[: self.slice_num], **self.kwargs),
            self.save_path / f"{out_name}_B.png",
        )
        save_image(
            make_grid(outputs[: self.slice_num], **self.kwargs),
            self.save_path / f"{out_name}_C.png",
        )

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.save_path = None
