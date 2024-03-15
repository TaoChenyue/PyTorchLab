from typing import Any, Literal

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from pytorchlab.typehints import OutputsDict

__all__ = ["LossCallback"]


class LossCallback(Callback):
    def __init__(
        self,
        prog_bar: bool = True,
        on_epoch: bool = True,
        on_step: bool = False,
        sync_dist: bool = True,
    ) -> None:
        super().__init__()
        self.prog_bar = prog_bar
        self.on_epoch = on_epoch
        self.on_step = on_step
        self.sync_dist = sync_dist

    def _batch_end(
        self,
        mode: Literal["train", "val", "test"],
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: OutputsDict,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        losses = outputs.get("losses", {})
        pl_module.log_dict(
            {f"{mode}_{k}": v for k, v in losses.items()},
            prog_bar=self.prog_bar if mode in ["train", "val"] else False,
            sync_dist=self.sync_dist,
            on_epoch=self.on_epoch,
            on_step=self.on_step,
        )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: OutputsDict,
        batch: Any,
        batch_idx: int,
    ) -> None:
        losses = outputs.get("losses", {})
        pl_module.log_dict(
            {f"train_{k}": v for k, v in losses.items()},
            prog_bar=self.prog_bar,
            sync_dist=self.sync_dist,
            on_epoch=self.on_epoch,
            on_step=self.on_step,
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
        losses = outputs.get("losses", {})
        pl_module.log_dict(
            {f"val_{k}": v for k, v in losses.items()},
            prog_bar=self.prog_bar,
            sync_dist=self.sync_dist,
            on_epoch=self.on_epoch,
            on_step=self.on_step,
        )

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: OutputsDict,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        losses = outputs.get("losses", {})
        pl_module.log_dict(
            {f"test_{k}": v for k, v in losses.items()},
            prog_bar=self.prog_bar,
            sync_dist=self.sync_dist,
            on_epoch=self.on_epoch,
            on_step=self.on_step,
        )
