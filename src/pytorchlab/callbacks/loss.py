from typing import Any, Literal, Mapping

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor


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

    def _get_losses(self, outputs: Tensor | Mapping[str, Any] | None):
        if not isinstance(outputs, Mapping):
            return None
        losses = {}
        for k, v in outputs.items():
            if "loss" in k:
                losses[k] = v
        return losses

    def _batch_end(
        self,
        mode: Literal["train", "val", "test"],
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        losses = self._get_losses(outputs)
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
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        return self._batch_end(
            "train",
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
        )

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
            "val",
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
            # dataloader_idx=dataloader_idx,
        )

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
            "test",
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
            # dataloader_idx=dataloader_idx,
        )
