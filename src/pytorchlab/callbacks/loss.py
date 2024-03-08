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

    def _batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
        mode: Literal["train", "val", "test"] = "train",
    ) -> None:
        if outputs is None:
            return
        if isinstance(outputs, Tensor):
            loss = outputs
        if isinstance(outputs, Mapping):
            loss = outputs.get("loss", None)
            if loss is None:
                return
        pl_module.log(
            f"{mode}_loss",
            loss,
            prog_bar=self.prog_bar if mode == "train" else False,
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
            trainer, pl_module, outputs, batch, batch_idx, mode="train"
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
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
            mode="val",
            dataloader_idx=dataloader_idx,
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
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
            mode="test",
            dataloader_idx=dataloader_idx,
        )
