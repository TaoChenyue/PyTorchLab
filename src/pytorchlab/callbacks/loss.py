from typing import Any

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from pytorchlab.typehints import OutputsDict
from pytorchlab.utils.state import get_stage

__all__ = [
    "LossCallback",
]


class LossCallback(Callback):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def _batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: OutputsDict,
    ):
        losses = outputs.get("losses", {})
        pl_module.log_dict(
            {f"{get_stage(trainer)}_{k}": v for k, v in losses.items()},
            prog_bar=True,
            sync_dist=True,
        )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: OutputsDict,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._batch_end(trainer, pl_module, outputs)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: OutputsDict,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._batch_end(trainer, pl_module, outputs)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: OutputsDict,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._batch_end(trainer, pl_module, outputs)
