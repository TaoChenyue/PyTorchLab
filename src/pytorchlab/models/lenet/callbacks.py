from typing import Any

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics import MetricCollection
from torchmetrics.classification import (AUROC, Accuracy, F1Score, Precision,
                                         Recall, Specificity)


class LeNetCallback(Callback):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # //////////////////////////////////////////////////
        # Metrics
        self.metrics = MetricCollection(
            {
                "precision": Precision(task="multiclass", num_classes=num_classes),
                "recall": Recall(task="multiclass", num_classes=num_classes),
                "speicificity": Specificity(task="multiclass", num_classes=num_classes),
                "f1score": F1Score(task="multiclass", num_classes=num_classes),
                "accuracy": Accuracy(task="multiclass", num_classes=num_classes),
                "auroc": AUROC(task="multiclass", num_classes=num_classes),
            }
        )
        
    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.metrics.to(pl_module.device)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        _, y = batch
        self.metrics.forward(outputs, y)

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        metrics = self.metrics.compute()
        pl_module.log_dict(
            metrics,
            sync_dist=True,
        )
        self.metrics.reset()
        
    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.metrics.to(pl_module.device)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        _, y = batch
        self.metrics.forward(outputs, y)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        metrics = self.metrics.compute()
        pl_module.log_dict(
            metrics,
            sync_dist=True,
        )
        self.metrics.reset()
