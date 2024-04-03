from typing import Any

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    AUROC,
    Accuracy,
    F1Score,
    Precision,
    Recall,
    Specificity,
)

from pytorchlab.typehints import OutputsDict

__all__ = ["MetricsClassificationCallback"]


class MetricsClassificationCallback(Callback):
    def __init__(
        self,
        name: str,
        num_classes: int = 10,
    ):
        """
        Record metrics:[precision,recall,speicificity,f1score,accuracy,auroc] of classification in validation or test stage.

        Args:
            num_classes (int, optional): number of classes. Defaults to 10.
        """
        super().__init__()
        self.name = name
        self.num_classes = num_classes
        self.metrics_dict = self.get_metrics(name)

    def get_metrics(self, name: str):
        return MetricCollection(
            {
                f"{name}_precision": Precision(
                    task="multiclass", num_classes=self.num_classes
                ),
                f"{name}_recall": Recall(
                    task="multiclass", num_classes=self.num_classes
                ),
                f"{name}_speicificity": Specificity(
                    task="multiclass", num_classes=self.num_classes
                ),
                f"{name}_f1score": F1Score(
                    task="multiclass", num_classes=self.num_classes
                ),
                f"{name}_accuracy": Accuracy(
                    task="multiclass", num_classes=self.num_classes
                ),
                f"{name}_auroc": AUROC(task="multiclass", num_classes=self.num_classes),
            }
        )

    def get_labels(self, outputs: OutputsDict):
        y = outputs.get("inputs", {}).get("labels", {}).get(self.name, None)
        pred = outputs.get("outputs", {}).get("labels", {}).get(self.name, None)
        if y is None or pred is None:
            raise ValueError(
                f"{self.name} not found in labels dict of inputs and outputs"
            )
        return pred, y

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.metrics_dict.to(pl_module.device)
        self.metrics_dict.reset()

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: OutputsDict,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.metrics_dict.update(*self.get_labels(outputs))

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        metrics = self.metrics_dict.compute()
        pl_module.log_dict(
            metrics,
            sync_dist=True,
        )

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.metrics_dict.to(pl_module.device)
        self.metrics_dict.reset()

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.metrics_dict.update(*self.get_labels(outputs))

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        metrics = self.metrics_dict.compute()
        pl_module.log_dict(
            metrics,
            sync_dist=True,
        )
