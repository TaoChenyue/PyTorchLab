from typing import Any

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics import MetricCollection
from torchmetrics.classification import AUROC, Accuracy, F1Score, Precision, Recall

from pytorchlab.typehints import OutputsDict
from pytorchlab.utils.state import get_stage

__all__ = [
    "MetricsBinaryClassificationCallback",
    "MetricsMultiClassificationCallback",
]


class MetricsBinaryClassificationCallback(Callback):
    def __init__(
        self,
        name: str,
    ):
        """
        Record metrics:[precision,recall,speicificity,f1score,accuracy,auroc] of classification in validation or test stage.

        Args:
            num_classes (int, optional): number of classes. Defaults to 10.
        """
        super().__init__()
        self.name = name
        self.metrics_dict = self.get_metrics(name)

    def get_metrics(self, name: str) -> MetricCollection:
        return MetricCollection(
            {
                f"{name}_precision": Precision(task="binary"),
                f"{name}_recall": Recall(task="binary"),
                f"{name}_f1score": F1Score(task="binary"),
                f"{name}_accuracy": Accuracy(task="binary"),
                f"{name}_auroc": AUROC(task="binary"),
            }
        )

    def get_labels(self, outputs: OutputsDict):
        y = outputs.get("inputs", {}).get(self.name, None)
        pred = outputs.get("outputs", {}).get(self.name, None)
        if y is None or pred is None:
            raise ValueError(
                f"{self.name} not found in labels dict of inputs and outputs"
            )
        return pred, y

    def _start(self, pl_module: LightningModule):
        self.metrics_dict.to(pl_module.device)

    def _epoch_start(self):
        self.metrics_dict.reset()

    def _batch_end(self, outputs: OutputsDict):
        self.metrics_dict.update(*self.get_labels(outputs))

    def _epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        metrics = self.metrics_dict.compute()
        metrics_log = {f"{get_stage(trainer)}_{k}": v for k, v in metrics.items()}
        pl_module.log_dict(
            metrics_log,
            sync_dist=True,
        )

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._start(pl_module)

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self._epoch_start()

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: OutputsDict,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._batch_end(outputs)

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self._epoch_end(trainer, pl_module)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._start(pl_module)

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._epoch_start()

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._batch_end(outputs)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._epoch_end(trainer, pl_module)


class MetricsMultiClassificationCallback(MetricsBinaryClassificationCallback):
    def __init__(self, name: str, num_classes: int):
        self.num_classes = num_classes
        super().__init__(name)

    def get_metrics(self, name: str) -> MetricCollection:
        return MetricCollection(
            {
                f"{name}_precision": Precision(
                    task="multiclass", num_classes=self.num_classes
                ),
                f"{name}_recall": Recall(
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
