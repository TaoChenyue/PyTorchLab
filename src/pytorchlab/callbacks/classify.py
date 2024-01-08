import json
from pathlib import Path
from typing import Any

import torch
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
from torchvision.utils import save_image


class ClassifyMetricsCallback(Callback):
    def __init__(self, num_classes: int = 10):
        """
        Record metrics:[precision,recall,speicificity,f1score,accuracy,auroc] of classification in validation or test stage.

        Args:
            num_classes (int, optional): number of classes. Defaults to 10.
        """
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

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
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


class ClassifyShowImageCallback(Callback):
    def __init__(self, num_classes: int = 10, classes: list[str] = None):
        """
        Show image and answer of classification in prediction stage.

        Args:
            num_classes (int, optional): number of classes. Defaults to 10.
            classes (list[str], optional): name of classes. If None, use number instead. Defaults to None.
        """
        super().__init__()
        self.num_classes = num_classes
        if classes == None:
            self.classes = [str(i) for i in range(num_classes)]
        else:
            self.classes = classes
        self.save_path: Path = None

    def on_predict_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        log_path = pl_module.logger.log_dir
        if log_path is None:
            return
        tag = f"{pl_module.__class__.__name__}_images"
        self.save_path = Path(log_path) / tag
        self.save_path.mkdir(parents=True, exist_ok=True)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        x, y = batch
        pred = outputs
        pred = torch.argmax(pred, dim=1)
        save_image(x, self.save_path / f"batch={batch_idx}_image.png")
        ans_dict = {}
        for i, (yy, pp) in enumerate(zip(y, pred)):
            ans_dict[i] = {"label": self.classes[yy], "pred": self.classes[pp]}
        json.dump(
            ans_dict,
            open(
                self.save_path
                / f"batch={batch_idx}_answer_{'✔' if y==pred else '❌'}.json",
                "w",
                encoding="utf-8",
            ),
        )
