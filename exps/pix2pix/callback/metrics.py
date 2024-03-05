from typing import Any

from lightning import LightningModule, Trainer
from lightning.pytorch import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics import MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


class IQAMetricsCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.metrics = MetricCollection(
            {
                "psnr": PeakSignalNoiseRatio(data_range=1.0),
                "ssim": StructuralSimilarityIndexMeasure(data_range=1.0),
            }
        )

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self.metrics.to(pl_module.device)
        self.metrics.reset()

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        x, y = batch[0:2]
        self.metrics.forward(outputs, y)
        
        
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.log_dict(
            self.metrics.compute(),
            sync_dist=True,
            prog_bar=True,
        )

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.metrics.to(pl_module.device)
        self.metrics.reset()

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        x, y = batch[0:2]
        self.metrics.forward(outputs, y)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.log_dict(
            self.metrics.compute(),
            sync_dist=True,
        )