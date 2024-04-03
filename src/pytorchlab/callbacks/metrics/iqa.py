from typing import Any

from lightning import LightningModule, Trainer
from lightning.pytorch import Callback
from torchmetrics import MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from pytorchlab.typehints import OutputsDict

__all__ = ["MetricsImageQualityCallback"]


class MetricsImageQualityCallback(Callback):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.metrics_dict = self.get_metrics(name)

    def get_metrics(self, name: str):
        return MetricCollection(
            {
                f"{name}_psnr": PeakSignalNoiseRatio(data_range=1.0),
                f"{name}_ssim": StructuralSimilarityIndexMeasure(data_range=1.0),
            }
        )

    def get_images(self, outputs: OutputsDict):
        input_images = outputs.get("inputs", {}).get("images", {}).get(self.name, None)
        output_images = (
            outputs.get("outputs", {}).get("images", {}).get(self.name, None)
        )
        if input_images is None or output_images is None:
            raise ValueError(
                f"Input and output images with key {self.name} must be provided for IQA metrics."
            )
        return input_images, output_images

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
        self.metrics_dict.update(*self.get_images(outputs))

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
        outputs: OutputsDict,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.metrics_dict.update(*self.get_images(outputs))

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        metrics = self.metrics_dict.compute()
        pl_module.log_dict(
            metrics,
            sync_dist=True,
        )
