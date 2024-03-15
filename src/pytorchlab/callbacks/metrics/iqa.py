from typing import Any

from lightning import LightningModule, Trainer
from lightning.pytorch import Callback
from torchmetrics import MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from pytorchlab.typehints import OutputsDict

__all__ = ["MetricsIQACallback"]


class MetricsIQACallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.metrics_dict: dict[str, MetricCollection] | None = None

    def get_metrics(self, name: str):
        return MetricCollection(
            {
                f"{name}_psnr": PeakSignalNoiseRatio(data_range=1.0),
                f"{name}_ssim": StructuralSimilarityIndexMeasure(data_range=1.0),
            }
        )

    def get_images(self, outputs: OutputsDict):
        input_images = outputs.get("inputs", {}).get("images", {})
        output_images = outputs.get("outputs", {}).get("images", {})
        return {
            k: (output_images[k], input_images[k])
            for k in output_images.keys()
            if k in input_images.keys()
        }

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.metrics_dict = None

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: OutputsDict,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        images = self.get_images(outputs)
        if self.metrics_dict is None:
            self.metrics_dict = {k: self.get_metrics(k).to(pl_module.device) for k in images.keys()}
        for k, v in images.items():
            self.metrics_dict[k].update(*v)
            
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        for k, v in self.metrics_dict.items():
            metrics = v.compute()
            pl_module.log_dict(
                metrics,
                sync_dist=True,
            )
            v.reset()

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.metrics_dict = None

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: OutputsDict,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        images = self.get_images(outputs)
        if self.metrics_dict is None:
            self.metrics_dict = {k: self.get_metrics(k).to(pl_module.device) for k in images.keys()}
        for k, v in images.items():
            self.metrics_dict[k].update(*v)
            
    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        for k, v in self.metrics_dict.items():
            metrics = v.compute()
            pl_module.log_dict(
                metrics,
                sync_dist=True,
            )
            v.reset()
