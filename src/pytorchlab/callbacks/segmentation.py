from typing import Any

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch import Callback
from torchmetrics import MetricCollection

from pytorchlab.metrics import SegmentationIOUMetric
from pytorchlab.typehints import OutputsDict
from pytorchlab.utils.state import get_stage

__all__ = ["MetricsSegmentationCallback"]


class MetricsSegmentationCallback(Callback):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.metrics_dict = self.get_metrics(name)

    def get_metrics(self, name: str):
        return MetricCollection(
            {
                f"{name}_iou": SegmentationIOUMetric(),
            }
        )

    def get_images(self, outputs: OutputsDict):
        input_images = outputs.get("inputs", {}).get(self.name, None)
        output_images = outputs.get("outputs", {}).get(self.name, None)

        if input_images is None or output_images is None:
            raise ValueError(
                f"Input and output images with key {self.name} must be provided for metrics."
            )
        return input_images, output_images

    def _start(self, pl_module: LightningModule):
        self.metrics_dict.to(pl_module.device)
        self.metrics_dict.reset()

    def _batch_end(self, outputs: OutputsDict):
        self.metrics_dict.update(*self.get_images(outputs))

    def _epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        metrics = self.metrics_dict.compute()
        metrics_log = {f"{get_stage(trainer)}_{k}": v for k, v in metrics.items()}
        pl_module.log_dict(
            metrics_log,
            sync_dist=True,
        )

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._start(pl_module)

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

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: OutputsDict,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._batch_end(outputs)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._epoch_end(trainer, pl_module)


if __name__ == "__main__":
    a = torch.zeros(1, 1, 256, 256)
    a[:, :, 128:, :] = 1
    b = torch.zeros(1, 1, 256, 256)
    b[:, :, :, 128:] = 1
    print(
        torch.sum(torch.where((a > 0.5) & (b > 0.5), 1, 0))
        / torch.sum(torch.where((a > 0.5) | (b > 0.5), 1, 0))
    )
    metric = SegmentationIOUMetric()
    metric.update(a, b)
    print(metric.compute())
