from pathlib import Path
from typing import Any

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics import MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.utils import make_grid, save_image


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
        x, y = batch
        self.metrics.reset()
        self.metrics.forward(outputs, y)
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
        x, y = batch
        self.metrics.forward(outputs, y)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.log_dict(
            self.metrics.compute(),
            sync_dist=True,
        )


class ShowImageCallback(Callback):
    def __init__(
        self,
        name: str | None = None,
        num_images: int = 4,
        batch: int = 0,
        **kwargs,
    ) -> None:
        self.save_path = None
        self.name = name
        self.batch = batch
        self.num_images = num_images
        self.kwargs = kwargs

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        log_path = trainer.logger.log_dir
        if log_path is None:
            return
        if self.name is None:
            name = f"{pl_module.__class__.__name__}_images"
        else:
            name = self.name
        self.save_path = Path(log_path) / name
        self.save_path.mkdir(exist_ok=True, parents=True)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.save_path is None:
            return
        if batch_idx == self.batch:
            x, y = batch
            x = x[: self.num_images + 1]
            y = y[: self.num_images + 1]
            out_name = f"epoch_{pl_module.current_epoch}.png"
            images = make_grid(
                torch.cat((x, y, outputs[: self.num_images + 1]), dim=-1), **self.kwargs
            )
            save_image(
                images,
                self.save_path / out_name,
            )
