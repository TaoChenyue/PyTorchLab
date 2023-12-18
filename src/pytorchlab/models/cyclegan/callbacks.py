from pathlib import Path

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchvision.utils import make_grid, save_image


class ShowImageCallback(Callback):
    def __init__(
        self,
        name: str = "cyclegan_image",
        limit_batches: int = 1,
        num_images: int = 4,
        nrow: int = 2,
        padding: int = 2,
    ) -> None:
        self.save_path = None
        self.name = name
        self.limit_batches = limit_batches
        self.num_images = num_images
        self.nrow = nrow
        self.padding = padding

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        log_path = trainer.logger.log_dir
        if log_path is None:
            return
        self.save_path = Path(log_path) / self.name
        self.save_path.mkdir(exist_ok=True, parents=True)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.save_path is None:
            return
        if batch_idx < self.limit_batches:
            x, y = batch
            x = x[: self.num_images]
            y = y[: self.num_images]
            image = make_grid(
                torch.cat((x, y, outputs[: self.num_images]), dim=-1),
                nrow=self.nrow,
                padding=self.padding,
            )
            save_image(
                image,
                self.save_path
                / f"epoch={pl_module.current_epoch}batch={batch_idx}.png",
            )
