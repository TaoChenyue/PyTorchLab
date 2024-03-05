from pathlib import Path
from typing import Any, Mapping

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from torch import Tensor
from torchvision.utils import make_grid, save_image


class CGANCallback(Callback):
    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        nums: int = 9,
        detail: bool = False,
        **kwargs,
    ):
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.nums = nums
        self.detail = detail
        self.kwargs = kwargs

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        z = torch.randn(self.nums * self.num_classes, self.latent_dim).to(
            pl_module.device
        )
        label = torch.tensor(
            [i for i in range(self.num_classes) for _ in range(self.nums)]
        ).to(pl_module.device)
        with torch.no_grad():
            pl_module.eval()
            images = pl_module(z, label)
            pl_module.train()
        images = make_grid(images, **self.kwargs)
        log_path = pl_module.logger.log_dir
        if log_path is None:
            return
        tag = f"{pl_module.__class__.__name__}_images"
        save_path = Path(log_path) / tag / "train"
        save_path.mkdir(exist_ok=True, parents=True)
        if self.detail:
            out_name = f"epoch_{trainer.current_epoch}.png"
            save_name = save_path / out_name
            save_image(images, save_name)
        save_image(images, save_path / "latest.png")

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: torch.Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        x = batch[0]
        images = make_grid(
            torch.cat(
                [x, outputs],
                dim=-1,
            ),
            **self.kwargs,
        )
        log_path = pl_module.logger.log_dir
        if log_path is None:
            return
        tag = f"{pl_module.__class__.__name__}_images"
        save_path = Path(log_path) / tag / "test"
        save_path.mkdir(exist_ok=True, parents=True)
        out_name = f"batch_{batch_idx}_dataloader_{dataloader_idx}.png"
        save_name = save_path / out_name
        save_image(images, save_name)
