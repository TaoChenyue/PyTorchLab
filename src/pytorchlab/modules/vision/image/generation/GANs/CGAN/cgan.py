from typing import Sequence

import torch
from jsonargparse import lazy_instance
from lightning.pytorch import LightningModule
from torch import nn
from torch.optim.lr_scheduler import ConstantLR

from pytorchlab.typehints import LRSchedulerCallable, OptimizerCallable

__all__ = ["CGANModule"]


class CGANModule(LightningModule):
    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        generator: nn.Module,
        discriminator: nn.Module,
        criterion: nn.Module = lazy_instance(nn.MSELoss),
        optimizer_g: OptimizerCallable = torch.optim.Adam,
        optimizer_d: OptimizerCallable = torch.optim.Adam,
        lr_g: LRSchedulerCallable = ConstantLR,
        lr_d: LRSchedulerCallable = ConstantLR,
    ) -> None:
        super().__init__()
        # do not optimize discriminator
        self.automatic_optimization = False
        # init model
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.generator = generator
        self.discriminator = discriminator

        self.criterion = criterion
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.lr_g = lr_g
        self.lr_d = lr_d

    def forward(self, z: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return self.generator(z, label)

    def configure_optimizers(self):
        optimizer_g = self.optimizer_g(self.generator.parameters())
        optimizer_d = self.optimizer_d(self.discriminator.parameters())
        lr_g = self.lr_g(optimizer_g)
        lr_d = self.lr_d(optimizer_d)
        return [
            {"optimizer": optimizer_g, "lr_scheduler": lr_g},
            {"optimizer": optimizer_d, "lr_scheduler": lr_d},
        ]

    def training_step(self, batch: Sequence[torch.Tensor], batch_idx: int):
        x = batch[0]
        y = batch[1]

        # zero grad generator optimizer
        optimizer_g: torch.optim.Optimizer = self.optimizers()[0]
        optimizer_g.zero_grad()

        g_loss, generate_images = self.generator_step(batch)

        # loss backward
        self.manual_backward(g_loss)
        # update generator optimizer
        optimizer_g.step()
        lr_g: torch.optim.lr_scheduler.LRScheduler = self.lr_schedulers()[0]
        lr_g.step()

        # zero grad discriminator optimizer
        optimizer_d: torch.optim.Optimizer = self.optimizers()[1]
        optimizer_d.zero_grad()

        d_loss = self.discriminator_step(batch)

        self.manual_backward(d_loss)
        optimizer_d.step()
        lr_d: torch.optim.lr_scheduler.LRScheduler = self.lr_schedulers()[1]
        lr_d.step()

        return {
            "g_loss": g_loss,
            "d_loss": d_loss,
            "outputs": {"images": [generate_images]},
        }

    def generator_step(self, batch: Sequence[torch.Tensor]) -> torch.Tensor:
        batch_size = batch[0].shape[0]
        # Sample noise
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        # Sample labels
        label = torch.randint(0, self.num_classes, (batch_size,)).to(self.device)
        # Generate images
        generated_imgs: torch.Tensor = self(z, label)
        # ground truth result (all true)
        valid = torch.ones(batch_size, 1).to(self.device)
        g_loss: torch.Tensor = self.criterion(
            self.discriminator(generated_imgs, label), valid
        )
        return g_loss, generated_imgs

    def discriminator_step(self, batch: Sequence[torch.Tensor]) -> torch.Tensor:
        x, y = batch[0:2]
        batch_size = x.size(0)

        # Sample noise
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        # Sample labels
        label = torch.randint(0, self.num_classes, (batch_size,)).to(self.device)
        # Generate images
        generated_imgs: torch.Tensor = self(z, label)
        # ground truth/ fake result
        real = torch.ones(batch_size, 1).to(self.device)
        fake = torch.zeros(batch_size, 1).to(self.device)
        real_loss = self.criterion(self.discriminator(x, y), real)
        fake_loss = self.criterion(self.discriminator(generated_imgs, label), fake)
        d_loss = (real_loss + fake_loss) / 2

        return d_loss
