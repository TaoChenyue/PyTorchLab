from typing import Sequence

import torch
from jsonargparse import lazy_instance
from lightning.pytorch import LightningModule
from torch import Tensor, nn
from torch.optim.lr_scheduler import ConstantLR

from pytorchlab.typehints import LRSchedulerCallable, OptimizerCallable

__all__ = ["GANModule"]


class GANModule(LightningModule):
    def __init__(
        self,
        latent_dim: int,
        generator: nn.Module,
        discriminator: nn.Module,
        criterion: nn.Module = lazy_instance(nn.MSELoss),
        optimizer_g: OptimizerCallable = torch.optim.Adam,
        optimizer_d: OptimizerCallable = torch.optim.Adam,
        lr_g: LRSchedulerCallable = ConstantLR,
        lr_d: LRSchedulerCallable = ConstantLR,
    ) -> None:
        super().__init__()
        # do not optimize model automatically
        self.automatic_optimization = False
        # init model
        self.latent_dim = latent_dim
        self.generator = generator
        self.discriminator = discriminator
        self.criterion = criterion
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.lr_g = lr_g
        self.lr_d = lr_d

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z)

    def configure_optimizers(self):
        optimizer_g = self.optimizer_g(self.generator.parameters())
        lr_g = self.lr_g(optimizer_g)
        optimizer_d = self.optimizer_d(self.discriminator.parameters())
        lr_d = self.lr_d(optimizer_d)
        return [
            {"optimizer": optimizer_g, "lr_scheduler": lr_g},
            {"optimizer": optimizer_d, "lr_scheduler": lr_d},
        ]

    def _get_sample(self, batch_size: int) -> torch.Tensor:
        return torch.randn(batch_size, self.latent_dim).to(self.device)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x = batch[0]
        # zero grad generator optimizer
        optimizer_g: torch.optim.Optimizer = self.optimizers()[0]
        optimizer_g.zero_grad()
        # generation
        g_loss, generated_imgs = self.generator_step(batch)
        # loss backward
        self.manual_backward(g_loss)
        # update generator optimizer
        optimizer_g.step()
        lr_g = self.lr_schedulers()[0]
        lr_g.step()
        # zero grad discriminator optimizer
        optimizer_d: torch.optim.Optimizer = self.optimizers()[1]
        optimizer_d.zero_grad()
        # discrimination
        d_loss = self.discriminator_step(batch)
        # loss backward
        self.manual_backward(d_loss)
        # update discriminator optimizer
        optimizer_d.step()
        lr_d = self.lr_schedulers()[1]
        lr_d.step()

        return {
            "g_loss": g_loss,
            "d_loss": d_loss,
            "outputs": {"images": [generated_imgs]},
        }

    def generator_step(self, batch: Sequence[Tensor]) -> torch.Tensor:
        batch_size = batch[0].size(0)
        # Sample noise
        z = self._get_sample(batch_size=batch_size)
        # Generate images
        generated_imgs: torch.Tensor = self(z)
        # ground truth result (all true)
        valid = torch.ones(batch_size, 1).to(self.device)
        g_loss: torch.Tensor = self.criterion(self.discriminator(generated_imgs), valid)
        return g_loss, generated_imgs

    def discriminator_step(self, batch: Sequence[Tensor]) -> torch.Tensor:
        batch_size = batch[0].size(0)
        # Sample noise
        z = self._get_sample(batch_size=batch_size)
        # Generate images
        generated_imgs: torch.Tensor = self(z)
        # ground truth/ fake result
        real = torch.ones(batch_size, 1).to(self.device)
        fake = torch.zeros(batch_size, 1).to(self.device)
        real_loss = self.criterion(self.discriminator(batch[0]), real)
        fake_loss = self.criterion(self.discriminator(generated_imgs), fake)
        d_loss = (real_loss + fake_loss) / 2
        return d_loss
