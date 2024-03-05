from typing import Any, Sequence

import torch
from jsonargparse import lazy_instance
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn
from torch.optim.lr_scheduler import ConstantLR

from pytorchlab.type_hint import LRSchedulerCallable, OptimizerCallable


class Pix2Pix(LightningModule):
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        criterion_gan: nn.Module = lazy_instance(torch.nn.MSELoss),
        criterion_image: nn.Module = lazy_instance(torch.nn.L1Loss),
        lambda_image_loss: float = 100,
        optimizer_g: OptimizerCallable = torch.optim.Adam,
        optimizer_d: OptimizerCallable = torch.optim.Adam,
        lr_g: LRSchedulerCallable = lazy_instance(ConstantLR, factor=1.0),
        lr_d: LRSchedulerCallable = lazy_instance(ConstantLR, factor=1.0),
    ):
        super().__init__()
        # do not optimize model automatically
        self.automatic_optimization = False
        # init model
        self.criterion_gan = criterion_gan
        self.criterion_image = criterion_image
        self.lambda_image_loss = lambda_image_loss
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.lr_g = lr_g
        self.lr_d = lr_d

        self.generator = generator
        self.discriminator = discriminator

    def forward(self, x):
        return self.generator(x)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer_g = self.optimizer_g(self.generator.parameters())
        optimizer_d = self.optimizer_d(self.discriminator.parameters())
        lr_g = self.lr_g(optimizer_g)
        lr_d = self.lr_d(optimizer_d)
        return [
            {"optimizer": optimizer_g, "lr_scheduler": lr_g},
            {"optimizer": optimizer_d, "lr_scheduler": lr_d},
        ]

    def training_step(
        self, batch: Sequence[torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        # zero grad generator optimizer
        optimizer_g: torch.optim.Optimizer = self.optimizers()[0]
        optimizer_g.zero_grad()

        g_loss = self.generator_step(batch)

        self.manual_backward(g_loss)
        optimizer_g.step()
        lr_g = self.lr_schedulers()[0]
        lr_g.step()

        # zero grad generator optimizer
        optimizer_d: torch.optim.Optimizer = self.optimizers()[1]
        optimizer_d.zero_grad()

        d_loss = self.discriminator_step(batch)

        self.manual_backward(d_loss)
        optimizer_d.step()
        lr_d = self.lr_schedulers()[1]
        lr_d.step()

        self.log_dict(
            {
                "g_loss": g_loss,
                "d_loss": d_loss,
                # "lr_g": self.optimizer_g.param_groups[0]["lr"],
                # "lr_d": self.optimizer_d.param_groups[0]["lr"],
            },
            sync_dist=True,
            prog_bar=True,
        )

    def generator_step(self, batch: Sequence[torch.Tensor]):
        real_A, real_B = batch[0:2]
        fake_B = self.generator(real_A)
        output: torch.Tensor = self.discriminator(torch.cat((real_A, fake_B), dim=1))
        valid = torch.ones(size=output.shape).to(self.device)
        gan_loss = self.criterion_gan(
            output,
            valid,
        )
        image_loss = self.criterion_image(fake_B, real_B)
        g_loss = gan_loss + self.lambda_image_loss * image_loss

        return g_loss

    def discriminator_step(self, batch: Sequence[torch.Tensor]):
        real_A, real_B = batch[0:2]
        # real loss
        output_real: torch.Tensor = self.discriminator(
            torch.cat((real_A, real_B), dim=1)
        )
        valid = torch.ones(size=output_real.shape).to(self.device)
        real_loss = self.criterion_gan(
            output_real,
            valid,
        )
        # fake loss
        fake_B = self.generator(real_A)
        output_fake: torch.Tensor = self.discriminator(
            torch.cat((real_A, fake_B), dim=1)
        )
        fake = torch.zeros(size=output_fake.shape).to(self.device)
        fake_loss = self.criterion_gan(
            output_fake,
            fake,
        )
        d_loss = real_loss + fake_loss

        return d_loss

    def validation_step(self, batch: Sequence[torch.Tensor], batch_idx: int) -> Any:
        real_A, real_B = batch[0:2]
        fake_B = self(real_A)
        return fake_B

    def test_step(
        self,
        batch: Sequence[torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Any:
        real_A, real_B = batch[0:2]
        fake_B = self(real_A)
        return fake_B

    def predict_step(
        self,
        batch: Sequence[torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Any:
        real_A, real_B = batch[0:2]
        fake_B = self(real_A)
        return fake_B
