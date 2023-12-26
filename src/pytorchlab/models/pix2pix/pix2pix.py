from typing import Any

import torch
from jsonargparse import lazy_instance
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn

from pytorchlab.models._base.gan import NLayerDiscriminator, ResNetGenerator
from pytorchlab.type_hint import LRSchedulerCallable, OptimizerCallable


class Pix2Pix(LightningModule):
    def __init__(
        self,
        generator: nn.Module = lazy_instance(ResNetGenerator),
        discriminator: nn.Module = lazy_instance(NLayerDiscriminator),
        criterion_gan: nn.Module = lazy_instance(nn.MSELoss),
        criterion_image: nn.Module = lazy_instance(nn.L1Loss),
        lambda_image_loss: float = 100,
        optimizer_g: OptimizerCallable = torch.optim.Adam,
        optimizer_d: OptimizerCallable = torch.optim.Adam,
        lr_g: LRSchedulerCallable | None = None,
        lr_d: LRSchedulerCallable | None = None,
    ):
        super().__init__()
        # do not optimize model automatically
        self.automatic_optimization = False
        # init model
        self.criterion_gan = criterion_gan
        self.criterion_image = criterion_image
        self.lambda_image_loss = lambda_image_loss
        self._optimizer_g = optimizer_g
        self._optimizer_d = optimizer_d
        self._lr_g = lr_g
        self._lr_d = lr_d

        self.generator = generator
        self.discriminator = discriminator

    def forward(self, x):
        return self.generator(x)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        self.optimizer_g = self._optimizer_g(self.generator.parameters())
        dict_g = {"optimizer": self.optimizer_g}
        self.optimizer_d = self._optimizer_d(self.discriminator.parameters())
        dict_d = {"optimizer": self.optimizer_d}
        if self._lr_g is not None:
            self.lr_g = self._lr_g(self.optimizer_g)
            dict_g["lr_scheduler"] = self.lr_g
        if self._lr_d is not None:
            self.lr_d = self._lr_d(self.optimizer_d)
            dict_d["lr_scheduler"] = self.lr_d
        return (
            dict_g,
            dict_d,
        )

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        g_loss = self.generator_step(batch)
        d_loss = self.discriminator_step(batch)
        self.log_dict(
            {
                "g_loss": g_loss,
                "d_loss": d_loss,
                "lr_g": self.optimizer_g.param_groups[0]["lr"],
                "lr_d": self.optimizer_d.param_groups[0]["lr"],
            },
            sync_dist=True,
            prog_bar=True,
        )

    def generator_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        # zero grad generator optimizer
        self.optimizer_g.zero_grad()

        real_A, real_B = batch
        fake_B = self.generator(real_A)
        output: torch.Tensor = self.discriminator(torch.cat((real_A, fake_B), dim=1))
        valid = torch.ones(size=output.shape).to(self.device)
        gan_loss = self.criterion_gan(
            output,
            valid,
        )
        image_loss = self.criterion_image(fake_B, real_B)
        g_loss = gan_loss + self.lambda_image_loss * image_loss
        self.manual_backward(g_loss)
        self.optimizer_g.step()
        if self._lr_g is not None:
            self.lr_g.step()
        return g_loss

    def discriminator_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        # zero grad generator optimizer
        self.optimizer_d.zero_grad()
        real_A, real_B = batch
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
        self.manual_backward(d_loss)
        self.optimizer_d.step()
        if self._lr_d is not None:
            self.lr_d.step()
        return d_loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Any:
        real_A, _ = batch
        fake_B = self(real_A)
        return fake_B

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Any:
        real_A, _ = batch
        fake_B = self(real_A)
        return fake_B

    def predict_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Any:
        real_A, _ = batch
        fake_B = self(real_A)
        return fake_B
