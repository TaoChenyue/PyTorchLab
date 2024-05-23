from typing import Any

import torch
from jsonargparse import lazy_instance
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn
from torch.optim.lr_scheduler import ConstantLR

from pytorchlab.typehints import (
    ImageDatasetItem,
    LRSchedulerCallable,
    OptimizerCallable,
    OutputDict,
    OutputsDict,
)


class Pix2PixGeneratorLoss(nn.Module):
    def __init__(
        self,
        criterion_gan: nn.Module = lazy_instance(torch.nn.MSELoss),
        criterion_image: nn.Module = lazy_instance(torch.nn.L1Loss),
        lambda_gan: float = 1,
        lambda_image: float = 100,
    ) -> None:
        super().__init__()
        self.criterion_gan = criterion_gan
        self.criterion_image = criterion_image
        self.lambda_gan = lambda_gan
        self.lambda_image = lambda_image

    def forward(
        self,
        fake_output: torch.Tensor,
        fake_images: torch.Tensor,
        target_images: torch.Tensor,
    ) -> torch.Tensor:
        valid = torch.ones_like(fake_output)
        loss_gan = self.criterion_gan(fake_output, valid)
        loss_image = self.criterion_image(fake_images, target_images)
        return self.lambda_gan * loss_gan + self.lambda_image * loss_image


class Pix2PixDiscriminatorLoss(nn.Module):
    def __init__(self, criterion: nn.Module = lazy_instance(torch.nn.MSELoss)):
        super().__init__()
        self.criterion = criterion

    def forward(
        self,
        fake_output: torch.Tensor,
        real_output: torch.Tensor,
    ):
        valid = torch.ones_like(real_output)
        loss_real = self.criterion(real_output, valid)
        fake = torch.zeros_like(fake_output)
        loss_fake = self.criterion(fake_output, fake)
        return (loss_real + loss_fake) / 2


class Pix2PixModule(LightningModule):
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        criterion_generator: Pix2PixGeneratorLoss = lazy_instance(Pix2PixGeneratorLoss),
        criterion_discriminator: Pix2PixDiscriminatorLoss = lazy_instance(
            Pix2PixDiscriminatorLoss
        ),
        optimizer_g: OptimizerCallable = torch.optim.Adam,
        optimizer_d: OptimizerCallable = torch.optim.Adam,
        lr_g: LRSchedulerCallable = ConstantLR,
        lr_d: LRSchedulerCallable = ConstantLR,
    ):
        """
        _summary_

        OutputsDict(
            losses={"g_loss": g_loss, "d_loss": d_loss},
            inputs=OutputDict(
                images={
                    "image": batch["image1"],
                    "reconstruct": batch["image2"],
                }
            ),
            outputs=OutputDict(
                images={"reconstruct": fake_B},
            ),
        )

        Args:
            generator (nn.Module): _description_
            discriminator (nn.Module): _description_
            criterion_generator (Pix2PixGeneratorLoss, optional): _description_. Defaults to lazy_instance(Pix2PixGeneratorLoss).
            criterion_discriminator (Pix2PixDiscriminatorLoss, optional): _description_. Defaults to lazy_instance( Pix2PixDiscriminatorLoss ).
            optimizer_g (OptimizerCallable, optional): _description_. Defaults to torch.optim.Adam.
            optimizer_d (OptimizerCallable, optional): _description_. Defaults to torch.optim.Adam.
            lr_g (LRSchedulerCallable, optional): _description_. Defaults to ConstantLR.
            lr_d (LRSchedulerCallable, optional): _description_. Defaults to ConstantLR.
        """
        super().__init__()
        # do not optimize model automatically
        self.automatic_optimization = False
        # init model
        self.criterion_generator = criterion_generator
        self.criterion_discriminator = criterion_discriminator
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

    def training_step(self, batch: ImageDatasetItem, batch_idx: int) -> STEP_OUTPUT:
        # zero grad generator optimizer
        optimizer_g: torch.optim.Optimizer = self.optimizers()[0]
        optimizer_g.zero_grad()

        g_loss, _ = self.generator_step(batch)

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

        return OutputsDict(losses={"g_loss": g_loss, "d_loss": d_loss})

    def generator_step(self, batch: ImageDatasetItem):
        real_A = batch["image"]
        real_B = batch["image2"]
        fake_B = self.generator(real_A)
        output: torch.Tensor = self.discriminator(torch.cat((real_A, fake_B), dim=1))
        g_loss = self.criterion_generator(output, fake_B, real_B)

        return g_loss, fake_B

    def discriminator_step(self, batch: ImageDatasetItem):
        real_A = batch["image"]
        real_B = batch["image2"]
        fake_B = self.generator(real_A)
        # real loss
        output_real: torch.Tensor = self.discriminator(
            torch.cat((real_A, real_B), dim=1)
        )
        output_fake: torch.Tensor = self.discriminator(
            torch.cat((real_A, fake_B), dim=1)
        )
        d_loss = self.criterion_discriminator(output_fake, output_real)
        return d_loss

    def _step(
        self,
        batch: ImageDatasetItem,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Any:
        g_loss, fake_B = self.generator_step(batch)
        d_loss = self.discriminator_step(batch)
        return OutputsDict(
            losses={"g_loss": g_loss, "d_loss": d_loss},
            inputs=OutputDict(
                images={
                    "image": batch["image"],
                    "reconstruct": batch["image2"],
                }
            ),
            outputs=OutputDict(
                images={"reconstruct": fake_B},
            ),
        )

    def validation_step(
        self,
        batch: ImageDatasetItem,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Any:
        return self._step(batch, batch_idx, dataloader_idx)

    def test_step(
        self,
        batch: ImageDatasetItem,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Any:
        return self._step(batch, batch_idx, dataloader_idx)

    def predict_step(
        self,
        batch: ImageDatasetItem,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Any:
        return self._step(batch, batch_idx, dataloader_idx)
