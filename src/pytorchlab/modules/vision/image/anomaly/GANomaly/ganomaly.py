from typing import Any, Sequence

import torch
from jsonargparse import lazy_instance
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn
from torch.optim.lr_scheduler import ConstantLR

from pytorchlab.typehints import LRSchedulerCallable, OptimizerCallable


class GANomalyGeneratorLoss(nn.Module):
    def __init__(
        self,
        criterion_gan: nn.Module = lazy_instance(torch.nn.MSELoss),
        criterion_image: nn.Module = lazy_instance(torch.nn.L1Loss),
        criterion_code: nn.Module = lazy_instance(torch.nn.SmoothL1Loss),
        lambda_gan: float = 1,
        lambda_image: float = 50,
        lambda_code: float = 1,
    ) -> None:
        super().__init__()
        self.criterion_gan = criterion_gan
        self.criterion_image = criterion_image
        self.criterion_code = criterion_code
        self.lambda_gan = lambda_gan
        self.lambda_image = lambda_image
        self.lambda_code = lambda_code

    def forward(
        self,
        fake_output: torch.Tensor,
        fake_images: torch.Tensor,
        target_images: torch.Tensor,
        latent_i: torch.Tensor,
        latent_o: torch.Tensor,
    ) -> torch.Tensor:
        valid = torch.ones_like(fake_output)
        loss_gan = self.criterion_gan(fake_output, valid)
        loss_image = self.criterion_image(fake_images, target_images)
        loss_code = self.criterion_code(latent_i, latent_o)
        return (
            self.lambda_gan * loss_gan
            + self.lambda_image * loss_image
            + self.lambda_code * loss_code
        )


class GANomalyDiscriminatorLoss(nn.Module):
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


class GANomalyGenerator(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        encoder2: nn.Module,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder2 = encoder2

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_i = self.encoder(x)
        image = self.decoder(latent_i)
        latent_o = self.encoder2(image)
        return image, latent_i, latent_o


class GANomalyModule(LightningModule):
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        criterion_generator: GANomalyGeneratorLoss = lazy_instance(
            GANomalyGeneratorLoss
        ),
        criterion_discriminator: GANomalyDiscriminatorLoss = lazy_instance(
            GANomalyDiscriminatorLoss
        ),
        optimizer_g: OptimizerCallable = torch.optim.Adam,
        optimizer_d: OptimizerCallable = torch.optim.Adam,
        lr_g: LRSchedulerCallable = ConstantLR,
        lr_d: LRSchedulerCallable = ConstantLR,
    ):
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

    def training_step(
        self, batch: Sequence[torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        # zero grad generator optimizer
        optimizer_g: torch.optim.Optimizer = self.optimizers()[0]
        optimizer_g.zero_grad()

        g_loss, fake_B, latent_i, latent_o = self.generator_step(batch)

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

        return {"g_loss": g_loss, "d_loss": d_loss}

    def generator_step(self, batch: Sequence[torch.Tensor]):
        real_A, real_B = batch[0:2]
        fake_B, latent_i, latent_o = self.generator(real_A)
        output: torch.Tensor = self.discriminator(torch.cat((real_A, fake_B), dim=1))
        g_loss = self.criterion_generator(output, fake_B, real_B, latent_i, latent_o)
        return g_loss, fake_B, latent_i, latent_o

    def discriminator_step(self, batch: Sequence[torch.Tensor]):
        real_A, real_B = batch[0:2]
        fake_B, latent_i, latent_o = self.generator(real_A)
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
        batch: Sequence[torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Any:
        real_A, real_B, labels = batch[0:3]
        g_loss, fake_B, latent_i, latent_o = self.generator_step(batch)
        d_loss = self.discriminator_step(batch)
        score = torch.mean(torch.pow(latent_i - latent_o, 2), dim=(1, 2, 3)).view(-1)
        return {
            "g_loss": g_loss,
            "d_loss": d_loss,
            "inputs": {"images": [real_B], "labels": labels},
            "outputs": {
                "images": [fake_B],
            },
            "metrics": {"anomaly_score": score},
        }

    def validation_step(
        self,
        batch: Sequence[torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Any:
        return self._step(batch, batch_idx, dataloader_idx)

    def test_step(
        self,
        batch: Sequence[torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Any:
        return self._step(batch, batch_idx, dataloader_idx)

    def predict_step(
        self,
        batch: Sequence[torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Any:
        return self._step(batch, batch_idx, dataloader_idx)
