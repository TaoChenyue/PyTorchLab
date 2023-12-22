from typing import Any, Callable
from pytorchlab.type_hint import LossCallable, OptimizerCallable

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn

from pytorchlab.models._base.gan import ResNetGenerator, NLayerDiscriminator


class Pix2Pix_ResNet(LightningModule):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        height: int,
        width: int,
        depth: int = 2,
        num_blocks: int = 6,
        nf: int = 64,
        dropout: float = 0,
        padding_cls: Callable = nn.ZeroPad2d,
        norm_cls: Callable = nn.BatchNorm2d,
        criterion_gan: LossCallable = nn.MSELoss,
        criterion_image: LossCallable = nn.L1Loss,
        lambda_image_loss: float = 100,
        optimizer_g: OptimizerCallable = torch.optim.Adam,
        optimizer_d: OptimizerCallable = torch.optim.Adam,
    ):
        super().__init__()
        # do not optimize model automatically
        self.automatic_optimization = False
        # init model
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.criterion_gan = criterion_gan()
        self.criterion_image = criterion_image()
        self.lambda_image_loss = lambda_image_loss
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d

        self.generator = ResNetGenerator(
            in_channel,
            out_channel,
            depth=depth,
            num_blocks=num_blocks,
            ngf=nf,
            dropout=dropout,
            padding_cls=padding_cls,
            norm_cls=norm_cls,
        )
        self.discriminator = NLayerDiscriminator(
            channel=in_channel + out_channel,
            ndf=nf,
            depth=depth,
            norm_cls=norm_cls,
        )

        self.d_out_height = height // 2**depth
        self.d_out_width = width // 2**depth

    def forward(self, x):
        return self.generator(x)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer_g = self.optimizer_g(self.generator.parameters())
        optimizer_d = self.optimizer_d(self.discriminator.parameters())
        return optimizer_g, optimizer_d

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        g_loss = self.generator_step(batch)
        d_loss = self.discriminator_step(batch)
        self.log_dict(
            {
                "g_loss": g_loss,
                "d_loss": d_loss,
            },
            sync_dist=True,
            prog_bar=True,
        )

    def generator_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        # zero grad generator optimizer
        optimizer_g: torch.optim.Optimizer = self.optimizers()[0]
        optimizer_g.zero_grad()

        real_A, real_B = batch
        batch_size = real_A.size(0)
        valid = torch.ones(batch_size, 1, self.d_out_height, self.d_out_width).to(
            self.device
        )
        fake_B = self.generator(real_A)
        gan_loss = self.criterion_gan(
            self.discriminator(torch.cat((real_A, fake_B), dim=1)),
            valid,
        )
        image_loss = self.criterion_image(fake_B, real_B)
        g_loss = gan_loss + self.lambda_image_loss * image_loss
        self.manual_backward(g_loss)
        optimizer_g.step()
        return g_loss

    def discriminator_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        # zero grad generator optimizer
        optimizer_d: torch.optim.Optimizer = self.optimizers()[1]
        optimizer_d.zero_grad()
        real_A, real_B = batch
        batch_size = real_A.size(0)
        valid = torch.ones(batch_size, 1, self.d_out_height, self.d_out_width).to(
            self.device
        )
        fake = torch.zeros(batch_size, 1, self.d_out_height, self.d_out_width).to(
            self.device
        )
        fake_B = self.generator(real_A)
        real_loss = self.criterion_gan(
            self.discriminator(torch.cat((real_A, real_B), dim=1)),
            valid,
        )
        fake_loss = self.criterion_gan(
            self.discriminator(torch.cat((real_A, fake_B), dim=1)),
            fake,
        )
        d_loss = real_loss + fake_loss
        self.manual_backward(d_loss)
        optimizer_d.step()
        return d_loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Any:
        real_A, real_B = batch
        fake_B = self(real_A)
        return fake_B

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Any:
        real_A, real_B = batch
        fake_B = self(real_A)
        return fake_B

    def predict_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Any:
        real_A, real_B = batch
        fake_B = self(real_A)
        return fake_B
