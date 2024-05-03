from typing import Any

import torch
from jsonargparse import lazy_instance
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn
from torch.optim.lr_scheduler import ConstantLR

from pytorchlab.experiments.pix2pix.criterions import DiscriminatorLoss, GeneratorLoss
from pytorchlab.experiments.pix2pix.torch_model import PixelDiscriminator, UnetGenerator
from pytorchlab.typehints import (
    ImageDatasetItem,
    LRSchedulerCallable,
    OptimizerCallable,
    OutputsDict,
)

__all__ = [
    "Pix2PixModule",
]


class Pix2PixModule(LightningModule):
    def __init__(
        self,
        generator: nn.Module = lazy_instance(UnetGenerator),
        discriminator: nn.Module = lazy_instance(PixelDiscriminator),
        criterion_g: nn.Module = lazy_instance(GeneratorLoss),
        criterion_d: nn.Module = lazy_instance(DiscriminatorLoss),
        optimizer_g: OptimizerCallable = torch.optim.Adam,
        optimizer_d: OptimizerCallable = torch.optim.Adam,
        lr_g: LRSchedulerCallable = ConstantLR,
        lr_d: LRSchedulerCallable = ConstantLR,
    ):
        """
        Image-to-Image Translation with Conditional Adversarial Networks
        DOI:
            - arxiv: https://doi.org/10.48550/arXiv.1611.07004
            - IEEE: https://doi.org/10.1109/CVPR.2017.632

        Args:
            generator (nn.Module, optional): _description_. Defaults to lazy_instance(Generator).
            discriminator (nn.Module, optional): _description_. Defaults to lazy_instance(Discriminator).
            criterion_g (nn.Module, optional): _description_. Defaults to lazy_instance(GeneratorLoss).
            criterion_d (nn.Module, optional): _description_. Defaults to lazy_instance(DiscriminatorLoss).
            optimizer_g (OptimizerCallable, optional): _description_. Defaults to torch.optim.Adam.
            optimizer_d (OptimizerCallable, optional): _description_. Defaults to torch.optim.Adam.
            lr_g (LRSchedulerCallable, optional): _description_. Defaults to ConstantLR.
            lr_d (LRSchedulerCallable, optional): _description_. Defaults to ConstantLR.
        """
        super().__init__()
        # do not optimize model automatically
        self.automatic_optimization = False
        # init model
        self.criterion_generator = criterion_g
        self.criterion_discriminator = criterion_d
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

    def _step(self, batch: ImageDatasetItem, train: bool = False):
        real_A = batch["image"]
        real_B = batch["generation"]
        if train:
            # train generator
            optimizer_g: torch.optim.Optimizer = self.optimizers()[0]
            optimizer_g.zero_grad()

        fake_B: torch.Tensor = self.generator(real_A)
        output: torch.Tensor = self.discriminator(torch.cat((real_A, fake_B), dim=1))
        g_loss = self.criterion_generator(output, fake_B, real_B)

        if train:
            self.manual_backward(g_loss)
            optimizer_g.step()

        if train:
            # train discriminator
            optimizer_d: torch.optim.Optimizer = self.optimizers()[1]
            optimizer_d.zero_grad()

        # real loss
        output_real: torch.Tensor = self.discriminator(
            torch.cat((real_A, real_B), dim=1)
        )
        output_fake: torch.Tensor = self.discriminator(
            torch.cat((real_A, fake_B.detach()), dim=1)
        )
        d_loss = self.criterion_discriminator(output_fake, output_real)

        if train:
            self.manual_backward(d_loss)
            optimizer_d.step()

        return OutputsDict(
            losses={"g_loss": g_loss, "d_loss": d_loss},
            inputs={
                "image": real_A,
                "generation": real_B,
            },
            outputs={
                "generation": fake_B,
            },
        )

    def training_step(self, batch, batch_idx):
        return self._step(batch, train=True)

    def on_train_epoch_end(self) -> None:
        for lr_scheduler in self.lr_schedulers():
            lr_scheduler.step()

    def validation_step(
        self,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Any:
        return self._step(batch)

    def test_step(
        self,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Any:
        return self._step(batch)
