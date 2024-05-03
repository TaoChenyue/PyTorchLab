import itertools
from typing import Any

import torch
from jsonargparse import lazy_instance
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn
from torch.optim.lr_scheduler import ConstantLR

from pytorchlab.experiments.cyclegan.criterions import DiscriminatorLoss, GeneratorLoss
from pytorchlab.experiments.cyclegan.torch_model import (
    NlayerDiscriminator,
    ResidualGenerator,
)
from pytorchlab.typehints import (
    ImageDatasetItem,
    LRSchedulerCallable,
    OptimizerCallable,
    OutputsDict,
)

__all__ = [
    "CycleGANModule",
]


class CycleGANModule(LightningModule):
    def __init__(
        self,
        generator_a: nn.Module = lazy_instance(ResidualGenerator),
        generator_b: nn.Module = lazy_instance(ResidualGenerator),
        discriminator_a: nn.Module = lazy_instance(NlayerDiscriminator),
        discriminator_b: nn.Module = lazy_instance(NlayerDiscriminator),
        criterion_g: nn.Module = lazy_instance(GeneratorLoss),
        criterion_d: nn.Module = lazy_instance(DiscriminatorLoss),
        optimizer_g: OptimizerCallable = torch.optim.Adam,
        optimizer_d: OptimizerCallable = torch.optim.Adam,
        lr_g: LRSchedulerCallable = ConstantLR,
        lr_d: LRSchedulerCallable = ConstantLR,
    ):
        """
        Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
        DOI:
            - arxiv: https://doi.org/10.48550/arXiv.1703.10593
            - IEEE: https://doi.org/10.1109/ICCV.2017.244

        Args:
            generator_a (nn.Module, optional): _description_. Defaults to lazy_instance(ResidualGenerator).
            generator_b (nn.Module, optional): _description_. Defaults to lazy_instance(ResidualGenerator).
            discriminator_a (nn.Module, optional): _description_. Defaults to lazy_instance(NlayerDiscriminator).
            discriminator_b (nn.Module, optional): _description_. Defaults to lazy_instance(NlayerDiscriminator).
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

        self.generator_a = generator_a
        self.generator_b = generator_b
        self.discriminator_a = discriminator_a
        self.discriminator_b = discriminator_b

    def forward(self, x):
        return self.generator_a(x)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer_g = self.optimizer_g(
            itertools.chain(
                self.generator_a.parameters(), self.generator_b.parameters()
            )
        )
        optimizer_d = self.optimizer_d(
            itertools.chain(
                self.discriminator_a.parameters(), self.discriminator_b.parameters()
            )
        )
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

        fake_B: torch.Tensor = self.generator_a(real_A)
        rec_A: torch.Tensor = self.generator_b(fake_B)
        idt_A: torch.Tensor = self.generator_b(real_A)

        fake_A: torch.Tensor = self.generator_b(real_B)
        rec_B: torch.Tensor = self.generator_a(fake_A)
        idt_B: torch.Tensor = self.generator_a(real_B)

        output_A: torch.Tensor = self.discriminator_a(fake_B)
        output_B: torch.Tensor = self.discriminator_b(fake_A)

        g_loss_A: torch.Tensor = self.criterion_generator(
            output_A, real_A, rec_A, idt_A
        )
        g_loss_B: torch.Tensor = self.criterion_generator(
            output_B, real_B, rec_B, idt_B
        )
        g_loss: torch.Tensor = g_loss_A + g_loss_B

        if train:
            self.manual_backward(g_loss)
            optimizer_g.step()

        if train:
            # train discriminator
            optimizer_d: torch.optim.Optimizer = self.optimizers()[1]
            optimizer_d.zero_grad()

        # real loss
        output_A_real: torch.Tensor = self.discriminator_a(real_B)
        output_A_fake: torch.Tensor = self.discriminator_a(fake_B.detach())
        d_loss_A = self.criterion_discriminator(output_A_fake, output_A_real)

        output_B_real: torch.Tensor = self.discriminator_b(real_A)
        output_B_fake: torch.Tensor = self.discriminator_b(fake_A.detach())
        d_loss_B = self.criterion_discriminator(output_B_fake, output_B_real)

        d_loss = d_loss_A + d_loss_B

        if train:
            self.manual_backward(d_loss)
            optimizer_d.step()

        return OutputsDict(
            losses={
                "g_loss": g_loss,
                "d_loss": d_loss,
            },
            inputs={
                "real_A": real_A,
                "real_B": real_B,
            },
            outputs={
                "fake_A": fake_A,
                "fake_B": fake_B,
                "rec_A": rec_A,
                "rec_B": rec_B,
                "idt_A": idt_A,
                "idt_B": idt_B,
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
