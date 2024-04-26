import torch
from jsonargparse import lazy_instance
from lightning.pytorch import LightningModule
from torch import nn
from torch.optim.lr_scheduler import ConstantLR

from pytorchlab.experiments.gan.torch_model import (
    Discriminator,
    Generator,
    _Discriminator,
    _Generator,
)
from pytorchlab.typehints import (
    ImageDatasetItem,
    LRSchedulerCallable,
    OptimizerCallable,
    OutputsDict,
)

__all__ = ["GANModule"]


class GANModule(LightningModule):
    def __init__(
        self,
        latent_dim: int,
        generator: _Generator = lazy_instance(Generator),
        discriminator: _Discriminator = lazy_instance(Discriminator),
        criterion: nn.Module = lazy_instance(nn.BCELoss),
        optimizer_g: OptimizerCallable = torch.optim.Adam,
        optimizer_d: OptimizerCallable = torch.optim.Adam,
        lr_g: LRSchedulerCallable = ConstantLR,
        lr_d: LRSchedulerCallable = ConstantLR,
    ) -> None:
        """
        Generative Adversarial Networks
        DOI:
            - arxiv: https://doi.org/10.48550/arXiv.1406.2661
            - IEEE: https://doi.org/10.1109/ICCCNT56998.2023.10306417

        Args:
            latent_dim (int): dimension of latent code
            generator (nn.Module): module for generate images
            discriminator (nn.Module): module for discriminate images
            criterion (nn.Module, optional): criterion function for gan. Defaults to lazy_instance(nn.BCELoss).
            optimizer_g (OptimizerCallable, optional): optimizer for generator. Defaults to torch.optim.Adam.
            optimizer_d (OptimizerCallable, optional): optimizer for discriminator. Defaults to torch.optim.Adam.
            lr_g (LRSchedulerCallable, optional): learning strategy for generator. Defaults to ConstantLR.
            lr_d (LRSchedulerCallable, optional): learning strategy for discriminator. Defaults to ConstantLR.
        """
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

    def training_step(self, batch: ImageDatasetItem, batch_idx: int):
        # train generator
        optimizer_g: torch.optim.Optimizer = self.optimizers()[0]
        optimizer_g.zero_grad()

        z = torch.randn(batch["image"].size(0), self.latent_dim).to(self.device)
        fake_image: torch.Tensor = self(z)
        fake_output: torch.Tensor = self.discriminator(fake_image)
        valid = torch.ones_like(fake_output)

        loss_generator = self.criterion(fake_output, valid)

        self.manual_backward(loss_generator)
        optimizer_g.step()
        lr_g = self.lr_schedulers()[0]
        lr_g.step()

        # train discriminator
        optimizer_d: torch.optim.Optimizer = self.optimizers()[1]
        optimizer_d.zero_grad()

        z = torch.randn(batch["image"].size(0), self.latent_dim).to(self.device)
        fake_image: torch.Tensor = self(z)
        fake_output: torch.Tensor = self.discriminator(fake_image)
        real_output: torch.Tensor = self.discriminator(batch["image"])
        valid = torch.ones_like(fake_output)
        fake = torch.zeros_like(fake_output)

        loss_fake = self.criterion(fake_output, fake)
        loss_real = self.criterion(real_output, valid)

        loss_discriminator = (loss_fake + loss_real) / 2
        # loss backward
        self.manual_backward(loss_discriminator)
        # update discriminator optimizer
        optimizer_d.step()
        lr_d = self.lr_schedulers()[1]
        lr_d.step()

        return OutputsDict(
            losses={
                "g_loss": loss_generator,
                "d_loss": loss_discriminator,
            },
            inputs={"image": batch["image"], "latent_code": z},
            outputs={"image": fake_image},
        )
