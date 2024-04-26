import torch
from jsonargparse import lazy_instance
from lightning.pytorch import LightningModule
from torch import nn
from torch.optim.lr_scheduler import ConstantLR

from pytorchlab.experiments.cgan.torch_model import (
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

__all__ = [
    "CGANModule",
]


class CGANModule(LightningModule):
    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        generator: _Generator = lazy_instance(Generator),
        discriminator: _Discriminator = lazy_instance(Discriminator),
        criterion: nn.Module = lazy_instance(nn.BCELoss),
        optimizer_g: OptimizerCallable = torch.optim.Adam,
        optimizer_d: OptimizerCallable = torch.optim.Adam,
        lr_g: LRSchedulerCallable = ConstantLR,
        lr_d: LRSchedulerCallable = ConstantLR,
    ) -> None:
        """
        Conditional Generative Adversarial Nets
        DOI:
            - arxiv: https://doi.org/10.48550/arXiv.1411.1784

        Args:
            latent_dim (int): dimension of latent code
            num_classes (int): number of classes
            generator (nn.Module): module for generate images
            discriminator (nn.Module): module for discriminate images
            criterion (nn.Module, optional): criterion function for gan. Defaults to lazy_instance(nn.BCELoss).
            optimizer_g (OptimizerCallable, optional): optimizer for generator. Defaults to torch.optim.Adam.
            optimizer_d (OptimizerCallable, optional): optimizer for discriminator. Defaults to torch.optim.Adam.
            lr_g (LRSchedulerCallable, optional): learning strategy for generator. Defaults to ConstantLR.
            lr_d (LRSchedulerCallable, optional): learning strategy for discriminator. Defaults to ConstantLR.
        """
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

    def training_step(self, batch: ImageDatasetItem, batch_idx: int):
        image = batch["image"]
        label = batch["label"]
        batch_size = image.size(0)
        # train generator
        optimizer_g: torch.optim.Optimizer = self.optimizers()[0]
        optimizer_g.zero_grad()

        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_label = torch.randint_like(label, low=0, high=self.num_classes)
        fake_image: torch.Tensor = self(z, fake_label)
        fake_output = self.discriminator(fake_image, fake_label)
        valid = torch.ones_like(fake_output)
        loss_generator: torch.Tensor = self.criterion(fake_output, valid)

        self.manual_backward(loss_generator)
        optimizer_g.step()
        lr_g: torch.optim.lr_scheduler.LRScheduler = self.lr_schedulers()[0]
        lr_g.step()

        # train discriminator
        optimizer_d: torch.optim.Optimizer = self.optimizers()[1]
        optimizer_d.zero_grad()

        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_label = torch.randint_like(label, low=0, high=self.num_classes)
        fake_image: torch.Tensor = self(z, fake_label)
        fake_output = self.discriminator(fake_image, fake_label)
        real_output = self.discriminator(image, label)
        valid = torch.ones_like(fake_output)
        fake = torch.zeros_like(fake_output)
        loss_real = self.criterion(real_output, valid)
        loss_fake = self.criterion(fake_output, fake)
        loss_discriminator = (loss_real + loss_fake) / 2

        self.manual_backward(loss_discriminator)
        optimizer_d.step()
        lr_d: torch.optim.lr_scheduler.LRScheduler = self.lr_schedulers()[1]
        lr_d.step()

        return OutputsDict(
            losses={
                "g_loss": loss_generator,
                "d_loss": loss_discriminator,
            },
            inputs={"image": batch["image"], "label": label},
            outputs={"image": fake_image},
        )
