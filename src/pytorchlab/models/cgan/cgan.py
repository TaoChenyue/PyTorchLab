import torch
from jsonargparse import lazy_instance
from lightning.pytorch import LightningModule
from torch import nn

from pytorchlab.lr_scheduler import KeepLR
from pytorchlab.modules.gans.discriminator.linear import ConditionalLinearDiscriminator
from pytorchlab.modules.gans.generator.linear import ConditionalLinearGenerator
from pytorchlab.type_hint import LRSchedulerCallable, ModuleCallable, OptimizerCallable


class CGAN(LightningModule):
    def __init__(
        self,
        generator: nn.Module = lazy_instance(
            ConditionalLinearGenerator,
            channel=1,
            height=28,
            width=28,
            num_classes=10,
            latent_dim=100,
        ),
        discriminator: nn.Module = lazy_instance(
            ConditionalLinearDiscriminator,
            channel=1,
            height=28,
            width=28,
            num_classes=10,
        ),
        criterion: ModuleCallable = nn.BCELoss,
        optimizer_g: OptimizerCallable = torch.optim.Adam,
        optimizer_d: OptimizerCallable = torch.optim.Adam,
        lr_g: LRSchedulerCallable = KeepLR,
        lr_d: LRSchedulerCallable = KeepLR,
    ) -> None:
        super().__init__()
        # do not optimize moddiscriminator
        self.automatic_optimization = False
        # init model
        self.latent_dim = getattr(generator, "latent_dim", None)
        if self.latent_dim is None:
            raise NotImplementedError("latent_dim is not defined in generator")
        ncs_g = getattr(generator, "num_classes", None)
        if ncs_g is None:
            raise NotImplementedError("num_classes is not defined in generator")
        ncs_d = getattr(discriminator, "num_classes", None)
        if ncs_d is None:
            raise NotImplementedError("num_classes is not defined in discriminator")
        if ncs_g != ncs_d:
            raise ValueError("num_classes in generator and discriminator must be equal")

        self.num_classes = ncs_g

        self.generator = generator
        self.discriminator = discriminator

        self.criterion = criterion()
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

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        g_loss = self.generator_step(x.size(0))
        d_loss = self.discriminator_step(x, y)
        self.log_dict(
            {
                "g_loss": g_loss,
                "d_loss": d_loss,
            },
            sync_dist=True,
            prog_bar=True,
        )

    def generator_step(self, batch_size: int) -> torch.Tensor:
        """generate an image that discriminator regard it as groundtruth

        Args:
            batch_size (int): size for one batch

        Returns:
            torch.Tensor: generator loss
        """
        # zero grad generator optimizer
        optimizer_g: torch.optim.Optimizer = self.optimizers()[0]
        optimizer_g.zero_grad()
        # Sample noise
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        # Sample labels
        label = torch.randint(0, self.num_classes, (batch_size,)).to(self.device)
        # Generate images
        generated_imgs: torch.Tensor = self(z, label)
        # ground truth result (all true)
        valid = torch.ones(batch_size, 1).to(self.device)
        g_loss: torch.Tensor = self.criterion(
            self.discriminator(generated_imgs, label), valid
        )
        # loss backward
        self.manual_backward(g_loss)
        # update generator optimizer
        optimizer_g.step()
        lr_g: torch.optim.lr_scheduler.LRScheduler = self.lr_schedulers()[0]
        lr_g.step()
        return g_loss

    def discriminator_step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """discriminate whether x is ground truth or not

        Args:
            x (torch.Tensor): image tensor

        Returns:
            torch.Tensor: discriminator loss
        """
        batch_size = x.size(0)
        # zero grad discriminator optimizer
        optimizer_d: torch.optim.Optimizer = self.optimizers()[1]
        optimizer_d.zero_grad()
        # Sample noise
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        # Sample labels
        label = torch.randint(0, self.num_classes, (batch_size,)).to(self.device)
        # Generate images
        generated_imgs: torch.Tensor = self(z, label)
        # ground truth/ fake result
        real = torch.ones(batch_size, 1).to(self.device)
        fake = torch.zeros(batch_size, 1).to(self.device)
        real_loss = self.criterion(self.discriminator(x, y), real)
        fake_loss = self.criterion(self.discriminator(generated_imgs, label), fake)
        d_loss = real_loss + fake_loss
        self.manual_backward(d_loss)
        optimizer_d.step()
        lr_d: torch.optim.lr_scheduler.LRScheduler = self.lr_schedulers()[1]
        lr_d.step()
        return d_loss
