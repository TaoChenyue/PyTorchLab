import itertools
from typing import Any

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn

from pytorchlab.type_hint import ModuleCallable, OptimizerCallable

from .components import Discriminator, GeneratorResNet, init_weights


class CycleGAN(LightningModule):
    def __init__(
        self,
        channel: int = 3,
        height: int = 256,
        width: int = 256,
        n_residual_blocks: int = 4,
        base_features: int = 64,
        lambda_id: float = 5.0,
        lambda_cycle: float = 10.0,
        criterion_gan: ModuleCallable = nn.BCELoss,
        criterion_cycle: ModuleCallable = nn.L1Loss,
        criterion_identity: ModuleCallable = nn.L1Loss,
        optimizer_g: OptimizerCallable = torch.optim.Adam,
        optimizer_d_a: OptimizerCallable = torch.optim.Adam,
        optimizer_d_b: OptimizerCallable = torch.optim.Adam,
    ) -> None:
        super().__init__()
        # do not optimize model automatically
        self.automatic_optimization = False
        # init model
        self.height = height
        self.width = width
        self.channel = channel
        self.G_AB = GeneratorResNet(
            channel=channel,
            num_residual_blocks=n_residual_blocks,
            base_features=base_features,
        )
        self.G_BA = GeneratorResNet(
            channel=channel,
            num_residual_blocks=n_residual_blocks,
            base_features=base_features,
        )
        self.D_A = Discriminator(
            channel=channel,
            base_features=base_features,
        )
        self.D_B = Discriminator(
            channel=channel,
            base_features=base_features,
        )
        self.init_weights()
        self.lambda_id = lambda_id
        self.lambda_cycle = lambda_cycle
        self.criterion_gan = criterion_gan()
        self.criterion_cycle = criterion_cycle()
        self.criterion_identity = criterion_identity()
        self.optimizer_g = optimizer_g
        self.optimizer_d_a = optimizer_d_a
        self.optimizer_d_b = optimizer_d_b

    def init_weights(self) -> None:
        init_weights(self.G_AB)
        init_weights(self.G_BA)
        init_weights(self.D_A)
        init_weights(self.D_B)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.G_AB(z)

    def configure_optimizers(self):
        optimizer_g = self.optimizer_g(
            itertools.chain(self.G_AB.parameters(), self.G_BA.parameters())
        )
        optimizer_d_a = self.optimizer_d_a(self.D_A.parameters())
        optimizer_d_b = self.optimizer_d_b(self.D_B.parameters())
        return optimizer_g, optimizer_d_a, optimizer_d_b

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        g_loss = self.generator_step(batch)
        d_loss = self.discriminator_step(batch)
        self.log_dict(
            {
                "g_loss": g_loss,
                "d_loss": d_loss,
            },
            sync_dist=True,
        )

    def generator_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """generate an image that discriminator regard it as groundtruth

        Args:
            batch_size (int): size for one batch

        Returns:
            torch.Tensor: generator loss
        """
        real_A, real_B = batch
        valid = torch.ones(
            (real_A.size(0), 1, self.height // 2**4, self.width // 2**4)
        ).to(self.device)
        # zero grad generator optimizer
        optimizer_g: torch.optim.Optimizer = self.optimizers()[0]
        optimizer_g.zero_grad()
        # identity loss
        loss_id_A = self.criterion_identity(self.G_BA(real_A), real_A)
        loss_id_B = self.criterion_identity(self.G_AB(real_B), real_B)
        loss_identity = loss_id_A + loss_id_B
        # gan loss
        fake_B = self.G_AB(real_A)
        loss_gan_AB = self.criterion_gan(self.D_B(fake_B), valid)
        fake_A = self.G_BA(real_B)
        loss_gan_BA = self.criterion_gan(self.D_A(fake_A), valid)
        loss_gan = loss_gan_AB + loss_gan_BA
        # reconstruct loss
        recov_A = self.G_BA(fake_B)
        loss_cycle_A = self.criterion_cycle(recov_A, real_A)
        recov_B = self.G_AB(fake_A)
        loss_cycle_B = self.criterion_cycle(recov_B, real_B)
        loss_cycle = loss_cycle_A + loss_cycle_B
        # total loss
        g_loss = (
            loss_gan + self.lambda_id * loss_identity + self.lambda_cycle * loss_cycle
        )
        # loss backward
        self.manual_backward(g_loss)
        # update generator optimizer
        optimizer_g.step()
        return g_loss

    def discriminator_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """discriminate whether x is ground truth or not

        Args:
            x (torch.Tensor): image tensor

        Returns:
            torch.Tensor: discriminator loss
        """
        real_A, real_B = batch
        optimizer_d_a, optimizer_d_b = self.optimizers()[1:3]
        # Discriminator A
        optimizer_d_a.zero_grad()
        valid = torch.ones(
            (real_A.size(0), 1, self.height // 2**4, self.width // 2**4)
        ).to(self.device)
        fake = torch.zeros(
            (real_A.size(0), 1, self.height // 2**4, self.width // 2**4)
        ).to(self.device)
        # Real loss
        loss_real = self.criterion_gan(self.D_A(real_A), valid)
        # Fake loss (on batch of previously generated samples)
        fake_A = self.G_BA(real_B)
        loss_fake = self.criterion_gan(self.D_A(fake_A), fake)
        # Total loss
        loss_D_A = loss_real + loss_fake
        self.manual_backward(loss_D_A)
        optimizer_d_a.step()
        # Discriminator B
        optimizer_d_b.zero_grad()
        loss_real = self.criterion_gan(self.D_B(real_B), valid)
        fake_B = self.G_AB(real_A)
        loss_fake = self.criterion_gan(self.D_B(fake_B), fake)
        loss_D_B = loss_real + loss_fake
        d_loss = loss_D_A + loss_D_B
        return d_loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        real_A, real_B = batch
        fake_B = self.G_AB(real_A)
        return fake_B

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        real_A, real_B = batch
        fake_B = self.G_AB(real_A)
        return fake_B

    def predict_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Any:
        real_A, real_B = batch
        fake_B = self.G_AB(real_A)
        return fake_B
