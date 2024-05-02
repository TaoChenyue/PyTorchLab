import torch
from jsonargparse import lazy_instance
from torch import nn

__all__ = [
    "GeneratorLoss",
    "DiscriminatorLoss",
]


class GeneratorLoss(nn.Module):
    def __init__(
        self,
        criterion_gan: nn.Module = lazy_instance(torch.nn.BCEWithLogitsLoss),
        criterion_cycle: nn.Module = lazy_instance(torch.nn.L1Loss),
        criterion_identity: nn.Module = lazy_instance(torch.nn.L1Loss),
        lambda_gan: float = 1,
        lambda_cycle: float = 10,
        lambda_identity: float = 5,
    ) -> None:
        super().__init__()
        self.criterion_gan = criterion_gan
        self.criterion_cycle = criterion_cycle
        self.criterion_identity = criterion_identity
        self.lambda_gan = lambda_gan
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def forward(
        self,
        fake_output: torch.Tensor,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        identity_images: torch.Tensor,
    ) -> torch.Tensor:
        valid = torch.ones_like(fake_output)
        loss_gan = self.criterion_gan(fake_output, valid)
        loss_cycle = self.criterion_cycle(fake_images, real_images)
        loss_identity = self.criterion_identity(identity_images, real_images)
        return (
            self.lambda_gan * loss_gan
            + self.lambda_cycle * loss_cycle
            + self.lambda_identity * loss_identity
        )


class DiscriminatorLoss(nn.Module):
    def __init__(
        self, criterion: nn.Module = lazy_instance(torch.nn.BCEWithLogitsLoss)
    ):
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
