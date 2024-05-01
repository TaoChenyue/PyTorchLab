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
