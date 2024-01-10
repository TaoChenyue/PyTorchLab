from abc import ABCMeta

import torch
from torch import nn
from torch.nn import functional as F


class _GradientLoss(nn.Module, metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()

    def get_left(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-1, :]

    def get_right(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, 1:, :]

    def get_up(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :, :-1]

    def get_down(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :, 1:]

    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class GradientLoss(_GradientLoss):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        a_x = torch.abs(self.get_left(a) - self.get_right(a))
        a_y = torch.abs(self.get_up(a) - self.get_down(a))
        b_x = torch.abs(self.get_left(b) - self.get_right(b))
        b_y = torch.abs(self.getup(b) - self.get_down(b))
        return F.mse_loss(a_x, b_x) + F.mse_loss(a_y, b_y)


class TVLoss(_GradientLoss):
    def __init__(self) -> None:
        """
        Total Variant Loss
        """
        super().__init__()

    def forward(self, a: torch.Tensor):
        return F.mse_loss(self.get_right(a), self.get_left(a)) + F.mse_loss(
            self.get_down(a), self.get_up(a)
        )
