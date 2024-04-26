from typing import Sequence

import torch
from jsonargparse import lazy_instance
from torch import nn

from pytorchlab.typehints import ModuleCallable

__all__ = ["ConvTranspose2dBlock", "SequentialConvTranspose2dBlock"]


class ConvTranspose2dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        norm: ModuleCallable = nn.Identity,
        activation: nn.Module = lazy_instance(nn.ReLU, inplace=True),
    ):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = norm(out_channels)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class SequentialConvTranspose2dBlock(nn.Module):
    def __init__(
        self,
        paras: Sequence[tuple[int, int, int, int, int]],
        norm: ModuleCallable = nn.Identity,
        activation: nn.Module = lazy_instance(nn.ReLU, inplace=True),
    ):
        super().__init__()
        self.model = nn.Sequential(
            *[
                ConvTranspose2dBlock(
                    in_channels=para[0],
                    out_channels=para[1],
                    kernel_size=para[2],
                    stride=para[3],
                    padding=para[4],
                    norm=norm,
                    activation=activation,
                )
                for para in paras
            ]
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)
