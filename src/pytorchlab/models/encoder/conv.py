from typing import Sequence

import torch
from jsonargparse import lazy_instance
from torch import nn

from pytorchlab.typehints import ModuleCallable

__all__ = ["Conv2dBlock", "SequentialConv2dBlock", "SequentialConvPool2dBlock"]


class Conv2dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int | tuple[int, int, int, int] = 1,
        norm: ModuleCallable = nn.Identity,
        activation: nn.Module = lazy_instance(nn.ReLU, inplace=True),
        padding_method: ModuleCallable = nn.ZeroPad2d,
    ):
        super().__init__()
        self.padding = padding_method(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.norm = norm(out_channels)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.padding(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class SequentialConv2dBlock(nn.Module):
    def __init__(
        self,
        paras: Sequence[tuple[int, int, int, int, int]],
        norm: ModuleCallable = nn.Identity,
        activation: nn.Module = lazy_instance(nn.ReLU, inplace=True),
        padding_method: ModuleCallable = nn.ZeroPad2d,
    ):
        super().__init__()
        self.model = nn.Sequential(
            *[
                Conv2dBlock(
                    in_channels=para[0],
                    out_channels=para[1],
                    kernel_size=para[2],
                    stride=para[3],
                    padding=para[4],
                    norm=norm,
                    activation=activation,
                    padding_method=padding_method,
                )
                for para in paras
            ]
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)


class SequentialConvPool2dBlock(nn.Module):
    def __init__(
        self,
        paras: Sequence[tuple[int, int, int, int, int]],
        norm: ModuleCallable = nn.Identity,
        activation: nn.Module = lazy_instance(nn.ReLU, inplace=True),
        padding_method: ModuleCallable = nn.ZeroPad2d,
        pool: nn.Module = lazy_instance(nn.MaxPool2d, kernel_size=2),
    ):
        super().__init__()
        self.model = nn.Sequential(
            *[
                nn.Sequential(
                    Conv2dBlock(
                        in_channels=para[0],
                        out_channels=para[1],
                        kernel_size=para[2],
                        stride=para[3],
                        padding=para[4],
                        norm=norm,
                        activation=activation,
                        padding_method=padding_method,
                    ),
                    pool,
                )
                for para in paras
            ]
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)
