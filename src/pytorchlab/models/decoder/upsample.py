from typing import Sequence

import torch
from jsonargparse import lazy_instance
from torch import nn

from pytorchlab.models.encoder.conv import Conv2dBlock
from pytorchlab.typehints import ModuleCallable

__all__ = ["Upsample2dBlock", "SequentialUpsample2dBlock"]


class Upsample2dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        scale_factor: float = 2,
        mode: str = "bilinear",
        align_corners: bool = True,
        size: Sequence[int] | None = None,
        norm: ModuleCallable = nn.Identity,
        activation: nn.Module = lazy_instance(nn.ReLU, inplace=True),
    ):
        super().__init__()
        self.deconv = nn.Upsample(
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )
        self.conv_block = Conv2dBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm=norm,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        x = self.conv_block(x)
        return x


class SequentialUpsample2dBlock(nn.Module):
    def __init__(
        self,
        paras: Sequence[tuple[int, int, int, int, int]],
        norm: ModuleCallable = nn.Identity,
        activation: nn.Module = lazy_instance(nn.ReLU, inplace=True),
    ):
        super().__init__()
        self.model = nn.Sequential(
            *[
                Upsample2dBlock(
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
