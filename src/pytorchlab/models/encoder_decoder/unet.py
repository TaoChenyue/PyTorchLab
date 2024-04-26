from typing import Optional

import torch
from jsonargparse import lazy_instance
from torch import Tensor, nn

from pytorchlab.models.encoder.conv import Conv2dBlock
from pytorchlab.typehints import ModuleCallable

__all__ = [
    "UNetSkipConnectionBlock",
]


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


class UNetSkipConnectionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        sub_channels: int,
        out_channels: int,
        norm: ModuleCallable = nn.BatchNorm2d,
        submodule: Optional["UNetSkipConnectionBlock"] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.down_block = nn.Sequential(
            nn.Conv2d(in_channels, sub_channels, kernel_size=4, stride=2, padding=1),
            norm(sub_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.submodule = nn.Identity() if submodule is None else submodule
        c = sub_channels if submodule is None else submodule.out_channels
        self.up_block = nn.Sequential(
            nn.ConvTranspose2d(
                c, out_channels - in_channels, kernel_size=4, stride=2, padding=1
            ),
            norm(out_channels - in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        down = self.down_block(x)
        sub = self.submodule(down)
        up = self.up_block(sub)
        return torch.cat([x, up], dim=1)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        nf: int = 64,
        depth: int = 7,
        hold_depth: int = 2,
        norm: ModuleCallable = nn.BatchNorm2d,
        activation: nn.Module = lazy_instance(nn.Sigmoid),
    ):
        super().__init__()

        df_num = 2**hold_depth
        unetblock = UNetSkipConnectionBlock(
            in_channels=nf * df_num,
            sub_channels=nf * df_num,
            out_channels=nf * df_num * 2,
            norm=norm,
        )
        for _ in range(depth - hold_depth - 1):
            unetblock = UNetSkipConnectionBlock(
                in_channels=nf * df_num,
                sub_channels=nf * df_num,
                out_channels=nf * df_num * 2,
                norm=norm,
                submodule=unetblock,
            )
        for i in range(hold_depth):
            unetblock = UNetSkipConnectionBlock(
                in_channels=nf * (2 ** (hold_depth - i - 1)),
                sub_channels=nf * (2 ** (hold_depth - i)),
                out_channels=nf * (2 ** (hold_depth - i)),
                norm=norm,
                submodule=unetblock,
            )
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, nf, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            unetblock,
            nn.Conv2d(nf * 2, out_channels, kernel_size=3, padding=1),
            activation,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


if __name__ == "__main__":
    b = UNet(
        in_channels=3,
        out_channels=3,
    )
    x = torch.randn(1, 3, 512, 512)
    y = b(x)
    print(y.shape)
