import torch
from jsonargparse import lazy_instance
from torch import Tensor, nn

from pytorchlab.models.decoder.convtranspose import ConvTranspose2dBlock
from pytorchlab.models.encoder.conv import Conv2dBlock
from pytorchlab.typehints import ModuleCallable


class UNetSkipConnection2dBlock(nn.Module):
    def __init__(
        self,
        last_channel: int,
        channel: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        submodule: nn.Module = lazy_instance(nn.Identity),
        norm: ModuleCallable = nn.Identity,
        down_activation: nn.Module = lazy_instance(nn.ReLU, inplace=True),
        up_activation: nn.Module = lazy_instance(nn.ReLU, inplace=True),
    ):
        super().__init__()
        self.down_block = Conv2dBlock(
            in_channels=last_channel,
            out_channels=channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm=norm,
            activation=down_activation,
        )
        self.submodule = submodule
        self.up_block = ConvTranspose2dBlock(
            in_channels=channel if isinstance(submodule, nn.Identity) else channel * 2,
            out_channels=last_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm=norm,
            activation=up_activation,
        )

    def forward(self, x):
        down = self.down_block(x)
        sub = self.submodule(down)
        up = self.up_block(sub)
        return torch.cat([x, up], dim=1)


class UNet2d(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        nf: int = 64,
        depth: int = 8,
        hold_depth: int = 3,
        norm: ModuleCallable = nn.Identity,
        down_activation: nn.Module = lazy_instance(nn.ReLU, inplace=True),
        up_activation: nn.Module = lazy_instance(nn.Tanh),
        submodule: nn.Module = lazy_instance(nn.Identity),
    ):
        super().__init__()

        df_num = 2**hold_depth
        unetblock = UNetSkipConnection2dBlock(
            last_channel=nf * df_num,
            channel=nf * df_num,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm=norm,
            down_activation=down_activation,
            up_activation=down_activation,
            submodule=submodule,
        )
        for _ in range(depth - hold_depth - 1):
            unetblock = UNetSkipConnection2dBlock(
                last_channel=nf * df_num,
                channel=nf * df_num,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                norm=norm,
                down_activation=down_activation,
                up_activation=down_activation,
                submodule=unetblock,
            )
        for i in range(hold_depth):
            unetblock = UNetSkipConnection2dBlock(
                last_channel=nf * (2 ** (hold_depth - i - 1)),
                channel=nf * (2 ** (hold_depth - i)),
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                norm=norm,
                down_activation=down_activation,
                up_activation=down_activation,
                submodule=unetblock,
            )
        self.model = nn.Sequential(
            nn.Conv2d(in_channel, nf, kernel_size=3, padding=1),
            down_activation,
            unetblock,
            nn.Conv2d(nf * 2, out_channel, kernel_size=3, padding=1),
            up_activation,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
