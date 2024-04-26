import torch
from jsonargparse import lazy_instance
from torch import Tensor, nn

from pytorchlab.models.decoder.convtranspose import ConvTranspose2dBlock
from pytorchlab.models.encoder.conv import Conv2dBlock
from pytorchlab.typehints import ModuleCallable

__all__ = [
    "AutoEncoder2dBlock",
    "AutoEncoder2d",
]


class AutoEncoder2dBlock(nn.Module):
    def __init__(
        self,
        last_channel: int,
        channel: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        norm: ModuleCallable = nn.Identity,
        down_activation: nn.Module = lazy_instance(nn.ReLU, inplace=True),
        up_activation: nn.Module = lazy_instance(nn.ReLU, inplace=True),
        submodule: nn.Module = lazy_instance(nn.Identity),
    ) -> None:
        super().__init__()

        self.encoder = Conv2dBlock(
            in_channels=last_channel,
            out_channels=channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm=norm,
            activation=down_activation,
        )
        self.decoder = ConvTranspose2dBlock(
            in_channels=channel,
            out_channels=last_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm=norm,
            activation=up_activation,
        )
        self.submodule = submodule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.submodule(x)
        x = self.decoder(x)
        return x


class AutoEncoder2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
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
        aeblock = AutoEncoder2dBlock(
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
            aeblock = AutoEncoder2dBlock(
                last_channel=nf * df_num,
                channel=nf * df_num,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                norm=norm,
                down_activation=down_activation,
                up_activation=down_activation,
                submodule=aeblock,
            )
        for i in range(hold_depth):
            aeblock = AutoEncoder2dBlock(
                last_channel=nf * (2 ** (hold_depth - i - 1)),
                channel=nf * (2 ** (hold_depth - i)),
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                norm=norm,
                down_activation=down_activation,
                up_activation=down_activation,
                submodule=aeblock,
            )
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, nf, kernel_size=3, padding=1),
            down_activation,
            aeblock,
            nn.Conv2d(nf, out_channels, kernel_size=3, padding=1),
            up_activation,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
