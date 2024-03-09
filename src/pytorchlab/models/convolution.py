import torch
from torch import nn

from pytorchlab.type_hint import ModuleCallable


class Conv2dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        norm: ModuleCallable | None = None,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.Identity() if norm is None else norm(out_channels)
        self.activation = nn.ReLU(inplace=True) if activation is None else activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class ConvTranspose2dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        norm: ModuleCallable | None = None,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.Identity() if norm is None else norm(out_channels)
        self.activation = nn.ReLU(inplace=True) if activation is None else activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class AutoEncoder2dBlock(nn.Module):
    def __init__(
        self,
        last_channel: int,
        channel: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        norm: ModuleCallable | None = None,
        activation: nn.Module | None = None,
        out_activation: nn.Module | None = None,
        sub_module: nn.Module | None = None,
    ) -> None:
        super().__init__()
        activation = nn.ReLU(inplace=True) if activation is None else activation
        out_activation = (
            nn.ReLU(inplace=True) if out_activation is None else out_activation
        )

        self.encoder = Conv2dBlock(
            in_channels=last_channel,
            out_channels=channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm=norm,
            activation=activation,
        )
        self.decoder = ConvTranspose2dBlock(
            in_channels=channel,
            out_channels=last_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm=norm,
            activation=out_activation,
        )
        self.sub_module = nn.Identity() if sub_module is None else sub_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.sub_module(x)
        x = self.decoder(x)
        return x
