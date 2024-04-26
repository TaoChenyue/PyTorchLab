from typing import Optional

import torch
from jsonargparse import lazy_instance
from torch import Tensor, nn

from pytorchlab.typehints import ModuleCallable

__all__ = [
    "AutoEncoderBlock",
    "AutoEncoder",
]


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: ModuleCallable = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            norm(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: ModuleCallable = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            ),
            norm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class AutoEncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        sub_channels: int,
        out_channels: int,
        norm: ModuleCallable = nn.Identity,
        submodule: Optional["AutoEncoderBlock"] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder = EncoderBlock(
            in_channels=in_channels,
            out_channels=sub_channels,
            norm=norm,
        )
        self.submodule = nn.Identity() if submodule is None else submodule
        c = sub_channels if submodule is None else submodule.out_channels
        self.decoder = DecoderBlock(
            in_channels=c,
            out_channels=out_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.submodule(x)
        x = self.decoder(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        nf: int = 64,
        depth: int = 7,
        hold_depth: int = 2,
        norm: ModuleCallable = nn.BatchNorm2d,
        activation: nn.Module = lazy_instance(nn.Tanh),
    ):
        super().__init__()

        df_num = 2**hold_depth
        aeblock = AutoEncoderBlock(
            in_channels=nf * df_num,
            sub_channels=nf * df_num,
            out_channels=nf * df_num,
            norm=norm,
        )
        for _ in range(depth - hold_depth - 1):
            aeblock = AutoEncoderBlock(
                in_channels=nf * df_num,
                sub_channels=nf * df_num,
                out_channels=nf * df_num,
                norm=norm,
                submodule=aeblock,
            )
        for i in range(hold_depth):
            aeblock = AutoEncoderBlock(
                in_channels=nf * (2 ** (hold_depth - i - 1)),
                sub_channels=nf * (2 ** (hold_depth - i)),
                out_channels=nf * (2 ** (hold_depth - i - 1)),
                norm=norm,
                submodule=aeblock,
            )
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, nf, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            aeblock,
            nn.Conv2d(nf, out_channels, kernel_size=3, padding=1),
            activation,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


if __name__ == "__main__":
    t = torch.randn(1, 3, 256, 256)
    b = AutoEncoder(3, 3)
    print(b(t).shape)
