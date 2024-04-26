import torch
from torch import nn

from pytorchlab.experiments.autoencoder.torch_model import EncoderBlock
from pytorchlab.experiments.unet.torch_model import UNet

__all__ = [
    "_Generator",
    "Generator",
    "_Discriminator",
    "Discriminator",
]


class _Generator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels


class Generator(_Generator):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__(in_channels, out_channels)
        self.model = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            depth=7,
            hold_depth=2,
            norm=nn.InstanceNorm2d,
            activation=nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class _Discriminator(nn.Module):
    def __init__(
        self,
        in_channels: int,
    ):
        super().__init__()
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("_Discriminator not implemented yet!")


class Discriminator(_Discriminator):
    def __init__(
        self,
        in_channels: int,
    ) -> None:
        super().__init__(in_channels)
        self.model = nn.Sequential(
            EncoderBlock(in_channels, 64, norm=nn.InstanceNorm2d),
            EncoderBlock(64, 128, norm=nn.InstanceNorm2d),
            EncoderBlock(128, 64, norm=nn.InstanceNorm2d),
            EncoderBlock(64, 32, norm=nn.InstanceNorm2d),
            EncoderBlock(32, 1, norm=nn.InstanceNorm2d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x


if __name__ == "__main__":
    t = torch.randn(1, 3, 256, 256)
    g = Generator(3, 3)
    t_g = g(t)
    print(t_g.shape)
    d = Discriminator(3)
    t_d = d(t_g)
    print(t_d.shape)
