import torch
from jsonargparse import lazy_instance
from torch import nn

from pytorchlab.type_hint import ModuleCallable


class UNetSkipConnectionBlock(nn.Module):
    def __init__(
        self,
        last_channel: int,
        channel: int,
        dropout: float = 0.0,
        submodule: nn.Module | None = None,
        norm_cls: ModuleCallable = nn.BatchNorm2d,
        down_relu: nn.Module = lazy_instance(
            nn.LeakyReLU, negative_slope=0.2, inplace=True
        ),
        up_relu: nn.Module = lazy_instance(nn.ReLU, inplace=True),
    ):
        super().__init__()
        down_conv = nn.Conv2d(
            last_channel,
            channel,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        down_norm: nn.Module = norm_cls(channel)
        up_conv = nn.ConvTranspose2d(
            channel * (1 if submodule is None else 2),
            last_channel,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        up_norm: nn.Module = norm_cls(last_channel)
        layers: list[nn.Module] = [down_relu, down_conv, down_norm]
        if submodule is not None:
            layers.append(submodule)
        layers += [up_relu, up_conv, up_norm]
        if dropout != 0:
            layers += [nn.Dropout(dropout)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return torch.cat([x, self.model(x)], dim=1)


class UNetGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 8,
        ngf: int = 64,
        norm_cls: ModuleCallable = nn.BatchNorm2d,
        down_relu: nn.Module = lazy_instance(
            nn.LeakyReLU, negative_slope=0.2, inplace=True
        ),
        up_relu: nn.Module = lazy_instance(nn.ReLU, inplace=True),
        dropout: float = 0.5,
    ):
        super().__init__()
        unet_block = UNetSkipConnectionBlock(
            last_channel=ngf * 8,
            channel=ngf * 8,
            submodule=None,
            norm_cls=norm_cls,
            down_relu=down_relu,
            up_relu=up_relu,
        )
        for _ in range(depth - 5):
            unet_block = UNetSkipConnectionBlock(
                last_channel=ngf * 8,
                channel=ngf * 8,
                dropout=dropout,
                submodule=unet_block,
                norm_cls=norm_cls,
                down_relu=down_relu,
                up_relu=up_relu,
            )
        for i in range(2, -1, -1):
            unet_block = UNetSkipConnectionBlock(
                last_channel=ngf * (1 << i),
                channel=ngf * (1 << (i + 1)),
                submodule=unet_block,
                norm_cls=norm_cls,
                down_relu=down_relu,
                up_relu=up_relu,
            )
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, ngf, 4, 2, 1),
            unet_block,
            up_relu,
            nn.ConvTranspose2d(ngf * 2, out_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)
