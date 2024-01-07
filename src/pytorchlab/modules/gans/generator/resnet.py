from jsonargparse import lazy_instance
from torch import nn

from pytorchlab.type_hint import ModuleCallable


class ResNetBlock(nn.Module):
    def __init__(
        self,
        channel: int,
        dropout: float = 0.0,
        padding_cls: ModuleCallable = nn.ReflectionPad2d,
        norm_cls: ModuleCallable = nn.BatchNorm2d,
        activation: nn.Module = lazy_instance(nn.ReLU, inplace=True),
    ):
        """residual block

        Args:
            channel (int): channel of input image
            num_blocks (int, optional): number of residual blocks. Defaults to 2.
            dropout (float, optional): dropout rate. Defaults to 0.0.
            padding_cls (ModuleCallable, optional): function name for padding. Defaults to nn.ReflectionPad2d.
            norm_cls (ModuleCallable, optional): function name for normalize. Defaults to nn.InstanceNorm2d.
            activation (nn.Module, optional): module for activate layer. Defaults to nn.ReLU(inplace=True).
        """
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(2):
            layers += [
                padding_cls(1),
                nn.Conv2d(
                    channel,
                    channel,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                ),
                norm_cls(channel),
                activation,
                nn.Dropout(dropout),
            ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = x + self.model(x)
        return out


class ResNetGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 2,
        num_blocks: int = 6,
        ngf: int = 64,
        dropout: float = 0.0,
        padding_cls: ModuleCallable = nn.ReflectionPad2d,
        norm_cls: ModuleCallable = nn.BatchNorm2d,
        activation: nn.Module = lazy_instance(nn.ReLU, inplace=True),
    ):
        super().__init__()
        layers: list[nn.Module] = []
        layers += [
            padding_cls(1),
            nn.Conv2d(
                in_channels,
                ngf,
                kernel_size=3,
                stride=1,
            ),
            norm_cls(ngf),
            activation,
        ]
        for i in range(depth):
            mult = 2**i
            layers += [
                padding_cls(1),
                nn.Conv2d(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=0,
                ),
                norm_cls(ngf * mult * 2),
                activation,
            ]
        mult = 2**depth
        for i in range(num_blocks):
            layers += [
                ResNetBlock(
                    channel=ngf * mult,
                    dropout=dropout,
                    padding_cls=padding_cls,
                    norm_cls=norm_cls,
                    activation=activation,
                )
            ]
        for i in range(depth):
            mult = 2 ** (depth - i)
            layers += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    ngf * mult // 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                norm_cls(ngf * mult // 2),
                activation,
            ]
        layers += [
            padding_cls(3),
            nn.Conv2d(
                ngf,
                out_channels,
                kernel_size=7,
                padding=0,
            ),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
