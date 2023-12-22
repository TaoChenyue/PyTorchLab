from torch import nn
from typing import Callable


class ResNetBlock(nn.Module):
    def __init__(
        self,
        channel: int,
        num_blocks: int = 2,
        dropout: float = 0.0,
        padding_cls: Callable = nn.ZeroPad2d,
        norm_cls: Callable = nn.BatchNorm2d,
        activation: nn.Module = nn.ReLU(inplace=True),
    ):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(num_blocks):
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
        channel: int,
        out_channel: int,
        depth: int = 2,
        num_blocks: int = 6,
        n_res_blocks: int = 2,
        ngf: int = 64,
        dropout: float = 0.0,
        padding_cls: Callable = nn.ZeroPad2d,
        norm_cls: Callable = nn.BatchNorm2d,
        activation: nn.Module = nn.ReLU(inplace=True),
    ):
        super().__init__()
        layers: list[nn.Module] = []
        layers += [
            padding_cls(1),
            nn.Conv2d(
                channel,
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
                    num_blocks=n_res_blocks,
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
                out_channel,
                kernel_size=7,
                padding=0,
            ),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    import torch

    x = torch.randn(10, 3, 32, 32)
    g = ResNetGenerator(
        channel=3,
        out_channel=3,
    )
    y = g(x)
    print(y.shape)
