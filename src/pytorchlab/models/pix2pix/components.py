import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        num_residual_blocks: int = 4,
        base_features: int = 64,
    ) -> None:
        super().__init__()

        layers = [
            nn.Sequential(
                nn.ReflectionPad2d(in_channel),
                nn.Conv2d(in_channel, base_features, 2 * in_channel + 1),
                nn.InstanceNorm2d(base_features),
                nn.ReLU(inplace=True),
            )
        ]
        for i in range(1, 3, 1):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(base_features * i, base_features * 2 * i, 3, 2, 1),
                    nn.InstanceNorm2d(base_features * 2 * i),
                    nn.ReLU(inplace=True),
                )
            )
        for _ in range(num_residual_blocks):
            layers.append(ResidualBlock(base_features * 4))

        for i in range(2, 0, -1):
            layers.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(
                        base_features * 2 * i, base_features * i, 3, stride=1, padding=1
                    ),
                    nn.InstanceNorm2d(base_features * i),
                    nn.ReLU(inplace=True),
                )
            )
        layers.append(
            nn.Sequential(
                nn.ReflectionPad2d(out_channel),
                nn.Conv2d(base_features, out_channel, 2 * out_channel + 1),
                nn.Tanh(),
            )
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        base_features: int = 64,
    ):
        super(Discriminator, self).__init__()

        # self.output_shape = (1, height // 2**4, width // 2**4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(
                in_channel + out_channel, base_features, normalize=False
            ),
            *discriminator_block(base_features, base_features * 2),
            *discriminator_block(base_features * 2, base_features * 4),
            *discriminator_block(base_features * 4, base_features * 8),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(base_features * 8, 1, 4, padding=1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        z = torch.cat((x, y), dim=1)
        return self.model(z)
