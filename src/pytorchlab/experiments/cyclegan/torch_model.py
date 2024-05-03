import torch
from torch import nn
from torchvision import transforms

__all__ = [
    "ResidualGenerator",
    "NlayerDiscriminator",
]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1),
            nn.InstanceNorm2d(in_channels),
        )

    def forward(self, x):
        return x + self.model(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.model(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.model(x)


class ResidualGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        features: list[int] = [64, 128, 256],
        num_residual_blocks: int = 9,
    ):
        super().__init__()
        self.downs = nn.ModuleList()
        self.downs.append(
            nn.Sequential(
                nn.ReflectionPad2d(padding=3),
                nn.Conv2d(in_channels, features[0], kernel_size=7),
                nn.InstanceNorm2d(features[0]),
                nn.ReLU(inplace=True),
            )
        )
        for idx in range(1, len(features)):
            self.downs.append(Down(features[idx - 1], features[idx]))

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(features[-1]) for _ in range(num_residual_blocks)]
        )

        self.ups = nn.ModuleList()
        for idx in range(1, len(features)):
            self.ups.append(Up(features[-idx], features[-idx - 1]))

        self.ups.append(
            nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(features[0], out_channels, 7),
                nn.Tanh(),
            )
        )

    def forward(self, x: torch.Tensor):
        samples = []
        for down in self.downs:
            x = down(x)
            samples.append(x)

        x = self.residual_blocks(x)

        for up, sample in zip(self.ups, samples[::-1]):
            if x.shape != sample.shape:
                x = transforms.Resize(size=sample.shape[-2:])(x)
            x = up(x)
        return x


class NlayerDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3, features: list[int] = [64, 128, 256, 512]):
        super().__init__()
        self.model = nn.ModuleList()
        for feature in features:
            self.model.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, feature, kernel_size=4, stride=2, padding=1),
                    nn.InstanceNorm2d(feature),
                    nn.ReLU(inplace=True),
                )
            )
            in_channels = feature
        self.model.append(nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0))
        self.model = nn.Sequential(*self.model)

    def forward(self, x: torch.Tensor):
        return self.model(x)


if __name__ == "__main__":
    t = torch.randn(1, 3, 183, 183)
    g = ResidualGenerator(3, 3)
    print(g)
    d = NlayerDiscriminator(3, [64, 128, 256])
    t_g = g(t)
    t_d = d(t)
    print(t_g.shape, t_d.shape)
