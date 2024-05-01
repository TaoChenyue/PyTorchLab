import torch
from torch import nn
from torchvision import transforms

__all__ = [
    "UnetGenerator",
    "PixelDiscriminator",
]


class Down(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=4,
        stride=2,
        padding=1,
        normalize: bool = True,
    ):
        super().__init__()
        if normalize:
            norm = nn.InstanceNorm2d(out_channels)
        else:
            norm = nn.Identity()
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            norm,
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.model(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        if x.shape != skip.shape:
            x = transforms.Resize(size=skip.shape[-2:])(x)
        x = torch.cat([x, skip], dim=1)
        return self.model(x)


class UnetGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: list[int] = [64, 128, 256] + [512] * 4,
    ):
        super().__init__()
        self.first_conv = Down(in_channels, features[0], 3, 1, 1, normalize=False)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], out_channels, 3, 1, 1),
            nn.Tanh(),
        )

        for idx in range(1, len(features)):
            self.downs.append(Down(features[idx - 1], features[idx]))

        self.bottleneck = Down(
            features[-1], features[-1], kernel_size=1, stride=1, padding=1
        )

        for idx in range(1, len(features)):
            self.ups.append(Up(2 * features[-idx], features[-idx - 1]))

    def forward(self, x: torch.Tensor):
        x = self.first_conv(x)
        shape = x.shape[-2:]
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)

        x = self.bottleneck(x)

        for up, skip in zip(self.ups, skip_connections[::-1]):
            x = up(x, skip)

        return self.final_conv(transforms.Resize(size=shape)(x))


class PixelDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 4, features: list[int] = [64, 128, 256]):
        super().__init__()
        self.model = nn.ModuleList()
        for feature in features:
            self.model.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, feature, kernel_size=1, stride=1, padding=0),
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
    g = UnetGenerator(3, 3)
    print(g)
    d = PixelDiscriminator(3, [64, 128, 256])
    t_g = g(t)
    t_d = d(t)
    print(t_g.shape, t_d.shape)
