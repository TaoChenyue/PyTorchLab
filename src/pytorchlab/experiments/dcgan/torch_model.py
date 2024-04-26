import torch
from torch import nn

__all__ = [
    "_Generator",
    "Generator",
    "_Discriminator",
    "Discriminator",
]


class _Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        channels: int,
        height: int,
        width: int,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.height = height
        self.width = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("_Generator not implemented yet!")

    def conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def last_block(self, in_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, self.channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )


class Generator(_Generator):
    def __init__(
        self,
        latent_dim: int = 100,
        channels: int = 1,
        height: int = 28,
        width: int = 28,
    ) -> None:
        super().__init__(latent_dim, channels, height, width)
        init_height = height // 4
        init_width = width // 4
        self.l1 = nn.Linear(latent_dim, 128 * init_height * init_width)
        self.bn1 = nn.BatchNorm2d(128)
        self.model = nn.Sequential(
            self.conv_block(128, 128),
            self.conv_block(128, 64),
            self.last_block(64),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = x.view(x.size(0), 128, self.height // 4, self.width // 4)
        x = self.bn1(x)
        x = self.model(x)
        return x


class _Discriminator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        channels: int,
        height: int,
        width: int,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.height = height
        self.width = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("_Discriminator not implemented yet!")

    def conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def last_block(self, in_features: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid(),
        )


class Discriminator(_Discriminator):
    def __init__(
        self,
        latent_dim: int = 100,
        channels: int = 1,
        height: int = 28,
        width: int = 28,
    ) -> None:
        super().__init__(latent_dim, channels, height, width)
        init_height = height // 4
        init_width = width // 4
        self.model = nn.Sequential(
            self.conv_block(channels, 256),
            self.conv_block(256, 128),
        )
        self.last = self.last_block(128 * init_height * init_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.last(x)
        return x


if __name__ == "__main__":
    t = torch.randn(2, 100)
    g = Generator(100, 1, 28, 28)
    t_g = g(t)
    print(t_g.shape)
    d = Discriminator(100, 1, 28, 28)
    t_d = d(t_g)
    print(t_d.shape)
