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
        num_classes: int,
        channels: int,
        height: int,
        width: int,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.channels = channels
        self.height = height
        self.width = width

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("_Generator not implemented yet!")

    def linear_block(self, in_features: int, out_features: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def last_block(self, in_features: int):
        return nn.Sequential(
            nn.Linear(
                in_features=in_features,
                out_features=self.channels * self.height * self.width,
            ),
            nn.Tanh(),
        )


class Generator(_Generator):
    def __init__(
        self,
        latent_dim: int = 100,
        num_classes: int = 10,
        channels: int = 1,
        height: int = 28,
        width: int = 28,
    ) -> None:
        super().__init__(latent_dim, num_classes, channels, height, width)
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            self.linear_block(latent_dim + num_classes, 128),
            self.linear_block(128, 256),
            self.linear_block(256, 512),
            self.linear_block(512, 1024),
            self.last_block(1024),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = self.label_emb(y)
        x = torch.cat([x, y], dim=1)
        x = self.model(x)
        x = x.view(x.size(0), self.channels, self.height, self.width)
        return x


class _Discriminator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        channels: int,
        height: int,
        width: int,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.channels = channels
        self.height = height
        self.width = width

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("_Discriminator not implemented yet!")

    def linear_block(self, in_features: int, out_features: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_features, out_features),
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
        num_classes: int = 10,
        channels: int = 1,
        height: int = 28,
        width: int = 28,
    ) -> None:
        super().__init__(latent_dim, num_classes, channels, height, width)
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            self.linear_block(channels * height * width + num_classes, 512),
            self.linear_block(512, 256),
            self.last_block(256),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = self.label_emb(y)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, y], dim=1)
        x = self.model(x)
        return x


if __name__ == "__main__":
    t = torch.randn(2, 100)
    l = torch.randint(0, 10, (2,))
    g = Generator(100, 10, 1, 28, 28)
    t_g = g(t, l)
    print(t_g.shape)
    d = Discriminator(100, 10, 1, 28, 28)
    t_d = d(t_g, l)
    print(t_d.shape)
