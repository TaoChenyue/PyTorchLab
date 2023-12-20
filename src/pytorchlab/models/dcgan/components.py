import torch
from torch import nn


def init_weights(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d | nn.ConvTranspose2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)


class UpsampleBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, out_features, 3),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features, out_features, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        channel: int,
        num_blocks: int = 2,
        base_features: int = 64,
        max_features: int = 512,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.channel = channel
        self.out_size = 4 << num_blocks

        out_features = base_features

        layers: list[nn.Module] = [
            nn.Sequential(
                nn.ConvTranspose2d(latent_dim, out_features, 4),
                nn.BatchNorm2d(out_features),
                nn.LeakyReLU(0.2, inplace=True),
            )
        ]

        for _ in range(num_blocks):
            next_features = min(max_features, out_features * 2)
            layers.append(UpsampleBlock(out_features, next_features))
            out_features = min(next_features, max_features)

        layers.append(
            nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_features, channel, kernel_size=3),
                nn.Tanh(),
            )
        )

        self.net = nn.Sequential(*layers)

        init_weights(self.net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        generator forward

        Args:
            x (torch.Tensor): (batch_size,latent_dim)

        Returns:
            torch.Tensor: (batch_size,channel,size,size)
        """
        x = x.unsqueeze(dim=-1).unsqueeze(dim=-1)
        x = self.net(x)
        return x


class Discriminator(nn.Module):
    def __init__(
        self,
        channel: int,
        size: int,
        num_blocks: int,
        base_features: int = 64,
        max_features: int = 512,
    ):
        """
        discriminate whether image is generated or groundtruth

        Args:
            channel (tuple[int,int,int]): channel of input image
            size (int): size of input image
            channel_list (list[int], optional): channels of hidden layers.
        """
        super().__init__()
        self.channel = channel
        self.size = size
        self.num_blocks = num_blocks
        self.out_size = size >> num_blocks

        out_features = base_features
        layers: list[nn.Module] = [
            nn.Sequential(
                nn.Conv2d(channel, out_features, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_features),
                nn.ReLU(inplace=True),
            )
        ]

        for _ in range(num_blocks):
            next_features = min(out_features * 2, max_features)
            layers.append(DownBlock(out_features, next_features))
            out_features *= 2
            out_features = min(out_features, max_features)

        layers.append(
            nn.Sequential(
                nn.Conv2d(out_features, 1, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid(),
            )
        )

        self.net = nn.Sequential(*layers)
        init_weights(self.net)

    def forward(self, x: torch.Tensor):
        x = self.net(x)
        return x
