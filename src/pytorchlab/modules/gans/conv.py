from torch import Tensor, nn

from pytorchlab.type_hint import ModuleCallable
from jsonargparse import lazy_instance


class ConvGenerator(nn.Module):
    def __init__(
        self,
        channel: int,
        latent_dim: int = 100,
        hidden_layers: list[tuple[int, int, int, int]] = [
            (128, 4, 2, 1),
            (64, 4, 2, 1),
            (32, 4, 2, 1),
            (16, 4, 2, 1),
        ],
        norm_cls: ModuleCallable = nn.BatchNorm2d,
        activation: nn.Module = lazy_instance(nn.ReLU,inplace=True),
        out_activation: nn.Module = lazy_instance(nn.Tanh),
    ):
        super().__init__()
        self.latent_dim = latent_dim
        hidden_layers = [(latent_dim, 4, 1, 0)] + hidden_layers + [(channel, 1, 1, 1)]
        layers: list[nn.Module] = []
        for i in range(1, len(hidden_layers)):
            layers.append(
                nn.ConvTranspose2d(
                    hidden_layers[i - 1][0],
                    hidden_layers[i][0],
                    kernel_size=hidden_layers[i - 1][1],
                    stride=hidden_layers[i - 1][2],
                    padding=hidden_layers[i - 1][3],
                )
            )

            if i == len(hidden_layers) - 1:
                layers.append(out_activation)
            else:
                layers.append(norm_cls(hidden_layers[i][0]))
                layers.append(activation)
        self.model = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        z = z.unsqueeze(dim=-1).unsqueeze(dim=-1)
        return self.model(z)


class ConvDiscriminator(nn.Module):
    def __init__(
        self,
        channel: int,
        hidden_layers: list[tuple[int, int, int, int]] = [
            (16, 4, 2, 1),
            (32, 4, 2, 1),
            (64, 4, 2, 1),
            (128, 4, 2, 1),
        ],
        dropout: float = 0.5,
        norm_cls: ModuleCallable = nn.BatchNorm2d,
        activation: nn.Module = lazy_instance(nn.ReLU,inplace=True),
    ):
        super().__init__()
        hidden_layers = [(channel, 4, 2, 1)] + hidden_layers + [(1, 1, 1, 1)]
        layers: list[nn.Module] = []
        for i in range(1, len(hidden_layers)):
            layers.append(
                nn.Conv2d(
                    hidden_layers[i - 1][0],
                    hidden_layers[i][0],
                    kernel_size=hidden_layers[i - 1][1],
                    stride=hidden_layers[i - 1][2],
                    padding=hidden_layers[i - 1][3],
                )
            )

            if i < len(hidden_layers) - 1:
                layers.append(norm_cls(hidden_layers[i][0]))
                layers.append(activation)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        return self.model(z)
    
    
class NLayerDiscriminator(nn.Module):
    def __init__(
        self,
        channel: int,
        ndf: int = 64,
        depth: int = 3,
        norm_cls: ModuleCallable = nn.BatchNorm2d,
        activation: nn.Module = lazy_instance(nn.ReLU,inplace=True),
    ):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(channel, ndf, kernel_size=4, stride=2, padding=1),
            activation,
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, depth):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            layers += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                norm_cls(ndf * nf_mult),
                activation,
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**depth, 8)
        layers += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            norm_cls(ndf * nf_mult),
            activation,
        ]

        layers += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=3, stride=1, padding=1),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(
        self,
        channel: int,
        ndf: int = 64,
        norm_cls: ModuleCallable = nn.BatchNorm2d,
        activation: nn.Module = lazy_instance(nn.ReLU,inplace=True),
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channel, ndf, kernel_size=1, stride=1, padding=0),
            activation,
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0),
            norm_cls(ndf * 2),
            activation,
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.model(x)