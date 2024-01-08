from torch import Tensor, nn

from pytorchlab.type_hint import ModuleCallable


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
        activation: nn.Module | None = None,
    ):
        """
        Discriminator with convolution layers.

        Args:
            channel (int): channel of input image
            hidden_layers (list[tuple[int, int, int, int]], optional): list of (features,kener_size,stride,padding). Defaults to [ (16, 4, 2, 1), (32, 4, 2, 1), (64, 4, 2, 1), (128, 4, 2, 1), ].
            norm_cls (ModuleCallable, optional): class of normalization. Defaults to nn.BatchNorm2d.
            activation (nn.Module, optional): activation layer. Defaults to lazy_instance(nn.ReLU, inplace=True).
        """
        super().__init__()
        activation = activation or nn.ReLU(inplace=True)
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

            if i != len(hidden_layers) - 1:
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
        activation: nn.Module | None = None,
    ):
        """PatchGAN discriminator

        Args:
            channel (int): channel of input image
            ndf (int, optional): number of filters on the first layer. Defaults to 64.
            depth (int, optional): depth of discriminator,output size equals to size//2**depth. Defaults to 3.
            norm_cls (ModuleCallable, optional): function name for norm. Defaults to nn.BatchNorm2d.
            activation (nn.Module, optional): module for activate layer. Defaults to nn.LeakyReLU(0.2, inplace=True).
        """
        super().__init__()
        activation = activation or nn.LeakyReLU(negative_slope=0.2, inplace=True)
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
        activation: nn.Module | None = None,
    ):
        super().__init__()
        activation = activation or nn.LeakyReLU(negative_slope=0.2, inplace=True)
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
