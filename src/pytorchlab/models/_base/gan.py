import torch
from torch import nn
from pytorchlab.type_hint import ModuleCallable
from jsonargparse import lazy_instance


class LinearGenerator(nn.Module):
    def __init__(
        self,
        channel: int,
        height: int,
        width: int,
        latent_dim: int,
        hidden_layers: list[int] = [256, 512, 1024],
        activation: nn.Module = lazy_instance(nn.LeakyReLU, negative_slope=0.2),
        output_activation: nn.Module = lazy_instance(nn.Tanh),
    ):
        super().__init__()
        self.channel = channel
        self.height = height
        self.width = width
        hidden_layers = [latent_dim] + hidden_layers + [channel * height * width]
        layers: list[nn.Module] = []
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            if i == len(hidden_layers) - 1:
                layers.append(output_activation)
            else:
                layers.append(activation)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), self.channel, self.height, self.width)
        return x


class LinearDiscriminator(nn.Module):
    def __init__(
        self,
        channel: int,
        height: int,
        width: int,
        hidden_layers: list[int] = [1024, 512, 256],
        activation: nn.Module = lazy_instance(nn.LeakyReLU, negative_slope=0.2),
        dropout: float = 0.3,
        output_activation: nn.Module = lazy_instance(nn.Sigmoid),
    ):
        super().__init__()
        self.channel = channel
        self.height = height
        self.width = width
        hidden_layers = [channel * height * width] + hidden_layers + [1]
        layers: list[nn.Module] = []
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            if i == len(hidden_layers) - 1:
                layers.append(output_activation)
            else:
                layers.append(activation)
                layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.model(x)
        return x


class ConvGenerator(nn.Module):
    def __init__(
        self,
        channel: int,
        latent_dim: int = 100,
        hidden_layers: list[int] = [128, 64, 32, 16],
        activation: nn.Module = lazy_instance(nn.ReLU, inplace=True),
        out_activation: nn.Module = lazy_instance(nn.Tanh),
    ):
        super().__init__()
        hidden_layers = hidden_layers + [channel]
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(
                latent_dim,
                hidden_layers[0],
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_layers[0]),
            activation,
        ]
        for i in range(1, len(hidden_layers)):
            layers.append(
                nn.ConvTranspose2d(
                    hidden_layers[i - 1],
                    hidden_layers[i],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )

            if i == len(hidden_layers) - 1:
                layers.append(out_activation)
            else:
                layers.append(nn.BatchNorm2d(hidden_layers[i]))
                layers.append(activation)
        self.model = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.unsqueeze(dim=-1).unsqueeze(dim=-1)
        return self.model(z)


class ResNetBlock(nn.Module):
    def __init__(
        self,
        channel: int,
        dropout: float = 0.0,
        padding_cls: ModuleCallable = nn.ReflectionPad2d,
        norm_cls: ModuleCallable = nn.InstanceNorm2d,
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
        channel: int,
        out_channel: int,
        depth: int = 2,
        num_blocks: int = 6,
        ngf: int = 64,
        dropout: float = 0.0,
        padding_cls: ModuleCallable = nn.ReflectionPad2d,
        norm_cls: ModuleCallable = nn.InstanceNorm2d,
        activation: nn.Module = lazy_instance(nn.ReLU, inplace=True),
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


class NLayerDiscriminator(nn.Module):
    def __init__(
        self,
        channel: int,
        ndf: int = 64,
        depth: int = 3,
        norm_cls: ModuleCallable = nn.InstanceNorm2d,
        activation: nn.Module = lazy_instance(
            nn.LeakyReLU, negative_slope=0.2, inplace=True
        ),
    ):
        """PatchGAN discriminator

        Args:
            channel (int): channel of input image
            ndf (int, optional): number of filters on the first layer. Defaults to 64.
            depth (int, optional): depth of discriminator,output size equals to size//2**depth. Defaults to 3.
            norm_cls (ModuleCallable, optional): function name for norm. Defaults to nn.InstanceNorm2d.
            activation (nn.Module, optional): module for activate layer. Defaults to nn.LeakyReLU(0.2, inplace=True).
        """
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
            nn.Sigmoid(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)


if __name__ == "__main__":
    import torch

    x = torch.randn(10, 3, 64, 64)
    g = ResNetGenerator(
        channel=3,
        out_channel=3,
    )
    y = g(x)
    d = NLayerDiscriminator(
        channel=3,
        depth=4,
    )
    print(d)
    out = d(y)
    print(out.shape)
