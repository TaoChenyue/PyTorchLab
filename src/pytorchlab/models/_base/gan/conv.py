from torch import nn
import torch


class ConvGenerator(nn.Module):
    def __init__(
        self,
        channel: int,
        latent_dim: int = 100,
        hidden_layers: list[int] = [128, 64, 32, 16],
        activation: nn.Module = nn.ReLU(inplace=True),
        out_activation: nn.Module = nn.Tanh(),
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


class ConvDiscriminator(nn.Module):
    def __init__(
        self,
        channel: int,
        hidden_layers: list[int] = [16, 32, 64, 128],
        activation: nn.Module = nn.LeakyReLU(0.2),
        out_activation: nn.Module = nn.Sigmoid(),
    ):
        super().__init__()
        hidden_layers = [channel] + hidden_layers + [1]
        layers: nn.Module = []
        for i in range(1, len(hidden_layers)):
            if i == len(hidden_layers) - 1:
                layers += [
                    nn.Conv2d(
                        hidden_layers[i - 1],
                        hidden_layers[i],
                        kernel_size=4,
                        stride=1,
                        padding=0,
                        bias=False,
                    ),
                    out_activation,
                ]
            else:
                layers += [
                    nn.Conv2d(
                        hidden_layers[i - 1],
                        hidden_layers[i],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_layers[i]),
                    activation,
                ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        return x.view(x.size(0), -1)


def init_weights(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d | nn.ConvTranspose2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    x = torch.rand(10, 100)
    g = ConvGenerator(3)
    d = ConvDiscriminator(3)
    y = g(x)
    print(y.shape)
    z = d(y)
    print(z.shape)
