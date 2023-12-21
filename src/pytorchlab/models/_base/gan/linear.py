from torch import nn


class LinearGenerator(nn.Module):
    def __init__(
        self,
        channel: int,
        height: int,
        width: int,
        latent_dim: int,
        hidden_layers: list[int] = [256, 512, 1024],
        activation: nn.Module = nn.LeakyReLU(0.2),
        output_activation: nn.Module = nn.Tanh(),
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
        activation: nn.Module = nn.LeakyReLU(0.2),
        dropout: float = 0.3,
        output_activation: nn.Module = nn.Sigmoid(),
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


if __name__ == "__main__":
    import torch

    x = torch.randn(1, 100)
    g = LinearGenerator(1, 28, 28, 100)
    y = g(x)
    print(y.shape)

    x = torch.randn(1, 1, 28, 28)
    d = LinearDiscriminator(1, 28, 28)
    y = d(x)
    print(y.shape)
