import torch
from torch import nn


class Generator(nn.Module):
    def __init__(
        self,
        channel: int,
        height: int,
        width: int,
        num_classes: int,
        latent_dim: int,
        hidden_layers: list[int] = [256, 512, 1024],
        activation: nn.Module = nn.LeakyReLU(0.2),
        output_activation: nn.Module = nn.Tanh(),
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, num_classes)
        self.channel = channel
        self.height = height
        self.width = width
        hidden_layers = (
            [latent_dim + num_classes] + hidden_layers + [channel * height * width]
        )
        layers: list[nn.Module] = []
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            if i == len(hidden_layers) - 1:
                layers.append(output_activation)
            else:
                layers.append(activation)
        self.model = nn.Sequential(*layers)

    def forward(self, latent_code, label):
        # Concatenate label embedding and latent code
        label_embedding = self.embedding(label)
        combined = torch.cat([latent_code, label_embedding], dim=-1)

        # Generate images from latent code and label embedding
        out = self.model(combined)
        out = out.view(out.size(0), self.channel, self.height, self.width)
        return out


class Discriminator(nn.Module):
    def __init__(
        self,
        channel: int,
        height: int,
        width: int,
        num_classes: int,
        hidden_layers: list[int] = [1024, 512, 256],
        activation: nn.Module = nn.LeakyReLU(0.2),
        dropout: float = 0.3,
        output_activation: nn.Module = nn.Sigmoid(),
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, num_classes)
        self.channel = channel
        self.height = height
        self.width = width
        hidden_layers = [channel * height * width + num_classes] + hidden_layers + [1]
        layers: list[nn.Module] = []
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            if i == len(hidden_layers) - 1:
                layers.append(output_activation)
            else:
                layers.append(activation)
                layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, img, label):
        label_embedding = self.embedding(label)
        img = img.view(img.size(0), -1)
        x = torch.cat([img, label_embedding], dim=-1)
        x = self.model(x)
        return x


if __name__ == "__main__":
    batch_size = 5
    g = Generator(
        channel=1,
        height=28,
        width=28,
        num_classes=10,
        latent_dim=100,
    )
    code = torch.randn(batch_size, 100)
    label = torch.randint(0, 10, (batch_size,))
    img = g(code, label)
    print(img.shape)
    d = Discriminator(
        channel=1,
        height=28,
        width=28,
        num_classes=10,
    )
    out = d(img, label)
    print(out.shape)
