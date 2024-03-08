import torch
from jsonargparse import lazy_instance
from torch import nn


class LinearGenerator(nn.Module):
    def __init__(
        self,
        channel: int,
        height: int,
        width: int,
        latent_dim: int,
        bn: bool = True,
        hidden_layers: list[int] = [256],
        activation: nn.Module = lazy_instance(
            nn.LeakyReLU, negative_slope=0.2, inplace=True
        ),
        out_activation: nn.Module = lazy_instance(nn.Tanh),
    ):
        super().__init__()
        self.channel = channel
        self.height = height
        self.width = width
        self.latent_dim = latent_dim

        def linearblock(
            in_features: int,
            out_features: int,
            bn=True,
            activation: nn.Module = nn.ReLU(),
        ) -> nn.Module:
            layers = [nn.Linear(in_features, out_features)]
            if bn:
                layers.append(nn.BatchNorm1d(out_features))
            layers.append(activation)
            return nn.Sequential(*layers)

        hidden_layers = [latent_dim] + hidden_layers

        layers: list[nn.Module] = [
            linearblock(
                hidden_layers[i - 1], hidden_layers[i], bn=bn, activation=activation
            )
            for i in range(1, len(hidden_layers))
        ]
        layers.append(
            linearblock(
                hidden_layers[-1],
                channel * height * width,
                bn=False,
                activation=out_activation,
            )
        )
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
        hidden_layers: list[int] = [256],
        dropout: float = 0.3,
        bn: bool = True,
        activation: nn.Module = lazy_instance(
            nn.LeakyReLU, negative_slope=0.2, inplace=True
        ),
        out_activation: nn.Module = lazy_instance(nn.Sigmoid),
    ):
        super().__init__()
        self.channel = channel
        self.height = height
        self.width = width

        def linearblock(
            in_features: int,
            out_features: int,
            bn: bool = True,
            dropout: float = 0.3,
            activation: nn.Module = nn.ReLU(),
        ) -> nn.Module:
            layers = [nn.Linear(in_features, out_features)]
            if bn:
                layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.Dropout(dropout))
            layers.append(activation)
            return nn.Sequential(*layers)

        hidden_layers = [channel * height * width] + hidden_layers
        layers: list[nn.Module] = [
            linearblock(
                hidden_layers[i - 1],
                hidden_layers[i],
                bn=bn,
                dropout=dropout,
                activation=activation,
            )
            for i in range(1, len(hidden_layers))
        ]
        layers.append(
            linearblock(
                hidden_layers[-1],
                1,
                bn=False,
                dropout=0,
                activation=out_activation,
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.model(x)
        return x


class ConditionalLinearGenerator(nn.Module):
    def __init__(
        self,
        channel: int,
        height: int,
        width: int,
        num_classes: int,
        latent_dim: int,
        hidden_layers: list[int] = [256, 512, 1024],
        activation: nn.Module = lazy_instance(nn.ReLU, inplace=True),
        out_activation: nn.Module = lazy_instance(nn.Tanh),
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, num_classes)
        self.channel = channel
        self.height = height
        self.width = width
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        hidden_layers = (
            [latent_dim + num_classes] + hidden_layers + [channel * height * width]
        )
        layers: list[nn.Module] = []
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            if i == len(hidden_layers) - 1:
                layers.append(out_activation)
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


class ConditionalLinearDiscriminator(nn.Module):
    def __init__(
        self,
        channel: int,
        height: int,
        width: int,
        num_classes: int,
        hidden_layers: list[int] = [1024, 512, 256],
        dropout: float = 0.3,
        activation: nn.Module = lazy_instance(nn.ReLU, inplace=True),
        out_activation: nn.Module = lazy_instance(nn.Sigmoid),
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, num_classes)
        self.channel = channel
        self.height = height
        self.width = width
        self.num_classes = num_classes
        hidden_layers = [channel * height * width + num_classes] + hidden_layers + [1]
        layers: list[nn.Module] = []
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            if i == len(hidden_layers) - 1:
                layers.append(out_activation)
            else:
                layers.append(activation)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, img, label):
        label_embedding = self.embedding(label)
        img = img.view(img.size(0), -1)
        x = torch.cat([img, label_embedding], dim=-1)
        x = self.model(x)
        return x
