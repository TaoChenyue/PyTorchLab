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
        hidden_layers: list[int] = [256, 512, 1024],
        activation: nn.Module = lazy_instance(nn.LeakyReLU, negative_slope=0.2),
        out_activation: nn.Module = lazy_instance(nn.Tanh),
    ):
        """
        Generator with linear layers

        Args:
            channel (int): channel of image to generate
            height (int): height of image to generate
            width (int): width of image to generate
            latent_dim (int): dimension of noise sample
            hidden_layers (list[int], optional): features of hidden layers. Defaults to [256, 512, 1024].
            activation (nn.Module, optional): activation layer used in hidden layers. Defaults to lazy_instance(nn.LeakyReLU, negative_slope=0.2).
            out_activation (nn.Module, optional): activation layer used in the last layer. Defaults to lazy_instance(nn.Tanh).
        """
        super().__init__()
        self.channel = channel
        self.height = height
        self.width = width
        self.latent_dim = latent_dim
        hidden_layers = [latent_dim] + hidden_layers + [channel * height * width]
        layers: list[nn.Module] = []
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            if i == len(hidden_layers) - 1:
                layers.append(out_activation)
            else:
                layers.append(activation)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), self.channel, self.height, self.width)
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
        activation: nn.Module = lazy_instance(nn.LeakyReLU, negative_slope=0.2),
        out_activation: nn.Module = lazy_instance(nn.Tanh),
    ):
        """
        Conditional generator with linear layers.

        Args:
            channel (int): channel of image to generate
            height (int): height of image to generate
            width (int): width of image to generate
            num_classes (int): number of classes
            latent_dim (int): dimension of noise sample
            hidden_layers (list[int], optional): features of hidden layers. Defaults to [256, 512, 1024].
            activation (nn.Module, optional): activation layer used in hidden layers. Defaults to nn.LeakyReLU(0.2).
            out_activation (nn.Module, optional): activation layer used in the last layer. Defaults to nn.Tanh().
        """
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
