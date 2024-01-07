import torch
from jsonargparse import lazy_instance
from torch import nn


class LinearDiscriminator(nn.Module):
    def __init__(
        self,
        channel: int,
        height: int,
        width: int,
        hidden_layers: list[int] = [1024, 512, 256],
        dropout: float = 0.3,
        activation: nn.Module = lazy_instance(nn.LeakyReLU, negative_slope=0.2),
        out_activation: nn.Module = lazy_instance(nn.Sigmoid),
    ):
        """
        Discriminator with linear layers.

        Args:
            channel (int): channel of input image
            height (int): height of input image
            width (int): width of input image
            hidden_layers (list[int], optional): features of hidden layers. Defaults to [1024, 512, 256].
            dropout (float, optional): dropout rate. Defaults to 0.3.
            activation (nn.Module, optional): activation layer used in hidden layers. Defaults to lazy_instance(nn.LeakyReLU, negative_slope=0.2).
            out_activation (nn.Module, optional): activation layer used in the last layer. Defaults to lazy_instance(nn.Sigmoid).
        """
        super().__init__()
        self.channel = channel
        self.height = height
        self.width = width
        hidden_layers = [channel * height * width] + hidden_layers + [1]
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

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.model(x)
        return x


class ConditionalLinearDiscriminator(nn.Module):
    def __init__(
        self,
        channel: int,
        height: int,
        width: int,
        num_classes: int,
        hidden_layers: list[int] = [1024, 512, 256],
        dropout: float = 0.3,
        activation: nn.Module = lazy_instance(nn.LeakyReLU, negative_slope=0.2),
        out_activation: nn.Module = lazy_instance(nn.Sigmoid),
    ):
        """
        Conditional discriminator with linear layers.

        Args:
            channel (int): channel of input image
            height (int): height of input image
            width (int): width of input image
            num_classes (int): number of classes
            hidden_layers (list[int], optional): features of hidden layers. Defaults to [1024, 512, 256].
            dropout (float, optional): dropout rate. Defaults to 0.3.
            activation (nn.Module, optional): activation layer used in hidden layers. Defaults to lazy_instance(nn.LeakyReLU, negative_slope=0.2).
            out_activation (nn.Module, optional): activation layer used in the last layer. Defaults to lazy_instance(nn.Sigmoid).
        """
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
