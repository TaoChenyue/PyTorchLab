import torch
from torch import nn


def linear_block(
    in_features: int,
    out_features: int,
    drop_out: float | None = None,
    norm: bool = True,
    activation: nn.Module = nn.ReLU(inplace=True),
) -> list[nn.Module]:
    """
    Linear + (Dropout1d) + (BatchNorm1d) + activation

    Args:
        in_features (int): number of input features
        out_features (int): number of output features
        drop_out (float | None, optional): drop_out rate(0.2 to 0.5 is recommand). Defaults to None.
        norm (bool, optional): batchnorm or not. Defaults to True.
        activation (nn.Module, optional): activate function. Defaults to nn.ReLU(inplace=True).

    Returns:
        list[nn.Module]: linear block
    """

    layers: list[nn.Module] = [nn.Linear(in_features, out_features)]
    if isinstance(drop_out, float):
        layers.append(nn.Dropout1d(drop_out))
    if norm:
        layers.append(nn.BatchNorm1d(out_features))
    layers.append(activation)
    return layers


class Generator(nn.Module):
    def __init__(
        self,
        channel: int,
        height: int,
        width: int,
        latent_dim: int = 10,
        hidden_layers: int = 3,
        hidden_layer_nodes: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.channel = channel
        self.height = height
        self.width = width

        out_features = channel * height * width

        layers: list[nn.Module] = [
            nn.Sequential(
                *linear_block(
                    self.latent_dim,
                    hidden_layer_nodes,
                    norm=False,
                )
            )
        ]
        for _ in range(hidden_layers):
            layers.append(
                nn.Sequential(
                    *linear_block(
                        hidden_layer_nodes,
                        hidden_layer_nodes,
                    )
                )
            )
        layers.append(
            nn.Sequential(
                nn.Linear(hidden_layer_nodes, out_features),
                nn.Tanh(),
            )
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)
        x = self.net(x)
        x = x.view(x.size(0), self.channel, self.height, self.width)
        return x


class Discriminator(nn.Module):
    def __init__(
        self,
        channel: int,
        height: int,
        width: int,
        hidden_layers: int = 3,
        hidden_layer_nodes: int = 64,
    ):
        """
        discriminate whether image is generated or groundtruth

        Args:
            in_shape (tuple[int,int,int]): shape of input image
            channel_list (list[int], optional): channels of hidden layers.
        """
        super().__init__()
        in_features: int = channel * height * width

        layers = [
            nn.Sequential(
                *linear_block(
                    in_features,
                    hidden_layer_nodes,
                    norm=False,
                )
            )
        ]
        for _ in range(hidden_layers):
            layers.append(
                nn.Sequential(
                    *linear_block(
                        hidden_layer_nodes,
                        hidden_layer_nodes,
                        norm=False,
                    )
                )
            )
        layers.append(
            nn.Sequential(
                nn.Linear(hidden_layer_nodes, 1),
                nn.Sigmoid(),
            )
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)
        x = self.net(x)
        return x
