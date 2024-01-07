from jsonargparse import lazy_instance
from torch import Tensor, nn

from pytorchlab.type_hint import ModuleCallable


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
        activation: nn.Module = lazy_instance(nn.ReLU, inplace=True),
        out_activation: nn.Module = lazy_instance(nn.Tanh),
    ):
        """
        Generator with convolution layers.

        Args:
            channel (int): channel of output image
            latent_dim (int, optional): dimension of noise sample. Defaults to 100.
            hidden_layers (list[tuple[int, int, int, int]], optional): list of (features,kener_size,stride,padding). Defaults to [ (128, 4, 2, 1), (64, 4, 2, 1), (32, 4, 2, 1), (16, 4, 2, 1), ].
            norm_cls (ModuleCallable, optional): class of normalization. Defaults to nn.BatchNorm2d.
            activation (nn.Module, optional): activation layer in hidden layers. Defaults to lazy_instance(nn.ReLU, inplace=True).
            out_activation (nn.Module, optional): activation layer in the last layer. Defaults to lazy_instance(nn.Tanh).
        """
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
