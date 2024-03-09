from torch import Tensor, nn

from pytorchlab.models.convolution import AutoEncoder2dBlock
from pytorchlab.type_hint import ModuleCallable


class AutoEncoder2d(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        nf: int = 64,
        depth: int = 8,
        hold_depth: int = 3,
        norm: ModuleCallable = None,
        activation: nn.Module | None = None,
        out_activation: nn.Module | None = None,
    ):
        super().__init__()
        activation = activation or nn.ReLU(inplace=True)
        out_activation = out_activation or nn.Tanh()

        df_num = 2**hold_depth
        aeblock = AutoEncoder2dBlock(
            last_channel=nf * df_num,
            channel=nf * df_num,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm=norm,
            activation=activation,
            out_activation=activation,
        )
        for _ in range(depth - hold_depth - 1):
            aeblock = AutoEncoder2dBlock(
                last_channel=nf * df_num,
                channel=nf * df_num,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                sub_module=aeblock,
                norm=norm,
                activation=activation,
                out_activation=activation,
            )
        for i in range(hold_depth):
            aeblock = AutoEncoder2dBlock(
                last_channel=nf * (2 ** (hold_depth - i - 1)),
                channel=nf * (2 ** (hold_depth - i)),
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                sub_module=aeblock,
                norm=norm,
                activation=activation,
                out_activation=activation,
            )
        self.model = nn.Sequential(
            nn.Conv2d(in_channel, nf, kernel_size=3, padding=1),
            activation,
            aeblock,
            nn.Conv2d(nf, out_channel, kernel_size=3, padding=1),
            out_activation,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
