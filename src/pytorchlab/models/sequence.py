from torch import nn


class SequenceBlock(nn.Module):
    def __init__(
        self,
        block: nn.Module,
        depth: int = 2,
    ):
        super().__init__()
        self.block = block
        self.model = nn.Sequential(*[block for _ in range(depth)])

    def forward(self, x):
        return self.model(x)
