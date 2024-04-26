from typing import TypedDict

from torch import Tensor

__all__ = [
    "OutputsDict",
]


class OutputsDict(TypedDict):
    inputs: dict[str, Tensor]
    outputs: dict[str, Tensor]
    loss: Tensor
    losses: dict[str, Tensor]
