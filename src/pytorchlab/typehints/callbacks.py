from typing import TypedDict

from torch import Tensor


class OutputDict(TypedDict):
    images: dict[str, Tensor]
    labels: dict[str, Tensor]
    metrics: dict[str, Tensor]


class OutputsDict(TypedDict):
    inputs: OutputDict
    outputs: OutputDict
    loss: Tensor
    losses: dict[str, Tensor]
