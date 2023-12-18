from typing import Callable, Iterable

from torch import nn
from torch.optim import Optimizer

OptimizerCallable = Callable[[Iterable], Optimizer]
LossCallable = Callable[[Iterable], nn.Module]
