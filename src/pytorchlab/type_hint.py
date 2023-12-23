from typing import Callable, Iterable

from torch import nn
from torch.optim import Optimizer

OptimizerCallable = Callable[[Iterable], Optimizer]
ModuleCallable = Callable[[Iterable], nn.Module]
