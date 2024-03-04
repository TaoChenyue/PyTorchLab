from typing import Callable, Iterable

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

ModuleCallable = Callable[[Iterable], nn.Module]
OptimizerCallable = Callable[[Iterable], Optimizer]
LRSchedulerCallable = Callable[[Iterable], LRScheduler]
