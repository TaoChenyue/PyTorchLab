from abc import ABCMeta
from typing import Callable, Iterable

from torch import Tensor
from torchvision.transforms import transforms

from pytorchlab.datamodules.abc import ABCDataModule


class VisionDataModule(ABCDataModule, metaclass=ABCMeta):
    def default_transforms(self) -> Callable[[Iterable], Tensor]:
        return transforms.Compose([transforms.ToTensor()])

    def default_target_transforms(self) -> Callable[[Iterable], Tensor] | None:
        return None
