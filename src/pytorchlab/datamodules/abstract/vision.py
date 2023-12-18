from abc import ABCMeta
from typing import Callable, Iterable

from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from pytorchlab.datamodules.abstract.basic import BasicDataModule


class VisionDataModule(BasicDataModule, metaclass=ABCMeta):
    def default_transforms(self) -> Callable[[Iterable], Tensor]:
        return transforms.Compose([transforms.ToTensor()])

    def default_target_transforms(self) -> Callable[[Iterable], Tensor] | None:
        return None


class TorchVisionDataModule(VisionDataModule, metaclass=ABCMeta):
    dataset_cls: Callable[[Iterable], Dataset]

    def prepare_data(self) -> None:
        """Saves files to root."""
        self.dataset_cls(self.train_root, train=True, download=True)
        self.dataset_cls(self.test_root, train=False, download=True)

    def entire_train_dataset(
        self,
        transforms: Callable[[Iterable], Tensor],
        target_transforms: Callable[[Iterable], Tensor] | None,
    ) -> Dataset:
        return self.dataset_cls(
            self.train_root,
            train=True,
            transform=transforms,
            target_transform=target_transforms,
        )

    def entire_test_dataset(
        self,
        transforms: Callable[[Iterable], Tensor],
        target_transforms: Callable[[Iterable], Tensor] | None,
    ) -> Dataset:
        return self.dataset_cls(
            self.test_root,
            train=False,
            transform=transforms,
            target_transform=target_transforms,
        )
