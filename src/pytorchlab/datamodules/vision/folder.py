from typing import Callable, Iterable

from torch import Tensor
from torch.utils.data import Dataset
from torchvision import datasets

from pytorchlab.datamodules.vision.abc import VisionDataModule


class ImageFolderDataModule(VisionDataModule):
    def entire_train_dataset(
        self,
        transforms: Callable[[Iterable], Tensor],
        target_transforms: Callable[[Iterable], Tensor] | None,
    ) -> Dataset:
        return datasets.ImageFolder(
            self.train_root.as_posix(),
            transform=transforms,
            target_transform=target_transforms,
        )

    def entire_test_dataset(
        self,
        transforms: Callable[[Iterable], Tensor],
        target_transforms: Callable[[Iterable], Tensor] | None,
    ) -> Dataset:
        return datasets.ImageFolder(
            self.test_root.as_posix(),
            transform=transforms,
            target_transform=target_transforms,
        )
