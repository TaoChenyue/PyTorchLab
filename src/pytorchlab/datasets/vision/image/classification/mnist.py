from typing import Callable

from torch.utils.data import Dataset
from torchvision import datasets

from pytorchlab.typehints.datasets import ImageDatasetItem

__all__ = ["MNISTDataset", "FashionMNISTDataset"]


class MNISTDataset(Dataset):
    def __init__(
        self,
        root: str = "dataset",
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        super().__init__()
        self.dataset = datasets.MNIST(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return ImageDatasetItem(image=image, label=label)


class FashionMNISTDataset(Dataset):
    def __init__(
        self,
        root: str = "dataset",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        super().__init__()
        self.dataset = datasets.FashionMNIST(
            root=root,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return ImageDatasetItem(image=image, label=label)
