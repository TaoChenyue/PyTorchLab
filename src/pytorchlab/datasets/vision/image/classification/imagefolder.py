from typing import Any, Callable

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.datasets.folder import default_loader

from pytorchlab.typehints.datasets import ImageDatasetItem

__all__ = ["ImageFolderDataset"]


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        root: str = "dataset",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        loader: Callable[[str], Any] | None = default_loader,
        is_valid_file: Callable[[str], bool] | None = None,
    ) -> None:
        super().__init__()
        self.dataset = datasets.ImageFolder(
            root=root,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            is_valid_file=is_valid_file,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return ImageDatasetItem(image=image, label=label)
