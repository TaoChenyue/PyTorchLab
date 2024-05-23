import random
from pathlib import Path
from typing import Any, Callable

from PIL import Image
from torch.utils.data import Dataset

from pytorchlab.typehints import ImageDatasetItem

__all__ = ["ImagePairDataset"]


class ImagePairDataset(Dataset):
    def __init__(
        self,
        root: str,
        A_name: str,
        B_name: str,
        shuffle: bool = False,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.root_A = self.root / A_name
        self.root_B = self.root / B_name
        self.transform = transform
        self.target_transform = target_transform if target_transform else transform

        if not self.root_A.exists():
            raise FileNotFoundError(f"No such path: {self.root_A}")
        if not self.root_B.exists():
            raise FileNotFoundError(f"No such path: {self.root_A}")

        self.paths_A = sorted(
            [x for x in self.root_A.glob("*")],
            key=lambda x: x.stem,
        )
        self.paths_B = sorted(
            [x for x in self.root_B.glob("*")],
            key=lambda x: x.stem,
        )
        if shuffle:
            random.shuffle(self.paths_B)

    def __len__(self):
        return max(len(self.paths_A), len(self.paths_B))

    def __getitem__(self, index) -> Any:
        path_A = self.paths_A[index % len(self.paths_A)]
        img_A = Image.open(path_A)
        path_B = self.paths_B[index % len(self.paths_B)]
        img_B = Image.open(path_B)
        if self.transform is not None:
            img_A = self.transform(img_A)
        if self.target_transform is not None:
            img_B = self.target_transform(img_B)
        return ImageDatasetItem(
            image=img_A,
            image2=img_B,
        )
