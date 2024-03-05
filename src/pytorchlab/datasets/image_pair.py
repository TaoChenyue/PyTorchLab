import random
from pathlib import Path
from typing import Any, Callable, Iterable

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class ImagePairDataset(Dataset):
    def __init__(
        self,
        root: str,
        A_name: str,
        B_name: str,
        shuffle:bool = False,
        transform: Callable[[Iterable], Tensor] = None,
        target_transform: Callable[[Iterable], Tensor] = None,
    ) -> None:
        """
        Dataset in form of (image_A, image_B)

        Args:
            root (str): root path of dataset
            A_name (str): directory name of image_A below root path
            B_name (str): directory name of image_B below root path
            transform (Callable[[Iterable],Tensor], optional): transformation for image_A. Defaults to None.
            target_transform (Callable[[Iterable],Tensor], optional): transformation for image_B. Defaults to None.
        """
        super().__init__()
        self.root = Path(root)
        self.root_A = self.root / A_name
        self.root_B = self.root / B_name
        self.transform = transform
        self.target_transform = target_transform if target_transform else transform

        if not self.root_A.exists() or not self.root_B.exists():
            raise FileNotFoundError("Dataset not found")

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
        img_A = self.paths_A[index % len(self.paths_A)]
        img_A = Image.open(img_A)
        img_B = self.paths_B[index % len(self.paths_B)]
        img_B = Image.open(img_B)
        if self.transform is not None:
            img_A = self.transform(img_A)
        if self.target_transform is not None:
            img_B = self.target_transform(img_B)
        return img_A, img_B