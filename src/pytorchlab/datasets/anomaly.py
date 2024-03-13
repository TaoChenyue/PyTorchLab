from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from torch.utils.data import Dataset


class AnomalyDataset(Dataset):
    def __init__(
        self,
        root: str,
        normal_name: str = "good",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.file_paths: list[Path] = []
        for directory in self.root.iterdir():
            if not directory.is_dir():
                continue
            self.file_paths += [x for x in directory.iterdir()]
        self.normal_name = normal_name
        self.transform = transform
        self.target_transform = (
            target_transform if target_transform is not None else transform
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index) -> Any:
        img: Image.Image = Image.open(self.file_paths[index])
        img_A = img.copy()
        img_B = img.copy()
        img_cls = self.file_paths[index].parent.name
        if img_cls == self.normal_name:
            label = 0
        else:
            label = 1
        if self.transform is not None:
            img_A = self.transform(img_A)
        if self.target_transform is not None:
            img_B = self.target_transform(img_B)
        return img_A, img_B, torch.tensor(label)
