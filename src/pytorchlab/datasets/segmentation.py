from pathlib import Path
from typing import Any, Callable, Iterable

import yaml
from PIL import Image
from torch.utils.data import Dataset, default_collate

from pytorchlab.typehints import ImageDatasetItem

__all__ = [
    "SegmentationCollateFn",
    "SegmentationDataset",
]


class SegmentationCollateFn:
    def __init__(self) -> None:
        pass

    def __call__(self, batch: Iterable[ImageDatasetItem]) -> ImageDatasetItem:
        return default_collate(
            [{"image": image, "mask": mask} for image, mask in batch]
        )


class SegmentationDataset(Dataset):
    def __init__(
        self,
        root: str,
        images: str | list[str] = "images",
        segmentations: str | list[str] = "segmentations",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        """
        Segmentation dataset.

        Args:
            root (str): root path for images and segmentations
            images (str | list[str], optional): If file, means image list, if directory, means all files under it. Defaults to "images".
            segmentations (str | list[str], optional): If file, means segmentation list, if directory, means all files under it.. Defaults to "segmentations".
            transform (Callable | None, optional): transformation for images. Defaults to None.
            target_transform (Callable | None, optional): transformation for segmentations. Defaults to None.

        Raises:
            FileNotFoundError: No such root path
        """
        super().__init__()
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"No such path: {self.root}")
        self.transform = transform
        self.target_transform = target_transform if target_transform else transform
        self.paths = self.load_imgs(images)
        self.paths_seg = self.load_imgs(segmentations)

    def load_imgs(self, path: str | list[str]) -> list[Path]:
        if isinstance(path, str):
            img_path = Path(path)
            if img_path.is_file():
                if img_path.suffix in [".yaml", ".yml"]:
                    img_paths = yaml.safe_load(open(img_path))
                    img_paths = [self.root / x for x in img_paths]
                    return img_paths
                elif img_path.suffix in [".txt"]:
                    img_paths = [x.strip() for x in open(img_path).readlines()]
                    img_paths = [self.root / x for x in img_paths]
                    return img_paths
                else:
                    raise ValueError(f"Unsupported file format: {img_path.suffix}")
            else:
                img_paths = [x for x in img_path.rglob("*") if x.is_file()]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index) -> Any:
        path = self.paths[index]
        img = Image.open(path)
        path_seg = self.paths_seg[index]
        img_seg = Image.open(path_seg)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            img_seg = self.target_transform(img_seg)
        return img, img_seg
