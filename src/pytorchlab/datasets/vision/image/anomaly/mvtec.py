from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

from pytorchlab.typehints import ImageDatasetItem

__all__ = ["MvtecMask", "MvtecAnomalyDataset"]


class MvtecMask:
    def __init__(self, transform: Callable | None = None) -> None:
        self.transform = ToTensor() if transform is None else transform

    def get_mask_path(self, path: str) -> Path:
        path: Path = Path(path)
        cls_name = path.parent.name
        mask_path = (
            path.parent.parent.parent
            / "ground_truth"
            / cls_name
            / (path.stem + "_mask.png")
        )
        return mask_path

    def __call__(self, path: str):
        mask_path = self.get_mask_path(path)
        if not mask_path.exists():
            # raise FileNotFoundError(f"{mask_path} does not exist.")
            origin_image = Image.open(path).convert("L")
            mask_image = Image.new("L", origin_image.size, 0)
        else:
            mask_image = Image.open(mask_path).convert("L")
        mask_image = self.transform(mask_image)
        return mask_image


class MvtecAnomalyDataset(Dataset):
    def __init__(
        self,
        root: str,
        normal_names: list[str] = ["good"],
        transform: Callable | None = None,
        get_mask: Callable | None = None,
    ) -> None:
        super().__init__()
        self.dataset = ImageFolder(
            root=root,
            transform=transform,
        )
        self.get_mask = get_mask
        self.normal_names = normal_names

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> ImageDatasetItem:
        image, label = self.dataset[index]
        image_path = self.dataset.imgs[index][0]
        name = self.dataset.classes[label]
        item = ImageDatasetItem(
            image=image,
            label=0 if name in self.normal_names else 1,
        )
        if self.get_mask is not None:
            mask = self.get_mask(image_path)
            item["image2"] = mask
        return item
