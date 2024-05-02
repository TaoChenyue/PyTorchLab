import random
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml
from PIL import Image
from torch.utils.data import Dataset, default_collate

from pytorchlab.typehints import ImageDatasetItem

__all__ = [
    "ImageGenerationCollateFn",
    "ImageGenerationDataset",
]


class ImageGenerationCollateFn:
    def __init__(self) -> None:
        pass

    def __call__(self, batch: Iterable[ImageDatasetItem]) -> ImageDatasetItem:
        return default_collate(
            [{"image": image, "generation": image2} for image, image2 in batch]
        )


class ImageGenerationDataset(Dataset):
    def __init__(
        self,
        root: str,
        images: str | list[str] = "images",
        generations: str | list[str] = "generations",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        use_random: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.images = self.load_imgs(images)
        self.generations = self.load_imgs(generations)
        self.transform = transform
        self.target_transform = target_transform if target_transform else transform
        self.use_random = use_random

    def load_imgs(self, images: str | list[str]) -> list[Path]:
        if isinstance(images, str):
            img_path = self.root / images
            assert img_path.exists(), "No such path: {}".format(img_path)
            if img_path.is_dir():
                return [x for x in img_path.iterdir()]
            else:
                img_path = Path(images)
                assert img_path.suffix in [".yml", ".yaml"], "Only support yaml format"
                return yaml.safe_load(open(img_path, "r"))
        elif isinstance(images, list):
            return [x for x in images]
        else:
            raise ValueError("Unsupported type of images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index) -> Any:
        image = self.images[index]
        image = Image.open(image)
        if self.use_random:
            generation = random.choice(self.generations)
        else:
            generation = self.generations[index]
        generation = Image.open(generation)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            generation = self.target_transform(generation)
        return image, generation
