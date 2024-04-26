from typing import Iterable

from torch.utils.data import default_collate

from pytorchlab.typehints.datasets import ImageDatasetItem

__all__ = [
    "ClassificationCollateFn",
]


class ClassificationCollateFn:
    def __init__(self) -> None:
        pass

    def __call__(self, batch: Iterable[ImageDatasetItem]) -> ImageDatasetItem:
        return default_collate(
            [{"image": image, "label": label} for image, label in batch]
        )
