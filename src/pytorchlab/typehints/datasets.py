from typing import TypedDict

from torch import Tensor

__all__ = [
    "ImageDatasetItem",
]


class ImageDatasetItem(TypedDict, total=False):
    image: Tensor  # source image
    generation: Tensor  # for generation
    mask: Tensor  # for segmentation
    label: Tensor  # for classification
    score: Tensor  # for quality assessment
