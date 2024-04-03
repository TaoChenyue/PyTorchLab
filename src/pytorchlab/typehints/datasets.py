from typing import TypedDict

from torch import Tensor


class ImageDatasetItem(TypedDict, total=False):
    image: Tensor  # source image
    image2: Tensor  # for generation
    label: Tensor  # for classification
    score: Tensor  # for quality assessment
