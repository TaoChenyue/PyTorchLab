from typing import TypedDict

from torch import Tensor


class ImageClassifyItem(TypedDict):
    image: Tensor
    label: Tensor


class ImagePairItem(TypedDict):
    image1: Tensor
    image2: Tensor


class ImageAnomalyItem(TypedDict):
    image: Tensor
    # ground truth
    mask: Tensor
    # for classification
    label: Tensor
