import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from typing import Literal

__all__ = [
    "Image2SementicLabel",
    "RandomColormap",
]


class Image2SementicLabel(object):
    def __init__(self, num_classes: int = 1):
        self.num_classes = num_classes

    def __call__(self, img: Image.Image) -> torch.Tensor:
        if self.num_classes == 1:
            return transforms.ToTensor()(img)
        img_np = np.array(img, dtype=np.uint8)
        if img_np.ndim == 3:
            img_np = img_np.squeeze(axis=-1)
        img_np = np.where(img_np >= self.num_classes, 0, img_np)
        img = torch.tensor(img_np, dtype=torch.long)
        return img.unsqueeze(dim=0)


class RandomColormap(object):
    def __init__(self, num_classes: int = 1, seed: int = 1234):
        self.num_classes = num_classes
        self.channel_R = torch.randint(
            0,
            255,
            (num_classes,),
            generator=torch.Generator().manual_seed(seed),
        )
        self.channel_G = torch.randint(
            0,
            255,
            (num_classes,),
            generator=torch.Generator().manual_seed(seed * 2),
        )
        self.channel_B = torch.randint(
            0,
            255,
            (num_classes,),
            generator=torch.Generator().manual_seed(seed * 3),
        )

    def __call__(self, img: torch.Tensor):
        if self.num_classes == 1:
            return img
        R = self.channel_R.to(img.device)[img]
        G = self.channel_G.to(img.device)[img]
        B = self.channel_B.to(img.device)[img]
        return torch.cat([R, G, B], dim=-3) / 255.0
