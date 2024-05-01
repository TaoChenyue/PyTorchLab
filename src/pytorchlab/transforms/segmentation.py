import numpy as np
import torch
from PIL import Image

__all__ = [
    "Image2SementicLabel",
    "RandomColormap",
]


class Image2SementicLabel(object):
    def __init__(self, num_classes: int = 1):
        self.num_classes = num_classes

    def __call__(self, img: Image.Image) -> torch.Tensor:
        assert len(img.split()) == 1, "The input image must be a grayscale image."
        img_np = torch.tensor(np.array(img, dtype=np.uint8))
        if self.num_classes == 1:
            return torch.where(
                img_np == 1,
                torch.ones_like(img_np, dtype=torch.float32),
                torch.zeros_like(img_np, dtype=torch.float32),
            ).unsqueeze(dim=0)
        return torch.cat(
            [
                torch.where(
                    img_np == i,
                    torch.ones_like(img_np, dtype=torch.float32),
                    torch.zeros_like(img_np, dtype=torch.float32),
                ).unsqueeze(dim=0)
                for i in range(self.num_classes)
            ],
            dim=-3,
        )


class RandomColormap(object):
    def __init__(self, num_classes: int = 1, seed: int = 1234):
        self.num_classes = num_classes
        self.channels = torch.randint(
            0,
            255,
            (num_classes, 3),
            generator=torch.Generator().manual_seed(seed),
        )

    def __call__(self, img: torch.Tensor):
        if self.num_classes == 1:
            return img
        img = torch.argmax(img, dim=-3, keepdim=True)
        return (
            self.channels.to(img.device)[img].squeeze(dim=1).permute(0, 3, 1, 2) / 255.0
        )
