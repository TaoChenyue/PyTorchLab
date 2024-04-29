import numpy as np
import torch
from PIL import Image
from torchvision import transforms

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
        assert len(img.split()) == 1, "The input image must be a grayscale image."
        img_np = torch.tensor(np.array(img, dtype=np.uint8))
        img_onehot = torch.cat(
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
        return img_onehot


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
        img = torch.argmax(img, dim=-3, keepdim=True)
        R = self.channel_R.to(img.device)[img]
        G = self.channel_G.to(img.device)[img]
        B = self.channel_B.to(img.device)[img]
        return torch.cat([R, G, B], dim=-3) / 255.0
