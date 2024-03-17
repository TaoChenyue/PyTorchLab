from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from pytorchlab.transforms import GaussianNoise, PepperSaltNoise


@pytest.fixture
def lena():
    Path("output").mkdir(exist_ok=True, parents=True)
    image = Image.open("images/lena.jpg")
    return transforms.ToTensor()(image).unsqueeze(dim=0)


def test_PepperSaltNoise(lena):
    image_out = [PepperSaltNoise(p=0.1 * i)(lena) for i in range(6)]
    save_image(
        torch.cat(image_out, dim=0),
        f"output/lena_pepper_salt_noise.png",
        nrow=3,
        padding=2,
    )


def test_GaussianNoise(lena):
    image_out = [GaussianNoise(mean=0, std=0.1 * i)(lena) for i in range(9)]
    save_image(
        torch.cat(image_out, dim=0),
        f"output/lena_gaussian_noise.png",
        nrow=3,
        padding=2,
    )
