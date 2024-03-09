import pytest


@pytest.fixture
def lena():
    from pathlib import Path

    from PIL import Image
    from torchvision import transforms

    Path("output").mkdir(exist_ok=True, parents=True)
    image = Image.open("images/lena.jpg")
    return transforms.ToTensor()(image).unsqueeze(dim=0)


def test_PepperSaltNoise(lena):
    import torch
    from torchvision.utils import save_image

    from pytorchlab.transforms.noise import PepperSaltNoise

    image_out = [PepperSaltNoise(p=0.1 * i)(lena) for i in range(6)]
    save_image(
        torch.cat(image_out, dim=0),
        f"output/lena_pepper_salt_noise.png",
        nrow=3,
        padding=2,
    )


def test_GaussianNoise(lena):
    import torch
    from torchvision.utils import save_image

    from pytorchlab.transforms.noise import GaussianNoise

    image_out = [GaussianNoise(mean=0, std=0.1 * i)(lena) for i in range(9)]
    save_image(
        torch.cat(image_out, dim=0),
        f"output/lena_gaussian_noise.png",
        nrow=3,
        padding=2,
    )
