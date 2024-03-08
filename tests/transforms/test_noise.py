import pytest


@pytest.fixture
def lena():
    from PIL import Image
    from torchvision import transforms
    from pathlib import Path

    Path("output").mkdir(exist_ok=True, parents=True)
    image = Image.open("images/lena.jpg")
    return transforms.ToTensor()(image).unsqueeze(dim=0)


def test_PepperSaltNoise(lena):
    from pytorchlab.transforms.noise import PepperSaltNoise
    from torchvision.utils import save_image
    import torch

    image_out = [PepperSaltNoise(p=0.1 * i)(lena.clone().detach()) for i in range(6)]
    save_image(
        torch.cat(image_out, dim=0),
        f"output/lena_pepper_salt_noise.png",
        nrow=3,
        padding=2,
    )
