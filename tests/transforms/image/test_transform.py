from pathlib import Path

import pytest
from PIL import Image

from pytorchlab.transforms import (
    GrayCLAHETransform,
    GrayGammaTransform,
    GrayHETransform,
    GrayLogTransform,
    GrayTransform,
    ImageTransform,
)


@pytest.fixture
def lena():
    Path("output").mkdir(exist_ok=True, parents=True)
    image = Image.open("images/lena.jpg")
    return image


def test_ImageTransform(lena):
    ImageTransform()(lena).save("output/lena_ImageTransform.png")


def test_GrayTransform(lena):
    GrayTransform()(lena).save("output/lena_GrayTransform.png")


def test_GrayLogTransform(lena):
    GrayLogTransform(v=2)(lena).save("output/lena_GrayLogTransform.png")


def test_GrayGammaTransform(lena):
    GrayGammaTransform(gamma=2.2)(lena).save("output/lena_GrayGammaTransform.png")


def test_GrayHETransform(lena):
    GrayHETransform()(lena).save("output/lena_GrayHETransform.png")


def test_GrayCLAHETransform(lena):
    GrayCLAHETransform()(lena).save("output/lena_GrayCLAHETransform.png")
