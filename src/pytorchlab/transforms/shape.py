from PIL import Image

__all__ = [
    "ImageCrop",
]


class ImageCrop(object):
    def __init__(self, box: tuple[int, int, int, int]):
        super().__init__()
        self.box = box

    def __call__(self, img: Image.Image) -> Image.Image:
        return img.crop(self.box)
