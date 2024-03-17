import cv2
import numpy as np
from PIL import Image


class ImageTransform:
    def __call__(self, image: Image.Image) -> Image.Image:
        return image


class GrayTransform(ImageTransform):
    def __init__(self) -> None:
        super().__init__()

    def image_to_numpy(self, image: Image.Image):
        return np.asarray(image.convert("L"), dtype=np.uint8)

    def numpy_to_image(self, image: np.ndarray) -> Image.Image:
        return Image.fromarray(np.uint8(self.range(image) * 255))

    def range(self, image: np.ndarray) -> np.ndarray:
        """
        range image into [0,1]

        Args:
            image (np.ndarray): image

        Returns:
            np.ndarray: image with range
        """
        image = image - np.min(image)
        image = image / np.max(image)
        return image

    def __call__(self, image: Image.Image) -> Image.Image:
        image = self.image_to_numpy(image)
        image = self.core(image)
        return self.numpy_to_image(image)

    def core(self, image: np.ndarray) -> np.ndarray:
        return image


class GrayLogTransform(GrayTransform):
    def __init__(self, v: float = 1.0):
        super().__init__()
        self.v = v

    def core(self, image: np.ndarray) -> np.ndarray:
        image = self.range(image)
        return np.log(1.0 + self.v * image)


class GrayGammaTransform(GrayTransform):
    name = "gamma"

    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def core(self, image: np.ndarray) -> np.ndarray:
        image = self.range(image)
        return np.power(image, self.gamma)


class GrayHETransform(GrayTransform):
    def __init__(self) -> None:
        super().__init__()

    def core(self, image: np.ndarray) -> np.ndarray:
        return cv2.equalizeHist(image)


class GrayCLAHETransform(GrayTransform):
    def __init__(
        self, clipLimit: float = 4.0, tileGridSize: tuple[int, int] = (12, 12)
    ):
        super().__init__()
        self.clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

    def core(self, image: np.ndarray) -> np.ndarray:
        return self.clahe.apply(image)
