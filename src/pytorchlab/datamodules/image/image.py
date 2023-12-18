from typing import Any, Callable, Literal

from torch.utils.data import Dataset

from pytorchlab.datamodules.abstract.vision import VisionDataModule
from pytorchlab.datamodules.image.datasets import ImagePairDataset


class ImagePairDataModule(VisionDataModule):
    def __init__(
        self,
        A_name: str,
        B_name: str,
        suffix_list: list[str] = [".jpg", ".png", ".bmp"],
        mode_A: Literal["RGB", "L"] = "RGB",
        mode_B: Literal["RGB", "L"] = "RGB",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.A_name = A_name
        self.B_name = B_name
        self.suffix_list = suffix_list
        self.mode_A = mode_A
        self.mode_B = mode_B

    def entire_train_dataset(
        self,
        transforms: Callable[..., Any],
        target_transforms: Callable[..., Any],
    ) -> Dataset:
        return ImagePairDataset(
            root=self.train_root.as_posix(),
            A_name=self.A_name,
            B_name=self.B_name,
            suffix_list=self.suffix_list,
            mode_A=self.mode_A,
            mode_B=self.mode_B,
            transform=transforms,
            target_transform=target_transforms,
        )

    def entire_test_dataset(
        self,
        transforms: Callable[..., Any],
        target_transforms: Callable[..., Any],
    ) -> Dataset:
        return ImagePairDataset(
            self.test_root.as_posix(),
            A_name=self.A_name,
            B_name=self.B_name,
            suffix_list=self.suffix_list,
            mode_A=self.mode_A,
            mode_B=self.mode_B,
            transform=transforms,
            target_transform=target_transforms,
        )
