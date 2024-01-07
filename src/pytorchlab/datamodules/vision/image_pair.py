from pathlib import Path
from typing import Any, Callable, Literal, overload

from torch.utils.data import Dataset

from pytorchlab.datamodules.vision.abc import VisionDataModule
from pytorchlab.datasets.vision.image_pair import ImagePairDataset


class ImagePairDataModule(VisionDataModule):
    @overload
    def __init__(
        self,
        A_name: str,
        B_name: str,
        suffix_list: list[str] = [".jpg", ".png", ".bmp"],
        mode_A: Literal["RGB", "L"] = "RGB",
        mode_B: Literal["RGB", "L"] = "RGB",
        train_root: str | Path = "dataset",
        test_root: str | Path = "dataset",
        val_in_train: bool = True,
        val_split: int | float = 0.2,
        split_seed: int | None = 42,
        num_workers: int = 0,
        batch_size: int = 32,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        transforms: Callable | None = None,
        target_transforms: Callable | None = None,
        train_transforms: Callable | None = None,
        train_target_transforms: Callable | None = None,
        test_transforms: Callable | None = None,
        test_target_transforms: Callable | None = None,
    ):
        ...

    def __init__(
        self,
        A_name: str,
        B_name: str,
        suffix_list: list[str] = [".jpg", ".png", ".bmp"],
        mode_A: Literal["RGB", "L"] = "RGB",
        mode_B: Literal["RGB", "L"] = "RGB",
        **kwargs,
    ) -> None:
        """
        Image pair datamodule in form of (image_A, image_B)

        Args:
            A_name (str): Directory name of image_A
            B_name (str): Directory name of image_B
            suffix_list (list[str], optional): suffix list to filter files. Defaults to [".jpg", ".png", ".bmp"].
            mode_A (Literal[&quot;RGB&quot;, &quot;L&quot;], optional): image_A should be RGB or gray image. Defaults to "RGB".
            mode_B (Literal[&quot;RGB&quot;, &quot;L&quot;], optional): image_B should be RGB or gray image. Defaults to "RGB".
            **kwargs: see VisionDataModule.__doc__
        """
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
