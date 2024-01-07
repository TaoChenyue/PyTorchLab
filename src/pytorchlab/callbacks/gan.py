from pathlib import Path
from typing import overload

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torchvision.utils import make_grid, save_image


class GANCallback(Callback):
    @overload
    def __init__(
        self,
        latent_dim: int,
        nums: int = 9,
        tag: str | None = None,
        override: bool = False,
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = False,
        value_range: tuple[int, int] | None = None,
        scale_each: bool = False,
        pad_value: float = 0.0,
    ):
        ...

    def __init__(
        self,
        latent_dim: int,
        nums: int = 9,
        tag: str | None = None,
        override: bool = False,
        **kwargs,
    ):
        """generate images on GAN

        Args:
            latent_dim(int): dimension of latent code
            nums (int, optional): number of images. Defaults to 8.
            tag(str, optional): tag name of directory for saving images.If None, use class name. Default to None.
            **kwargs: Other arguments are documented in make_grid
        """
        self.latent_dim = latent_dim
        self.nums = nums
        self.tag = tag
        self.kwargs = kwargs
        self.override = override

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        z = torch.randn(self.nums, self.latent_dim).to(pl_module.device)
        with torch.no_grad():
            pl_module.eval()
            images = pl_module(z)
            pl_module.train()
        images = make_grid(images, **self.kwargs)
        log_path = pl_module.logger.log_dir
        if log_path is None:
            return
        if self.tag is None:
            tag = f"{pl_module.__class__.__name__}_images"
        else:
            tag = self.tag
        save_path = Path(log_path) / tag
        save_path.mkdir(exist_ok=True, parents=True)
        out_name = (
            f"image.png" if self.override else f"epoch_{trainer.current_epoch}.png"
        )
        save_name = save_path / out_name
        save_image(images, save_name)


class ConditionalGANCallback(Callback):
    @overload
    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        nums: int = 5,
        tag: str | None = None,
        override: bool = False,
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = False,
        value_range: tuple[int, int] | None = None,
        scale_each: bool = False,
        pad_value: float = 0.0,
    ):
        ...

    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        nums: int = 5,
        tag: str | None = None,
        override: bool = False,
        **kwargs,
    ):
        """generate images on GAN

        Args:
            latent_dim(int): dimension of latent code
            nums (int, optional): number of images. Defaults to 8.
            num_classes (int, optional): number of classes. Defaults to 10.
            tag(str, optional): tag name of directory for saving images.If None, use class name. Default to None.
            **kwargs: Other arguments are documented in make_grid
        """
        self.latent_dim = latent_dim
        self.nums = nums
        self.num_classes = num_classes
        self.tag = tag
        self.kwargs = kwargs
        self.override = override

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        z = torch.randn(self.nums * self.num_classes, self.latent_dim).to(
            pl_module.device
        )
        label = torch.tensor(
            [i for i in range(self.num_classes) for _ in range(self.nums)]
        ).to(pl_module.device)
        with torch.no_grad():
            pl_module.eval()
            images = pl_module(z, label)
            pl_module.train()
        images = make_grid(images, **self.kwargs)
        log_path = pl_module.logger.log_dir
        if log_path is None:
            return
        if self.tag is None:
            tag = f"{pl_module.__class__.__name__}_images"
        else:
            tag = self.tag
        save_path = Path(log_path) / tag
        save_path.mkdir(exist_ok=True, parents=True)
        out_name = (
            f"image.png" if self.override else f"epoch_{trainer.current_epoch}.png"
        )
        save_name = save_path / out_name
        save_image(images, save_name)
