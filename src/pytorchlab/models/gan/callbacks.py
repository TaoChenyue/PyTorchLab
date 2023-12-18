from pathlib import Path

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torchvision.utils import make_grid, save_image


class GANCallback(Callback):
    def __init__(
        self,
        latent_dim: int,
        nums: int = 8,
        tag: str = "gan_images",
        **kwargs,
    ):
        """generate images on GAN

        Args:
            latent_dim(int): dimension of latent code
            nums (int, optional): number of images. Defaults to 8.
            tag(str, optional): tag name of directory for saving images. Default to "gan_images".
            save_on_tensorboard(bool, optional): save image on tensorboard. Default to True.
            save_on_directory(bool, optional): save image on directory. Default to True.
            **kwargs: Other arguments are documented in make_grid
        """
        self.latent_dim = latent_dim
        self.nums = nums
        self.tag = tag
        self.kwargs = kwargs

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
        save_path = Path(log_path) / self.tag
        save_path.mkdir(exist_ok=True, parents=True)
        save_name = save_path / f"epoch_{trainer.current_epoch}.png"
        save_image(images, save_name)
