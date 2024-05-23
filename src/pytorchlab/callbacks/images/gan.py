import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torchvision.utils import make_grid, save_image

from pytorchlab.callbacks.utils.save_path import get_save_dir

__all__ = ["GANCallback"]


class GANCallback(Callback):
    def __init__(
        self,
        latent_dim: int,
        nums: int = 8,
        **kwargs,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.nums = nums
        self.kwargs = kwargs

    def _sample_images(self, pl_module: LightningModule, train: bool = True):
        z = torch.randn(self.nums, self.latent_dim).to(pl_module.device)
        if train:
            with torch.no_grad():
                pl_module.eval()
                images = pl_module(z)
                pl_module.train()
        else:
            images = pl_module(z)
        return images

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        save_path = get_save_dir("val", trainer, pl_module)
        if save_path is None:
            return

        images = self._sample_images(pl_module)
        images = make_grid(images, **self.kwargs)

        out_name = f"output.png"
        save_name = save_path / out_name
        save_image(images, save_name)
