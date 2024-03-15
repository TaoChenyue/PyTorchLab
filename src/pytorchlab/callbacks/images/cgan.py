import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torchvision.utils import make_grid, save_image

from pytorchlab.callbacks.utils.save_path import get_save_dir

__all__ = ["CGANCallback"]


class CGANCallback(Callback):
    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        nums: int = 8,
        **kwargs,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.nums = nums
        self.kwargs = kwargs

    def _sample_images(self, pl_module: LightningModule, train: bool = True):
        z = torch.randn(self.nums * self.num_classes, self.latent_dim).to(
            pl_module.device
        )
        label = torch.tensor(
            [i for i in range(self.num_classes) for _ in range(self.nums)]
        ).to(pl_module.device)
        if train:
            with torch.no_grad():
                pl_module.eval()
                images = pl_module(z, label)
                pl_module.train()
        else:
            images = pl_module(z, label)
        return images

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        images = self._sample_images(pl_module)
        images = make_grid(images, **self.kwargs)
        save_path = get_save_dir("train", trainer, pl_module)
        save_image(images, save_path / "output.png")
