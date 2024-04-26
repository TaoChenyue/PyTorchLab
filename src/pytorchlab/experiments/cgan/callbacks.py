import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback
from torchvision.utils import make_grid, save_image

from pytorchlab.experiments.gan.lightning_model import GANModule
from pytorchlab.utils.state import get_epoch_save_path, get_stage

__all__ = [
    "CGANCallback",
]


class CGANCallback(Callback):
    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        nums: int = 8,
        show_epoch: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.nums = nums
        self.show_epoch = show_epoch
        self.kwargs = kwargs

    def _epoch_end(self, trainer: Trainer, pl_module: GANModule):
        save_path = get_epoch_save_path(trainer, pl_module, self.show_epoch)
        if save_path is None:
            return

        z = torch.randn(self.nums * self.num_classes, self.latent_dim).to(
            pl_module.device
        )
        label = torch.tensor(
            [i for i in range(self.num_classes) for _ in range(self.nums)]
        ).to(pl_module.device)

        if get_stage(trainer) == "train":
            with torch.no_grad():
                pl_module.eval()
                images = pl_module(z, label)
                pl_module.train()
        else:
            images = pl_module(z, label)
        images = make_grid(images, **self.kwargs)

        out_name = f"output.png"
        save_name = save_path / out_name
        save_image(images, save_name)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: GANModule) -> None:
        self._epoch_end(trainer, pl_module)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: GANModule,
        outputs: torch.Any,
        batch: torch.Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        return self._epoch_end(trainer, pl_module)
