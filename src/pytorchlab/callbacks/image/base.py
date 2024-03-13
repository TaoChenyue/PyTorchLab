from pathlib import Path
from typing import Literal

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback


class BaseImageCallback(Callback):
    def __init__(self, on_epoch: bool = False):
        super().__init__()
        self.on_epoch = on_epoch

    def _get_log_dir(self, pl_module: LightningModule) -> Path | None:
        log_path = pl_module.logger.log_dir
        if log_path is None:
            return None
        return Path(log_path)

    def _get_save_dir(
        self,
        mode: Literal["train", "val", "test", "predict"],
        trainer: Trainer,
        pl_module: LightningModule,
        batch_idx: int = 0,
        dataloader_idx: int = 0,
    ) -> Path | None:
        log_dir = self._get_log_dir(pl_module)
        if log_dir is None:
            return None
        tag: str = f"{pl_module.__class__.__name__}_images"
        log_dir = log_dir / tag
        if mode in ["train", "val"]:
            save_dir = (
                (
                    log_dir
                    / f"{mode}_images"
                    / f"dataloader={dataloader_idx}"
                    / f"epoch={trainer.current_epoch}"
                )
                if self.on_epoch
                else (
                    log_dir
                    / f"{mode}_images"
                    / f"dataloader={dataloader_idx}"
                    / "_latest"
                )
            )

        else:
            save_dir = (
                log_dir
                / f"{mode}_images"
                / f"dataloader={dataloader_idx}"
                / f"batch={batch_idx}"
            )
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir
