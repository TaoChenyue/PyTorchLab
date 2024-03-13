from pathlib import Path
from typing import Literal

from lightning import LightningModule, Trainer


def get_save_dir(
    mode: Literal["train", "val", "test", "predict"],
    trainer: Trainer,
    pl_module: LightningModule,
    batch_idx: int = 0,
    dataloader_idx: int = 0,
) -> Path | None:
    log_dir = pl_module.logger.log_dir
    if log_dir is None:
        return None
    log_dir = Path(log_dir)
    tag: str = f"{pl_module.__class__.__name__}"
    log_dir = log_dir / tag / f"{mode}_images"
    if mode in ["train", "val"]:
        log_dir = log_dir / f"epoch={trainer.current_epoch}"
    if mode in ["val", "test", "predict"]:
        log_dir = log_dir / f"dataloader={dataloader_idx}"
    save_dir = log_dir / f"batch={batch_idx}"
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir
