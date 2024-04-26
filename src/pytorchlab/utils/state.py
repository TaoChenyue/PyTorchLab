from pathlib import Path

from lightning import LightningModule, Trainer
from lightning.pytorch import Trainer


def get_stage(trainer: Trainer):
    stage = trainer.state.stage
    if stage is None:
        return ""
    return stage.value


def get_epoch_save_path(
    trainer: Trainer,
    pl_module: LightningModule,
    show_epoch: bool = False,
) -> Path:
    log_dir = pl_module.logger.log_dir
    if log_dir is None:
        return None
    log_dir = Path(log_dir)
    save_path = log_dir / get_stage(trainer)
    if show_epoch:
        save_path = save_path / f"epoch={trainer.current_epoch}"
    save_path.mkdir(exist_ok=True, parents=True)
    return save_path


def get_batch_save_path(
    trainer: Trainer,
    pl_module: LightningModule,
    batch_idx: int = 0,
    dataloader_idx: int = 0,
    show_epoch: bool = False,
):
    save_path = get_epoch_save_path(trainer, pl_module, show_epoch)
    if save_path is None:
        return None
    save_path = save_path / f"dataloader={dataloader_idx}" / f"batch={batch_idx}"
    save_path.mkdir(exist_ok=True, parents=True)
    return save_path
