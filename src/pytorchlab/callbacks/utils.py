from pathlib import Path

from lightning import LightningModule, Trainer


def get_epoch_save_path(
    trainer: Trainer,
    pl_module: LightningModule,
    on_epoch: bool = False,
    default: str = "output",
) -> Path:
    log_dir = pl_module.logger.log_dir
    if log_dir is None:
        log_dir = Path(default)
    else:
        log_dir = Path(log_dir)
    stage = trainer.state.stage
    if stage is None:
        stage = ""
    else:
        stage = str(stage)
    if on_epoch:
        epoch = f"epoch={trainer.current_epoch}"
    else:
        epoch = "_epoch"
    save_path = log_dir / stage / epoch
    save_path.mkdir(exist_ok=True, parents=True)
    return save_path


def get_batch_save_path(
    trainer: Trainer,
    pl_module: LightningModule,
    batch_idx: int = 0,
    dataloader_idx: int = 0,
    on_epoch: bool = False,
    default: str = "output",
):
    save_path = get_epoch_save_path(trainer, pl_module, on_epoch, default)
    save_path = save_path / f"dataloader={dataloader_idx}" / f"batch={batch_idx}"
    save_path.mkdir(exist_ok=True, parents=True)
    return save_path
