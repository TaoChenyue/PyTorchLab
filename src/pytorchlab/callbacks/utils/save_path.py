from pathlib import Path

from lightning import LightningModule, Trainer


def get_epoch_save_path(
    trainer: Trainer,
    pl_module: LightningModule,
) -> Path:
    log_dir = pl_module.logger.log_dir
    if log_dir is None:
        return None
    else:
        log_dir = Path(log_dir)
    stage = trainer.state.stage
    if stage is None:
        stage = ""
    else:
        stage = stage.value
    save_path = log_dir / stage
    save_path.mkdir(exist_ok=True, parents=True)
    return save_path


def get_batch_save_path(
    trainer: Trainer,
    pl_module: LightningModule,
    batch_idx: int = 0,
    dataloader_idx: int = 0,
):
    save_path = get_epoch_save_path(trainer, pl_module)
    if save_path is None:
        return None
    save_path = save_path / f"dataloader={dataloader_idx}" / f"batch={batch_idx}"
    save_path.mkdir(exist_ok=True, parents=True)
    return save_path
