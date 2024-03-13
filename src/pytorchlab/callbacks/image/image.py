from typing import Any, Literal, Mapping, Sequence

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor
from torchvision.utils import make_grid, save_image

from pytorchlab.callbacks.image.utils import get_save_dir


class ImageCallback(Callback):
    def __init__(
        self,
        batch: int = 0,
        image_slice: tuple[int | None, int | None, int | None] = (None, None, None),
        **kwargs,
    ) -> None:
        super().__init__()
        self.batch: int = batch
        self.image_slice = slice(*image_slice)
        self.kwargs = kwargs

    def _get_images(
        self, outputs: Tensor | Mapping[str, Any] | None
    ) -> list[Tensor] | None:
        if not isinstance(outputs, Mapping):
            return [], []
        input_images = outputs.get("inputs", {}).get("images", [])
        output_images = outputs.get("outputs", {}).get("images", [])
        return input_images, output_images

    def _batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Sequence[Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
        mode: Literal["val", "test", "predict"] = "val",
    ) -> None:
        if mode == "val" and batch_idx != self.batch:
            return

        input_images, output_images = self._get_images(outputs)

        save_dir = get_save_dir(
            mode,
            trainer,
            pl_module,
            batch_idx,
            dataloader_idx,
        )
        if save_dir is None:
            return

        images_dict = {
            f"input_{index}.png": input_images[index][self.image_slice]
            for index in range(len(input_images))
        }
        images_dict.update(
            {
                f"output_{index}.png": output_images[index][self.image_slice]
                for index in range(len(output_images))
            }
        )

        for name, image in images_dict.items():
            save_image(make_grid(image, **self.kwargs), save_dir / name)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        return self._batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
            mode="val",
        )

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        return self._batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
            mode="test",
        )

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        return self._batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
            mode="predict",
        )
