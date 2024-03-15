from typing import Callable

from jsonargparse import lazy_instance
from torch import Tensor

from pytorchlab.transforms import GaussianNoise
from pytorchlab.typehints import ImageAnomalyItem, OutputDict, OutputsDict

from .ae import AutoEncoder2dModule


class DenoiseAutoEncoder2dModule(AutoEncoder2dModule):
    def __init__(
        self,
        noise_transform: Callable[[Tensor], Tensor] = lazy_instance(
            GaussianNoise,
            mean=0,
            std=0.1,
        ),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.noise_transform = noise_transform

    def _step(self, batch: ImageAnomalyItem, batch_idx: int, dataloader_idx: int = 0):
        x = batch["image"]
        x_noise = self.noise_transform(x).detach()
        pred = self(x_noise)
        loss = self.criterion(pred, x)
        return OutputsDict(
            loss=loss,
            losses={"loss": loss},
            inputs=OutputDict(images={"image": x_noise, "reconstructed": x}),
            outputs=OutputDict(images={"reconstructed": pred}),
        )
