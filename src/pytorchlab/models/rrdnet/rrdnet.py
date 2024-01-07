from typing import Any

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

from .components import RRDNet as _RRDNet
from .criterions import (illumination_smooth_loss, noise_loss,
                         reconstruction_loss, reflectance_smooth_loss)


class RRDNet(LightningModule):
    def __init__(
        self,
        channel: int = 3,
        illu_factor=1,
        reflect_factor=1,
        noise_factor=5000,
        reffac=1,
        gamma=0.4,
    ) -> None:
        super().__init__()
        self.illu_factor = illu_factor
        self.reflect_factor = reflect_factor
        self.noise_factor = noise_factor
        self.reffact = reffac
        self.gamma = gamma
        self.net = _RRDNet(in_channel=channel)

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch
        illumination, reflectance, noise = self(x)
        loss_recons = reconstruction_loss(x, illumination, reflectance, noise)
        loss_illu = illumination_smooth_loss(x, illumination)
        loss_reflect = reflectance_smooth_loss(x, illumination, reflectance)
        loss_noise = noise_loss(x, illumination, reflectance, noise)

        loss = (
            loss_recons
            + self.illu_factor * loss_illu
            + self.reflect_factor * loss_reflect
            + self.noise_factor * loss_noise
        )
        self.log_dict(
            {
                "loss": loss,
            },
            sync_dist=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch
        illumination, reflectance, noise = self(x)
        adjust_illu = torch.pow(illumination, self.gamma)
        res_image = adjust_illu * ((x - noise) / illumination)
        res_image = torch.clamp(res_image, min=0, max=1)
        return res_image

    def test_step(self, batch, batch_idx):
        x, y = batch
        illumination, reflectance, noise = self(x)
        adjust_illu = torch.pow(illumination, self.gamma)
        res_image = adjust_illu * ((x - noise) / illumination)
        res_image = torch.clamp(res_image, min=0, max=1)
        return res_image

    def predict_step(self, batch, batch_idx) -> Any:
        x, y = batch
        illumination, reflectance, noise = self(x)
        adjust_illu = torch.pow(illumination, self.gamma)
        res_image = adjust_illu * ((x - noise) / illumination)
        res_image = torch.clamp(res_image, min=0, max=1)
        return res_image
