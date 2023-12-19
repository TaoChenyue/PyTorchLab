import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

from pytorchlab.models.zerodce.components import enhance_net_nopool

from .criterions import L_TV, L_color, L_exp, L_spa


class ZeroDCE(LightningModule):
    def __init__(self, channel: int, patch_size: int = 16, mean_val: float = 0.6):
        super().__init__()
        self.net = enhance_net_nopool(channel)
        self.loss_tv = L_TV()
        self.loss_spa = L_spa()
        self.loss_color = L_color()
        self.loss_exp = L_exp(patch_size, mean_val)

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def training_step(self, batch, batch_idx):
        x, y = batch
        enhance_image1, enhanced_image, A = self(x)
        loss_tv = 200 * self.loss_tv(A)
        loss_spa = torch.mean(self.loss_spa(enhance_image1, enhanced_image))
        loss_col = 5 * torch.mean(self.loss_color(enhanced_image))
        loss_exp = 10 * torch.mean(self.loss_exp(enhanced_image))
        loss = loss_tv + loss_spa + loss_col + loss_exp
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
        enhance_image1, enhanced_image, A = self(x)
        return enhanced_image

    def test_step(self, batch, batch_idx):
        x, y = batch
        enhance_image1, enhanced_image, A = self(x)
        return enhanced_image

    def predict_step(self, batch, batch_idx):
        x, y = batch
        enhance_image1, enhanced_image, A = self(x)
        return enhanced_image
