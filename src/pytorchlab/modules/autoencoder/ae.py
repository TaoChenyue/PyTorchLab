from typing import Sequence

import torch
from jsonargparse import lazy_instance
from lightning.pytorch import LightningModule
from torch import Tensor, nn

from pytorchlab.modules.autoencoder.components import AutoEncoder2d
from pytorchlab.type_hint import ModuleCallable


class AutoEncoder2dModule(LightningModule):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        nf: int = 64,
        depth: int = 8,
        hold_depth: int = 3,
        norm: ModuleCallable = None,
        activation: nn.Module = lazy_instance(nn.ReLU),
        out_activation: nn.Module = lazy_instance(nn.Tanh),
        criterion: nn.Module = lazy_instance(nn.MSELoss),
    ):
        super().__init__()
        self.model = AutoEncoder2d(
            in_channel=in_channel,
            out_channel=out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            nf=nf,
            depth=depth,
            hold_depth=hold_depth,
            norm=norm,
            activation=activation,
            out_activation=out_activation,
        )
        self.criterion = criterion

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)
    
    def training_step(self, batch:Sequence[Tensor], batch_idx:int)->Tensor:
        x,y = batch[0:2]
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log_dict(
            {
                "train_loss": loss
            },
            prog_bar=True,
            sync_dist=True,
        )
        return loss
    
    def validation_step(self,batch: Sequence[Tensor],batch_idx:int,)->Tensor:
        x,y = batch[0:2]
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log_dict(
            {
                "val_loss": loss
            },
        )
        return loss
    
if __name__ == "__main__":
    