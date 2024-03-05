from functools import partial
from pathlib import Path

import torch
import torchvision
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from pytorchlab.callbacks.generation import GANCallback
from pytorchlab.datamodules import BaseDataModule
from pytorchlab.models.gan import GAN
from pytorchlab.modules.gans import LinearDiscriminator, LinearGenerator

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--batch_size",type=int,default=256)
    parser.add_argument("--num_workers",type=int,default=20)
    parser.add_argument("--max_epochs",type=int,default=100)
    parser.add_argument("--devices",type=int,nargs="+",default=[1])
    parser.add_argument("--ckpt_path",type=str,default=None)
    parser.add_argument("--interpolate",type=int,default=10)
    parser.add_argument("--seed",type=int,default=None)
    return parser.parse_args()

def latent_space_interpolate(model: GAN, num_images: int = 10):
    model.eval()
    latent_space = torch.randn((1, model.latent_dim, 2))
    latent_space = torch.nn.functional.interpolate(
        latent_space, size=num_images, mode="linear"
    )
    # print(latent_space.shape)
    save_path = Path("tmp") / "gan"
    save_path.mkdir(parents=True, exist_ok=True)
    save_image(
        torch.cat([model(latent_space[:, :, i].to(model.device)) for i in range(num_images)], dim=0),
        save_path / "latent_space_interpolate.png",
    )


if __name__ == "__main__":
    channel = 1
    height = 28
    width = 28
    
    args = parse_args()
    
    if args.ckpt_path is None:
        load_func = GAN
        load_kwargs = {}
    else:
        load_func = GAN.load_from_checkpoint
        load_kwargs = {"checkpoint_path": args.ckpt_path}

    model = load_func(
        latent_dim=args.latent_dim,
        generator=LinearGenerator(
            channel=channel,
            height=height,
            width=width,
            latent_dim=args.latent_dim,
            bn=False,
            hidden_layers=[256, 512],
            activation=nn.LeakyReLU(negative_slope=0.2, inplace=True),
            out_activation=nn.Tanh(),
        ),
        discriminator=LinearDiscriminator(
            channel=channel,
            height=height,
            width=width,
            hidden_layers=[512, 256],
            dropout=0.3,
            bn=False,
            activation=nn.LeakyReLU(negative_slope=0.2, inplace=True),
            out_activation=nn.Sigmoid(),
        ),
        criterion=nn.BCELoss(),
        optimizer_g=partial(torch.optim.Adam, lr=0.0001, betas=(0.5, 0.999)),
        optimizer_d=partial(torch.optim.Adam, lr=0.0001, betas=(0.5, 0.999)),
        lr_g=partial(torch.optim.lr_scheduler.ConstantLR, factor=1.0),
        lr_d=partial(torch.optim.lr_scheduler.ConstantLR, factor=1.0),
        **load_kwargs,
    )

    transform = torchvision.transforms.ToTensor()
    datamodule = BaseDataModule(
        train_dataset=MNIST(root="dataset", train=True, transform=transform),
        test_dataset=MNIST(root="dataset", train=False, transform=transform),
        split=0,
        batch_size=args.batch_size,
        num_workers=20,
    )

    trainer = Trainer(
        max_epochs=args.max_epochs,
        devices=args.devices,
        logger=TensorBoardLogger(
            save_dir="logs/gan",
            name="mnist",
        ),
        callbacks=[GANCallback(latent_dim=args.latent_dim, nums=100, nrow=10, padding=2)],
    )
    
    if args.seed is not None:
        seed_everything(seed=args.seed)

    # train
    # trainer.fit(model=model, datamodule=datamodule)
    # test
    # trainer.test(model=model, datamodule=datamodule)
    # visualize
    latent_space_interpolate(model,args.interpolate)
