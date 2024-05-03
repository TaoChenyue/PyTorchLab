from lightning.pytorch.cli import LightningCLI

def cli():
    LightningCLI(parser_kwargs={"parser_mode": "omegaconf"})