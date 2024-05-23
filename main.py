from lightning.pytorch.cli import LightningCLI

def main():
    cli = LightningCLI(parser_kwargs={"parser_mode": "omegaconf"})

if __name__ == "__main__":
    main()
