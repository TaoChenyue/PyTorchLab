import torch
from torchvision.transforms import GaussianBlur


class PepperSaltNoise(object):
    def __init__(
        self, p: float = 0.05, pepper: float | None = None, salt: float | None = None
    ) -> None:
        self.p = p
        self.pepper = pepper
        self.salt = salt

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.rand_like(x.index_select(dim=-3,index=torch.tensor([0])))
        noise = noise.repeat_interleave(repeats=x.shape[-3],dim=-3)
        salt = (
            torch.max(x)
            if self.salt is None
            else torch.tensor(self.salt).to(x.device)
        )
        pepper = (
            torch.min(x)
            if self.pepper is None
            else torch.tensor(self.pepper).to(x.device)
        )
        x[noise < self.p / 2] = pepper
        x[noise > 1 - self.p / 2] = salt
        return x
