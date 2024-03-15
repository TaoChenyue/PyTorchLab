import torch

__all__ = ["PepperSaltNoise", "GaussianNoise"]


class PepperSaltNoise(object):
    def __init__(
        self, p: float = 0.05, pepper: float | None = None, salt: float | None = None
    ) -> None:
        self.p = p
        self.pepper = pepper
        self.salt = salt

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone().detach()
        noise = torch.rand_like(
            x.index_select(dim=-3, index=torch.tensor(0, device=x.device))
        )
        noise = noise.repeat_interleave(repeats=x.shape[-3], dim=-3)
        salt = (
            torch.max(x) if self.salt is None else torch.tensor(self.salt).to(x.device)
        )
        pepper = (
            torch.min(x)
            if self.pepper is None
            else torch.tensor(self.pepper).to(x.device)
        )
        x[noise < self.p / 2] = pepper
        x[noise > 1 - self.p / 2] = salt
        return x


class GaussianNoise(object):
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(
            x.index_select(dim=-3, index=torch.tensor(0, device=x.device))
        )
        noise = noise.repeat_interleave(repeats=x.shape[-3], dim=-3)
        return x + self.mean + self.std * noise
