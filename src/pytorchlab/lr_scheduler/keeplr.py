from torch.optim.lr_scheduler import LRScheduler


class KeepLR(LRScheduler):
    def __init__(self, optimizer, last_epoch=-1, verbose=False):
        """
        Not change learning rate

        Args:
            optimizer (_type_): optimizer to keep
            last_epoch (int, optional): last epoch. Defaults to -1.
            verbose (bool, optional): verbose. Defaults to False.
        """
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]
