
from torch.optim.lr_scheduler import (LRScheduler,
                                      _warn_get_lr_called_within_step)
from torch.optim.optimizer import Optimizer


class ExponentialLRWithMin(LRScheduler):

    _is_initial: bool = False

    def __init__(
        self,
        optimizer: Optimizer,
        gamma: float,
        min_lr: float,
        last_epoch: int = -1,
    ) -> None:  # noqa: D107
        self.gamma = gamma
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Compute the learning rate of each parameter group."""
        _warn_get_lr_called_within_step(self)

        # when loading from a checkpoint, we don't want _initial_step (called from the constructor)
        # to update the lr one more step ahead of itself.
        if self._is_initial:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [max(group["lr"] * self.gamma, self.min_lr) for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [max(base_lr * self.gamma**self.last_epoch, self.min_lr) for base_lr in self.base_lrs]
