from lightning.pytorch.callbacks import WeightAveraging
from torch.optim.swa_utils import get_ema_avg_fn

# Taken from the documentation available at https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html
class EMAWeightAveraging(WeightAveraging):
    def __init__(self, decay: float = 0.9999, start_step: int = 10_000):
        super().__init__(avg_fn=get_ema_avg_fn(decay))
        self.start_step = start_step

    def should_update(self, step_idx=None, epoch_idx=None):
        return (step_idx is not None) and (step_idx >= self.start_step)