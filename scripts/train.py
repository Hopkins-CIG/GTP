import torch
from lightning.pytorch.cli import LightningCLI

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    cli = LightningCLI()