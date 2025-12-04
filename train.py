import sys
import lightning
import torch
from lightning.pytorch.cli import LightningCLI,LightningArgumentParser

from crossgoose.data import FlowDataModule
from crossgoose.models import CrossGooseModel
from crossgoose.utils import get_timestamp, random_adjective


def cli_main():
    torch.set_float32_matmul_precision('medium')
    random_name = f"{get_timestamp()}_{random_adjective()}-goose"
    cli = LightningCLI(
        model_class=CrossGooseModel,
        datamodule_class=FlowDataModule,
        args=sys.argv[1:] + [f'--trainer.logger.version={random_name}']
    )


if __name__ == "__main__":
    cli_main()
