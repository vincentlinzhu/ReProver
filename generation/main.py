"""Script for training the tactic generator."""

import os
from loguru import logger
from pytorch_lightning.cli import LightningCLI

from generation.datamodule import GeneratorDataModule
from generation.model import RetrievalAugmentedGenerator
from common import CONFIG_LINK_ARGUMENTS

class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        for k, v in CONFIG_LINK_ARGUMENTS["generator"].items():
            parser.link_arguments(k, v)


def main() -> None:
    logger.info(f"PID: {os.getpid()}")
    cli = CLI(RetrievalAugmentedGenerator, GeneratorDataModule)
    print("Configuration: \n", cli.config)


if __name__ == "__main__":
    main()