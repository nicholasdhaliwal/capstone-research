# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
#


import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import hydra
import submitit
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict, read_write
from stopes.core import Requirements, StopesModule

from lcm.train.common import get_trainer
from lcm.utils.common import setup_conf

setup_conf()


class TrainModule(StopesModule):
    def requirements(self) -> Requirements:
        return self.config.requirements

    def run(self, iteration_value: Optional[Any] = None, iteration_index: int = 0):
        # Add module.name to the config's log_folder
        with read_write(self.config):
            self.config.log_folder = Path(self.config.log_folder) / self.name()

        trainer = get_trainer(self.config)

        # trainer should have a run() method
        trainer.run()

    def should_retry(
        self,
        ex: Exception,
        attempt: int,
        iteration_value: Optional[Any] = None,
        iteration_index: int = 0,
    ) -> bool:
        # Before retrying the failed train run, clean the environment to make sure
        # fs2 ProcessGroupGang can set up properly without raising error if the
        # gang is not set up reliably
        with submitit.helpers.clean_env():
            return "ValueError" not in str(ex)

    def name(self):
        """
        implement this if you want to give a fancy name to your job
        """
        name = self.config.get(
            "experiment_name", f"{self.__class__.__name__}_{self.sha_key()[:10]}"
        )
        return name


@dataclass
class TrainingConfig:
    trainer: DictConfig
    launcher: DictConfig
    dry_run: bool = False


async def run(config: TrainingConfig):
    # dump the all config to the outputs config log
    dump_dir = Path(config.launcher.config_dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.resolve(config)  # type: ignore
    # XXX: do we want to promote datasets configs from thier names to the final params
    OmegaConf.save(
        config=config,
        f=str(dump_dir / "all_config.yaml"),
    )

    train_config = config.trainer

    # If launcher.cluster = debug set debug in the trainer to True
    with open_dict(train_config):
        if config.launcher.cluster == "debug":
            train_config.debug = True
        train_config.log_folder = config.launcher.log_folder

    if getattr(config, "dry_run", False):
        trainer = get_trainer(train_config)
        print(f"Trainer: {trainer}")
        print(f"Train config: {getattr(trainer, 'config')}")

        return

    launcher = hydra.utils.instantiate(config.launcher)

    train_module = TrainModule(train_config)
    wait_on = launcher.schedule(train_module)

    await wait_on


@hydra.main(
    version_base="1.2",
    config_path="../../recipes/train",
    config_name="defaults.yaml",
)
def main(config: TrainingConfig) -> None:
    """
    Launch a train module from CLI.

    Example:

    ```sh
    python -m lcm.train +pretrain=mse
    ```

    in this example, `pretrain` is a folder under the `recipes` directory and `mse`
    is a yaml file with the trainer configuration.
    This yaml file must be in the `trainer` package (i.e. start with the `# @package trainer`
    hydra directive).
    It must contain a `__trainer__` entry defining the constructor for the trainer.

    You can use `-c job` to see the configuration without running anything. You can use
    `dry_run=true` to initialize the trainer from the configuration and make sure it's correct
    without running the actual training. To debug the jobs, you can use `launcher.cluster=debug`
    """
    asyncio.run(run(config))


if __name__ == "__main__":
    main()
