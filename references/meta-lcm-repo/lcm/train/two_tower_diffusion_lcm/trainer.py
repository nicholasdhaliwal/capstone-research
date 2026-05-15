# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
#

from dataclasses import dataclass, field
from typing import Union

from lcm.models.two_tower_diffusion_lcm.builder import TwoTowerDiffusionLCModelConfig
from lcm.models.two_tower_diffusion_lcm.loader import (
    load_two_tower_diffusion_lcm_model,
)
from lcm.train.lcm.trainer import LCMTrainer, LCMTrainerBuilder, LCMTrainingConfig
from lcm.train.two_tower_diffusion_lcm.criterion import (
    TowerDiffusionLCMCriterionConfig,
)


@dataclass
class TwoTowerDiffusionLCMTrainingConfig(LCMTrainingConfig):
    model_config_or_name: Union[TwoTowerDiffusionLCModelConfig, str, None] = None
    """The model configuration or name to train."""

    criterion: TowerDiffusionLCMCriterionConfig = field(  # type: ignore
        default_factory=lambda: TowerDiffusionLCMCriterionConfig()
    )


class DiffusionLCMTrainerBuilder(LCMTrainerBuilder):
    config: TwoTowerDiffusionLCMTrainingConfig

    def __init__(self, config: TwoTowerDiffusionLCMTrainingConfig):
        super().__init__(config)

    @property
    def model_loader(self):
        """A fairseq2 ModelLoader"""
        return load_two_tower_diffusion_lcm_model


def prepare_two_tower_diffusion_lcm_trainer(
    config: TwoTowerDiffusionLCMTrainingConfig,
) -> LCMTrainer:
    """Create an LCM Trainer.
    :param config: The training configuration.
    """
    return DiffusionLCMTrainerBuilder(config).build_trainer()
