# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

from dataclasses import dataclass
from typing import Any

from fairseq2.typing import override

from lcm.evaluation.predictors.lcm import LCMConfig, LCMPredictor
from lcm.inference.two_tower_diffusion_lcm import (
    DiffusionLCMGeneratorOptions,
    TwoTowerDiffusionLCMGenerator,
)


@dataclass(unsafe_hash=True)
class TwoTowerDiffusionLCMConfig(DiffusionLCMGeneratorOptions, LCMConfig):
    @classmethod
    def predictor_class(cls):
        return TwoTowerDiffusionLCMPredictor


class TwoTowerDiffusionLCMPredictor(LCMPredictor):
    """
    A predictor that wraps LCMGenerator and format the output for evaluation
    """

    config: TwoTowerDiffusionLCMConfig

    def __init__(
        self,
        config: TwoTowerDiffusionLCMConfig,
        **kwargs: Any,
    ):
        super().__init__(config, **kwargs)

    def build_generator(self, model):
        self.generator = TwoTowerDiffusionLCMGenerator(  # type: ignore
            model=model, options=self.config, eos_vec=self.eos_vec
        )

    @override
    @staticmethod
    def from_config(config: LCMConfig, **kwargs) -> "TwoTowerDiffusionLCMPredictor":
        return TwoTowerDiffusionLCMPredictor(config=config, **kwargs)  # type: ignore
