# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

from fairseq2.config_registry import ConfigRegistry
from fairseq2.logging import get_log_writer
from fairseq2.typing import DataType, Device
from torch.nn import Module

from lcm.models.sonar_normalizer import SonarNormalizer, load_sonar_normalizer_model

logger = get_log_writer(__name__)


"""
An abstract LCM model class for the bare minimum
"""

ABSTRACT_LCM_MODEL_TYPE = "abstract_lcm"


@dataclass
class AbstractLCModelConfig:
    model_type: str = ABSTRACT_LCM_MODEL_TYPE

    sonar_embed_dim: int = 1024

    sonar_normalizer_name: Optional[str] = None


lcm_archs = ConfigRegistry[AbstractLCModelConfig]()
lcm_arch = lcm_archs.decorator


class AbstractLCModel(Module):
    """Asbtract Class for LCM models"""

    def __init__(
        self,
        config: AbstractLCModelConfig,
    ) -> None:
        """
        Asbtract LCM model
        """
        super().__init__()

        self.config = config

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device


class AbstractLCModelBuilder:
    """Builds modules of an LCM"""

    config: AbstractLCModelConfig
    device: Optional[Device]
    dtype: Optional[DataType]

    def __init__(
        self,
        config: AbstractLCModelConfig,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param config:
            The configuration.
        :param device:
            The device on which to initialize modules.
        :param dtype:
            The data type of module parameters and buffers.
        """
        self.config = config

        self.device, self.dtype = device, dtype

    def build_sonar_normalizer(
        self,
    ) -> Optional[SonarNormalizer]:
        if self.config.sonar_normalizer_name is not None:
            logger.info(
                f"Building sonar_normalizer = {self.config.sonar_normalizer_name}"
            )
            return load_sonar_normalizer_model(
                self.config.sonar_normalizer_name,
                device=self.device,
                dtype=self.dtype,
            )
        return None

    @abstractmethod
    def build_model(self) -> AbstractLCModel:
        """Build a model."""
        ...
