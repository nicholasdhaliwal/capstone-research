# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

import logging
from typing import Any, Dict

from fairseq2.models.config_loader import StandardModelConfigLoader
from fairseq2.models.loader import StandardModelLoader, load_model
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from lcm.models.base_lcm.builder import (
    BASE_LCM_MODEL_TYPE,
    BaseLCModelConfig,
    create_base_lcm_model,
    lcm_archs,
)
from lcm.utils.model_type_registry import ModelTypeConfig, lcm_model_type_registry

logger = logging.getLogger(__name__)


def convert_lcm_checkpoint(
    checkpoint: Dict[str, Any], config: BaseLCModelConfig
) -> Dict[str, Any]:
    # For DDP checkpoints
    # We need to first remove the prefix "module." from state dict keys.
    consume_prefix_in_state_dict_if_present(checkpoint["model"], "module.")
    return checkpoint


load_base_lcm_config = StandardModelConfigLoader(
    family=BASE_LCM_MODEL_TYPE,
    config_kls=BaseLCModelConfig,
    arch_configs=lcm_archs,
)

load_base_lcm_model = StandardModelLoader(
    config_loader=load_base_lcm_config,
    factory=create_base_lcm_model,
    checkpoint_converter=convert_lcm_checkpoint,
    restrict_checkpoints=False,
)

load_model.register(BASE_LCM_MODEL_TYPE, load_base_lcm_model)

lcm_model_type_registry.register(
    ModelTypeConfig(
        model_type=BASE_LCM_MODEL_TYPE,
        config_loader=load_base_lcm_config,
        model_factory=create_base_lcm_model,
        model_loader=load_base_lcm_model,
    )
)
