# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#


from fairseq2.models.config_loader import StandardModelConfigLoader
from fairseq2.models.loader import StandardModelLoader, load_model

from lcm.models.base_lcm.loader import convert_lcm_checkpoint
from lcm.models.two_tower_diffusion_lcm.builder import (
    TWO_TOWER_DIFFUSION_LCM_MODEL_TYPE,
    TwoTowerDiffusionLCModelConfig,
    create_two_tower_diffusion_lcm_model,
    lcm_archs,
)
from lcm.utils.model_type_registry import ModelTypeConfig, lcm_model_type_registry

load_two_tower_diffusion_lcm_config = StandardModelConfigLoader(
    family=TWO_TOWER_DIFFUSION_LCM_MODEL_TYPE,
    config_kls=TwoTowerDiffusionLCModelConfig,
    arch_configs=lcm_archs,
)


load_two_tower_diffusion_lcm_model = StandardModelLoader(  # type: ignore # FIXME
    config_loader=load_two_tower_diffusion_lcm_config,
    factory=create_two_tower_diffusion_lcm_model,
    checkpoint_converter=convert_lcm_checkpoint,
    restrict_checkpoints=False,
)

load_model.register(
    TWO_TOWER_DIFFUSION_LCM_MODEL_TYPE, load_two_tower_diffusion_lcm_model
)

lcm_model_type_registry.register(
    ModelTypeConfig(
        model_type=TWO_TOWER_DIFFUSION_LCM_MODEL_TYPE,
        config_loader=load_two_tower_diffusion_lcm_config,
        model_factory=create_two_tower_diffusion_lcm_model,
        model_loader=load_two_tower_diffusion_lcm_model,
    )
)
