# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

# We import all the model types in order to populate the model type registry
from lcm.models.base_lcm.loader import BASE_LCM_MODEL_TYPE
from lcm.models.two_tower_diffusion_lcm.loader import (
    TWO_TOWER_DIFFUSION_LCM_MODEL_TYPE,
)

__all__ = [
    "BASE_LCM_MODEL_TYPE",
    "TWO_TOWER_DIFFUSION_LCM_MODEL_TYPE",
]
