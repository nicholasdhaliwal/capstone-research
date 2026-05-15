# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#


from fairseq2.models.config_loader import StandardModelConfigLoader
from fairseq2.models.loader import StandardModelLoader, load_model

from lcm.models.sonar_normalizer.builder import (
    SonarNormalizerConfig,
    create_sonar_normalizer,
    sonar_normalizer_archs,
)

load_sonar_normalizer_config = StandardModelConfigLoader(
    family="sonar_normalizer",
    config_kls=SonarNormalizerConfig,
    arch_configs=sonar_normalizer_archs,
)

load_sonar_normalizer_model = StandardModelLoader(
    config_loader=load_sonar_normalizer_config,
    factory=create_sonar_normalizer,
    restrict_checkpoints=False,
)

load_model.register("sonar_normalizer", load_sonar_normalizer_model)
