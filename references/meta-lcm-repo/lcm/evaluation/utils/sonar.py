# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#


from functools import lru_cache
from typing import Optional

from fairseq2.typing import Device
from sonar.inference_pipelines.text import (
    TextToEmbeddingModelPipeline,
    TextToTextModelPipeline,
)

from lcm.datasets.configs import SonarEncoderConfig

from .distributed import get_gang

# We fix the sonar encoder for the LCM prompt
eng_config = SonarEncoderConfig(
    tokenizer="text_sonar_basic_encoder",
    encoder="text_sonar_basic_encoder",
    lang="eng_Latn",
)


@lru_cache(maxsize=2)
def text_encoder(
    config: SonarEncoderConfig = eng_config, device: Optional[Device] = None
):
    """Load a text embedding pipleine with a sonar encoder"""
    if device is None:
        gang = get_gang()
        device = gang.device

    return TextToEmbeddingModelPipeline(
        encoder=config.encoder,
        tokenizer=config.tokenizer,
        device=device,
    )


@lru_cache
def text_translator():
    t2t_model = TextToTextModelPipeline(
        encoder="text_sonar_basic_encoder",
        decoder="text_sonar_basic_decoder",
        tokenizer="text_sonar_basic_encoder",
    )
    return t2t_model
