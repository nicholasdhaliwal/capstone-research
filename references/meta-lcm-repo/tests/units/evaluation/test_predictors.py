# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

import pytest

from lcm.evaluation.predictors import build_predictor
from lcm.evaluation.predictors.huggingface import (
    HuggingfacePredictor,
    HuggingfacePredictorConfig,
)

HF_PREDICTORS = [
    HuggingfacePredictorConfig(
        model_name="google/pegasus-x-base",
        model_class="PegasusXForConditionalGeneration",
        tokenizer_name="google/pegasus-x-large",
        tokenizer_class="PreTrainedTokenizerFast",
    )
]


@pytest.mark.parametrize("config", HF_PREDICTORS)
def test_hf_predictor(config):
    predictor = build_predictor(config)
    assert isinstance(predictor, HuggingfacePredictor)
