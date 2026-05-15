# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#


import os

import pytest
from fairseq2.gang import FakeGang

from lcm.evaluation.api import Scorer, ScorerConfig
from lcm.evaluation.metrics import get_scorer
from lcm.evaluation.utils.data_utils import load_jsonl
from tests.common import device

REFERENCE_FREE_METRICS = [
    "sentence_fluency",
    # "sentence_perplexity",   # skip due to recent error accessing the pbulic model in HF hub
    # "round_trip_translation",  # skip due to large model, run in a separate node
    "word_repetition",
    "token_repetition",
]

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS", "") == "true"


@pytest.mark.parametrize("scorer_type", REFERENCE_FREE_METRICS)
def test_reference_free_scorer(simple_json_dataset, scorer_type):
    examples = load_jsonl(simple_json_dataset)[:3]
    config = ScorerConfig(
        scorer_type=scorer_type,
        inputs="input_text",  # type: ignore
        params={"batch_size": 3},
    )
    metric_fn = get_scorer(config, gang=FakeGang(device=device))
    assert isinstance(metric_fn, Scorer)
    result = metric_fn(examples)
    for metric_name in metric_fn.outputs:
        assert metric_name in result and len(result[metric_name]) == 3


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS, reason="Skip tests that download big models in CI"
)
def test_round_trip_translation(simple_json_dataset):
    examples = load_jsonl(simple_json_dataset)[:3]
    config = ScorerConfig(
        scorer_type="round_trip_translation",
        inputs="input_text",  # type: ignore
        params={"batch_size": 3},
    )
    metric_fn = get_scorer(config, gang=FakeGang(device=device))
    assert isinstance(metric_fn, Scorer)
    result = metric_fn(examples)
    for metric_name in metric_fn.outputs:
        assert metric_name in result and len(result[metric_name]) == 3


@pytest.mark.skip(reason="long runtime")
@pytest.mark.parametrize("question_id", range(1, 7))
def test_seahorse(simple_json_dataset, question_id):
    examples = load_jsonl(simple_json_dataset)[:3]
    config = ScorerConfig(
        scorer_type="seahorse",
        model_name=f"google/seahorse-large-q{question_id}",
        inputs=("input_text", "target_text"),  # type: ignore
        params={"batch_size": 3},
    )
    metric_fn = get_scorer(config, gang=FakeGang(device=device))
    assert isinstance(metric_fn, Scorer)
    result = metric_fn(examples)

    assert (
        f"seahorse-q{question_id}" in result
        and len(result[f"seahorse-q{question_id}"]) == 3
    )
