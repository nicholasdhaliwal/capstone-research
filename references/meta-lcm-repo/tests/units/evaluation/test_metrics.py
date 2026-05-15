# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#
# fmt: off

from typing import List

import pytest

from lcm.evaluation.metrics import (
    bleu,
    exact_match,
    exact_match_f1,
    rouge_score,
    sentence_bleu,
)
from lcm.evaluation.utils.common import evaluate


@pytest.mark.parametrize(
    "x,y,expected", [("yes", ["yes", "YES"], 1.0), ("no", ["yes", "YES"], 0.0)]
)
def test_exact_match(x, y, expected) -> None:
    assert exact_match(x, y) == expected


def test_exact_match_f1() -> None:
    precision, recall = exact_match_f1("yes", ["yes", "YES"])
    assert precision == 1.0
    assert recall == 1.0
    precision, recall = exact_match_f1("no", ["yes", "YES"])
    assert precision == 0.0
    assert recall == 0.0


@pytest.mark.skip(reason="flaky test")
def test_rouge_score() -> None:
    types: List[str] = ["rouge1", "rouge2", "rougeL"]

    reference = "The cat is on the mat"
    candidate = ["The cat is on the mat", "A dog is near the mat"]

    result = rouge_score(reference, candidate, types)
    assert list(result.values()) == [1.0, 1.0, 1.0]

    candidate = ["A dog is near the mat"]
    results = rouge_score(reference, candidate, types)

    assert results["rouge1"] == 0.5
    assert round(result["rouge2"], 1) == 1.0
    assert results["rougeL"] == 0.5


def test_bleu() -> None:
    x: str = "The quick brown fox jumps over the lazy dog."
    y: List[str] = [
        "A quick brown fox jumps over a lazy dog.",
        "The fast brown fox jumps over a sleeping dog.",
    ]
    expected_bleu = 52.53
    assert pytest.approx(bleu(x, y), abs=0.02) == expected_bleu

    expected_bleu = 54.1
    assert pytest.approx(sentence_bleu(x, y), abs=0.02) == expected_bleu


def test_evaluate() -> None:
    """Test that a python function can be wrapped into a MetricFn successfully"""

    examples = {
        "prediction": [
            "Billy Bob . They are on trial for tax fraud",
            "Billy Bob . They are on trial for tax fraud",
        ],
        "targets": [
            ["Billy Bob . Are they really on trial for tax"],
            ["Billy Bob . They are on trial for tax fraud"],
        ],
    }
    expected_outputs = {
        "exact_match": [0.0, 1.0],
        "f1": [0.7, 1.0],
    }

    metric_fn = evaluate(
        exact_match_f1,
        inputs=("prediction", "targets"),
        outputs=("exact_match", "f1"),
        collate=True,
    )

    assert metric_fn(examples) == expected_outputs

# fmt: on
