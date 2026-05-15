# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#


import pytest
from fairseq2.gang import FakeGang

from lcm.datasets.configs import SonarEncoderConfig
from lcm.evaluation.metrics.round_trip import round_trip_l2_distance, text_encoder


def test_round_trip_l2(monkeypatch):
    monkeypatch.setattr("lcm.evaluation.utils.sonar.get_gang", lambda: FakeGang())

    x = ["The quick brown fox jumps over the lazy dog."] * 2
    y = [
        "A quick brown fox jumps over a lazy dog.",
        "The fast brown fox jumps over a sleeping dog.",
    ]

    encoder_config = SonarEncoderConfig()
    encoder = text_encoder(encoder_config)
    encoded_y = encoder.predict(y, "eng_Latn", batch_size=2)

    scores = round_trip_l2_distance(x, encoded_y, encoder_config=encoder_config)

    assert pytest.approx(scores, abs=0.02) == [0.05, 0.11]
