# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#
# flake8: noqa

from pathlib import Path
from typing import Any, Dict, List

import pytest

from lcm.evaluation.utils.common import write_to_jsonl


def get_jsonl() -> List[Dict[str, Any]]:
    return [
        {
            "input_text": f"This is a long sentence that sometimes have extraordinally complex words that is unimaginatively ambiguous such as Sesquipedalianism. We want to test this with a shorter sentence that follows text {i}",
            "target_text": f"random target {i}",
        }
        for i in range(100)
    ]


@pytest.fixture
def simple_json_dataset(tmp_path: Path):
    file_path = tmp_path / "dataset.jsonl"
    write_to_jsonl(get_jsonl(), str(file_path))
    yield file_path
