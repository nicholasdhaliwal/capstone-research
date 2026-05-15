# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fairseq2.gang import FakeGang

from lcm.datasets.configs import (
    DataLoadingConfig,
    ParquetBatchFormat,
    ParquetDatasetConfig,
)

logger = logging.getLogger("lcm.test.units")


def mock_init_process_group(dconfig: Any, logger: logging.Logger):
    from tests.common import device

    return FakeGang(device=device)


def mock_get_gang():
    from tests.common import device

    return FakeGang(device=device)


def simple_table() -> pa.Table:
    d = {
        "0": "zero",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",  # fmt: skip
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
    }

    def num2text(x):
        return " ".join([d[i] for i in list(str(x))])

    data = {
        "cat": np.arange(1000) // 100,
        "id": np.arange(1000),
        "seq": [np.arange(i % 10) for i in range(1000)],
        "text": [f"random text {num2text(i)}" for i in range(1000)],
    }
    return pa.Table.from_pydict(data)


@pytest.fixture(autouse=True)
def patches(monkeypatch):
    # Change behaviour of the training pipelines for testing.
    # Please put all the patch in this fixture

    # LCM patch
    monkeypatch.setattr(
        "lcm.utils.distributed.init_process_group", mock_init_process_group
    )
    monkeypatch.setattr("lcm.train.trainer.init_process_group", mock_init_process_group)
    monkeypatch.setattr(
        "lcm.train.trainer.TrainerBuilder._setup_additional_logging", lambda _: None
    )
    monkeypatch.setattr("lcm.utils.logging.log_env_variables", lambda _: None)
    monkeypatch.setattr("lcm.train.trainer.log_env_variables", lambda _: None)
    monkeypatch.setattr("lcm.datasets.base.set_mkl_num_threads", lambda: None)
    monkeypatch.setattr("lcm.datasets.dataloader.set_mkl_num_threads", lambda: None)
    monkeypatch.setattr("lcm.evaluation.utils.common.setup_env", lambda: None)
    monkeypatch.setattr("lcm.evaluation.run.setup_env", lambda: None)

    monkeypatch.setattr("lcm.evaluation.cli.local.get_gang", mock_get_gang)


@pytest.fixture()
def simple_dataset(tmp_path: Path):
    pq.write_to_dataset(simple_table(), tmp_path, partition_cols=["cat"])
    yield tmp_path


@pytest.fixture()
def simple_data_config(simple_dataset):
    dlc = DataLoadingConfig(
        batch_size=10,
        seed=12,
        nb_epochs=1,
        shuffle=False,
        output_format=ParquetBatchFormat.pyarrow,
        min_length_of_sequences=None,
    )
    pdc = ParquetDatasetConfig(
        parquet_path=simple_dataset,
        columns=["id", "cat"],
        source_column="seq",
        source_text_column="text",
        nb_parallel_fragments=1,
        split_to_row_groups=False,
    )
    yield dlc, pdc
