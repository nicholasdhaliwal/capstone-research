# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#


from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch  # type: ignore
from stopes.utils.arrow_utils import nested_numpy_to_pyarrow

from lcm.datasets.configs import ColumnsNames, ParquetDatasetConfig


def simple_data(batch_size: int, split: str) -> pa.Table:
    # Creating a toy batch
    src_len, tgt_len = 17, 11
    sonar_dim, sonar_std = 1024, 0.006

    batch = {
        ColumnsNames.dataset_name.value: ["_train_dataset"] * batch_size,
        "split": [split] * batch_size,
    }
    table = pa.Table.from_pydict(batch)

    x = torch.randn(size=[batch_size, src_len, sonar_dim]) * sonar_std
    y = torch.randn(size=[batch_size, tgt_len, sonar_dim]) * sonar_std
    x_pa = nested_numpy_to_pyarrow([row.numpy() for row in x])
    y_pa = nested_numpy_to_pyarrow([row.numpy() for row in y])
    table = table.append_column("dummy_source_column", x_pa)
    table = table.append_column("dummy_target_column", y_pa)
    return table


@pytest.fixture()
def simple_train_dataset(tmp_path: Path):
    (tmp_path / "train").mkdir()
    pq.write_to_dataset(
        simple_data(10, "train"), tmp_path / "train", partition_cols=["split"]
    )

    yield ParquetDatasetConfig(
        parquet_path=str(tmp_path / "train"),
        source_column="dummy_source_column",
        target_column="dummy_target_column",
        filesystem_expr="pc.equal(pc.field('split'), 'train')",
    )


@pytest.fixture()
def simple_validation_dataset(tmp_path: Path):
    (tmp_path / "dev").mkdir()
    pq.write_to_dataset(
        simple_data(10, "dev"), tmp_path / "dev", partition_cols=["split"]
    )

    yield ParquetDatasetConfig(
        parquet_path=str(tmp_path / "dev"),
        source_column="dummy_source_column",
        target_column="dummy_target_column",
        filesystem_expr="pc.equal(pc.field('split'), 'dev')",
    )
