# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

import numpy as np
import pandas as pd
import pyarrow as pa

from lcm.datasets.parquet_utils import (
    compute_length_splits,
    filter_document_by_quality,
    filter_long_short_sentence_document,
    hierarchical_explode_table_with_max_length,
    pyarrow_table_to_torch_dict,
    renaming,
)


def test_nested_text_conversion():
    nested_input = pa.array([["abc", "efg"], ["xyz"]])
    tt = pa.Table.from_pydict({"nested_text": nested_input})
    converted = pyarrow_table_to_torch_dict(tt)
    # we want to keep this type unchanged
    assert isinstance(converted["nested_text"], pa.Array)


def test_filter_long_short_sentence_document():
    # Create a sample input table with a text column containing sentences
    df = {
        "text": [
            ["a" * 5, "b" * 10],
            ["a" * 2, "b" * 3],
            None,
            [],
            ["c" * 3],
            ["d" * 1, "e" * 10],
            ["f" * 2, "i" * 4],
        ]
    }
    table = pa.Table.from_pydict(df)

    # Call the filter_long_short_sentence_document function with the sample input table and max sentence length of 10
    result = filter_long_short_sentence_document(
        table, column="text", max_sentence_len=5, min_sentence_len=2
    )
    assert result.to_pydict() == {"text": [["aa", "bbb"], ["ccc"], ["ff", "iiii"]]}


def test_renaming():
    data = {
        "column1": [1, 2, 3, 4, 5],
        "column2": ["a", "b", "c", "d", "e"],
        "column3": [1.1, 2.2, 3.3, 4.4, 5.5],
    }

    table = data
    new_table = renaming(table, {"column1": "col1", "cc": "rr"}, "name")
    assert isinstance(new_table, dict)
    assert list(new_table.keys()) == ["col1", "column2", "column3", "_dataset_name"]

    table = pa.Table.from_pydict(data)
    new_table = renaming(table, {"column1": "col1", "cc": "rr"}, "name")
    assert isinstance(new_table, pa.Table)
    assert new_table.column_names == ["col1", "column2", "column3", "_dataset_name"]

    table = pa.Table.from_pydict(data).to_pandas()
    new_table = renaming(table, {"column1": "col1", "cc": "rr"}, "name")
    assert isinstance(new_table, pd.DataFrame)
    assert new_table.columns.tolist() == ["col1", "column2", "column3", "_dataset_name"]


def test_compute_length_splits_random_case():
    length_col = np.random.randint(0, 10, 1000, dtype=np.int32)
    max_tokens = 37
    splits = compute_length_splits(
        length_col, max_tokens=max_tokens, order_by_length=False, drop_long_sample=True
    )
    np.testing.assert_allclose(np.concatenate(splits), np.arange(len(length_col)))
    for split in splits:
        batch = length_col[split]
        assert len(batch) * batch.max() <= max_tokens
    nb_simple_splits = len(splits)

    splits = compute_length_splits(
        length_col, max_tokens=max_tokens, order_by_length=True, drop_long_sample=True
    )
    np.testing.assert_allclose(
        np.sort(np.concatenate(splits)), np.arange(len(length_col))
    )
    for split in splits:
        batch = length_col[split]
        assert len(batch) * batch.max() <= max_tokens
    assert nb_simple_splits > int(1.5 * len(splits))


def test_compute_length_splits_edge_case():
    length_col = np.array([2, 1, 3, 90, 2, 3, 1, 1], dtype=np.int32)
    splits = compute_length_splits(
        length_col, max_tokens=10, order_by_length=False, drop_long_sample=False
    )
    values = [length_col[ind].tolist() for ind in splits]
    expected_values = [[2, 1, 3], [2, 3, 1], [1], [90]]
    assert values == expected_values

    splits = compute_length_splits(
        length_col, max_tokens=8, order_by_length=False, drop_long_sample=False
    )
    values = [length_col[ind].tolist() for ind in splits]
    expected_values = [[2, 1], [3, 2], [3, 1], [1], [90]]
    assert values == expected_values

    splits = compute_length_splits(
        length_col, max_tokens=8, order_by_length=True, drop_long_sample=False
    )
    values = [length_col[ind].tolist() for ind in splits]
    expected_values = [[1, 1, 1, 2], [2, 3], [3], [90]]
    assert values == expected_values

    splits = compute_length_splits(
        length_col, max_tokens=10, order_by_length=True, drop_long_sample=False
    )
    values = [length_col[ind].tolist() for ind in splits]
    expected_values = [[1, 1, 1, 2, 2], [3, 3], [90]]
    assert values == expected_values


def test_filter_document_by_quality():
    # Create a sample input table with a text column containing sentences
    df = {
        "score": [
            [1, 2, -3, 5],
            [2, 3, 4],
            None,
            [],
            [3, 2, 1],
            [10, 10, 10],
            [-2, 1, 2],
        ]
    }
    table = pa.Table.from_pydict(df)

    result = filter_document_by_quality(
        table, column="score", min_score=0, max_score=4.9
    )
    assert result.to_pydict() == {"score": [[2, 3, 4], [3, 2, 1]]}

    result = filter_document_by_quality(
        table, column="score", max_score=4.8, min_score=None
    )
    assert result.to_pydict() == {"score": [[2, 3, 4], [3, 2, 1], [-2, 1, 2]]}


def test_hierarchical_explode_table_with_max_length():
    batch = pa.Table.from_pydict(
        {
            "index": [0, 1, 2],
            "text_sentences": [
                [[x] for x in "abcdefghijk"],
                [[x] for x in "qrstuvwxyz"],
                [[x] for x in "ABCDEFGHIJKL"],
            ],
            "text_page_lens": [
                [3, 1, 4, 1, 2],
                [2, 2, 1, 1, 1, 1, 2],
                [3, 3, 2, 2, 1, 1],
            ],
            "text_page_texts": [
                [-1, -2, -3, -4, -5],
                [-10, -20, -30, -40, -50, -60, -70],
                [-100, -200, -300, -400, -500, -600],
            ],
        }
    )
    wrap_batch = hierarchical_explode_table_with_max_length(
        batch,
        columns=["text_sentences"],
        page_len_column="text_page_lens",
        page_embs_columns=["text_page_texts"],
        max_seq_len=5,
    )

    expected_output = {
        "index": [0, 0, 0, 1, 1, 2, 2, 2],
        "text_sentences": [
            [["a"], ["b"], ["c"], ["d"]],
            [["e"], ["f"], ["g"], ["h"], ["i"]],
            [["j"], ["k"]],
            [["q"], ["r"], ["s"], ["t"], ["u"]],
            [["v"], ["w"], ["x"], ["y"], ["z"]],
            [["A"], ["B"], ["C"]],
            [["D"], ["E"], ["F"], ["G"], ["H"]],
            [["I"], ["J"], ["K"], ["L"]],
        ],
        "text_page_lens": [
            [3, 1],
            [4, 1],
            [2],
            [2, 2, 1],
            [1, 1, 1, 2],
            [3],
            [3, 2],
            [2, 1, 1],
        ],
        "text_page_texts": [
            [-1, -2],
            [-3, -4],
            [-5],
            [-10, -20, -30],
            [-40, -50, -60, -70],
            [-100],
            [-200, -300],
            [-400, -500, -600],
        ],
    }
    assert wrap_batch.to_pydict() == expected_output
