# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import gc
import typing as tp
from builtins import enumerate
from dataclasses import dataclass, field

import numba
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import torch
from stopes.modules.partitioned_data_mapper import BatchMapper
from stopes.modules.preprocess.sonar_text_embedding import (
    SonarTextBatchEmbedder,
    SonarTextEmbedderConfig,
)
from stopes.utils.arrow_utils import (
    apply_on_nested_array,
)
from wtpsplit import SaT, indices_to_sentences

from lcm.datasets.sentence_splitting import remove_emojis, resplit


@numba.jit(nopython=True)
def insert_elements(arr, max_diff):
    """
    Insert elements into an array to ensure no two consecutive elements have a difference greater than max_diff.

    Parameters:
    arr (numpy array): The original array of integers.
    max_diff (int): The maximum allowed difference between consecutive elements after insertion.

    Returns:
    numpy array: The modified array with additional elements inserted to satisfy the max_diff condition.
    """

    result = []
    for i in range(len(arr) - 1):
        result.append(arr[i])
        diff = arr[i + 1] - arr[i]
        if diff > max_diff:
            num_insert = int(diff // max_diff)
            step_size = diff / (num_insert + 1)
            last_val = arr[i]
            for j in range(1, num_insert + 1):
                val = round(last_val + step_size)
                if val < arr[i + 1]:
                    result.append(val)
                    last_val = val
    result.append(arr[-1])
    return np.array(result, dtype=np.int32)


@numba.jit(nopython=True)
def merge_small_intervals(
    lenghts: np.ndarray, min_merging_length: int = 2, max_merge_length: int = 15
):
    """
    Merge small intervals in a list of lengths.
    This function takes a list of lengths and merges any intervals that are smaller than or equal to `min_merging_length`
    into larger intervals. The merged intervals are limited to a maximum length of `max_merge_length`.
    Parameters:
    lengths (np.ndarray): A list of lengths to be merged.
    min_merging_length (int): The minimum length of an interval to be merged. Defaults to 2.
    max_merge_length (int): The maximum length of a merged interval. Defaults to 15.
    Returns:
    list: A list of merged lengths.

    Examples:
    >>> merge_small_intervals(np.array([1, 2, 3, 4, 5]))
    array([3, 3, 4, 5], dtype=int32)
    >>> merge_small_intervals(np.array([1, 1, 1, 1, 1]))
    array([5], dtype=int32)
    >>> merge_small_intervals(np.array([1, 2, 3, 2, 2, 2, 4, 1, 1, 5]))
    array([3, 3, 6, 4, 2, 5], dtype=int32)
    """
    merge_arr = []
    merge_len = 0

    for curr_len in lenghts:
        if curr_len <= min_merging_length and merge_len + curr_len <= max_merge_length:
            merge_len += curr_len
        else:
            if merge_len > 0:
                merge_arr.append(merge_len)
                merge_len = 0
            merge_arr.append(curr_len)
    if merge_len > 0:
        merge_arr.append(merge_len)

    return np.array(merge_arr, dtype=np.int32)


@numba.jit(nopython=True)
def find_closest_indices(arr1, arr2):
    """
    Find indices of the closest elements in arr2 for each element in arr1.

    Parameters:
    arr1 (numpy array): The array containing the elements for which we want to find the closest elements in arr2.
    arr2 (numpy array): The array in which we want to find the closest elements.

    Returns:
    indices (numpy array): The indices of the closest elements in arr2 for each element in arr1.
    """
    # Use searchsorted to find the indices where elements from arr1 should be inserted in arr2
    indices = np.searchsorted(arr2, arr1, side="left")

    indices_bis = np.clip(indices - 1, a_min=0, a_max=len(arr2) - 1)
    dist_one = np.abs(arr2[indices] - arr1)
    dist_bis = np.abs(arr2[indices_bis] - arr1)

    return np.where(dist_one < dist_bis, indices, indices_bis)


@dataclass
class SentenceSplitterConfig:
    columns: tp.List[str]
    model_name: str = "sat-6l"
    sentence_suffix: str = "_sentences"
    sentence_threshold: float = 0.01
    max_sentence_len: int = 256
    min_text_length: int = 10
    min_unique_chars: int = 0
    fallback_separators: tp.List[str] = field(
        default_factory=lambda: [
            "...",
            "\n",
            "!",
            "?",
            ";",
            ":",
            ".",
            ",",
            "\t",
            " ",
        ]
    )
    device: str = "cuda"
    remove_whitespace_before_inference: bool = False
    batch_size: int = 256
    block_size: int = 256
    stride: int = 256
    outer_batch_size: int = 1024
    verbose: bool = False
    pad_last_batch: bool = False


class SentenceSplitter(BatchMapper):
    def __init__(self, config: SentenceSplitterConfig):
        super().__init__(config)
        self.columns = config.columns
        device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        try:
            self.model = SaT(
                self.config.model_name,
                from_pretrained_kwargs={"local_files_only": True},
            )
        except Exception:
            self.model = SaT(self.config.model_name)

        if "cuda" in config.device:
            self.model.half()

        self.model.eval().to(device)

    @torch.inference_mode()
    def _resplit_long_sentences(self, col: pa.Array) -> pa.Array:
        mask = pc.greater_equal(pc.utf8_length(col), self.config.max_sentence_len)
        texts_to_resplit = col.filter(mask).to_pandas().to_list()

        resplit_sentences = []
        for text, probs in zip(
            texts_to_resplit,
            self.model.predict_proba(
                texts_to_resplit,
                stride=self.config.stride,
                block_size=self.config.block_size,
                batch_size=self.config.batch_size,
                pad_last_batch=self.config.pad_last_batch,
                remove_whitespace_before_inference=self.config.remove_whitespace_before_inference,
                outer_batch_size=self.config.outer_batch_size,
                verbose=self.config.verbose,
            ),
        ):
            nb_split = round(len(probs) / self.config.max_sentence_len) + 1
            sentence_threshold = np.partition(probs, -nb_split)[-nb_split]
            sentences = indices_to_sentences(
                text,
                np.where(probs >= sentence_threshold)[0],
                strip_whitespace=False,
            )
            resplit_sentences.append(sentences)

        # if not, hard resplit with some separators
        def _resplit(raw_sentences):
            for separator in self.config.fallback_separators:
                raw_sentences = [
                    subchunk.strip()
                    for sent in raw_sentences
                    for subchunk in resplit(
                        sent, max_length=self.config.max_sentence_len, sep=separator
                    )
                ]
            return raw_sentences

        np_mask = mask.to_pandas().to_numpy()
        full_text = col.to_pandas().to_list()

        output_sentences = []
        j = 0
        for i, text in enumerate(full_text):
            if np_mask[i]:
                output_sentences.append(_resplit(resplit_sentences[j]))
                j += 1
            else:
                output_sentences.append([text])

        return pa.array(output_sentences, type=pa.list_(pa.string()))

    def resplit_long_sentences(self, col: pa.Array) -> pa.Array:
        list_col = apply_on_nested_array(self._resplit_long_sentences, col)
        reflatten_col = pl.from_arrow(list_col).list.eval(pl.element().explode())  # type: ignore
        # remove single char repeated
        if self.config.min_unique_chars > 0:
            reflatten_col = reflatten_col.list.eval(
                pl.when(
                    pl.element().str.split("").list.n_unique()
                    > self.config.min_unique_chars
                )
                .then(pl.element())
                .drop_nulls()
            )
        return reflatten_col.to_arrow().cast(pa.list_(pa.string()))

    @torch.inference_mode()
    def basic_split_on_single_column(
        self,
        col: tp.Union[pa.Array, pa.ChunkedArray],
    ) -> tp.Union[pa.Array, pa.ChunkedArray]:
        if not (pa.types.is_large_string(col.type) or pa.types.is_string(col.type)):
            raise ValueError("Column must be of type string")

        texts = col.to_pandas().to_list()
        texts = list(map(remove_emojis, texts))

        long_texts = [t for t in texts if len(t) > self.config.min_text_length]
        keep_texts = [
            (idx, t)
            for idx, t in enumerate(texts)
            if len(t) <= self.config.min_text_length
        ]

        outputs = self.model.split(
            long_texts,
            threshold=self.config.sentence_threshold,
            stride=self.config.stride,
            block_size=self.config.block_size,
            batch_size=self.config.batch_size,
            pad_last_batch=self.config.pad_last_batch,
            remove_whitespace_before_inference=self.config.remove_whitespace_before_inference,
            outer_batch_size=self.config.outer_batch_size,
            verbose=self.config.verbose,
        )
        sentences = []
        for row in outputs:
            sentences.append([s.strip() for s in row if s.strip()])

        for idx, text in keep_texts:
            sentences.insert(idx, text)

        return pa.array(sentences, type=pa.list_(pa.string()))

    def __call__(self, table: pa.Table) -> pa.Table:
        for column in self.columns:
            sentence_array = self.basic_split_on_single_column(table[column])

            sentence_array = self.resplit_long_sentences(sentence_array)

            table = table.append_column(
                f"{column}{self.config.sentence_suffix}", sentence_array
            )

        return table


@dataclass
class FullPipelineConfig:
    splitter_config: SentenceSplitterConfig
    sonar_encoder_config: SonarTextEmbedderConfig
    min_text_length: int = 10


class FullPipeline(BatchMapper):
    """
    Creating sonar vectors from scratch.
    Making sentences splits.
    Computing sonar embeddings.

    Config example requires only one input column:
    - `text`

    Note also that text should not be empty!

    Example of config:

        splitter_config = SentenceSplitterConfig(
            columns=["text"],
            model_name="sat-3l",
            verbose=True,
            sentence_threshold=0.02,
            max_sentence_len=256,
        )
        sonar_encoder_config = SonarTextEmbedderConfig(
            column_config=[LangColumnConfig("text_sentences", lang_value="eng_Latn")],
            device="cuda",
        )

        full_config = FullPipelineConfig(
            splitter_config=splitter_config,
            sonar_encoder_config=sonar_encoder_config,
        )

    """

    def __init__(self, config: FullPipelineConfig):
        self.config = config
        self.splitter = SentenceSplitter(self.config.splitter_config)
        self.sonar_encoder = SonarTextBatchEmbedder(self.config.sonar_encoder_config)

    def __call__(self, batch: pa.Table) -> pa.Table:
        for col in self.config.splitter_config.columns:
            batch = batch.filter(
                pc.greater_equal(
                    pc.utf8_length(batch[col]), self.config.min_text_length
                )
            )

        batch = self.splitter(batch)
        batch = self.sonar_encoder(batch)
        gc.collect()
        return batch
