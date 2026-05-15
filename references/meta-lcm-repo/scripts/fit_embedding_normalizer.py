# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#


import argparse
from typing import List

import numpy as np
import pyarrow.compute as pc
import torch
from stopes.utils.arrow_utils import pyarrow_fixed_size_array_to_numpy
from tqdm.auto import tqdm

from lcm.datasets.configs import (
    DataLoadingConfig,
    ParquetBatchFormat,
    get_parquet_config_from_name,
)
from lcm.datasets.dataloading import (
    build_weighted_pipeline_with_renaming,
)
from lcm.models.sonar_normalizer import SonarNormalizer, SonarNormalizerConfig


def sample_sentences_from_mixed_sources(
    name_with_weights: List[str],
    max_nb_samples: int = 10**6,
    down_sample: int = 5,
    column: str = "_source_column",
) -> np.ndarray:
    ds_list = list(map(get_parquet_config_from_name, name_with_weights))

    dlc = DataLoadingConfig(
        max_tokens=10000,
        min_length_of_sequences=1,
        order_by_length=False,
        nb_prefetch=2,
        num_parallel_calls=2,
        nb_epochs=1,
        output_format=ParquetBatchFormat.pyarrow,
    )

    basic_iterator = build_weighted_pipeline_with_renaming(ds_list, dlc, 0, 1)

    nb_sentences = 0
    sentences_batch = []

    pbar = tqdm(total=None)
    for batch in tqdm(basic_iterator):
        vecs = pyarrow_fixed_size_array_to_numpy(pc.list_flatten(batch[column]))[
            ::down_sample
        ].astype(np.float32)
        sentences_batch.append(vecs)
        nb_sentences += len(vecs)
        pbar.update(len(vecs))
        if nb_sentences > max_nb_samples:
            break

    return np.vstack(sentences_batch)


def main(
    ds_mixture: List[str],
    save_path: str,
    max_nb_samples: int = 10**6,
):
    """
    Args example:

    ds_mixture = [
        "dataset1:5",
        "dataset2:10",
        "dataset3=train:2",
    ]
    save_path = f"/path/to/new/normalizer.pt"
    """
    embs = sample_sentences_from_mixed_sources(
        ds_mixture, max_nb_samples=max_nb_samples
    )
    normalizer = SonarNormalizer(SonarNormalizerConfig())
    normalizer.fit(torch.from_numpy(embs))

    torch.save(
        {
            "model": normalizer.state_dict(),
            "dataset_mixture": ds_mixture,
        },
        save_path,
    )
    print(f"Normalizer saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", nargs="+", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--max_nb_samples", type=int, default=10**6)
    args = parser.parse_args()
    main(args.ds, args.save_path, args.max_nb_samples)
