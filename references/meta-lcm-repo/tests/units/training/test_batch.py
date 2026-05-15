# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

import torch
from fairseq2.nn.padding import pad_seqs

from lcm.datasets.batch import EmbeddingsBatch

test_cases = [(0, 2), (2, 6), (0, 15), (3, 4), (1, 9)]


def test_unbatchon_embedding_batch():
    ragged_seq = [torch.randn((i**2 % 11, 3)) for i in range(100)]
    full_eb = EmbeddingsBatch(*pad_seqs(ragged_seq))
    full_eb_bis = EmbeddingsBatch(*pad_seqs(full_eb.unbatch()))

    assert (full_eb_bis.seqs == full_eb.seqs).all().item()
    assert (
        (full_eb_bis.padding_mask.seq_lens == full_eb.padding_mask.seq_lens)
        # type: ignore
        .all()
        .item()
    )
    assert (
        full_eb_bis.padding_mask._batch_seq_len == full_eb.padding_mask._batch_seq_len  # type: ignore
    )


def test_last_element_embedding_batch():
    ragged_seq = [torch.randn((i**2 % 11 + 1, 3)) for i in range(100)]
    full_eb = EmbeddingsBatch(*pad_seqs(ragged_seq))

    expected_ans = torch.stack([tt[-1] for tt in ragged_seq], dim=0)
    print(expected_ans.shape)
    found_ans = full_eb.get_last_element()
    print(found_ans.shape)

    assert (expected_ans == found_ans).all().item()
