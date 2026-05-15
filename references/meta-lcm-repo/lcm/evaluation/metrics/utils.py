# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

from itertools import islice
from typing import Iterator, Sequence, Tuple


def create_context_prediction_pairs(
    gt_docs: Sequence[Sequence[str]],
    pred_docs: Sequence[Sequence[str]],
    max_ctx_len: int,
) -> Iterator[Tuple[Sequence[str], str]]:
    """
    For each pair of documents (where a predicted document has 1:1 alignment with a suffix of a gt document),
    generate all pairs of a predicted sentence and up to max_ctx_len gt sentences immediately preceding it.
    This could be used to prepare data for evaluating the quality of teacher forced generation.
    """
    for gt_doc, pred_doc in zip(gt_docs, pred_docs):
        prefix_size = len(gt_doc) - len(pred_doc)
        for pred_idx, pred_sent in enumerate(pred_doc):
            full_ctx_len = pred_idx + prefix_size
            context_sents = gt_doc[max(0, full_ctx_len - max_ctx_len) : full_ctx_len]
            yield context_sents, pred_sent


def divide_chunks_as(iterable, reference_sequences):
    it = iter(iterable)
    chunks = [len(seq) for seq in reference_sequences]
    return [list(islice(it, c)) for c in chunks]
