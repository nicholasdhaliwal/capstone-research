# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
#


import torch
from fairseq2.models.sequence import SequenceBatch


def move_eos_to_the_end(
    batch: SequenceBatch, pad_token_id: int = 0, eos_token_id: int = 3
) -> SequenceBatch:
    """
    Convert a decoder-input batch (with the eos token in the beginning) to a decoder-output batch
    (with eos in the end) of the same shape.
    Note that this processing is missing two potentially critical issues:
    1) If the sequence end has been truncated away, EOS token will be appended erroneously.
    2) The language code token is still included in the loss computation (we may want to avoid it).
    """
    # strip the EOS token prepended to the input and add an empty token in the end
    seqs = torch.cat(
        [
            batch.seqs[:, 1:],
            torch.zeros_like(batch.seqs[:, :1]) + pad_token_id,
        ],
        dim=-1,
    )
    # fill the last real token in the batch with the eos value
    if batch.padding_mask:
        seqs[
            torch.arange(seqs.shape[0], dtype=torch.int32),
            batch.padding_mask.seq_lens - 1,
        ] = eos_token_id
    else:
        seqs[:, -1] = eos_token_id

    result = SequenceBatch(
        seqs=seqs,
        padding_mask=batch.padding_mask,
    )
    return result
