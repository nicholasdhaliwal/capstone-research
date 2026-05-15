# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

import math
from typing import Optional, final

import torch
from fairseq2.nn.transformer import (
    AbstractAttentionMask,
    AttentionMask,
    AttentionMaskFactory,
)
from fairseq2.typing import DataType, Device, override
from torch import Tensor

from lcm.nn.incremental_state import LCMIncrementalStateBag


def _get_shifted_causal_mask(
    seq_len: int,
    key_len: int,
    shift: int = 0,
    cf_guidance_prob: float = 0.0,
    zero_vector: bool = False,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> Tensor:
    causal_mask = torch.ones(
        (seq_len, key_len),
        device=device,
        dtype=dtype,
    )
    causal_mask.tril_(diagonal=shift)

    if cf_guidance_prob > 0.0:
        num_rows_to_drop = math.floor((seq_len - 1) * cf_guidance_prob)
        if num_rows_to_drop > 0:
            rows_to_drop = 1 + torch.randperm(seq_len - 1)[:num_rows_to_drop]
            if zero_vector:
                causal_mask[rows_to_drop, 1:] = 0
            else:
                causal_mask[rows_to_drop, :] = 0

    return causal_mask


class NoAttentionMaskFactory(AttentionMaskFactory):
    """Constructs instances of :class:`NoAttentionMask`."""

    @override
    def __call__(  # type: ignore
        self,
        seqs: Tensor,
        keys: Tensor,
        *,
        training: bool = True,
        state_bag: Optional[LCMIncrementalStateBag] = None,
        inference_without_caching: Optional[bool] = False,
        **kwargs,
    ) -> Optional[AttentionMask]:
        mask: NoAttentionMask

        attn_len: Optional[int] = seqs.size(1)
        seq_len = seqs.size(1)
        key_len = keys.size(1)

        mask = NoAttentionMask(
            seq_len=seq_len,
            key_len=key_len,
            attn_len=attn_len,
            device=seqs.device,
            dtype=seqs.dtype,
        )
        return mask

    def __repr__(self) -> str:
        return "NoAttentionMaskFactory()"


@final
class NoAttentionMask(AbstractAttentionMask):
    """
    Represents a diagonal attention mask, i.e attention
    on current position only.
    This turns the self-attention layer into an FFN
    """

    def __init__(
        self,
        seq_len: int,
        key_len: int,
        attn_len: Optional[int],
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param seq_len:
            The sequence length.
        """
        super().__init__()

        self.seq_len = seq_len

        self._device, self._dtype = device, dtype

    @override
    def _do_materialize(self) -> Tensor:
        mask = torch.eye((self.seq_len), device=self._device, dtype=self._dtype)
        mask.log_()
        return mask


class ShiftedCausalAttentionMaskFactory(AttentionMaskFactory):
    """
    Constructs instances of :class:`ShiftedCausalAttentionMask`
    """

    @override
    def __call__(  # type: ignore
        self,
        seqs: Tensor,
        keys: Tensor,
        *,
        source_lengths: Optional[Tensor] = None,
        cf_guidance_prob: float = 0.0,
        training: bool = True,
        state_bag: Optional[LCMIncrementalStateBag] = None,
        inference: bool = False,
    ) -> Optional[AttentionMask]:
        mask: Optional[ShiftedCausalAttentionMask]

        attn_len: Optional[int] = seqs.size(1)
        seq_len = seqs.size(1)
        key_len = keys.size(1)

        if inference:
            mask = None
        else:
            mask = ShiftedCausalAttentionMask(
                seq_len=seq_len,
                key_len=key_len,
                attn_len=attn_len,
                source_lengths=source_lengths,
                cf_guidance_prob=cf_guidance_prob,
                device=seqs.device,
                dtype=seqs.dtype,
            )

        return mask

    def __repr__(self) -> str:
        return "ShiftedCausalAttentionMask()"


@final
class ShiftedCausalAttentionMask(AbstractAttentionMask):
    """
    Represents a causal mask shifted by source_lengths

    In training time, Without source_lengths, the mask look like (e.g. seq_len = 5):

        [ 0., -inf, -inf, -inf, -inf, -inf],
        [ 0., 0., -inf, -inf, -inf, -inf],
        [ 0., 0., 0., -inf, -inf, -inf],
        [ 0., 0., 0., 0., -inf, -inf],
        [ 0., 0., 0., 0., 0., -inf]

    """

    def __init__(
        self,
        seq_len: int,
        key_len: int,
        attn_len: Optional[int],
        *,
        source_lengths: Optional[Tensor] = None,
        cf_guidance_prob: float = 0.0,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param seq_len:
            The sequence length.
        """
        super().__init__()

        self.seq_len = seq_len
        self.key_len = key_len
        self._source_lengths = source_lengths
        self._cf_guidance_prob = cf_guidance_prob
        self._device, self._dtype = device, dtype

    @override
    def _do_materialize(self) -> Tensor:
        if self._source_lengths is None:
            causal_mask = _get_shifted_causal_mask(
                seq_len=self.seq_len,
                key_len=self.key_len,
                shift=0,
                cf_guidance_prob=self._cf_guidance_prob,
                zero_vector=True,
                device=self._device,
                dtype=self._dtype,
            )

        else:
            causal_mask = torch.stack(
                [
                    _get_shifted_causal_mask(
                        seq_len=self.seq_len,
                        key_len=self.key_len,
                        shift=src_len,
                        cf_guidance_prob=self._cf_guidance_prob,
                        zero_vector=True,
                        device=self._device,
                        dtype=self._dtype,
                    )
                    for src_len in self._source_lengths
                ]
            ).unsqueeze(1)
            # bs x 1 (head) x seq_len x seq_len

        causal_mask.log_()

        return causal_mask
