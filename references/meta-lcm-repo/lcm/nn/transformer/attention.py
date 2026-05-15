# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

from typing import Optional, Tuple, final

import torch
import torch.nn as nn
from fairseq2.nn.ops import repeat_interleave
from fairseq2.nn.padding import PaddingMask
from fairseq2.nn.position_encoder import PositionEncoder
from fairseq2.nn.projection import Projection
from fairseq2.nn.transformer import (
    AttentionMask,
    AttentionMaskFactory,
    AttentionState,
    AttentionStateFactory,
    FullAttentionState,
    LayerNormFactory,
    StandardMultiheadAttention,
    create_standard_layer_norm,
)
from fairseq2.nn.transformer.attention import SDPA
from fairseq2.typing import DataType, Device, override
from torch import Tensor
from torch.nn.parameter import Parameter

# FIXME revert to fs2's standard state bag if possible
from lcm.nn.incremental_state import (
    LCMIncrementalStateBag,
)


@final
class QKNormMultiheadAttention(StandardMultiheadAttention):  # type: ignore
    """Represents a Transformer multi-head attention as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`
    with two additional layer-normalization for keys and queries
    as described in https://arxiv.org/pdf/2302.05442
    and other related work
    """

    kv_dim: int
    num_key_value_heads: int
    q_proj: Projection
    k_proj: Projection
    v_proj: Projection
    attn_mask_factory: Optional[AttentionMaskFactory]
    pos_encoder: Optional[PositionEncoder]
    bias_k: Optional[Parameter]
    bias_v: Optional[Parameter]
    add_zero_attn: bool
    sdpa: SDPA
    head_scale_weight: Optional[Parameter]
    output_proj: Projection
    state_factory: Optional[AttentionStateFactory]
    layer_norm_factory: Optional[LayerNormFactory]

    """
    For full parameters description see fairseq2/src/fairseq2/nn/transformer/multihead_attention.py
    Parameters of interest to us:
        :param num_key_value_heads:
            The number of key/value heads for Grouped Query Attention as
            described in :cite:t:`https://doi.org/10.48550/arXiv.2305.13245`.
            If ``None`` or set to ``num_heads``, it is equivalent to standard
            Multi Head Attention (MHA); if set to 1, it is equivalent to Multi
            Query Attention (MQA).

        :param enable_qk_layernorm:
            If True follow Q/K projections with LayerNorms

        :param weight_normalization:
            If True, wrap K/Q/V projections with weight normalization for regularization

        :param pos_encoder:
            For RoPE positional encoder that adds positional encoding to keys
            and queries before computing the attention scores
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        *,
        kv_dim: Optional[int] = None,
        num_key_value_heads: Optional[int] = None,
        q_proj: Optional[Projection] = None,
        k_proj: Optional[Projection] = None,
        v_proj: Optional[Projection] = None,
        attn_mask_factory: Optional[AttentionMaskFactory] = None,
        pos_encoder: Optional[PositionEncoder] = None,
        sdpa: Optional[SDPA] = None,
        scale_heads: bool = False,
        output_proj: Optional[Projection] = None,
        bias: bool = True,
        state_factory: Optional[AttentionStateFactory] = None,
        enable_qk_layernorm: bool = False,
        weight_normalization: bool = False,
        layer_norm_factory: Optional[LayerNormFactory] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        super().__init__(
            model_dim=model_dim,
            num_heads=num_heads,
            kv_dim=kv_dim,
            num_key_value_heads=num_key_value_heads,
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            attn_mask_factory=attn_mask_factory,
            pos_encoder=pos_encoder,
            sdpa=sdpa,
            scale_heads=scale_heads,
            output_proj=output_proj,
            bias=bias,
            state_factory=state_factory,
            device=device,
            dtype=dtype,
        )

        # wrap linear layers with weight norm
        if weight_normalization:
            self.k_proj = nn.utils.parametrizations.weight_norm(self.k_proj)
            self.q_proj = nn.utils.parametrizations.weight_norm(self.q_proj)
            self.v_proj = nn.utils.parametrizations.weight_norm(self.v_proj)

        self.enable_qk_layernorm = enable_qk_layernorm
        # initialize q-k LayerNorms if needed
        if self.enable_qk_layernorm:
            if layer_norm_factory is None:
                # use default LayerNorm factory
                layer_norm_factory = create_standard_layer_norm

            self.q_layer_norm = layer_norm_factory(
                model_dim, device=device, dtype=dtype
            )
            self.k_layer_norm = layer_norm_factory(
                self.kv_dim, device=device, dtype=dtype
            )

    @override
    def _project_q(  # type: ignore
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        state_bag: Optional[LCMIncrementalStateBag] = None,
    ) -> Tensor:
        # (N, S, M) -> (N, S, K_proj)
        q = self.q_proj(seqs)

        # normalize queries
        if self.enable_qk_layernorm:
            q = self.q_layer_norm(q)

        # (N, S, K_proj) -> (N, H, S, K_h)
        q = q.unflatten(-1, (self.num_heads, -1)).transpose(1, 2)

        if self.pos_encoder is not None:
            q = self.pos_encoder(
                q,
                padding_mask,
                state_bag=state_bag,
            )

        return q  # type: ignore[no-any-return]

    @override
    def _project_kv(  # type: ignore
        self,
        keys: Tensor,
        key_padding_mask: Optional[PaddingMask],
        values: Tensor,
        state_bag: Optional[LCMIncrementalStateBag] = None,
    ) -> Tuple[Tensor, Tensor]:
        # (N, S, K) -> (N, S, K_proj)
        k = self.k_proj(keys)

        # normalize keys
        if self.enable_qk_layernorm:
            k = self.k_layer_norm(k)

        # (N, S, V) -> (N, S, V_proj)
        v = self.v_proj(values)

        # (N, S, K_proj) -> (N, H, S, K_h)
        k = k.unflatten(-1, (self.num_key_value_heads, -1)).transpose(1, 2)
        # (N, S, V_proj) -> (N, H, S, V_h)
        v = v.unflatten(-1, (self.num_key_value_heads, -1)).transpose(1, 2)

        if self.pos_encoder is not None:
            k = self.pos_encoder(
                k,
                key_padding_mask,
                state_bag=state_bag,
            )

        return k, v

    @override
    def forward(  # type: ignore
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        keys: Tensor,
        key_padding_mask: Optional[PaddingMask],
        values: Tensor,
        *,
        attn_mask: Optional[AttentionMask] = None,
        state_bag: Optional[LCMIncrementalStateBag] = None,
    ) -> Tensor:
        # (N, S, M) -> (N, H, S, K_h)
        q = self._project_q(
            seqs,
            padding_mask,
            state_bag,
        )
        if self.training or state_bag is None:
            # k: (N, S_kv, M) -> (N, H_kv, S_kv, K_h)
            # v: (N, S_kv, M) -> (N, H_kv, S_kv, V_h)
            k, v = self._project_kv(
                keys,
                key_padding_mask,
                values,
            )
        else:
            if key_padding_mask is not None:
                raise ValueError(
                    "`key_padding_mask` must be `None` during incremental decoding."
                )

            # k: (N, S_step, M) -> (N, H_kv, S_step, K_h)
            # v: (N, S_step, M) -> (N, H_kv, S_step, V_h)
            k, v = self._project_kv(keys, key_padding_mask, values, state_bag)

            state = state_bag.get_state(self, AttentionState)  # type: ignore

            if state is None:
                state_factory = self.state_factory or FullAttentionState

                state = state_factory(
                    k, v, state_bag.max_num_steps, state_bag.capacity_increment
                )

                state_bag.set_state(self, state)
            else:
                state.append(k, v)

                # k: (N, H_kv, S_kv, K_h)
                # v: (N, H_kv, S_kv, V_h)

                k, v = state.get()

        # With Grouped Query Attention, each key/value head is repeated.
        if (num_query_groups := self.num_heads // self.num_key_value_heads) > 1:
            # (N, H_kv, S_kv, K_h) -> (N, H, S_kv, K_h)
            k = repeat_interleave(k, dim=1, repeat=num_query_groups)
            # (N, H_kv, S_kv, K_h) -> (N, H, S_kv, V_h)
            v = repeat_interleave(v, dim=1, repeat=num_query_groups)

        if self.attn_mask_factory is not None:
            attn_mask = self.attn_mask_factory(
                seqs, keys=keys, training=self.training, state_bag=state_bag
            )

        needs_weights = len(self._attn_weight_hooks) > 0

        # attn:         (N, H, S, V_h)
        # attn_weights: (N, H, S, S_kv)

        attn, attn_weights = self.sdpa(
            q,
            k,
            key_padding_mask,
            v,
            attn_mask=attn_mask,
            needs_weights=needs_weights,
        )

        if attn_weights is not None:
            for hook in self._attn_weight_hooks.values():
                hook(self, attn, attn_weights)

        # (N, H, S, V_h) -> (N, S, H, V_h)
        attn = attn.transpose(1, 2)

        if self.head_scale_weight is not None:
            attn = torch.einsum("nshv,h->nshv", attn, self.head_scale_weight)

        # (N, S, H, V_h) -> (N, S, V_proj)
        attn = attn.flatten(2, 3)

        # (N, S, V_proj) -> (N, S, M)

        attn = self.output_proj(attn)

        return attn  # type: ignore[no-any-return]

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        s = f"{s}, enable_qk_layernorm={self.enable_qk_layernorm}"

        return s
