# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

from typing import List, Optional, Tuple

from fairseq2.nn.padding import PaddingMask
from fairseq2.nn.transformer import (
    AttentionMask,
    AttentionMaskFactory,
    LayerNormFactory,
    StandardTransformerDecoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerNormOrder,
)
from fairseq2.typing import DataType, Device, override
from torch import Generator, Tensor
from torch.nn import Dropout, ModuleList

from lcm.nn.incremental_state import LCMIncrementalStateBag


class LCMStandardTransformerDecoderLayer(StandardTransformerDecoderLayer):  # type: ignore
    """Pass on `source_lengths` to StandardTransformerDecoderLayer's forward_pass."""

    @override
    def forward(  # type: ignore
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        self_attn_mask: Optional[AttentionMask] = None,
        encoder_output: Optional[Tensor] = None,
        encoder_padding_mask: Optional[PaddingMask] = None,
        *,
        state_bag: Optional[LCMIncrementalStateBag] = None,
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        seqs = self._forward_self_attn(
            seqs,
            padding_mask,
            self_attn_mask,
            state_bag,
        )

        seqs = self._forward_encoder_decoder_attn(
            seqs, padding_mask, encoder_output, encoder_padding_mask, state_bag
        )

        seqs = self._forward_ffn(seqs)

        return seqs, padding_mask

    @override
    def _forward_self_attn(  # type: ignore
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        self_attn_mask: Optional[AttentionMask],
        state_bag: Optional[LCMIncrementalStateBag],
    ) -> Tensor:
        residual = seqs

        if self.norm_order != TransformerNormOrder.POST:
            seqs = self.self_attn_layer_norm(seqs)

        seqs = self.self_attn(
            seqs,
            padding_mask,
            keys=seqs,
            key_padding_mask=padding_mask,
            values=seqs,
            attn_mask=self_attn_mask,
            state_bag=state_bag,
        )

        if self.self_attn_norm is not None:
            seqs = self.self_attn_norm(seqs)

        if self.self_attn_dropout is not None:
            seqs = self.self_attn_dropout(seqs)

        seqs = seqs + residual

        if self.norm_order == TransformerNormOrder.POST:
            seqs = self.self_attn_layer_norm(seqs)

        return seqs


class LCMTransformerDecoder(TransformerDecoder):
    def __init__(
        self,
        layers: List[TransformerDecoderLayer],
        layer_norm_factory: LayerNormFactory,
        self_attn_mask_factory: AttentionMaskFactory,
        use_causal_attn_mask: bool = True,
        generator: Optional[Generator] = None,
        dropout_p: float = 0.0,
        norm_order: TransformerNormOrder = TransformerNormOrder.POST,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        layer_list = ModuleList(layers)

        if not layer_list:
            raise ValueError("`layers` must be non-empty.")

        model_dim = layer_list[0].model_dim

        super().__init__(model_dim)

        self.self_attn_mask_factory = self_attn_mask_factory

        self.layers = layer_list

        self.generator = generator

        if norm_order != TransformerNormOrder.POST:
            self.layer_norm = layer_norm_factory(model_dim, device=device, dtype=dtype)
        else:
            self.register_module("layer_norm", None)

        if dropout_p > 0.0:
            self.dropout = Dropout(dropout_p)
        else:
            self.register_module("dropout", None)

        self.norm_order = norm_order

    @override
    def forward(  # type: ignore
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        encoder_output: Optional[Tensor] = None,
        encoder_padding_mask: Optional[PaddingMask] = None,
        *,
        state_bag: Optional[LCMIncrementalStateBag] = None,
        **kwargs,
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        """Pass on two additional arguments to StandardTransformerDecoder's forward_pass:"""
        num_layers = len(self.layers)

        self_attn_mask: Optional[AttentionMask] = None
        if self.self_attn_mask_factory is not None:
            self_attn_mask = self.self_attn_mask_factory(
                seqs,
                keys=seqs,
                training=self.training,
                state_bag=state_bag,
            )

        for layer_idx, layer in enumerate(self.layers):
            layer_output, layer_padding_mask = layer(
                seqs,
                padding_mask,
                self_attn_mask,
                encoder_output,
                encoder_padding_mask,
                state_bag=state_bag,
            )

            seqs, padding_mask = layer_output, layer_padding_mask

            for hook in self._layer_output_hooks.values():
                if not hook(layer_idx, seqs, padding_mask, num_layers):
                    break

        if self.layer_norm is not None:
            seqs = self.layer_norm(seqs)

        if self.dropout is not None:
            seqs = self.dropout(seqs)

        return seqs, padding_mask
