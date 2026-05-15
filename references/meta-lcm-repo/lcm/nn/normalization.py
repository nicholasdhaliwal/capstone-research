# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
#

from typing import Literal, Optional, final

import torch
from fairseq2.nn import LayerNorm, RMSNorm, StandardLayerNorm
from fairseq2.nn.transformer import LayerNormFactory, create_standard_layer_norm
from fairseq2.typing import DataType, Device, override

SUPPORTED_LN_TYPES = Literal["standard", "fp32", "rms", "unit"]


@final
class FP32LayerNorm(LayerNorm):
    """Applies Layer Normalization in single-precision."""

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w, b = self.weight, self.bias

        # cast input and params to float32
        fp32_x = x.float()
        fp32_w = w.float() if w is not None else None
        fp32_b = b.float() if b is not None else None

        y = torch.nn.functional.layer_norm(
            fp32_x, self.normalized_shape, fp32_w, fp32_b, self.eps
        )

        return y.type_as(x)


def build_rms_layer_norm(
    model_dim: int,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> LayerNorm:
    """Build an RMS Layer Normalization module."""
    return RMSNorm(model_dim, bias=False, device=device, dtype=dtype)


def build_fp32_layer_norm(
    model_dim: int,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> LayerNorm:
    """Build an Single-precision Layer Normalization module."""
    return FP32LayerNorm(model_dim, bias=False, device=device, dtype=dtype)


def build_unit_layer_norm(
    model_dim: int,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> LayerNorm:
    """Create an instance of :class:`StandardLayerNorm
    without learnable mean and variance`."""
    return StandardLayerNorm(
        model_dim,
        bias=False,
        elementwise_affine=False,
        device=device,
        dtype=dtype,
    )


def parse_layer_norm_factory(layer_normalization_style: str) -> LayerNormFactory:
    if layer_normalization_style == "rms":
        # Note that RMSNorm normalizes in single-precision by default
        return build_rms_layer_norm

    elif layer_normalization_style == "unit":
        return build_unit_layer_norm

    elif layer_normalization_style == "fp32":
        return build_fp32_layer_norm

    elif layer_normalization_style == "standard":
        return create_standard_layer_norm

    else:
        raise ValueError(f"Unsupported LayerNorm style {layer_normalization_style}")
