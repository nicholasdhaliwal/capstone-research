# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

from dataclasses import dataclass
from typing import Optional

import torch
from fairseq2.nn.projection import Linear
from fairseq2.typing import DataType, Device
from torch import Tensor
from torch.nn import Module

from lcm.nn.initialization import (
    SUPPORTED_INIT_TYPES,
    get_init_fn,
    parse_activation_fn,
)
from lcm.nn.normalization import SUPPORTED_LN_TYPES


@dataclass
class ProjectionConfig:
    dropout_p: float = 0.0
    """ The dropout probability applied to the module' output"""

    linear_bias: bool = True
    """ Whether or not the pre-linear layer has a bias term"""

    linear_init_fn: SUPPORTED_INIT_TYPES = "kaiming_uniform"

    weight_normalization: bool = False

    layer_normalization_style: SUPPORTED_LN_TYPES = "standard"

    activation_name: Optional[str] = None
    """the activation function to apply after fi any"""


class Projection(Module):
    """
    An output projecton module.
    """

    def __init__(
        self,
        output_dim: int,
        input_dim: int,
        config: ProjectionConfig,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        super().__init__()

        self.dtype = dtype

        init_fn = get_init_fn(config.linear_init_fn)

        lin = Linear(
            input_dim,
            output_dim,
            bias=config.linear_bias,
            device=device,
            dtype=dtype,
            init_fn=init_fn,
        )
        if config.weight_normalization:
            self.fc = torch.nn.utils.parametrizations.weight_norm(lin)
        else:
            self.fc = lin

        self.activation_fn = parse_activation_fn(config.activation_name)

        if self.activation_fn is not None:
            # some activation functions (e.g., PReLU) have parameters
            # and so we need to move them to the right device
            self.activation_fn.to(device)

    def forward(self, seqs: Tensor):
        seqs = self.fc(seqs)

        if self.activation_fn is not None:
            seqs = self.activation_fn(seqs)

        return seqs
