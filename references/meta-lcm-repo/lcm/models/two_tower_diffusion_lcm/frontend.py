# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from fairseq2.logging import get_log_writer
from fairseq2.nn import Embedding, LearnedPositionEncoder, PositionEncoder
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.padding import PaddingMask
from fairseq2.nn.projection import Linear
from fairseq2.typing import DataType, Device
from torch import Tensor
from torch.nn import Dropout, Module

from lcm.nn.initialization import SUPPORTED_INIT_TYPES, get_init_fn

logger = get_log_writer(__name__)


@dataclass
class EncoderFrontendConfig:
    dropout_p: float = 0.0
    """ The dropout probability applied to the module' output"""

    pre_linear_bias: bool = True
    """ Whether or not the pre-linear layer has a bias term"""

    pre_linear_init_fn: SUPPORTED_INIT_TYPES = "kaiming_uniform"

    weight_normalization: bool = False

    embedding_std: float = 1.0


class EncoderFrontend(Module):
    """
    A fronted for the context encoder in encoder-decoder LCMs
    """

    embed: Embedding
    pos_encoder: Optional[PositionEncoder]
    dropout: Optional[Dropout]

    def __init__(
        self,
        sonar_embed_dim: int,
        model_dim: int,
        config: EncoderFrontendConfig,
        pos_encoder: Optional[PositionEncoder],
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param sonar_embed_dim
            The embedding dimension of the sentence encoder, in this case SONAR
        :param model_dim
            The model embedding dimension
        :param config:
            A Frontend config. See `LCMFrontendConfig`
        :param pos_encoder:
            An optional position encoder.
        """

        super().__init__()

        self.sonar_embed_dim = sonar_embed_dim

        self.model_dim = model_dim

        self.device = device

        # Pre-linear to map to model dimension
        init_fn = get_init_fn(config.pre_linear_init_fn)

        lin = Linear(
            sonar_embed_dim,
            model_dim,
            bias=config.pre_linear_bias,
            device=device,
            dtype=dtype,
            init_fn=init_fn,
        )

        if config.weight_normalization:
            self.pre_linear = torch.nn.utils.parametrizations.weight_norm(lin)
        else:
            self.pre_linear = lin

        if pos_encoder is not None:
            if pos_encoder.encoding_dim != self.model_dim:
                raise ValueError(
                    f"`encoding_dim` of `pos_encoder` and `embedding_dim` of \
                    `embed` must be equal, but are {pos_encoder.encoding_dim} \
                    and {self.model_dim} instead."
                )

            self.pos_encoder = pos_encoder
        else:
            self.register_module("pos_encoder", None)

        if config.dropout_p > 0.0:
            self.dropout = Dropout(config.dropout_p)
        else:
            self.register_module("dropout", None)

        self.reset_parameters(embedding_std=config.embedding_std)

    def reset_parameters(self, embedding_std: float) -> None:
        """Initialize module parameters.
        The positional embeddings should be initialized with the
        same order of magnitude as the semantic embeddings, in order
        to make the early training as stable as possible.
        Otherwise, the positional and special token embeddings would
        flood out the semantic information.
        """
        logger.info(
            f"Initializing frontend embeddings (special and positional) ~ N(0, {embedding_std})"
        )
        if isinstance(self.pos_encoder, LearnedPositionEncoder):
            torch.nn.init.normal_(self.pos_encoder.weight, std=embedding_std)

    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        state_bag: Optional[IncrementalStateBag] = None,
        **kwargs,
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        """
        Apply pre-linear (if relevant) and add positional embeddings
        """

        # pre-linear if any:
        seqs = self.pre_linear(seqs)

        if self.pos_encoder is not None:
            seqs = self.pos_encoder(
                seqs,
                padding_mask,
                state_bag=state_bag,
                **kwargs,
            )

        if self.dropout is not None:
            seqs = self.dropout(seqs)

        return seqs, padding_mask
