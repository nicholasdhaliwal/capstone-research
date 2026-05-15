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

from lcm.models.sonar_normalizer.builder import SonarNormalizer
from lcm.nn.initialization import SONAR_STD, SUPPORTED_INIT_TYPES, get_init_fn

logger = get_log_writer(__name__)


@dataclass
class LCMFrontendConfig:
    dropout_p: float = 0.0
    """ The dropout probability applied to the module' output"""

    pre_linear_bias: bool = True
    """ Whether or not the pre-linear layer has a bias term"""

    pre_linear_init_fn: SUPPORTED_INIT_TYPES = "kaiming_uniform"

    scale_embeddings: bool = False
    """ Scale the embeddings by model_dim before
        adding positions (and before the pre_linear) """

    weight_normalization: bool = False

    embedding_std: float = SONAR_STD
    """Most SONAR embeddings have a distribution with the mean close to 0
    and std close to 0.006. Initializing embedding-like parameters (e.g. end-of-text vector)
    from a similar distribution is recommended, to minimize their disruption of the model training
    """


class LCMFrontend(Module):
    """
    A fronted for the LCM with positional embeddings
    """

    embed: Embedding
    scale: float
    pos_encoder: Optional[PositionEncoder]
    dropout: Optional[Dropout]

    def __init__(
        self,
        sonar_embed_dim: int,
        model_dim: int,
        config: LCMFrontendConfig,
        pos_encoder: Optional[PositionEncoder],
        timestep_embed_dim: int = 0,
        sonar_normalizer: Optional[SonarNormalizer] = None,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param sonar_embed_dim
            The embedding dimension of the sentence encoder, in this case SONAR
        :param model_dim
            The model embedding dimension
        :param timestep_embed_dim
            The embedding dimension of diffusion timesteps (if relevant, defaults to 0)
        :param config:
            A Frontend config. See `LCMFrontendConfig`
        :param pos_encoder:
            An optional position encoder.
        """

        super().__init__()

        self.sonar_embed_dim = sonar_embed_dim

        self.model_dim = model_dim

        self.device = device

        self.embed_scale: float = model_dim**0.5 if config.scale_embeddings else 1.0

        logger.info(f"Using LCMFrontend with embeddings scaler = {self.embed_scale}")

        # Optional sonar normalizer
        self.sonar_normalizer = sonar_normalizer

        # Pre-linear to map to model dimension

        init_fn = get_init_fn(config.pre_linear_init_fn)

        lin = Linear(
            sonar_embed_dim + timestep_embed_dim,
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

    def pre_forward(
        self, seqs: Tensor, diffusion_timesteps: Optional[Tensor] = None, **kwargs
    ) -> Tensor:
        return seqs

    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        state_bag: Optional[IncrementalStateBag] = None,
        diffusion_timesteps: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        """
        Apply pre-linear (if relevant) and add positional embeddings
        """

        # Normalize in standard LCM or add timestep embeddings in diffusion frontentd
        seqs = self.pre_forward(seqs, diffusion_timesteps, **kwargs)

        # pre-linear if any:
        seqs = self.pre_linear(self.embed_scale * seqs)

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
