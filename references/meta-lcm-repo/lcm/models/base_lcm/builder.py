# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

from dataclasses import dataclass, field
from typing import Optional

import torch.nn
from fairseq2.config_registry import ConfigRegistry
from fairseq2.logging import get_log_writer
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.transformer import AttentionMaskFactory, CausalAttentionMaskFactory
from fairseq2.typing import DataType, Device

from lcm.datasets.batch import EmbeddingsBatch
from lcm.models.abstract_lcm import (
    AbstractLCModel,
    AbstractLCModelBuilder,
    AbstractLCModelConfig,
)
from lcm.models.base_lcm.frontend import LCMFrontend, LCMFrontendConfig
from lcm.nn.initialization import parse_norm_order
from lcm.nn.normalization import parse_layer_norm_factory
from lcm.nn.projection import Projection, ProjectionConfig
from lcm.nn.transformer import (
    LCMTransformerDecoder,
    TransformerConfig,
    TransformerFactory,
)

logger = get_log_writer(__name__)

BASE_LCM_MODEL_TYPE = "base_lcm"


@dataclass
class BaseLCModelConfig(AbstractLCModelConfig):
    model_type: str = BASE_LCM_MODEL_TYPE

    max_seq_len: int = 2048

    model_dim: int = 1024

    model_output_dim: Optional[int] = None
    """If ``None`` use SONAR dimension as output_dim."""

    frontend: LCMFrontendConfig = field(default_factory=lambda: LCMFrontendConfig())
    """The fronted config. This module maps from `sonar_embed_dim` to `model_dim`
        and potentially adds positional embeddings"""

    lcm: TransformerConfig = field(default_factory=lambda: TransformerConfig())
    """The core lcm config. This is causal Transformer decoder"""

    postnet: ProjectionConfig = field(default_factory=lambda: ProjectionConfig())
    """The postnet config. A module mapping the output of the core lcm
       back to `sonar_embed_dim`"""


lcm_archs = ConfigRegistry[BaseLCModelConfig]()
lcm_arch = lcm_archs.decorator


class BaseLCModel(AbstractLCModel):
    """Base class for LCM models"""

    config: BaseLCModelConfig

    def __init__(
        self,
        config: BaseLCModelConfig,
        lcm: LCMTransformerDecoder,
        frontend: LCMFrontend,
        postnet: Projection,
    ) -> None:
        """
        Basic LCM model with :
            - fronted
            - lcm
            - postnet
        """
        super().__init__(config)

        self.frontend = frontend

        self.lcm = lcm

        self.postnet = postnet

        self.model_dim = lcm.model_dim

        self.sonar_embed_dim = config.sonar_embed_dim

    def forward(
        self,
        batch: EmbeddingsBatch,
        state_bag: Optional[IncrementalStateBag] = None,
        **kwargs,
    ) -> EmbeddingsBatch:
        """
        Scaling + Positions
        If a normalizer is provided, the features will be normalized in the
        frontend's pre_forward (e.g. MSE LCM) or in the criterion (Diffusion LCM)
        """
        seqs, padding_mask = self.frontend(
            batch.seqs,
            batch.padding_mask,
            diffusion_timesteps=batch.diffusion_timesteps,
            state_bag=state_bag,
            **kwargs,
        )

        # Core LCM
        seqs, padding_mask = self.lcm(
            seqs,
            padding_mask,
            state_bag=state_bag,
            **kwargs,
        )

        # Postnet:
        seqs = self.postnet(seqs)  # type: ignore

        return EmbeddingsBatch(seqs=seqs, padding_mask=padding_mask)

    def predict_next_sentence(
        self,
        batch: EmbeddingsBatch,
        sample: bool = False,
        temperature: float = 1.0,
        state_bag: Optional[IncrementalStateBag] = None,
        **kwargs,
    ) -> EmbeddingsBatch:
        """
        The method for predicting the next sentence embeddings.
        In the basic LCM, this is equivalent to just the forward method,
        but the derived architectures may have a different implementation.
        E.g. in VAE LCM, we run the VAE decoder on top of the `forward` results.

        Args:
            batch (EmbeddingsBatch): the sequence of concepts which
                the model should continue.
            sample (bool): whether to predict the single most probable next sentence
                or to sample from the predicted distribution.
            temperature (float): a positive float indicating the degree of diversity
                for the sampling (active only if `sample is True`).
        Returns:
            EmbeddingsBatch: the batch with predicted SONAR sentences.
        """
        # Normalize the input embeddings if we're expected to
        # normalize outside of the model's forward pass
        if self.frontend.sonar_normalizer is not None:
            batch = batch.normalize_seqs(self.frontend.sonar_normalizer)

        # TODO: implement efficient sampling of multiple candidates
        predicted_means = self.forward(batch, state_bag=state_bag, **kwargs)

        if sample and temperature > 0:
            noise = torch.randn_like(predicted_means.seqs) * temperature
            predicted_means.seqs = predicted_means.seqs + noise

        if self.frontend.sonar_normalizer is not None:
            predicted_means = predicted_means.denormalize_seqs(
                self.frontend.sonar_normalizer
            )

        return predicted_means


class BaseLCModelBuilder(AbstractLCModelBuilder):
    """Builds modules of a base LCM model"""

    config: BaseLCModelConfig
    device: Optional[Device]
    dtype: Optional[DataType]

    def __init__(
        self,
        config: BaseLCModelConfig,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        super().__init__(config=config, device=device, dtype=dtype)
        self.lcm_factory = TransformerFactory(
            model_dim=self.config.model_dim,
            max_seq_len=self.config.max_seq_len,
            config=self.config.lcm,
            device=device,
            dtype=dtype,
        )

        if config.model_output_dim is None:
            self.model_output_dim = self.config.sonar_embed_dim
        else:
            self.model_output_dim = config.model_output_dim

    def build_model(self) -> BaseLCModel:
        """Build a model."""

        frontend = self.build_frontend()

        lcm = self.build_core_lcm()

        postnet = self.build_postnet()

        return BaseLCModel(
            config=self.config,
            frontend=frontend,
            lcm=lcm,
            postnet=postnet,
        )

    def build_frontend(self) -> LCMFrontend:
        """Build the LCM front-end (i.e., prenet)."""

        return LCMFrontend(
            sonar_embed_dim=self.config.sonar_embed_dim,
            model_dim=self.config.model_dim,
            config=self.config.frontend,
            pos_encoder=self.lcm_factory.build_pos_encoder(),
            sonar_normalizer=self.build_sonar_normalizer(),
            device=self.device,
            dtype=self.dtype,
        )

    def build_postnet(self) -> Projection:
        return Projection(
            output_dim=self.model_output_dim,
            input_dim=self.config.model_dim,
            config=self.config.postnet,
            device=self.device,
            dtype=self.dtype,
        )

    def build_attention_mask_factory(self):
        self_attn_mask_factory: AttentionMaskFactory

        self_attn_mask_factory = CausalAttentionMaskFactory()

        return self_attn_mask_factory

    def build_core_lcm(self) -> LCMTransformerDecoder:
        """Build the core LCM module."""

        config = self.config.lcm

        layers = [self.lcm_factory.build_layer() for _ in range(config.num_layers)]

        self_attn_mask_factory = self.build_attention_mask_factory()

        if config.final_norm_order_style is None:
            # The final norm order style will be that of the layer-level norm order
            final_norm_order = parse_norm_order(config.norm_order_style)
        else:
            final_norm_order = parse_norm_order(config.final_norm_order_style)

        layer_norm_factory = parse_layer_norm_factory(config.layer_normalization_style)

        return LCMTransformerDecoder(
            layers,  # type: ignore
            self_attn_mask_factory=self_attn_mask_factory,
            norm_order=final_norm_order,
            layer_norm_factory=layer_norm_factory,
            dropout_p=config.final_dropout_p,
            device=self.device,
            dtype=self.dtype,
        )


def create_base_lcm_model(
    config: BaseLCModelConfig,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> BaseLCModel:
    """Create an LCM model.
    :param config:
        The configuration.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """
    return BaseLCModelBuilder(config, device=device, dtype=dtype).build_model()
