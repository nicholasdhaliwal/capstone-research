# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

from lcm.models.base_lcm.builder import (
    BaseLCModelConfig,
    LCMFrontendConfig,
    ProjectionConfig,
    TransformerConfig,
    lcm_arch,
)


# Every model must register a toy_{model_family}
@lcm_arch("toy_base_lcm")
def toy_base_lcm() -> BaseLCModelConfig:
    return BaseLCModelConfig(
        lcm=TransformerConfig(num_layers=2),
    )


@lcm_arch("base_lcm_1_6B")
def base_lcm_1_6B() -> BaseLCModelConfig:
    """Base 1.6B model
    Parameter Size: 1,647,635,456
    """
    model_dim: int = 2048
    num_attn_heads: int = 16
    return BaseLCModelConfig(
        max_seq_len=4096,
        model_dim=model_dim,
        sonar_embed_dim=1024,
        sonar_normalizer_name="dummy_sonar_normalizer",
        frontend=LCMFrontendConfig(),
        lcm=TransformerConfig(
            final_dropout_p=0.0,
            attention_dropout_p=0.0,
            dropout_p=0.1,
            mha_output_proj_bias=True,
            ffn_inner_dim=model_dim * 4,
            num_attn_heads=num_attn_heads,
            num_layers=32,
            pos_embedding_style="rope",
            use_swiglu=True,
            layer_normalization_style="rms",
        ),
        postnet=ProjectionConfig(),
    )
