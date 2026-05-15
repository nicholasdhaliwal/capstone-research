# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

import pytest
import torch

from lcm.datasets.batch import EmbeddingsBatch
from lcm.inference.lcm import LCMGenerator, LCMGeneratorOptions
from lcm.models.base_lcm import BaseLCModelConfig, create_base_lcm_model
from lcm.nn.transformer import TransformerConfig

torch.manual_seed(0)


@pytest.mark.parametrize("prefix_len", [1, 2, 8])
def test_kv_caching(prefix_len):
    """
    Test that KV caching works as expected.
    Special case if prefix_len = 1 in which case
    the generator's prefill is a no-op
    """
    # Sample input data
    batch_size = 1
    sonar_embed_dim = 4
    max_gen_len = 8
    # Create sample input tensor
    sample_input = torch.randn(batch_size, prefix_len, sonar_embed_dim)

    # Create an LCM model
    model_cfg = BaseLCModelConfig(
        sonar_embed_dim=sonar_embed_dim,
        model_dim=sonar_embed_dim,
        lcm=TransformerConfig(
            ffn_inner_dim=4 * sonar_embed_dim,
            num_layers=2,
            num_attn_heads=1,
        ),
    )

    model = create_base_lcm_model(model_cfg)
    eos_vec = torch.zeros(sonar_embed_dim)

    generator = LCMGenerator(
        model,
        eos_vec=eos_vec,
        options=LCMGeneratorOptions(
            eos_threshold=1,
            stop_on_repetition_cosine_threshold=1,
            sample_latent_variable=False,
            seed=0,
        ),
    )
    # Generate without caching
    lcm_output_no_cache = generator(
        EmbeddingsBatch(seqs=sample_input, padding_mask=None),
        max_gen_len=max_gen_len,
        disable_cache=True,
    )

    # Enable KV caching
    lcm_output_with_cache = generator(
        EmbeddingsBatch(seqs=sample_input, padding_mask=None),
        max_gen_len=max_gen_len,
        disable_cache=False,
    )

    # Check if the outputs are equal (indicating successful caching)
    assert torch.allclose(
        lcm_output_no_cache.hypotheses[0][0].seq,
        lcm_output_with_cache.hypotheses[0][0].seq,
        atol=1e-3,
    ), "Outputs with and without caching do not match"
