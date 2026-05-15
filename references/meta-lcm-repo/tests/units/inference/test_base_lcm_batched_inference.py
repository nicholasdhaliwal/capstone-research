# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

from random import randint
from typing import List

import pytest
import torch
from fairseq2.nn.padding import pad_seqs
from stopes.core.utils import batch as into_batches
from torch import Tensor

from lcm.datasets.batch import EmbeddingsBatch
from lcm.inference.lcm import LCMGenerator, LCMGeneratorOptions
from lcm.models.base_lcm import BaseLCModelConfig, create_base_lcm_model
from lcm.nn.transformer import TransformerConfig


@pytest.mark.parametrize(
    "disable_cache,eos_threshold",
    [(True, 0.3), (True, 1.1), (False, 0.3), (False, 1.1)],
)
def test_caching_and_batching(disable_cache, eos_threshold):
    """
    Test that batching works with and without batching
    and with a different stopping criterion to test the control of output length.
    """
    # Sample input data
    batch_size = 5
    sonar_embed_dim = 4
    max_prompt_len = 7
    max_gen_len = 8

    # Create sample input list
    sample_inputs: List[Tensor] = []

    for _ in range(batch_size):
        random_len = randint(1, max_prompt_len)
        sample_inputs.append(torch.randn(random_len, sonar_embed_dim))

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
            eos_threshold=eos_threshold,
            sample_latent_variable=False,
        ),
    )

    def generate(batch_size):
        print(
            f"Generating with a batch_size of {batch_size} - disable_cache={disable_cache} and eos_threshold={eos_threshold}"
        )
        lcm_outputs = []
        for batch in into_batches(sample_inputs, batch_size=3):
            padded_batch = EmbeddingsBatch(*pad_seqs(batch))

            lcm_output = generator(
                padded_batch,
                max_gen_len=max_gen_len,
                disable_cache=disable_cache,
            )
            lcm_outputs.extend([hyp[0].seq for hyp in lcm_output.hypotheses])

        return lcm_outputs

    lcm_output_with_batching = generate(batch_size=3)
    lcm_output_without_batching = generate(batch_size=1)

    # Check if the outputs are equal (indicating successful batching/caching)
    assert all(
        [
            torch.allclose(
                a,
                b,
                atol=1e-6,
            )
            for a, b in zip(lcm_output_with_batching, lcm_output_without_batching)
        ]
    ), "Outputs with and without batching do not match"


@pytest.mark.parametrize(
    "batch_size",
    [3, 1],
)
def test_single_input_stopping(batch_size):
    """
    Test that batching we don't stop prematurely with small batches
    """
    # Sample input data
    sonar_embed_dim = 4
    max_prompt_len = 4
    max_gen_len = 3

    # Create sample input list
    sample_inputs: List[Tensor] = []

    for _ in range(batch_size):
        random_len = randint(1, max_prompt_len)
        sample_inputs.append(torch.randn(random_len, sonar_embed_dim))

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
            sample_latent_variable=False,
            trim_hypotheses=False,
        ),
    )

    lcm_outputs = []
    for batch in into_batches(sample_inputs, batch_size=3):
        padded_batch = EmbeddingsBatch(*pad_seqs(batch))

        lcm_output = generator(
            padded_batch,
            max_gen_len=max_gen_len,
        )
        lcm_outputs.extend([hyp[0].seq for hyp in lcm_output.hypotheses])

    # checking that we didn't stop prematurely
    for output_seq, prompt_seq in zip(lcm_outputs, sample_inputs):
        assert output_seq.size(0) - prompt_seq.size(0) == max_gen_len
