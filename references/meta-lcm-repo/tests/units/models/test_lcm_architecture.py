# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#
"""
Testing that different combinations of parameters are supported
"""

import math

import pytest
import torch

from lcm.datasets.batch import EmbeddingsBatch, LCMInput
from lcm.models.base_lcm import (
    BaseLCModel,
    BaseLCModelConfig,
    create_base_lcm_model,
)
from lcm.models.base_lcm.builder import BASE_LCM_MODEL_TYPE
from lcm.models.sonar_normalizer.loader import load_sonar_normalizer_model
from lcm.models.two_tower_diffusion_lcm.builder import (
    TwoTowerDiffusionLCModelConfig,
    create_two_tower_diffusion_lcm_model,
)
from lcm.nn.transformer import TransformerConfig
from lcm.train.mse_lcm.criterion import (
    ReconstructionCriterionConfig,
    TargetMSECriterion,
)
from lcm.utils.model_type_registry import lcm_model_type_registry


def test_sonar_normalizer():
    assert load_sonar_normalizer_model("dummy_sonar_normalizer") is not None


@pytest.mark.parametrize(
    "model_cfg_kwargs, lcm_cfg_kwargs, criterion_cfg_kwargs",
    [
        ({}, {}, {}),  # default
        ({}, {"pos_embedding_style": "rope"}, {}),
        ({"model_dim": 712}, {"pos_embedding_style": "learned"}, {}),
    ],
)
def test_supervised_mse_forward(model_cfg_kwargs, lcm_cfg_kwargs, criterion_cfg_kwargs):
    """Testing that we can create an LCM model and do a forward step with supervised MSE criterion"""
    # Creating a toy model
    model_cfg = BaseLCModelConfig(
        **model_cfg_kwargs, lcm=TransformerConfig(**lcm_cfg_kwargs)
    )
    model = create_base_lcm_model(model_cfg)

    # Creating a criterion
    criterion_cfg = ReconstructionCriterionConfig(**criterion_cfg_kwargs)
    criterion = TargetMSECriterion(criterion_cfg, model=model)

    # Creating a toy batch
    batch_size, src_len, tgt_len = 5, 17, 11
    sonar_dim, sonar_std = 1024, 0.006
    x = torch.randn(size=[batch_size, src_len, sonar_dim]) * sonar_std
    y = torch.randn(size=[batch_size, tgt_len, sonar_dim]) * sonar_std
    batch = LCMInput(
        source=[row for row in x],
        target=[row for row in y],
        tokens=None,
    )
    # Doing the forward step
    loss_item = criterion(batch)

    # Testing that the output is adequate
    loss_value = loss_item.value.item()
    n_elements = loss_item.num_target_elements
    assert math.isfinite(loss_value)
    assert n_elements == batch_size * (tgt_len)


@pytest.mark.parametrize(
    "model_type_name",
    [BASE_LCM_MODEL_TYPE],
)
def test_next_sentence_prediction(model_type_name: str) -> None:
    """Testing that the `predict_next_sentence` method is really callable for the single-pass LCM model types."""

    # Creating a model of the given type, the toy architecture
    model_type_cfg = lcm_model_type_registry.get_config(model_type_name)
    arch_registry = model_type_cfg.config_loader._arch_configs  # type: ignore
    model_cfg = arch_registry.get(f"toy_{model_type_name}")
    model: BaseLCModel = model_type_cfg.model_factory(model_cfg)
    model.eval()

    # Creating a toy batch
    batch_size, src_len = 5, 17
    sonar_dim, sonar_std = 1024, 0.006
    x = torch.randn(size=[batch_size, src_len, sonar_dim]) * sonar_std
    batch = EmbeddingsBatch(x, padding_mask=None)

    # Feeding the batch to the model and testing that the next sentence prediction can be deterministic
    deterministic_prediction1 = model.predict_next_sentence(batch, sample=False).seqs
    assert deterministic_prediction1.shape == x.shape

    deterministic_prediction2 = model.predict_next_sentence(batch, sample=False).seqs
    assert torch.allclose(deterministic_prediction1, deterministic_prediction2)

    # Testing that the sampling is really random
    random_prediction = model.predict_next_sentence(
        batch,
        sample=True,
        temperature=0.01,
    ).seqs
    assert random_prediction.shape == x.shape
    assert not torch.allclose(random_prediction, deterministic_prediction1)


@pytest.mark.parametrize(
    "model_cfg_kwargs, criterion_cfg_kwargs",
    [
        (
            {
                "sonar_normalizer_name": "dummy_sonar_normalizer",
                "sonar_embed_dim": 1024,
                "model_dim": 1024,
            },
            {},
        ),
    ],
)
def test_supervised_diffusion_forward(model_cfg_kwargs, criterion_cfg_kwargs):
    """Testing that we can create an LCM model and do a forward step with supervised MSE criterion"""
    from lcm.train.two_tower_diffusion_lcm.criterion import (
        DiffusionNextSentFinetuningCriterion,
        TowerDiffusionLCMCriterionConfig,
    )

    # Creating a toy model
    model_cfg = TwoTowerDiffusionLCModelConfig(
        **model_cfg_kwargs,
    )
    model = create_two_tower_diffusion_lcm_model(model_cfg)

    # Creating a criterion
    criterion_cfg = TowerDiffusionLCMCriterionConfig(**criterion_cfg_kwargs)
    criterion = DiffusionNextSentFinetuningCriterion(criterion_cfg, model=model)

    # Creating a toy batch
    batch_size, src_len, tgt_len = 5, 17, 11
    sonar_dim, sonar_std = 1024, 0.006
    x = torch.randn(size=[batch_size, src_len, sonar_dim]) * sonar_std
    y = torch.randn(size=[batch_size, tgt_len, sonar_dim]) * sonar_std

    batch = LCMInput(
        source=[row[: i + 1] for i, row in enumerate(x)],
        target=[row[: i + 1] for i, row in enumerate(y)],
    )
    # Doing the forward step
    loss_item = criterion(batch)

    # Testing that the output is adequate
    loss_value = loss_item.value.item()
    n_elements = loss_item.num_target_elements
    assert math.isfinite(loss_value)
    assert n_elements == batch_size * (batch_size + 1) // 2
