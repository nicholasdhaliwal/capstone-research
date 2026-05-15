# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#
#

from typing import Optional

from lcm.datasets.configs import ParquetDatasetConfig
from lcm.evaluation.api import EOSConfig
from lcm.evaluation.metrics.common import rouge_score
from lcm.evaluation.tasks import register_task
from lcm.evaluation.tasks.base import GenerationTaskConfig
from lcm.evaluation.utils.common import evaluate
from lcm.evaluation.utils.data_utils import (
    ParquetTestDataLoader,
    default_embed_prompt,
    default_lcm_postprocess,
)


@register_task(
    "lcm_generation",
    data_loader_type=ParquetTestDataLoader,
)
def get_task_config_lcm(
    dataset: ParquetDatasetConfig,
    max_gen_len: int = 128,
    max_gen_len_ratio: Optional[float] = None,
    max_prompt_len: int = 2048,
    eos_config: Optional[EOSConfig] = None,
) -> GenerationTaskConfig:
    return GenerationTaskConfig(
        dataset=dataset,
        prompt_func=default_embed_prompt,  # type: ignore
        postprocess_fn=default_lcm_postprocess,
        metric_fns=[
            evaluate(
                rouge_score,
                outputs=("rouge2", "rougeL", "rougeLsum"),
                types=("rouge2", "rougeL", "rougeLsum"),
            )
        ],
        max_gen_len=max_gen_len,
        max_gen_len_ratio=max_gen_len_ratio,
        max_prompt_len=max_prompt_len,
        eos_config=eos_config,
    )


@register_task(
    "finetuning_data_lcm.validation",
    data_loader_type=ParquetTestDataLoader,
)
def get_validation_task_config_lcm(
    dataset: ParquetDatasetConfig,
    max_gen_len: int = 128,
    max_gen_len_ratio: Optional[float] = None,
    max_prompt_len: int = 2048,
    eos_config: Optional[EOSConfig] = None,
) -> GenerationTaskConfig:
    dataset.name = "finetuning_data=validation"
    return GenerationTaskConfig(
        dataset=dataset,
        prompt_func=default_embed_prompt,  # type: ignore
        postprocess_fn=default_lcm_postprocess,
        metric_fns=[
            evaluate(
                rouge_score,
                outputs=("rouge2", "rougeL", "rougeLsum"),
                types=("rouge2", "rougeL", "rougeLsum"),
            )
        ],
        max_gen_len=max_gen_len,
        max_gen_len_ratio=max_gen_len_ratio,
        max_prompt_len=max_prompt_len,
        eos_config=eos_config,
    )
