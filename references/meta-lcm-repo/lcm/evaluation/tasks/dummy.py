# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#
#
# A dummy task that evaluates a dummy predictor

from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

import torch
from numpy.random import RandomState

from lcm.datasets.base import DataLoader
from lcm.datasets.configs import (
    ColumnsNames,
    DatasetConfigT,
    JSONDatasetConfig,
    ParquetDatasetConfig,
)
from lcm.evaluation.utils.data_utils import JSONTestDataLoader, ParquetTestDataLoader

from ..api import (
    EOSConfig,
    Example,
    Prediction,
    Predictor,
    Prompts,
    Task,
    TaskConfig,
    TaskResult,
    parse_task_configs,
)
from ..tasks import register_task
from ..utils.distributed import get_dp_group, mean_reduce_dict


@dataclass
class DummyTaskConfig(TaskConfig, Generic[DatasetConfigT]):
    dataset: DatasetConfigT
    eos_config: Optional[EOSConfig] = None
    prompt_func: Callable[..., Tuple] = lambda x: ("", x)  # noqa


class DummyTask(Task):
    """
    A simple dummy task to illustrate how to work with parquet or json
    data for evaluation. It ignores the predictor and simply returns
    the input back as the prediction
    """

    def __init__(
        self,
        config: DummyTaskConfig,
        data_loader: DataLoader,
    ):
        self.config = config
        self.data_loader = data_loader
        self.gang = self.data_loader.gang

    def compute_metrics(  # type: ignore
        self,
        predictions: Sequence[Prediction],
        prompts: Prompts,
        hypothesis_indices: List[Sequence],
        input: Sequence[Example],
        show_progress: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Calculate the sum of element-wise subtraction between
        prediction and the batched. This should be 0 for
        IdentityPredictor
        """
        prediction_embed = [predictions[idx[0]].embed for idx in hypothesis_indices]
        # input_batch should be a NestedTensorAsPrompt by now
        assert isinstance(prompts, Iterable)
        assert isinstance(self.config, DummyTaskConfig)
        l2_dist_values = []
        for i, seq in enumerate(prompts):  # type: ignore
            l2_dist = (seq - prediction_embed[i]).pow(2).sum(-1).sqrt()
            l2_dist_values.append(l2_dist.mean().item())
        return {
            "metrics": {"m1": l2_dist_values},
            "prediction_embed": prediction_embed,
            "prompts": prompts,
        }

    def run(  # type: ignore
        self,
        predictor: Predictor,
        random_state: Optional["RandomState"] = None,  # noqa unused
        show_progress: bool = False,
        **kwargs: Any,
    ) -> TaskResult:
        dataset = self.data_loader.iterate_batches()

        raw_results: List[Dict[str, Any]] = []
        accumulated_result: Dict[str, List[float]] = defaultdict(list)
        for _, batch in enumerate(dataset):
            text_prompts, embed_prompts = self.config.prompt_func(batch)

            # Assume:
            # For n prompts and k generations, Predictions is a list of nxk items, where
            # n-subseuence has k items, corresponding to generations of one prompt
            n = len(embed_prompts)
            indices = [[i] for i in range(n)]

            predictions = [
                Prediction(text=text, tokens=[0], embed=seq)
                for text, seq in zip(text_prompts, embed_prompts)
            ]
            batch_results = self.compute_metrics(
                predictions=predictions,
                prompts=embed_prompts,
                hypothesis_indices=indices,  # type: ignore
                input=batch,
            )
            raw_results.append({"text_prompts": text_prompts, **batch_results})

            for name, values in batch_results["metrics"].items():
                accumulated_result[name].extend(values)

        avg_results = mean_reduce_dict(accumulated_result, group=get_dp_group())
        return TaskResult(metrics=avg_results, raw_results=raw_results)


@register_task("dummy_generation", data_loader_type=ParquetTestDataLoader)  # type: ignore
def get_dummy_task_config(
    dataset: ParquetDatasetConfig,
    eos_config: Optional[EOSConfig] = None,
    **kwargs,
) -> DummyTaskConfig[ParquetDatasetConfig]:
    def dummy_prompt(batch):
        text_prompts = [""] * len(batch)
        embed_prompts = [x[ColumnsNames.source_column.value] for x in batch]
        return text_prompts, embed_prompts

    return DummyTaskConfig[ParquetDatasetConfig](
        dataset=dataset,
        eos_config=eos_config,
        prompt_func=dummy_prompt,
        **parse_task_configs(DummyTaskConfig, **kwargs),
    )


@register_task("dummy_json_generation", data_loader_type=JSONTestDataLoader)  # type: ignore
def get_dummy_json_task_config(
    dataset: JSONDatasetConfig,
    eos_config: Optional[EOSConfig] = None,
    **kwargs,
) -> DummyTaskConfig[JSONDatasetConfig]:
    def dummy_prompt(batch):
        text_prompts = [x[ColumnsNames.source_text_column.value] for x in batch]
        embed_prompts = torch.rand(len(batch), 1024)
        return text_prompts, embed_prompts

    dataset.prompt_template = "[INST] Prompt: {{ _source_text_column }}"
    return DummyTaskConfig[JSONDatasetConfig](
        dataset=dataset,
        eos_config=eos_config,
        prompt_func=dummy_prompt,
        **parse_task_configs(DummyTaskConfig, **kwargs),
    )
