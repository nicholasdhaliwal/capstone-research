# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

import logging
from typing import Iterable, List, Optional, Sequence

import torch

from lcm.evaluation.api import (
    PREDICTION_TEXT_COLUMN,
    Example,
    Prediction,
    Predictor,
    Prompts,
)
from lcm.evaluation.tasks import register_task
from lcm.evaluation.tasks.base import GenerationTaskConfig
from lcm.evaluation.utils.common import evaluate
from lcm.evaluation.utils.data_utils import ResultsDataLoader, ResultsDatasetConfig

logger = logging.getLogger("lcm.evaluation.test_judge")


class Judge(Predictor):
    """
    Example judge.

    A judge is any model that predicts from the outcome of an
    evaluation task
    """

    def __call__(
        self,
        prompts: Prompts,
        max_prompt_len: Optional[int] = None,
        max_gen_len: Optional[int] = None,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0,
        echo: bool = True,
        return_logprobs: bool = False,
        show_progress: bool = False,
        disable_cache: bool = False,
        **kwargs,
    ) -> Sequence[Prediction]:
        assert isinstance(prompts, Iterable)
        preds: List[Prediction] = []
        for seq in prompts:
            # The outcome of the dummy task is {"metrics": list, "prediction_embed": list, "prompts": list}
            # The judge will calculate the L2 distance between the means of the prediction and inputs previous
            # task
            assert "prompts" in seq and "prediction_embed" in seq
            prompts_mean = torch.mean(torch.stack(seq["prompts"]))  # type: ignore
            preds_mean = torch.mean(torch.stack(seq["prediction_embed"]))  # type: ignore
            preds.append(
                Prediction(text="judge", embed=preds_mean, tokens=prompts_mean)
            )  # type: ignore
        return preds


def dist(prediction: torch.Tensor, targets: torch.Tensor) -> float:
    return prediction.item() - targets.item()


def postprocess(x: Example) -> Example:
    # Get the best hypothesis
    prediction = x["prediction_embed"][0]

    prompts_mean = torch.mean(torch.stack(x["prompts"]))

    # Bause L2 distance is a batch-wise measure, we must add a dimension to it
    return {"prediction": prediction.unsqueeze(0), "targets": prompts_mean.unsqueeze(0)}


@register_task("l2_as_judge", data_loader_type=ResultsDataLoader)
def get_judge_task_config(dataset: ResultsDatasetConfig) -> GenerationTaskConfig:
    """
    Implement the L2 distance as a judge task, to illustrate the
    new API. This trivial task just reload the generation results from a
    next-sentence-prediction task, then recalculate the L2 from the averaged result

    Note that the name convention `dataset_dir` is mandatory for all tasks that
    require a dataset located locally.
    """
    dataset.source_text_column = PREDICTION_TEXT_COLUMN
    return GenerationTaskConfig(
        dataset=dataset,
        postprocess_fn=postprocess,
        metric_fns=[
            evaluate(dist, outputs=("dist")),
        ],
    )
