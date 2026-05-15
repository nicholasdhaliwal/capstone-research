# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import torch

from ..api import Prediction, Predictor, PredictorConfig, Prompts


@dataclass
class DummyPredictorConfig(PredictorConfig):
    @classmethod
    def predictor_class(cls):
        return DummyPredictor


class DummyPredictor:
    def __init__(self, config: DummyPredictorConfig, **kwargs):
        self.config = config
        if "eos_config" in kwargs:
            self.eos = "<EOS>"
        else:
            self.eos = " "

    @staticmethod
    def from_config(config: DummyPredictorConfig, **kwargs) -> "DummyPredictor":  # type: ignore
        return DummyPredictor(config, **kwargs)

    def __call__(  # type: ignore
        self,
        prompts: Prompts,
        max_prompt_len: Optional[int] = None,
        max_gen_len: Optional[int] = None,
        temperature: float = 0.0,
        disable_cache: bool = False,
        greedy: bool = True,
        top_p: float = 0.0,
        top_k: int = 0,
        echo: bool = True,
        return_logprobs: bool = False,
        show_progress: bool = False,
        num_generations: int = 1,
        **kwargs,
    ) -> Sequence[Prediction]:
        if return_logprobs:
            raise NotImplementedError("The Dummy predictor does not support logprobs")
        if greedy and num_generations > 1:
            raise ValueError("Greedy generation only works with beam size 1")
        # prompts should be a NestedTensor by now. Here we return a fake
        # prediction that shares the same embeddings and input, with empty text
        preds: List[Prediction] = []
        assert isinstance(prompts, Iterable)
        for seq in prompts:  #
            preds.extend(
                [Prediction(text=self.eos, tokens=[0], embed=seq)] * num_generations  # type: ignore
            )
        return preds


@dataclass
class DummyJudgeConfig(PredictorConfig):
    @classmethod
    def predictor_class(cls):
        return DummyJudge


class DummyJudge(Predictor):
    """
    Example judge.

    A judge is any model that predicts from the outcome of an evaluation task. In this example,
    we just average out the prediction from the previous (dummy) predictor
    """

    def __init__(self, config: DummyPredictorConfig, **kwargs):
        self.config = config

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
        num_generations: int = 1,
        **kwargs,
    ) -> Sequence[Prediction]:
        assert isinstance(prompts, Iterable)
        preds: List[Prediction] = []
        for seq in prompts:
            # The outcome of the dummy task is {"metrics": list, "prediction_embed": list, "prompts": list}
            # The judge will calculate the L2 distance between the means of the prediction and inputs previous
            # task
            assert "prompts" in seq and "prediction_embed" in seq
            preds_mean = torch.mean(torch.stack(seq["prediction_embed"]))  # type: ignore
            preds.append(Prediction(text="judge", embed=preds_mean, tokens=[1]))
        return preds

    @staticmethod
    def from_config(config: DummyJudgeConfig, **kwargs) -> "DummyJudge":  # type: ignore
        return DummyJudge(config)  # type: ignore
