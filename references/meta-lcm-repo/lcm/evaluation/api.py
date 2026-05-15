# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#


import abc
import gc
from dataclasses import dataclass, fields
from enum import Enum
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

import numpy as np
import torch
from fairseq2.gang import Gang
from fairseq2.nn.utils.module import to_device
from fairseq2.typing import CPU, DataType, Device
from numpy.random import RandomState

from lcm.datasets.configs import DatasetConfig
from lcm.utils.common import Batched, torch_type

Example = Dict[str, Any]
ExampleFn = Callable[[Example], Example]

# postprocessed prediction, normally contains the best hypothesis from the prediction text
PREDICTION_COLUMN = "prediction"

GROUND_TRUTH_COLUMN = "targets"

# prediction texts, normally this is a list that contains all hypotheses
PREDICTION_TEXT_COLUMN = "prediction_texts"

# prediction token ids, normally this is a list that contains all hypotheses
PREDICTION_TOKEN_ID_COLUMN = "prediction_token_ids"

# prediction tokens, normally this is a list that contains all hypotheses
PREDICTION_TOKEN_COLUMN = "prediction_tokens"

# prediction (sentence / token) embeddings, normally this is a list that
# contains all hypotheses
PREDICTION_EMBED_COLUMN = "prediction_embed"


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    role: Role
    content: str

    def __repr__(self) -> str:
        return f"{self.role.value.upper()}: {self.content}"

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role.value, "content": self.content}


Prompt = Union[str, List[int], torch.Tensor, Dict[str, Any]]
Predictor_T = TypeVar("Predictor_T", bound="Predictor")
Prompts = Union[Sequence[Prompt], Dict[str, Prompt], Batched]


@dataclass
class Prediction:
    text: Union[str, List[str]]
    embed: Optional[torch.Tensor] = None
    tokens: Optional[Union[List[int], List[str]]] = None
    logprobs: Optional[List[float]] = None
    text_offsets: Optional[List[int]] = None
    token_ids: Optional[List[int]] = None


@runtime_checkable
class Predictor(Protocol):
    """API for the predictor"""

    @staticmethod
    @abc.abstractmethod
    def from_config(
        config: "PredictorConfig",
        **kwargs: Any,
    ) -> "Predictor": ...

    @abc.abstractmethod
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
    ) -> Sequence[Prediction]: ...


@dataclass
class ScorerConfig:
    """Config to set up the model-based metrics dynamically"""

    scorer_type: str
    """Type of the scorer, as registered in lcm.evaluation.metrics._METRICS_CONFIG_MAP"""

    model_name: Optional[str] = None
    """Model name to load the scorer"""

    inputs: Optional[Tuple[str, ...]] = None
    """Name of input columns the model will use to calculate the metrics"""

    params: Optional[Dict[str, Any]] = None
    """Additional parameters"""


class Scorer:
    """API for a model-based metrics"""

    model: torch.nn.Module
    """The model used to calculate the metrics"""

    model_name: str
    """name to load the model"""

    inputs: Tuple[str, ...]
    """list of columns from the input examples that the model uses"""

    outputs: Tuple[str, ...]
    """Names of the metrics to report in the output results"""

    def __init__(
        self,
        model_name: str = "",
        inputs: Union[Tuple[str, ...], str] = PREDICTION_COLUMN,
        outputs: Union[Tuple[str, ...], str] = "",
        gang: Optional[Gang] = None,
        device: Device = CPU,
        dtype: Optional[DataType] = None,
        **kwargs,
    ):
        if gang is not None:
            device = gang.device
        self.device = device

        self.dtype = torch_type(dtype)
        self.model_name = model_name
        if isinstance(inputs, str):
            self.inputs = (inputs,)
        else:
            self.inputs = inputs
        if not outputs:
            outputs = self.default_outputs(model_name)
        if isinstance(outputs, str):
            self.outputs = (outputs,)
        else:
            self.outputs = outputs
        self.kwargs = kwargs
        self.init_model()

    def __init_subclass__(cls, **kwargs):
        def init_decorator(previous_init):
            @wraps(previous_init)
            def new_init(self, *args, **kwargs):
                previous_init(self, *args, **kwargs)
                if isinstance(self, cls):
                    self.__post_init__()

            return new_init

        cls.__init__ = init_decorator(cls.__init__)  # type: ignore

    def __post_init__(self):
        if hasattr(self, "inputs") and isinstance(self.inputs, str):
            self.inputs = (self.inputs,)
        if hasattr(self, "model") and self.model:
            self.model.eval()
            to_device(self.model, self.device)

    @abc.abstractmethod
    def init_model(self) -> None:
        raise NotImplementedError(
            f"`init_model` must be implemented in {self.__class__.__name__}"
        )

    @classmethod
    def default_outputs(cls, model_name: str) -> Tuple[str, ...]:
        """Generate the default output name for the model"""
        return (cls.__name__ + "-" + model_name.split("/")[-1],)

    @abc.abstractmethod
    def score_texts(
        self,
        texts: Sequence[str],
        references: Optional[Sequence[str]] = None,
        show_progress: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]: ...

    def __call__(
        self, examples: Sequence[Example], show_progress: bool = False
    ) -> Dict[str, Any]:
        texts = [x[self.inputs[0]] for x in examples]
        if len(self.inputs) > 1:
            references = [x[self.inputs[1]] for x in examples]
        else:
            references = None

        scores = self.score_texts(
            texts, references=references, show_progress=show_progress
        )

        if np.ndim(scores) == 1:
            scores = scores[None, :].T
        assert len(self.outputs) == scores.shape[1] , f"Expect {len(self.outputs)} metrics, get {scores.shape[1]}"  # fmt: skip
        metrics = {}
        for c in range(scores.shape[1]):
            if self.outputs[c]:  # the metric should be reported
                metrics[self.outputs[c]] = scores[:, c]

        return metrics

    def free_gpu(self):
        gc.collect()
        if torch.cuda.is_available():
            self.model.cpu()
            torch.cuda.empty_cache()


@dataclass
class AverageMetric:
    """
    Average metric with confidence interval.

    avg is the mean of a list of values
    count is the length of this list
    square is the mean of the squares of the values
    avg_ci_fn is a function applied to the bounds of the confidence interval
    """

    avg: float
    count: int
    square: float
    avg_ci_fn: Optional[Callable] = None

    @property
    def value(self):
        return self.avg_ci_fn(self.avg) if self.avg_ci_fn else self.avg

    def update(self, value: float, count: int, square: Optional[float] = None) -> None:
        self.avg = (self.avg * self.count + value * count) / (self.count + count)
        if square is None:
            assert count == 1
            square = value**2
        self.square = (self.square * self.count + square * count) / (self.count + count)
        self.count += count

    def compute_ci(
        self, confidence_level: float = 0.95
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        returns bounds of confidence interval: ci_lb ('lower_bound') and ci_ub ('upper_bound').
        Confidence interval is computed with error margins:
        z * s / sqrt(n), where:
        - P(-z <= X <= z) = confidence_level and X follows a t student low with self.count - 1 parameters.
        - s is the unbiased std estimate: (1/(n-1) sum((xi - mean(xi) ** 2))) ** 0.5

        example: first 100 integers as metric_values and confidence_level = 0.95:
        >>> avg_m = AverageMetric(0, 0, 0)
        >>> for i in range(100):
        >>>     avg_m.update(value=i, count=1)
        >>> avg_m.compute_ci() # mean is 49.5, std is 29.0115, self.count = 100, z = 1.98
        >>> (43.743, 55.257)
        """
        from scipy.stats import t as student_t  # type: ignore

        if self.count < 2:
            return None, None

        std = (self.count / (self.count - 1) * (self.square - (self.avg) ** 2)) ** 0.5
        scale = std / (self.count**0.5)
        lb, ub = student_t.interval(confidence_level, self.count - 1, loc=self.avg, scale=scale)  # fmt: skip
        if self.avg_ci_fn:
            lb, ub = self.avg_ci_fn(lb), self.avg_ci_fn(ub)
        return (lb, ub)


@dataclass
class TaskResult:
    metrics: Dict[str, AverageMetric]
    raw_results: Optional[List[Dict[str, Any]]] = None


@dataclass
class TaskConfig:
    """
    Generic task config.

    Each task is associated with a dataset and the data loading class, which defines
    how the data loader is constructed dynamically from the dataset.
    """

    dataset: DatasetConfig


def parse_task_configs(config_cls: Type[TaskConfig], **kwargs) -> Dict:
    """Helper functions to extract the config related to a Task config"""
    return {k.name: kwargs[k.name] for k in fields(config_cls) if k.name in kwargs}


TPredictor = TypeVar("TPredictor")


@dataclass
class EOSConfig:
    """
    Config of how (untrained) EOS is represented in the (generation)
    task. It can be a string or a saved vector

    TODO: The EOS config in evalation seems artificial. Maybe we put this into
    lcm.inference

    """

    text: Optional[str] = None
    ckpt: Optional[str] = None


@dataclass
class PredictorConfig:
    @classmethod
    @abc.abstractmethod
    def predictor_class(cls) -> TPredictor:  # type: ignore
        pass


class UnsupportedPredictorException(Exception):
    pass


class Task(abc.ABC):
    @abc.abstractmethod
    def run(
        self,
        predictor: Predictor,
        random_state: Optional[RandomState] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> TaskResult:
        """Run the task for a given predictor"""
        ...


MetricFn = Callable[[Example], Dict[str, float]]
GlobalMetricFn = Callable[[List[Example]], Dict[str, AverageMetric]]
ParallelMetricFn = Callable[[List[Example], bool], List[Dict[str, float]]]
AggregationFn = Callable[
    [Dict[str, Dict[str, AverageMetric]]], Dict[str, Dict[str, AverageMetric]]
]
