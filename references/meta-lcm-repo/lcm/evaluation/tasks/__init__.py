# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

import importlib
import inspect
import os
from copy import deepcopy

# flake8: noqa
from dataclasses import asdict
from functools import partial
from itertools import product
from typing import (
    TYPE_CHECKING,
    AbstractSet,
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
)

from lcm.datasets.configs import DatasetConfig, EvaluationDataLoadingConfig
from lcm.evaluation.api import Task, TaskConfig
from lcm.evaluation.utils.data_utils import (
    EvalDataLoader,
    JSONTestDataLoader,
)
from lcm.utils.common import promote_config, torch_type
from lcm.utils.data_utils import update_dataclass

if TYPE_CHECKING:
    from fairseq2.gang import Gang


def _filter_args(parameters, **kwargs):
    has_kwargs = (
        len({k: v for k, v in parameters.items() if v.kind == v.VAR_KEYWORD}) > 0
    )
    if has_kwargs:
        return kwargs
    else:
        return {k: v for k, v in kwargs.items() if k in parameters}


def instantiate(cls_name: Type, **kwargs):
    fn_to_check = cls_name.__init__
    defaults = inspect.signature(fn_to_check).parameters
    args = _filter_args(defaults, **kwargs)
    return cls_name(**args)


class TaskRegistry:
    _REGISTRY: Dict[str, Tuple[Callable[..., TaskConfig], Type[EvalDataLoader]]] = {}

    @staticmethod
    def names() -> AbstractSet[str]:
        return TaskRegistry._REGISTRY.keys()

    @staticmethod
    def register(
        name: str,
        func: Callable[..., TaskConfig],
        data_loader_type: Type[EvalDataLoader],
    ) -> None:
        if name in TaskRegistry._REGISTRY:
            raise ValueError(f"Config for task {name} already exists.")
        TaskRegistry._REGISTRY[name] = (func, data_loader_type)

    @staticmethod
    def get_config(name: str, **kwargs: Any) -> TaskConfig:
        if name not in TaskRegistry._REGISTRY:
            raise ValueError(f"No task registered under the name {name}")
        fn_to_check = TaskRegistry._REGISTRY[name][0]
        defaults = inspect.signature(fn_to_check).parameters
        args = _filter_args(defaults, **kwargs)
        return fn_to_check(**args)

    @staticmethod
    def get_task_args(name: str) -> Dict[str, inspect.Parameter]:
        defaults = inspect.signature(TaskRegistry._REGISTRY[name][0]).parameters
        # Ignore the `**kwargs` in the function signature
        defaults = {k: v for k, v in defaults.items() if v.kind != v.VAR_KEYWORD}  # type: ignore
        return defaults  # type: ignore

    @staticmethod
    def get_dataloader_type(name: str) -> Type[EvalDataLoader]:
        try:
            return TaskRegistry._REGISTRY[name][1]
        except KeyError as err:
            raise KeyError(f"Task {name} is not registered yet") from err

    @staticmethod
    def reset() -> None:
        TaskRegistry._REGISTRY = {}


def register_task(
    name: str,
    parameters: Optional[Dict[Union[str, Tuple[str, ...]], Iterable[Any]]] = None,
    data_loader_type: Type[EvalDataLoader] = JSONTestDataLoader,
) -> Callable[[Callable[..., TaskConfig]], Callable[..., TaskConfig]]:
    """Register the task name with the decorated task configuration callable."""

    def register(callable: Callable[..., TaskConfig]) -> Callable[..., TaskConfig]:
        if parameters is None:
            TaskRegistry.register(name, callable, data_loader_type)
        else:
            for values in product(*parameters.values()):
                param_dict: Dict[str, Any] = {}
                for keys, value in zip(parameters.keys(), values):
                    if isinstance(keys, tuple):
                        param_dict.update(zip(keys, value))
                    else:
                        param_dict[keys] = value
                task_name = name.format(**param_dict)
                TaskRegistry.register(
                    task_name, partial(callable, **param_dict), data_loader_type
                )
        return callable

    return register


def build_task(
    task_config: TaskConfig,
    data_loading_config: EvaluationDataLoadingConfig,
    data_loader_type: Type[EvalDataLoader],
    gang: Optional["Gang"] = None,
    dataset_config: Optional[DatasetConfig] = None,
    **kwargs,
) -> Task:
    config_cls_name = task_config.__class__.__name__
    try:
        module = __import__(
            task_config.__class__.__module__, fromlist=[config_cls_name]
        )
        cls_name = config_cls_name.replace("Config", "")
        task_cls = getattr(module, cls_name)

        # Build the dataset config.
        # If no dataset config is provided by the user at runtime, we use the
        # default config defined in each task registering function.
        # Otherwise we override the default configs with user-defined values
        dataset = asdict(task_config.dataset) if task_config.dataset else {}
        if dataset_config:
            update_dataclass(dataset_config, dataset)
        else:
            dataset_cls = data_loader_type.dataset_config()
            dataset_config = dataset_cls(**dataset)

        dtype = torch_type(kwargs.get("dtype", None))
        if dtype:
            data_loader = instantiate(
                data_loader_type,
                data_config=data_loading_config,  # type: ignore
                dataset=dataset_config,
                gang=gang,
                dtype=dtype,
            )
        else:
            data_loader = instantiate(
                data_loader_type,
                data_config=data_loading_config,  # type: ignore
                dataset=dataset_config,
                gang=gang,
            )
        return instantiate(
            task_cls, config=task_config, data_loader=data_loader, gang=gang, **kwargs
        )
    except ImportError:
        raise ValueError("No task class found for {config_cls_name}")


def build_dataset_config(
    task_name: str,
    dataset_args: Optional[Mapping[str, Any]] = None,
) -> DatasetConfig:
    """Create a dataset config instance corresponding to a `task_name` and has custom
    arguments defined in `dataset_args`
    """
    dataset_cls = TaskRegistry.get_dataloader_type(task_name).dataset_config()
    if not dataset_args:
        return dataset_cls()

    args = deepcopy(dataset_args)
    dataset_config: DatasetConfig = promote_config(args, dataset_cls)

    # If the custom dataset arguments are specified, we must ensure this takes precedence
    # over the default values defined inside each task. This is done by setting the flag
    # `silent_freeze` to surpress all subsequent inline updates of dataset values within
    # the "register_task" function
    dataset_config.freeze()

    return dataset_config


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        module = file[: file.find(".py")]
        importlib.import_module("lcm.evaluation.tasks." + module)
