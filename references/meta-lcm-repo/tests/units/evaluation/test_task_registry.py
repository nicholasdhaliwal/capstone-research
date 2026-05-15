# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# type: ignore

from dataclasses import dataclass

import pytest

from lcm.evaluation.tasks import TaskConfig, TaskRegistry
from lcm.evaluation.utils.data_utils import EvalDataLoader


@pytest.fixture()
def reset_task_registry():
    registry = TaskRegistry._REGISTRY
    TaskRegistry.reset()
    yield
    TaskRegistry._REGISTRY = registry


@dataclass
class TestTaskConfig(TaskConfig):
    foo: int = 0
    bar: str = "qux"


@pytest.mark.usefixtures("reset_task_registry")
def test_new_tasks() -> None:
    TaskRegistry.register(
        "task0", lambda: TestTaskConfig(None), data_loader_type=EvalDataLoader
    )
    TaskRegistry.register(
        "task1",
        lambda foo: TestTaskConfig(None, foo=foo),
        data_loader_type=EvalDataLoader,
    )
    TaskRegistry.register(
        "task2",
        lambda bar: TestTaskConfig(None, bar=bar),
        data_loader_type=EvalDataLoader,
    )

    assert TaskRegistry.names() == {"task0", "task1", "task2"}, TaskRegistry.names()
    assert TaskRegistry.get_config("task0") == TestTaskConfig(None, foo=0, bar="qux")
    assert TaskRegistry.get_config("task1", foo=4) == TestTaskConfig(None, foo=4)
    assert TaskRegistry.get_config("task2", bar="waldo") == TestTaskConfig(
        None, bar="waldo"
    )
