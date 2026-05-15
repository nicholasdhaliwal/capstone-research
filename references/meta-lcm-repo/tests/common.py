# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

from typing import Any, List, Union

import torch
from fairseq2.typing import Device
from torch import Tensor

# The default device that tests should use. Note that pytest can change it based
# on the provided command line arguments.
device = Device("cpu")


# The default deugging flag is False; Pytest can turn debuggin on
# via command line arguments `pytest --debug-training`
DEBUG = False


def assert_close(a: Tensor, b: Union[Tensor, List[Any]]) -> None:
    """Assert that ``a`` and ``b`` are element-wise equal within a tolerance."""
    if not isinstance(b, Tensor):
        b = torch.tensor(b, device=device, dtype=a.dtype)

    torch.testing.assert_close(a, b)  # type: ignore[attr-defined]


def assert_equal(a: Tensor, b: Union[Tensor, List[Any]]) -> None:
    """Assert that ``a`` and ``b`` are element-wise equal."""
    if not isinstance(b, Tensor):
        b = torch.tensor(b, device=device, dtype=a.dtype)

    torch.testing.assert_close(a, b, rtol=0, atol=0)  # type: ignore[attr-defined]
