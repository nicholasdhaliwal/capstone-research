#  Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
#

import ctypes
from abc import abstractmethod
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Optional,
    Protocol,
    Sized,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

import torch
from omegaconf import DictConfig, OmegaConf

root_working_dir = Path(__file__).parent.parent.parent


def set_mkl_num_threads():
    """Setting mkl num threads to 1, so that we don't get thread explosion."""
    mkl_rt = ctypes.CDLL("libmkl_rt.so")
    mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(1)))


def working_dir_resolver(p: str):
    """The omegaconf resolver that translates a relative path to the absolute path"""
    return "file://" + str(root_working_dir.joinpath(p).resolve())


def setup_conf():
    """Register the common Hydra config groups used in LCM (for now only the launcher)"""
    from stopes.pipelines import config_registry  # noqa

    recipe_root = Path(__file__).parent.parent.parent / "recipes"
    config_registry["lcm-common"] = "file://" + str((recipe_root / "common").resolve())
    config_registry["lcm-root"] = "file://" + str(recipe_root.resolve())

    # Register omegaconf resovlers
    OmegaConf.register_new_resolver("realpath", working_dir_resolver, replace=True)


def torch_type(
    dtype: Optional[Union[str, torch.dtype]] = None,
) -> Optional[torch.dtype]:
    # Convert dtyp string from the checkpoint to torch.dtype
    # https://github.com/pytorch/pytorch/issues/40471
    if dtype is None:
        return None

    if isinstance(dtype, torch.dtype):
        return dtype

    _dtype = eval(dtype)  # type: ignore
    assert isinstance(_dtype, torch.dtype), f"Invalid dtype value: {dtype}"
    return _dtype


@runtime_checkable
class Batched(Sized, Protocol):
    """Abstract class for batched data"""

    @abstractmethod
    def __getitem__(self, i: int) -> Any: ...


T = TypeVar("T")


def promote_config(config: Union[T, DictConfig, Dict], config_cls: Type[T]) -> T:
    if isinstance(config, (Dict, DictConfig)):
        import dacite

        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config)  # type: ignore

        return dacite.from_dict(
            data_class=config_cls,
            data=config,  # type: ignore
            config=dacite.Config(cast=[Path]),  # type: ignore
        )
    else:
        assert isinstance(config, config_cls), f"Unknown config type: {type(config)}"
        return config


def batched(inputs: Iterable, batch_size=10000) -> Iterable:
    batch = []
    for line in inputs:
        batch.append(line)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch
