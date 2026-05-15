# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

import importlib
import logging
import os
import subprocess
from logging import Logger
from types import ModuleType
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from fairseq2.gang import FakeGang, Gang, ProcessGroupGang
from fairseq2.typing import Device

from lcm.utils.distributed import init_torch_distributed, is_slurm_job, is_torch_run

from ..api import AverageMetric

_LOGGER: Logger = logging.getLogger()


def get_mpu_module() -> ModuleType:
    default_module_name = "fairscale.nn.model_parallel.initialize"
    module_name = os.environ.get("MPU_MODULE", default_module_name)
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        rank_zero_warn(f"Error when importing {module_name}: {e}")
        return importlib.import_module(default_module_name)


def init_model_parallel(model_parallel_size: int) -> None:
    mpu = get_mpu_module()
    if not dist.is_initialized():
        init_torch_distributed()

    if not mpu.model_parallel_is_initialized():
        mpu.initialize_model_parallel(model_parallel_size)


def reinit_model_parallel() -> None:
    from fairscale.nn.model_parallel import destroy_model_parallel  # type: ignore
    from llm_inference.models import nccl  # type: ignore

    destroy_model_parallel()
    nccl.get_mp_group.cache_clear()
    nccl.get_mp_src_rank.cache_clear()
    nccl.get_mp_rank.cache_clear()
    nccl.get_mp_world_size.cache_clear()


def clear_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_gang(max_attempt: int = 5) -> Gang:
    """Simple setup of fairseq2 Gang that works directly with torch.distributed"""
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        init_torch_distributed(backend="gloo", max_attempt=max_attempt)
        return FakeGang()

    if not dist.is_initialized():
        init_torch_distributed(backend="nccl", max_attempt=max_attempt)

    rank = get_local_rank()
    device = Device("cuda", index=rank)

    # As of PyTorch 2.0, FSDP fails to work if the default device is not set.
    torch.cuda.set_device(device)

    pg = get_dp_group()
    if pg:
        return ProcessGroupGang(pg=pg, device=device)
    else:
        return FakeGang()


def get_global_rank() -> int:
    if dist.is_initialized():
        return dist.get_rank()
    if is_torch_run():
        return int(os.environ["RANK"])
    if is_slurm_job():
        return int(os.environ["SLURM_PROCID"])
    return 0


def get_local_rank() -> int:
    if is_torch_run():
        return int(os.environ["LOCAL_RANK"])
    if is_slurm_job():
        return int(os.environ["SLURM_LOCALID"])
    return 0


def get_world_size() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    if is_torch_run():
        return int(os.environ["WORLD_SIZE"])
    if is_slurm_job():
        return int(os.environ["SLURM_NTASKS"])
    return 1


def get_master_addr() -> str:
    if is_torch_run():
        return os.environ["MASTER_ADDR"]
    if is_slurm_job():
        hostnames = subprocess.check_output(
            ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]]
        )
        return hostnames.split()[0].decode("utf-8")
    return "127.0.0.1"


def get_dp_rank() -> int:
    mpu = get_mpu_module()
    if mpu.model_parallel_is_initialized():
        return mpu.get_data_parallel_rank()
    return get_global_rank()


def get_dp_size() -> int:
    mpu = get_mpu_module()
    if mpu.model_parallel_is_initialized():
        return mpu.get_data_parallel_world_size()
    return get_world_size()


def get_mp_rank() -> int:
    mpu = get_mpu_module()
    if mpu.model_parallel_is_initialized():
        return mpu.get_model_parallel_rank()
    return 0


def get_mp_size() -> int:
    mpu = get_mpu_module()
    if mpu.model_parallel_is_initialized():
        return mpu.get_model_parallel_world_size()
    return 1


def get_dp_group() -> Optional[dist.ProcessGroup]:
    mpu = get_mpu_module()
    if mpu.model_parallel_is_initialized():
        return mpu.get_data_parallel_group()
    if dist.is_initialized():
        return dist.group.WORLD
    return None


def get_mp_group() -> Optional[dist.ProcessGroup]:
    mpu = get_mpu_module()
    if mpu.model_parallel_is_initialized():
        return mpu.get_model_parallel_group()
    return None


def rank_zero_debug(*args: Any, logger: Optional[Logger] = None, **kwargs: Any) -> None:
    if get_global_rank() != 0:
        return
    logger = logger or _LOGGER
    logger.debug(*args, **kwargs)


def rank_zero_info(*args: Any, logger: Optional[Logger] = None, **kwargs: Any) -> None:
    if get_global_rank() != 0:
        return
    logger = logger or _LOGGER
    logger.info(*args, **kwargs)


def rank_zero_warn(*args: Any, logger: Optional[Logger] = None, **kwargs: Any) -> None:
    if get_global_rank() != 0:
        return
    logger = logger or _LOGGER
    logger.warning(*args, **kwargs)


def rank_zero_error(*args: Any, logger: Optional[Logger] = None, **kwargs: Any) -> None:
    if get_global_rank() != 0:
        return
    logger = logger or _LOGGER
    logger.error(*args, **kwargs)


def mp_rank_zero_warn(
    *args: Any, logger: Optional[Logger] = None, **kwargs: Any
) -> None:
    if get_mp_rank() != 0:
        return
    logger = logger or _LOGGER
    logger.warning(*args, **kwargs)


def mp_rank_zero_debug(
    *args: Any, logger: Optional[Logger] = None, **kwargs: Any
) -> None:
    if get_mp_rank() != 0:
        return
    logger = logger or _LOGGER
    logger.debug(*args, **kwargs)


def mp_rank_zero_info(
    *args: Any, logger: Optional[Logger] = None, **kwargs: Any
) -> None:
    if get_mp_rank() != 0:
        return
    logger = logger or _LOGGER
    logger.info(*args, **kwargs)


def rank_zero_print(*args: Any, **kwargs: Any) -> None:
    if get_global_rank() != 0:
        return
    print(*args, **kwargs)


def all_reduce(
    tensor: torch.Tensor, op: str = "sum", group: Optional[dist.ProcessGroup] = None
) -> torch.Tensor:
    """All-reduces single scalar value if torch distributed is in use."""
    if not dist.is_available() or not dist.is_initialized():
        return tensor

    dop = None
    if op == "sum" or op == "mean":
        dop = dist.ReduceOp.SUM
    elif op == "min":
        dop = dist.ReduceOp.MIN
    elif op == "max":
        dop = dist.ReduceOp.MAX
    elif op == "product":
        dop = dist.ReduceOp.PRODUCT

    dist.all_reduce(tensor, op=dop, group=group)
    if op == "mean":
        tensor /= dist.get_world_size(group)
    return tensor


def gather_object(
    result: List[Any], dst: int = 0, group: Optional[dist.ProcessGroup] = None
) -> List[Any]:
    if not dist.is_initialized() or dist.get_world_size(group) == 1:
        return result
    import torch.distributed.distributed_c10d as c10d

    global_dst = c10d.get_global_rank(group or c10d.GroupMember.WORLD, dst)  # type: ignore
    output = [None for _ in range(dist.get_world_size(group))]
    results = output if get_global_rank() == global_dst else None
    dist.gather_object(result, results, dst=global_dst, group=group)
    return [item for res in results for item in res] if results else None  # type: ignore


def all_gather_object(
    object_list: List[Any], group: Optional[dist.ProcessGroup] = None
) -> List[Any]:
    if not dist.is_initialized() or dist.get_world_size(group) == 1:
        return object_list
    results = [None for _ in range(dist.get_world_size(group))]
    dist.all_gather_object(results, object_list, group=group)
    return [item for res in results for item in res]  # type: ignore


def broadcast_object_list(
    object_list: List[Any], src: int = 0, group: Optional[dist.ProcessGroup] = None
) -> None:
    if not dist.is_initialized() or dist.get_world_size(group) == 1:
        return
    import torch.distributed.distributed_c10d as c10d

    global_src = c10d.get_global_rank(group or c10d.GroupMember.WORLD, src)  # type: ignore
    dist.broadcast_object_list(object_list, src=global_src, group=group)


def mean_reduce_dict(
    data: Dict[str, List[float]],
    device: Union[str, torch.device] = "cuda",
    group: Optional[dist.ProcessGroup] = None,
) -> Dict[str, AverageMetric]:
    avg_results: Dict[str, AverageMetric] = {}
    for k, v in data.items():
        # If v is a list of list aggregate by position - hacky workaround for l2_distance in the sentpred task
        if isinstance(v[0], list):
            maxlen = max([len(x) for x in v])
            for pos in range(maxlen):
                vpos = [x[pos] for x in v if x[pos] == x[pos]]  # ignore nan
                stats = [sum(vpos), len(vpos), sum(x * x for x in vpos)]
                if not dist.is_initialized() or dist.get_world_size(group) == 1:
                    sum_v, len_v, sum_v2 = stats
                else:
                    tensor = torch.tensor(stats, device=device, dtype=torch.float32)
                    sum_v, len_v, sum_v2 = all_reduce(
                        tensor, op="sum", group=group
                    ).tolist()
                avg_results[f"{k}_{pos}"] = AverageMetric(
                    avg=sum_v / max(len_v, 1),
                    count=int(len_v),
                    square=sum_v2 / max(len_v, 1),
                )

        else:  # previous behaviour assuming a list of scalars to aggregate
            v = [x for x in v if x == x]  # ignore nan
            stats = [sum(v), len(v), sum(x * x for x in v)]
            if not dist.is_initialized() or dist.get_world_size(group) == 1:
                sum_v, len_v, sum_v2 = stats
            else:
                tensor = torch.tensor(stats, device=device, dtype=torch.float32)
                sum_v, len_v, sum_v2 = all_reduce(
                    tensor, op="sum", group=group
                ).tolist()
            avg_results[k] = AverageMetric(
                avg=sum_v / max(len_v, 1),
                count=int(len_v),
                square=sum_v2 / max(len_v, 1),
            )
    return avg_results
