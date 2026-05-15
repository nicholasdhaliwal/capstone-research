#  Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
#

import logging
import os
import random
import subprocess
import warnings
from datetime import timedelta
from functools import partial
from typing import Any, List, Literal, Optional, Set, Tuple, Type

import submitit
import torch
import torch.distributed as dist
from fairseq2.gang import Gang, ProcessGroupGang
from fairseq2.logging import get_log_writer
from fairseq2.nn.fsdp import (
    FSDP_LOW_MEMORY_POLICY,
    FSDP_STANDARD_MEMORY_POLICY,
    FSDP_VERY_LOW_MEMORY_POLICY,
    FSDPMemoryPolicy,
    FSDPWrapPolicy,
)
from fairseq2.nn.transformer import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn import Module

logger = get_log_writer(__name__)


SUPPORTED_FSDP_MEMORY_POLICIES = Literal["standard", "low", "very_low"]
SUPPORTED_FSDP_WRAP_POLICIES = Literal["layer", "stack", "model"]


def get_fsdp_memory_policy(
    policy: SUPPORTED_FSDP_MEMORY_POLICIES = "standard",
) -> FSDPMemoryPolicy:
    fsdp_memory_policy: FSDPMemoryPolicy
    if policy == "standard":
        fsdp_memory_policy = FSDP_STANDARD_MEMORY_POLICY
    elif policy == "low":
        fsdp_memory_policy = FSDP_LOW_MEMORY_POLICY
    elif policy == "very_low":
        fsdp_memory_policy = FSDP_VERY_LOW_MEMORY_POLICY
    else:
        raise ValueError("Unsupported policy {policy}. Choose from {}")

    return fsdp_memory_policy


def get_fsdp_wrap_policy(
    model: Module, wrap_granularity: SUPPORTED_FSDP_WRAP_POLICIES = "layer"
) -> Tuple[Optional[FSDPWrapPolicy], Optional[List[Module]]]:
    """Return the FSDP wrap policy for ``model`` along with ignored modules.

    :param model:
        The model to be wrapped.
    :param wrap_granularity:
        The granularity at which to wrap modules of ``model``.

          - 'layer': Wraps individual layers (e.g. :class:`TransformerDecoderLayer`).
          - 'stack': Wraps layer stacks (e.g. :class:`TransformerDecoder`).
          - 'model': Wraps ``model`` only.

      Copied over from fs2 to experiment easily with fsdp wrap policies
    """
    if wrap_granularity == "model":
        return None, None

    kls: Set[Type[Module]]

    if wrap_granularity == "stack":
        kls = {TransformerEncoder, TransformerDecoder}
    elif wrap_granularity == "layer":
        kls = {
            TransformerEncoderLayer,
            TransformerDecoderLayer,
        }
    else:
        raise ValueError(
            f"`wrap_granularity` must be 'layer', 'stack', or 'model', but is '{wrap_granularity}' instead."
        )

    wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls=kls)

    return wrap_policy, None


def init_process_group(config: Any, logger: logging.Logger) -> Gang:
    if getattr(config, "use_submitit", True):
        try:
            submitit.helpers.TorchDistributedEnvironment().export(overwrite=True)
            os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

        except RuntimeError:
            warnings.warn(
                "looks like you are not in a submitit/stopes job. \
                 You probably want to override use_submitit=false",
                stacklevel=2,
            )

    timeout = timedelta(minutes=15)

    gang = ProcessGroupGang.init_default_process_group(
        ok_initialized=False,
        timeout=timeout,
    )
    logger.info(f"Initialized gang with default process group (timeout={timeout})")

    return gang


def is_torch_run() -> bool:
    return os.environ.get("TORCHELASTIC_RUN_ID") is not None


def is_slurm_job() -> bool:
    return "SLURM_JOB_ID" in os.environ


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


def get_master_port(job_id: int) -> Optional[int]:
    if is_torch_run():
        return int(os.environ["MASTER_PORT"])
    else:
        MIN_MASTER_PORT, MAX_MASTER_PORT = (20000, 60000)
        rng = random.Random(job_id)
        return rng.randint(MIN_MASTER_PORT, MAX_MASTER_PORT)


def init_torch_distributed(
    backend: str = "cpu:gloo,cuda:nccl",
    port: Optional[str] = None,
    max_attempt: int = 5,
) -> None:
    if dist.is_initialized():
        return
    os.environ["RANK"] = str(get_global_rank())
    os.environ["WORLD_SIZE"] = str(get_world_size())

    master_addr = get_master_addr()

    # Allow max_attempt to be set directly via os environment variable
    # TORCH_DISTRIBUTED_PORT_ATTEMPTS
    if os.environ.get("TORCH_DISTRIBUTED_PORT_ATTEMPTS", None):
        max_attempt = int(os.environ["TORCH_DISTRIBUTED_PORT_ATTEMPTS"])
    attempt = 0
    while True:
        try:
            os.environ["MASTER_ADDR"] = master_addr
            if port is None:
                port = str(
                    get_master_port(job_id=int(os.environ.get("SLURM_JOB_ID", -1)))
                )
            os.environ["MASTER_PORT"] = port
            local_rank = get_local_rank()
            if "nccl" in backend:
                torch.cuda.set_device(local_rank)
            timeout = timedelta(hours=10)
            dist.init_process_group(backend=backend, timeout=timeout)
            break
        except (dist.DistNetworkError, RuntimeError) as e:
            attempt += 1
            if attempt == max_attempt:
                raise RuntimeError(
                    "Failed to initialize torch.distributed after 5 max attempts"
                ) from e
