# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Union

import hydra
import torch

from lcm.evaluation.utils.distributed import get_local_rank
from lcm.utils.common import torch_type

logger = logging.getLogger(__name__)


def infer_cache_dir() -> Optional[str]:
    if os.getenv("HF_HOME"):
        # or use `HF_HUB_CACHE`
        logger.info(f"Using env HF_HOME={os.environ['HF_HOME']}")
        return f"{os.environ['HF_HOME']}/hub"
    return None


def infer_offline_model_dir(
    dtype: Optional[Union[str, torch.dtype]] = None,
) -> Optional[str]:
    if os.getenv("HF_DOWNLOAD"):
        return os.environ["HF_DOWNLOAD"]
    return None


def download_model(
    model_name: str,
    model_class: str = "AutoModelForCausalLM",
    tokenizer_class: str = "AutoTokenizer",
    model_dtype: str = "torch.float32",
    model_dir: Optional[str] = None,
) -> None:
    token = os.getenv("HF_AUTH_TOKEN")
    assert token is not None, "set HF_AUTH_TOKEN path please."

    from huggingface_hub import login

    login(token=token)

    if model_dir is None:
        model_dir = infer_offline_model_dir(model_dtype)
    assert isinstance(model_dir, str), "Unknown model_dir"

    start_time = time.time()
    dtype = torch_type(model_dtype)
    model_cls = hydra.utils.get_class(f"transformers.{model_class}")
    tokenizer_cls = hydra.utils.get_class(f"transformers.{tokenizer_class}")

    tokenizer = tokenizer_cls.from_pretrained(model_name)  # type: ignore
    model = model_cls.from_pretrained(  # type: ignore
        model_name, trust_remote_code=True, torch_dtype=dtype
    )

    Path(model_dir).joinpath(model_name).mkdir(parents=True, exist_ok=True)
    print(f"Saving model to {model_dir}")

    model.save_pretrained(Path(model_dir).joinpath(model_name))
    tokenizer.save_pretrained(Path(model_dir).joinpath(model_name))
    print(f"Finish downloading in {time.time() - start_time} seconds.")


def infer_hf_device_memory(model_parallel: int) -> Dict[int, int]:
    """Infers maximum memory allocation for each GPU in model parallel group."""
    gpus_per_node = torch.cuda.device_count()
    start = model_parallel * get_local_rank() % gpus_per_node
    end = start + model_parallel
    max_memory = {
        i: torch.cuda.mem_get_info(i)[0] if start <= i < end else 0
        for i in range(gpus_per_node)
    }
    return max_memory


if __name__ == "__main__":
    from fire import Fire

    Fire(download_model)
