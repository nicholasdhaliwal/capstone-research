# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

from collections.abc import MutableMapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import torch
from fairseq2.gang import Gang
from fairseq2.logging import get_log_writer
from fairseq2.metrics import (
    MetricBag,
    format_as_float,
    format_as_int,
    format_as_seconds,
)
from fairseq2.metrics.recorder import (
    MetricRecorder,
    _metric_formatters,
    register_metric_formatter,
)
from fairseq2.typing import override
from torch import Tensor
from torch.cuda import _get_device_index
from torcheval.metrics import Max, Mean, Sum, Throughput

logger = get_log_writer(__name__)

format_as_percent = partial(format_as_int, postfix="%")


def flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = ".") -> Dict:
    """
    A helper function to flatten nested dictionaries
    Example. With a training config like
        config = {
        'data': {
            'training': {'batch_size': 10},
            'validation': {'batch_size': 2}
            },
        'model': {'model_dim': 1024},
        'use_fsdp': True
        }
        The flat config will be:
            {
            'data.training.batch_size': 10,
            'data.validation.batch_size': 2,
            'model.model_dim': 1024,
            'use_fsdp': True
            }
        This helper is used to convert our nested training config into a flat
        dictionary for Tensoarboard's HParams conusmption

    """
    items: List = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_allocated_gpu_memory(device):
    """
    Get allocated memory in GiB for GPU devices
    """
    if device.type == "cpu":
        return 0, 0
    device = _get_device_index(device, optional=True)
    memory_stats = torch.cuda.memory_stats(device=device)
    current_usage = memory_stats["allocated_bytes.all.current"] / (1024**3)
    peak_usage = memory_stats["allocated_bytes.all.peak"] / (1024**3)
    return current_usage, peak_usage


@dataclass
class LossTerm:
    """Dataclass for a batch loss term"""

    value: Tensor
    """The final loss to be optimized"""

    batch_size: int

    num_target_elements: Union[int, float]

    summands: Dict[str, Tuple[Any, Any]] = field(default_factory=lambda: {})
    """A dictionary of loss terms to record. Each term is a tuple of (loss, number of elements)
       The second term is optional; if None, we will use `num_target_elements` when aggregating"""


class LCMMetricBag(MetricBag):
    """Holds the common metrics of an LCM."""

    loss: Mean
    batch_size: Sum
    elements_per_batch: Mean
    elements_per_second: Throughput
    num_target_elements: Sum
    total_num_target_elements: Sum

    grad_norm: Mean

    def __init__(
        self, gang: Gang, loss_summands: Sequence[str] = [], reduction: str = "sum"
    ) -> None:
        """
        :param gang:
            The gang to sync metrics across all processes.
        """
        super().__init__(gang)

        # temporary fix:

        self.reduction = reduction

        d = gang.device

        # A temporary solution to track as many loss terms as we explore
        self.loss_summands = loss_summands

        self.register_metric("loss", Mean(device=d), persistent=False)

        # this is the effective batch size
        self.register_metric("batch_size", Sum(device=d), persistent=False)

        self.register_metric("elements_per_batch", Mean(device=d), persistent=False)

        self.register_metric(
            "elements_per_second", Throughput(device=d), persistent=False
        )

        self.register_metric("gpu_memory_usage", Max(device=d), persistent=False)

        self.register_metric("gpu_peak_memory_usage", Max(device=d), persistent=False)

        # self.register_metric("ram_percentage", Max(device=d), persistent=False)

        # self.register_metric("cpu_percentage", Max(device=d), persistent=False)

        for summand in self.loss_summands:
            self.register_metric(summand, Mean(device=d), persistent=False)

        # The number of target tokens in a parallel batch. Used for computing throughput
        self.register_metric("num_target_elements", Sum(device=d), persistent=False)

        # The total_num_target_elements is persistent and is supposed to track the
        # total number of tokens consumed since training started
        self.total_num_target_elements = Sum(device=d)

    def register_adaln_metric(self, module_name: str):
        for block in ["mha", "ffn"]:
            for tensor in [
                "shift",
                "scale",
                "gate",
            ]:
                self.register_metric(
                    f"{module_name}_{block}_{tensor}_mean",
                    Mean(device=self._gang.device),
                    persistent=False,
                )
                self.register_metric(
                    f"{module_name}_{block}_{tensor}_std",
                    Mean(device=self._gang.device),
                    persistent=False,
                )
                # formatters
                register_metric_formatter(
                    f"{module_name}_{block}_{tensor}_mean",
                    f"{module_name}_{block}_{tensor}_mean",
                    1000,
                    format_as_float,
                )
                register_metric_formatter(
                    f"{module_name}_{block}_{tensor}_std",
                    f"{module_name}_{block}_{tensor}_std",
                    1000,
                    format_as_float,
                )

    def register_module_metric(self, module_name: str):
        for tensor in [
            "input_gradient",
            "output_gradient",
            "input_activations",
            "output_activations",
        ]:
            self.register_metric(
                f"{module_name}_{tensor}_mean",
                Mean(device=self._gang.device),
                persistent=False,
            )
            self.register_metric(
                f"{module_name}_{tensor}_std",
                Mean(device=self._gang.device),
                persistent=False,
            )
            # formatters
            register_metric_formatter(
                f"{module_name}_{tensor}_mean",
                f"{module_name}_{tensor}_mean",
                1000,
                format_as_float,
            )
            register_metric_formatter(
                f"{module_name}_{tensor}_std",
                f"{module_name}_{tensor}_std",
                1000,
                format_as_float,
            )

    @torch.inference_mode()
    def update(
        self,
        losses: Sequence[LossTerm],
    ) -> None:
        """Update the metrics.

        :param output:
            The losses generated by the model for each batch
        :param elapsed_time:
            The total elapsed time to read and process batches
        """

        loss = torch.zeros((), dtype=torch.float64)

        loss_summands = {
            s: torch.zeros((), dtype=torch.float64) for s in self.loss_summands
        }
        # Denominator to normalize the loss summands, if -1,
        # we will default to normalizing with `num_target_elements`
        loss_summands_numel = {
            s: -torch.ones((), dtype=torch.long) for s in self.loss_summands
        }

        batch_size = torch.zeros((), dtype=torch.int64)

        num_target_elements = torch.zeros((), dtype=torch.int64)

        # Only in the case of using gradient accumulation that `losses` will be a non-singleton
        for batch_loss in losses:
            loss += float(batch_loss.value)

            for s in self.loss_summands:
                loss_term = batch_loss.summands.get(s, (0.0, None))
                loss_summands[s] += float(loss_term[0])
                if loss_term[1] is not None and not loss_term[1] == -1:
                    if loss_summands_numel[s] == -1:
                        loss_summands_numel[s] = torch.zeros((), dtype=torch.int64)
                    loss_summands_numel[s] += loss_term[1]

            batch_size += batch_loss.batch_size
            num_target_elements += batch_loss.num_target_elements

        # Misleading normalization in the metric bag with reduction == "mean"
        # Kept here for backward compatibility
        # Any normalization here is only for reporting and doesn't impact optimization
        if self.reduction == "sum":
            loss /= num_target_elements
            keys = list(loss_summands)
            for k in keys:
                denom = loss_summands_numel[k]
                if denom == -1:
                    denom = num_target_elements
                loss_summands[k] /= denom + 1e-6

        self.loss.update(loss, weight=num_target_elements)

        for s in loss_summands:
            weight = loss_summands_numel[s]
            if weight == -1:
                weight = num_target_elements
            getattr(self, s).update(loss_summands[s], weight=weight)

        self.batch_size.update(batch_size)

        self.elements_per_batch.update(num_target_elements)

        self.num_target_elements.update(num_target_elements)

        # update the cumulative metric
        self.total_num_target_elements.update(num_target_elements)

        # Get GPU memory usage
        gpu_memory_usage, gpu_peak_memory_usage = get_allocated_gpu_memory(
            self._gang.device
        )
        self.gpu_memory_usage.update(torch.tensor(gpu_memory_usage))
        self.gpu_peak_memory_usage.update(torch.tensor(gpu_peak_memory_usage))

    def reset_batch_metrics(self) -> None:
        """Reset the batch metrics to their initial state."""
        self.loss.reset()
        for s in self.loss_summands:
            getattr(self, s).reset()

        self.batch_size.reset()
        self.elements_per_batch.reset()
        self.elements_per_second.reset()
        self.grad_norm.reset()
        self.gpu_memory_usage.reset()
        self.gpu_peak_memory_usage.reset()
        # self.ram_percentage.reset()
        # self.cpu_percentage.reset()


## Weight and Biases recorder

try:
    import wandb  # type: ignore[import-not-found]
except ImportError:
    has_wandb = False
else:
    has_wandb = True


class LCMWandBRecorder(MetricRecorder):
    """Records metric values to Weights & Biases."""

    defined_runs: Set[str] = set()

    def __init__(
        self,
        project: Optional[str] = None,
        name: Optional[str] = None,
        output_dir: Optional[Path] = None,
        config: Dict[str, Any] = {},
        **kwargs,
    ) -> None:
        """
        :param project: A project to organise this run with other experiments, if none, the run will go under `uncategorized`.
        :param name: A unique name for your run, if none is given, a random name will be generated
        :param output_dir: The base directory under which to store the W&B files. You don't have to provide this.
        :param config: A dictionary of key-value pairs to be stored as the experiment's config. (akin to hparams in tb)
        :param kwargs: Additional arguments to pass to wandb.init()

        In order to use W&B, run `wandb login` from the command line and enter
        the API key when prompted.
        """
        if not has_wandb:
            log = get_log_writer(__name__)
            log.warning("wandb not found. Please install it with `pip install wandb`.")  # fmt: skip

            self._run = None
        else:
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
            self._run = wandb.init(  # type: ignore
                project=project,
                name=name,
                dir=output_dir,
                resume="allow",
                config=config,
                **kwargs,
            )

    def _define_run(self, run: str):
        if run in self.defined_runs:
            return
        # https://docs.wandb.ai/guides/track/log/customize-logging-axes/
        wandb.define_metric(f"{run}/step")
        wandb.define_metric(f"{run}/*", step_metric=f"{run}/step")

    @override
    def record_metrics(
        self,
        run: str,
        values: Mapping[str, Any],
        step_nr: Optional[int] = None,
        *,
        flush: bool = True,
    ) -> None:
        if self._run is None:
            return

        self._define_run(run)

        for name, value in values.items():
            formatter = _metric_formatters.get(name)
            if formatter is None:
                display_name = name
            else:
                display_name = formatter.display_name

            self._run.log({f"{run}/{display_name}": value, f"{run}/step": step_nr})

    @override
    def close(self) -> None:
        if self._run is not None:
            self._run.finish()


lcm_metric_formatters: Dict[str, Tuple[str, int, Callable[[Any], str]]] = {
    # fmt: off
    "loss": ("Loss", 100, format_as_float),
    "nll_loss": ("NLL Loss", 100, format_as_float),
    "mse_loss": ("MSE Loss", 100, format_as_float),
    "contrastive_loss": ("Contrastive Loss", 110, format_as_float),
    "reconstruction_loss": ("Reconstruction loss", 110, format_as_float),
    "unnormalized_reconstruction_loss": (
        "Unnormalized Reconstruction Loss",
        110,
        format_as_float,
    ),
    "kld": ("KLD loss", 110, format_as_float),
    "encoder_mse_loss": ("Encoder MSE loss", 110, format_as_float),
    "decoder_ce_loss": ("Decoder CE loss", 110, format_as_float),
    "elapsed_time": ("Elapsed Time", 500, format_as_seconds),
    "wall_time": ("Wall Time", 510, format_as_seconds),
    "lr": ("Learning Rate", 800, format_as_float),
    "loss_scale": ("Loss Scale", 810, format_as_float),
    "grad_norm": ("Grad norm", 810, format_as_float),
    "raw_grad_norm": ("Raw Grad norm", 815, format_as_float),
    "encoder_mse_scale": ("Encoder MSE loss scale", 850, format_as_float),
    "batch_size": ("Batch Size", 900, format_as_int),
    "elements_per_batch": ("Elements per Batch", 900, format_as_int),
    "elements_per_second": ("Elements per Second", 900, format_as_int),
    "num_examples": ("Number of Examples", 900, format_as_int),
    "num_source_elements": ("Number of Source Elements", 900, format_as_int),
    "num_target_elements": ("Number of Target Elements", 900, format_as_int),
    "total_num_target_elements": ("Accumulated Target Elements", 920, format_as_int),
    "gpu_memory_usage": ("GPU memory usage (GiB)", 910, format_as_float),
    "gpu_peak_memory_usage": ("GPU peak memory usage (GiB)", 910, format_as_float),
    "ram_percentage": ("RAM usage", 920, format_as_percent),
    "cpu_percentage": ("CPU usage", 920, format_as_percent),
    "mean_predicted_embeddings": ("mean_predicted_embeddings", 920, format_as_float),
    "std_predicted_embeddings": ("std_predicted_embeddings", 920, format_as_float),
    # fmt: on
}
for key in lcm_metric_formatters:
    register_metric_formatter(key, *lcm_metric_formatters[key], overwrite=True)
