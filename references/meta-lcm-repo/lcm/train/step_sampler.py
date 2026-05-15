# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.distributions as D
from fairseq2.logging import get_log_writer
from torch import Tensor

from lcm.nn.schedulers import DDIMScheduler

SUPPORTED_SAMPLERS = Literal["uniform", "beta"]
SUPPORTED_WEIGHTINGS = Literal["none", "clamp_snr"]

logger = get_log_writer(__name__)


def beta_function(a, b):
    result = torch.exp(torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b))
    return result


@dataclass
class StepsSamplerConfig:
    sampling: SUPPORTED_SAMPLERS = "uniform"
    weighting: SUPPORTED_WEIGHTINGS = "none"
    beta_a: float = 0.8
    beta_b: float = 1
    max_gamma: float = 5.0
    min_gamma: float = 0


class StepsSampler(object):
    def __init__(
        self,
        config: StepsSamplerConfig,
        noise_scheduler: DDIMScheduler,
    ):
        num_diffusion_train_steps = noise_scheduler.num_diffusion_train_steps
        weights: Optional[Tensor] = None

        if config.sampling == "uniform":
            weights = torch.ones(
                num_diffusion_train_steps,
            )

        elif config.sampling == "beta":
            # As motivated in https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00328.pdf
            a = torch.tensor([config.beta_a])
            b = torch.tensor([config.beta_b])
            # a=1, b=1 -> uniform
            # The paper empirically chooses b=1, a=0.8 < 1

            steps = (
                torch.arange(1, num_diffusion_train_steps + 1)
                / num_diffusion_train_steps
            )
            weights = (
                1 / beta_function(a, b) * (steps ** (a - 1)) * ((1 - steps) ** (b - 1))
            )

        assert weights is not None, "The sampling weights were not properly set!"
        logger.info(f"Training with sampling weights={weights}")

        self.distrib = D.Categorical(
            probs=weights / weights.sum(),
        )

        # setup weights for scaling:
        if config.weighting == "none":
            self.gamma_per_step = None

        elif config.weighting == "clamp_snr":
            # Min-SNR scheme from
            # https://arxiv.org/abs/2303.09556
            snrs = noise_scheduler.get_snrs()
            # gamma(t) = min(max_gamma, snr(t))
            self.gamma_per_step = torch.clamp(
                snrs, max=config.max_gamma, min=config.min_gamma
            )

        logger.info(f"Training with Gamma={self.gamma_per_step}")

    @property
    def _training_weights(self) -> Tensor:
        return self.distrib.probs

    def sample(self, size: torch.Size, device: torch.device):
        samples = self.distrib.sample(size).to(device)
        # print('Samples', samples)
        # print('Counts:', torch.bincount(samples.flatten()))
        return samples

    def get_loss_scales(self, steps):
        if self.gamma_per_step is None:
            return None

        # If we're using constant Gamma=1 (returning None), then the sum of
        # the loss scales is steps.numel(), to match the total mass,
        # we normalize the scales to sum to steps.numel()
        gamma = self.gamma_per_step.to(steps.device)[steps]
        gamma = gamma / gamma.sum() * steps.numel()
        return gamma
