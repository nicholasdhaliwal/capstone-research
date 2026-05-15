# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

# This code is based on https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py, which is distributed under the Apache 2.0 License.
# HuggingFace's diffusers DISCLAIMER: This code is strongly influenced by https://github.com/pesser/pytorch_diffusion
# and https://github.com/hojonathanho/diffusion

import math
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

import torch
from fairseq2.logging import get_log_writer
from fairseq2.typing import CPU
from torch import Tensor

logger = get_log_writer(__name__)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def logit(x):
    return math.log(x / (1 - x))


@dataclass
class DDIMSchedulerOutput:
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: Tensor
    pred_original_sample: Tensor


@dataclass
class DDIMSchedulerConfig:
    num_diffusion_train_steps: int = 1000
    """The number of diffusion steps to train the model."""

    beta_start: float = 0.0001
    """The starting `beta` value of inference."""

    beta_end: float = 0.02
    """The final `beta` value."""
    """In DDPM (https://arxiv.org/pdf/2006.11239), $\beta_t$ is increasing
    linearly from $\beta_1$ (`beta_start`)=1e−4 to $\beta_T$ (`beta_end`)=0.02.
    These constants were chosen to be small relative to data scaled to [−1, 1],
    ensuring that reverse and forward processes have approximately
    the same functional form while keeping the signal-to-noise ratio at $x_T$ as small as possible.
    Another common choice in HF:diffusers `beta_start=0.00085, beta_end=0.012,`
    Note that `beta_start` and `beta_end` are irrelevant for `squaredcos_cap_v2`
    """

    beta_schedule: Literal[
        "linear",
        "scaled_linear",
        "squaredcos_cap_v2",
        "sigmoid",
    ] = "squaredcos_cap_v2"
    """The beta schedule, a mapping from a beta range to a sequence of betas
        for stepping the model (length=`num_diffusion_train_steps`).
        Choose from:
        - `linear`: Linearly spaced betas between `beta_start` and `beta_end`.
            Referred to as `sqrt_linear` in stable-diffusion.
        - `scaled_linear`:  Squared values after linearly spacing form sqrt(beta_start) to sqrt(beta_end).
            Referred to as `linear` in stable-diffusion.
        -`squaredcos_cap_v2`: Creates a beta schedule that discretizes
            math:: $\bar alpha(t) = {cos((t/T + s) / (1+s) * \pi/2)}^2$, HF:diffusers sets `s` to 0.008.
            For the intuition behind how a cosine schedule compares to a linear schedule
            see Figure 3 of https://arxiv.org/pdf/2102.09672
        - `sigmoid` our sigmoid schedule (see Equation 14 of the LCM paper).
    """

    scaled_linear_exponent: float = 2.0
    """Exponent for the scaled linear beta schedule. Default is quadratic (scaled_linear_exponent=2)"""

    sigmoid_schedule_alpha: float = 1.5
    sigmoid_schedule_beta: float = 0
    """alpha and beta hyper-parameters of the sigmoid beta-schedule"""

    clip_sample: bool = False
    """Clip the predicted sample for numerical stability."""

    clip_sample_range: float = 1.0
    """The maximum magnitude for sample clipping. Valid only when `clip_sample=True`."""

    set_alpha_to_one: bool = True
    """Each diffusion step uses the alphas product value at that step and at the previous one. For the final step
    there is no previous alpha. When this option is `True` the previous alpha product is fixed to `1`,
    otherwise it uses the alpha value at step 0."""

    prediction_type: Literal["sample", "epsilon", "v_prediction"] = "sample"
    """If `sample`, the model predicts the clean ground truth embeddings.
       If `epsilon`, the model predicts the added noise of the diffusion process.
       If `v_epsilon`, the model predicts an interpolation of the ground truth clean
       embeddings and the added noise. As introduced in section 2.4 of the Imagen paper
       (https://imagen.research.google/video/paper.pdf)
    """

    thresholding: bool = False
    """Whether to use the "dynamic thresholding" method.
    This is unsuitable for latent-space diffusion models such as Stable Diffusion."""

    dynamic_thresholding_ratio: float = 0.995
    """The ratio for the dynamic thresholding method. Valid only when `thresholding=True`."""

    sample_max_value: float = 1.0
    """The threshold value for dynamic thresholding. Valid only when `thresholding=True`."""

    rescale_betas_zero_snr: bool = True
    """Whether to rescale the betas to have zero terminal SNR. This enables the
    model to generate very bright and dark samples instead of limiting it to samples
    with medium brightness. Loosely related to
    [`--offset_noise`](https://github.com/huggingface/diffusers/blob/74fd735eb073eb1d774b1ab4154a0876eb82f055/examples/dreambooth/train_dreambooth.py#L506)."""

    # Inference specific
    timestep_spacing: Literal["linspace", "leading", "trailing"] = "trailing"
    """The way the timesteps should be scaled. Refer to Table 2 of
        https://arxiv.org/abs/2305.08891 for more information."""


class DDIMScheduler:
    def __init__(self, config: DDIMSchedulerConfig):
        self.config = config

        # Make these 2 arguments easily accessible
        self.num_diffusion_train_steps = self.config.num_diffusion_train_steps

        self.prediction_type = self.config.prediction_type

        beta_schedule = self.config.beta_schedule

        if beta_schedule == "linear":
            self.betas = torch.linspace(
                self.config.beta_start,
                self.config.beta_end,
                self.num_diffusion_train_steps,
                dtype=torch.float32,
            )
        elif beta_schedule == "scaled_linear":
            # This schedule is very specific to the latent diffusion model.
            exponent = self.config.scaled_linear_exponent
            self.betas = (
                torch.linspace(
                    self.config.beta_start ** (1 / exponent),
                    self.config.beta_end ** (1 / exponent),
                    self.num_diffusion_train_steps,
                    dtype=torch.float32,
                )
                ** exponent
            )
        elif beta_schedule == "squaredcos_cap_v2":
            # Cosine schedule as introduced in
            # [Nichol and Dhariwal, 2021](https://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf)
            self.betas = betas_for_alpha_bar(
                self.num_diffusion_train_steps,
                alpha_transform_type="cosine",
            )

        elif beta_schedule == "sigmoid":
            self.betas = betas_for_alpha_bar(
                self.num_diffusion_train_steps,
                alpha_transform_type="sigmoid",
                sigmoid_alpha=self.config.sigmoid_schedule_alpha,
                sigmoid_beta=self.config.sigmoid_schedule_beta,
            )

        else:
            raise NotImplementedError(
                f"We do not recognize beta_schedule={beta_schedule}"
            )

        # Rescale for zero SNR
        if self.config.rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # At every step in ddim, we are looking into the previous alphas_cumprod
        # For the final step, there is no previous alphas_cumprod because we are already at 0
        # `set_alpha_to_one` decides whether we set this parameter simply to one or
        # whether we use the final alpha of the "non-previous" one.
        self.final_alpha_cumprod = (
            torch.tensor(1.0)
            if self.config.set_alpha_to_one
            else self.alphas_cumprod[0]
        )

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # timesteps for inference
        self.num_inference_steps: Optional[int] = None

    def _get_variance(self, timestep, prev_timestep):
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (
            1 - alpha_prod_t / alpha_prod_t_prev
        )
        return variance

    def get_variances(self) -> Tensor:
        alpha_prod_t = self.alphas_cumprod
        alpha_prod_t_prev = torch.cat(
            (torch.tensor([self.final_alpha_cumprod]), alpha_prod_t[:-1])
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (
            1 - alpha_prod_t / alpha_prod_t_prev
        )
        return variance

    def get_snrs(self) -> Tensor:
        alphas_cumprod = self.alphas_cumprod
        snr = alphas_cumprod / (1 - alphas_cumprod)
        return snr

    def _threshold_sample(self, sample: Tensor) -> Tensor:
        """
        "Dynamic thresholding: At each sampling step we set s to a certain
        percentile absolute pixel value in xt0 (the prediction of x_0 at timestep t),
        and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
        s. Dynamic thresholding pushes saturated pixels (those near -1 and 1)
        inwards, thereby actively preventing pixels from saturation at each step.
        We find that dynamic thresholding results in significantly better
        photorealism as well as better image-text alignment,
        especially when using very large guidance weights."

        https://arxiv.org/abs/2205.11487
        """
        dtype = sample.dtype
        batch_size, channels, *remaining_dims = sample.shape

        if dtype not in (torch.float32, torch.float64):
            sample = (
                sample.float()
            )  # upcast for quantile calculation, and clamp not implemented for cpu half

        # Flatten sample for doing quantile calculation along each image
        sample = sample.reshape(batch_size, -1)

        abs_sample = sample.abs()  # "a certain percentile absolute pixel value"

        s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(
            s, min=1, max=self.config.sample_max_value
        )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]
        s = s.unsqueeze(1)  # (batch_size, 1) because clamp will broadcast along dim=0
        sample = (
            torch.clamp(sample, -s, s) / s
        )  # "we threshold xt0 to the range [-s, s] and then divide by s"

        sample = sample.reshape(batch_size, channels, *remaining_dims)
        sample = sample.to(dtype)

        return sample

    def set_timesteps(
        self, num_inference_steps: int, device: Union[str, torch.device] = None
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
        """

        if num_inference_steps > self.config.num_diffusion_train_steps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.num_diffusion_train_steps`:"
                f" {self.num_diffusion_train_steps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.num_diffusion_train_steps} timesteps."
            )

        self.num_inference_steps = num_inference_steps

        # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
        # With T the number of training steps and S the number of inference steps

        if self.config.timestep_spacing == "linspace":
            # Linspace: flip round(linspace(1, T, S))
            # With T=1000 and S=10; [999, 888, 777, 666, 555, 444, 333, 222, 111,   0]
            timesteps = torch.linspace(
                0,
                self.config.num_diffusion_train_steps - 1,
                self.num_inference_steps,
                device=device,
                dtype=torch.long,
            )
            timesteps = torch.flip(timesteps, dims=(0,)).round()

        elif self.config.timestep_spacing == "leading":
            # Leading: flip arange(1, T + 1, floor(T /S))
            # With T=1000 and S=10: [900, 800, 700, 600, 500, 400, 300, 200, 100,  0]

            leading_step_ratio = (
                self.num_diffusion_train_steps // self.num_inference_steps
            )
            timesteps = torch.arange(
                start=0,
                end=self.num_diffusion_train_steps,
                step=leading_step_ratio,
                device=device,
                dtype=torch.long,
            )
            timesteps = torch.flip(timesteps, dims=(0,)).round()

        elif self.config.timestep_spacing == "trailing":
            # Trailing: round(flip(arange(T, 0, −T /S)))
            # With T=1000 and S=10: [999, 899, 799, 699, 599, 499, 399, 299, 199,  99]
            trailing_step_ratio: float = (
                self.num_diffusion_train_steps / self.num_inference_steps
            )
            # creates integer timesteps by multiplying by ratio
            timesteps = torch.arange(
                self.config.num_diffusion_train_steps,
                0,
                -trailing_step_ratio,
                device=device,
                dtype=torch.long,
            ).round()
            timesteps -= 1
        else:
            raise ValueError(
                f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'leading' or 'trailing'."
            )

        self.timesteps = timesteps
        logger.debug(
            f"With `{self.config.timestep_spacing}`, setting inference timesteps to {self.timesteps}"
        )

    def step(
        self,
        model_output: Tensor,
        timestep: int,
        sample: Tensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[Tensor] = None,
        prediction_type: Optional[str] = None,
        epsilon_scaling: Optional[float] = None,
    ) -> DDIMSchedulerOutput:
        """
        INFERENCE ONLY.
        Predict the sample from the previous timestep by reversing the SDE.
        This function propagates the diffusion
        process from the learned model outputs.

        Args:
            model_output (`Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`Tensor`):
                A current instance of a sample created by the diffusion process.
            eta (`float`):
                The weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`, defaults to `False`):
                If `True`, computes "corrected" `model_output` from the clipped predicted original sample. Necessary
                because predicted original sample is clipped to [-1, 1] when `self.config.clip_sample` is `True`. If no
                clipping has happened, "corrected" `model_output` would coincide with the one provided as input and
                `use_clipped_model_output` has no effect.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            variance_noise (`Tensor`):
                Alternative to generating noise with `generator` by directly providing the noise for the variance
                itself. Useful for methods such as [`CycleDiffusion`].
            prediction_type: Optional[str] if provided we step with a different prediction_type
                than the one in the config
            epsilon_scaling: Optional[float] if not None, the predicted epsilon will be scaled down by
                the provided factor as introduced in https://arxiv.org/pdf/2308.15321

        Returns:
            DDIMSchedulerOutput

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. Get previous step value (=t-1)
        prev_timestep = (
            timestep - self.config.num_diffusion_train_steps // self.num_inference_steps
        )

        # 2. Compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.final_alpha_cumprod
        )

        beta_prod_t = 1 - alpha_prod_t

        # 3. Compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prediction_type = prediction_type or self.prediction_type
        if prediction_type == "epsilon":
            pred_original_sample = (
                sample - beta_prod_t ** (0.5) * model_output
            ) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (
                sample - alpha_prod_t ** (0.5) * pred_original_sample
            ) / beta_prod_t ** (0.5)
        elif prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (
                beta_prod_t**0.5
            ) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (
                beta_prod_t**0.5
            ) * sample
        else:
            raise ValueError(
                f"prediction_type given as {prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        # 3.a epsilon scaling:
        if epsilon_scaling is not None:
            pred_epsilon = pred_epsilon / epsilon_scaling

        # 4. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 5. Compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** (0.5)
        if use_clipped_model_output:
            # the pred_epsilon is always re-derived from the clipped x_0 in Glide
            pred_epsilon = (
                sample - alpha_prod_t ** (0.5) * pred_original_sample
            ) / beta_prod_t ** (0.5)
        # 6. Compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (
            0.5
        ) * pred_epsilon
        # 7. Compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = (
            alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        )

        if eta > 0:
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                    " `variance_noise` stays `None`."
                )

            if variance_noise is None:
                variance_noise = randn_tensor(
                    model_output.shape,
                    generator=generator,
                    device=model_output.device,
                    dtype=model_output.dtype,
                )
            variance = std_dev_t * variance_noise
            prev_sample = prev_sample + variance

        return DDIMSchedulerOutput(
            prev_sample=prev_sample, pred_original_sample=pred_original_sample
        )

    def add_noise(
        self,
        original_samples: Tensor,
        noise: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        """TRAINING ONLY
        Forward noising process during training"""
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        # Move the self.alphas_cumprod to device to avoid redundant CPU to GPU data movement
        # for the subsequent add_noise calls
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device).to(torch.int32)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = (
            sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        )
        return noisy_samples

    def get_velocity(self, sample: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as sample
        self.alphas_cumprod = self.alphas_cumprod.to(device=sample.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=sample.dtype)
        timesteps = timesteps.to(sample.device).to(torch.int32)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity

    def get_epsilon(
        self, model_output: Tensor, sample: Tensor, timestep: int
    ) -> Tensor:
        """Given model inputs (sample) and outputs (model_output)
        Predict the noise residual according to the scheduler's
        prediction type"""

        pred_type = self.prediction_type

        alpha_prod_t = self.alphas_cumprod[timestep]

        beta_prod_t = 1 - alpha_prod_t

        if pred_type == "epsilon":
            return model_output

        elif pred_type == "sample":
            return (sample - alpha_prod_t ** (0.5) * model_output) / beta_prod_t ** (
                0.5
            )

        elif pred_type == "v_prediction":
            return (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"The scheduler's prediction type {pred_type} must be one of `epsilon`, `sample`, or `v_prediction`"
            )


def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = (
            generator.device.type
            if not isinstance(generator, list)
            else generator[0].device.type
        )
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = CPU
            if device != "mps":
                logger.info(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(
                f"Cannot generate a {device} tensor from a generator of type {gen_device_type}."
            )

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]  # type: ignore
        latents_list = [
            torch.randn(
                shape,
                generator=generator[i],
                device=rand_device,
                dtype=dtype,
                layout=layout,
            )
            for i in range(batch_size)
        ]
        latents = torch.cat(latents_list, dim=0).to(device)
    else:
        latents = torch.randn(
            shape, generator=generator, device=rand_device, dtype=dtype, layout=layout
        ).to(device)

    return latents


def betas_for_alpha_bar(
    num_diffusion_timesteps: int,
    max_beta: float = 0.999,
    alpha_transform_type: Literal["cosine", "exp", "sigmoid"] = "cosine",
    sigmoid_alpha: float = 1.5,
    sigmoid_beta: float = 0,
) -> Tensor:
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
        alpha_transform_type (`str`, *optional*, default to `cosine`): the type of noise schedule for alpha_bar.
                     Choose from `cosine` or `exp`
        sigmoid_alpha/sigmoid_beta: additional hyper-parameters for the sigmoid schedule

    Returns:
        betas (`Tensor`): the betas used by the scheduler to step the model outputs
    """
    if alpha_transform_type == "cosine":

        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "sigmoid":

        def alpha_bar_fn(t):
            epsilon = 1e-32
            return sigmoid(
                sigmoid_beta
                - sigmoid_alpha
                * logit(torch.clamp(torch.tensor(t), min=epsilon, max=1 - epsilon))
            )

    elif alpha_transform_type == "exp":

        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    else:
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


def rescale_zero_terminal_snr(betas: Tensor) -> Tensor:
    """
    Rescales betas to have zero terminal SNR Based on https://arxiv.org/pdf/2305.08891.pdf (Algorithm 1)

    Args:
        betas (`Tensor`):
            the betas that the scheduler is being initialized with.

    Returns:
        `Tensor`: rescaled betas with zero terminal SNR
    """
    # Convert betas to alphas_bar_sqrt
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # Revert cumprod
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas
