# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

from dataclasses import dataclass
from typing import Literal, Optional

import torch
from fairseq2.config_registry import ConfigRegistry
from fairseq2.typing import DataType, Device
from torch import Tensor
from torch.nn import Module


@dataclass
class SonarNormalizerConfig:
    dim: int = 1024
    """The dimension of the features to be normalized"""

    clip_proba: Optional[float] = None
    """
    If `clip_proba` is not None, `clip_min` and `clip_max` will
    be used to clip the features before normalizing.
    `clip_min` and `clip_max` correspond to the pre-computed `clip_proba`
    and `1-clip_proba` quantiles respectively.
    """

    with_fft: bool = False
    """
    Applying FFT transform at the raw input before all other transforms.
    """

    quantile_min: float = 0.25
    """The lower quantile used to measure the IQR when estimating the scale with a robust scaler"""

    quantile_max: float = 0.75
    """The upper quantile used to measure the IQR when estimating the scale with a robust scaler"""

    normalization_method: Literal["standard", "robust", "gaussian_robust"] = (
        "gaussian_robust"
    )
    """
    Dictates how the normalizer's scale is evaluated when fitting.
    (1) 'standard': center=mean, scale = std
    (2) 'robust':  center=median, scale = IQR = Qmax - Qmin
    (3) 'gaussian_robust': center=median, scale = IQR / k,
        where k=`stats.norm.ppf(q_max / 100.0) - stats.norm.ppf(q_min / 100.0)`
        i.e scale = scale = 0.7413 x IQR if q_min=0.25 and q_max=0.75.
        This is the robust normalization of https://arxiv.org/pdf/2307.05445
    """


sonar_normalizer_archs = ConfigRegistry[SonarNormalizerConfig]()
sonar_normalizer_arch = sonar_normalizer_archs.decorator


class FFTInterface:
    @staticmethod
    def fft_transform(embeddings: Tensor) -> Tensor:
        dtype = embeddings.dtype
        if dtype in [torch.float16, torch.bfloat16]:
            embeddings = embeddings.to(dtype=torch.float32)
        embeddings = torch.fft.rfft(embeddings, norm="backward")
        return torch.concat(
            [torch.real(embeddings), torch.imag(embeddings)[..., 1:-1]], dim=-1
        ).to(dtype)

    @staticmethod
    def fft_inverse_transform(embeddings: Tensor) -> Tensor:
        assert embeddings.shape[-1] % 2 == 0
        dtype = embeddings.dtype
        if dtype in [torch.float16, torch.bfloat16]:
            embeddings = embeddings.to(dtype=torch.float32)
        rr, im = torch.split(
            embeddings,
            [embeddings.shape[-1] // 2 + 1, embeddings.shape[-1] // 2 - 1],
            dim=-1,
        )
        im = torch.concat(
            [torch.zeros_like(im[..., :1]), im, torch.zeros_like(im[..., :1])], dim=-1
        )
        embeddings = torch.fft.irfft(rr + im * 1j)
        return embeddings.to(dtype)


class SonarNormalizer(FFTInterface, Module):
    """
    To perform efficient diffusion modeling, SONAR embeddings need to be
    normalized. This SonarNormalizer follows the robust normalization introduced in
    https://arxiv.org/abs/2307.05445
    Quoting from the paper: "Due to the very long-tailed feature distribution, typical mean and standard deviation statistics will be
    heavily biased. We thus propose a robust alternative based on the feature distribution quantiles. We
    take the median as the center of the distribution and approximate its scale using the Normalized
    InterQuartile Range (IQR) for a normal distribution: 0.7413 × IQR
    """

    def __init__(
        self,
        config: SonarNormalizerConfig,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        super().__init__()
        self.config = config

        self.register_buffer(
            "center", torch.zeros(config.dim, dtype=dtype, device=device)
        )
        self.register_buffer(
            "scale", torch.ones(config.dim, dtype=dtype, device=device)
        )
        if self.config.clip_proba is not None:
            self.register_buffer(
                "clip_min", torch.ones(config.dim, dtype=dtype, device=device)
            )
            self.register_buffer(
                "clip_max", torch.ones(config.dim, dtype=dtype, device=device)
            )

    def normalize(self, embeddings: Tensor) -> Tensor:
        if self.config.with_fft:
            embeddings = self.fft_transform(embeddings)

        embeddings = (embeddings - self.center) / self.scale
        if self.config.clip_proba is not None:
            embeddings = torch.clamp(embeddings, min=self.clip_min, max=self.clip_max)
        return embeddings

    def denormalize(self, embeddings: Tensor) -> Tensor:
        if self.config.clip_proba is not None:
            embeddings = torch.clamp(embeddings, min=self.clip_min, max=self.clip_max)

        embeddings = (embeddings * self.scale) + self.center
        if self.config.with_fft:
            embeddings = self.fft_inverse_transform(embeddings)
        return embeddings

    @torch.no_grad()
    def fit(self, embeddings: Tensor):
        if self.config.normalization_method in [
            "robust",
            "gaussian_robust",
        ]:
            from sklearn.preprocessing import RobustScaler

            _scaler = RobustScaler(
                unit_variance=self.config.normalization_method == "gaussian_robust",
                quantile_range=(self.config.quantile_min, self.config.quantile_max),
            )

        elif self.config.normalization_method == "standard":
            from sklearn.preprocessing import StandardScaler

            _scaler = StandardScaler()
        else:
            raise ValueError(
                f"Unrecognizable method {self.config.normalization_method} for scaling input features"
            )

        assert embeddings.shape[-1] == self.config.dim
        assert len(embeddings.shape) == 2

        if self.config.with_fft:
            embeddings = self.fft_transform(embeddings)

        embeddings = _scaler.fit_transform(embeddings.cpu().float().numpy())

        if self.config.normalization_method in [
            "robust",
            "gaussian_robust",
        ]:
            _center = _scaler.center_
            _scale = _scaler.scale_

        elif self.config.normalization_method == "standard":
            _center = _scaler.mean_
            _scale = _scaler.scale_

        self.center[:] = torch.tensor(
            _center, dtype=self.center.dtype, device=self.center.device
        )
        self.scale[:] = torch.tensor(
            _scale, dtype=self.scale.dtype, device=self.scale.device
        )

        if self.config.clip_proba is not None:
            self.clip_min[:] = torch.quantile(
                torch.tensor(embeddings), self.config.clip_proba, dim=0
            ).to(dtype=self.clip_min.dtype, device=self.clip_min.device)
            self.clip_max[:] = torch.quantile(
                torch.tensor(embeddings), 1 - self.config.clip_proba, dim=0
            ).to(dtype=self.clip_max.dtype, device=self.clip_max.device)


def create_sonar_normalizer(
    config: SonarNormalizerConfig,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> SonarNormalizer:
    """Create an LCM model.
    :param config:
        The configuration.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """
    return SonarNormalizer(config, device=device, dtype=dtype)
