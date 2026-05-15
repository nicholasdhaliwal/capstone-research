# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

from lcm.models.sonar_normalizer.builder import (
    SonarNormalizerConfig,
    sonar_normalizer_arch,
)


@sonar_normalizer_arch("base")
def _base_sonar_normalizer() -> SonarNormalizerConfig:
    """The base architecture for all center-and-scale normalizers
    regardless of how the center/scale are estimated"""
    return SonarNormalizerConfig(
        dim=1024,
    )


@sonar_normalizer_arch("base_page4k")
def _base_page_normalizer() -> SonarNormalizerConfig:
    return SonarNormalizerConfig(
        dim=4 * 1024,
    )


@sonar_normalizer_arch("base_fft")
def _base_fft_sonar_normalizer() -> SonarNormalizerConfig:
    return SonarNormalizerConfig(dim=1024, with_fft=True)


@sonar_normalizer_arch("clipping")
def _clipping_sonar_normalizer() -> SonarNormalizerConfig:
    return SonarNormalizerConfig(dim=1024, clip_proba=1e-4)


@sonar_normalizer_arch("clipping_fft")
def _clipping_fft_sonar_normalizer() -> SonarNormalizerConfig:
    return SonarNormalizerConfig(dim=1024, clip_proba=1e-4, with_fft=True)
