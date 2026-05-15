# Copyright (c) Meta Platforms, Inc. and affiliates.
#
#
#


from typing import List, Union

import torch

from lcm.datasets.configs import SonarEncoderConfig
from lcm.evaluation.metrics.similarity import cos_sim, l2_distance
from lcm.evaluation.utils.sonar import text_encoder


def round_trip_l2_distance(
    prediction_text: List[str],
    targets: torch.Tensor,
    encoder_config: SonarEncoderConfig,
    flatten: bool = False,
) -> Union[List[float], List[List[float]]]:
    """
    Calculate the L2 distance of a text and a vector by putting the text
    into the sonar space embedding
    """
    text2vec = text_encoder(encoder_config, device=targets.device)
    prediction_projected = text2vec.predict(
        prediction_text,
        source_lang=encoder_config.lang,
        batch_size=len(prediction_text),
    ).reshape(targets.shape)

    return l2_distance(prediction_projected, targets, flatten=flatten)


def round_trip_cos(
    prediction_text: List[str],
    targets: torch.Tensor,
    encoder_config: SonarEncoderConfig,
) -> Union[List[float], List[List[float]]]:
    """
    Calculate the cosine similarity between a text and the target embeddings
    (e.g. input embedding of the decoder that generates the text)
    """
    text2vec = text_encoder(encoder_config)
    prediction_projected = text2vec.predict(
        prediction_text,
        source_lang=encoder_config.lang,
        batch_size=len(prediction_text),
    ).reshape(targets.shape)

    return cos_sim(prediction_projected.numpy(), targets.numpy())
