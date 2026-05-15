# Copyright (c) Meta Platforms, Inc. and affiliates.
#
#
#

from typing import List, Union

import numpy as np
import torch


def l2_distance(
    prediction: torch.Tensor, targets: torch.Tensor, flatten: bool = False
) -> Union[List[float], List[List[float]]]:
    l2_dist = (prediction - targets).pow(2).sum(dim=-1).sqrt()
    l2_dist = l2_dist.squeeze()
    if flatten:
        l2_dist = torch.flatten(l2_dist)
    x = l2_dist.cpu().tolist()
    if isinstance(x, float):  # l2_dist is a torch scalar
        return [x]
    return x


def mse_constrative_accuracy(
    prediction: torch.Tensor,
    targets: torch.Tensor,
) -> List[float]:
    """
    Calculate the mse_loss between the predictions and groundtruth, each
    has the shape batch_size x seq_len x model_dim.

    Returns:
        a list of length batch_size x seq_len and scores are between 0 and
        1 (with 0.5 corresponding to a random model)
    """
    assert prediction.size() == targets.size()
    batch_size, seq_len, model_dim = prediction.size()
    preds_flat = prediction.reshape(batch_size * seq_len, model_dim)
    gt_flat = targets.reshape(batch_size * seq_len, model_dim)
    pos_dist = torch.nn.functional.mse_loss(preds_flat, gt_flat, reduction="none").sum(
        dim=-1
    )
    distractors_flat = torch.stack(
        [
            targets[torch.arange(batch_size) != batch_id, :].reshape(
                (batch_size - 1) * (seq_len), model_dim
            )
            for batch_id in range(batch_size)
            for j in range(seq_len)
        ],
        dim=0,
    )
    n_distractors = distractors_flat.shape[1]

    neg_dist = torch.nn.functional.mse_loss(
        preds_flat.unsqueeze(1).repeat(1, n_distractors, 1),
        distractors_flat,
        reduction="none",
    ).sum(dim=-1)
    ptw_acc = (neg_dist > pos_dist.unsqueeze(-1)).to(torch.float).mean(-1)
    return ptw_acc.cpu().tolist()


def nltk_sentence_bleu(prediction_tokens: List[int], target_tokens: List[int]) -> float:
    try:
        from nltk.translate.bleu_score import (  # type: ignore
            SmoothingFunction,
            sentence_bleu,
        )
    except (ImportError, ModuleNotFoundError):
        return -1.0

    return float(
        sentence_bleu(
            [target_tokens],
            prediction_tokens,
            smoothing_function=SmoothingFunction().method1,
        )
    )


def edit_distance(prediction_tokens: List[int], target_tokens: List[int]) -> float:
    # Get minimum edit distance between prediction and targets in the case of multiple targets
    try:
        import editdistance
    except (ImportError, ModuleNotFoundError):
        return -1.0

    return float(editdistance.eval(prediction_tokens, target_tokens))


def longest_common_substring(
    prediction_tokens: List[int], target_tokens: List[int]
) -> float:
    lengths = np.zeros((len(prediction_tokens), len(target_tokens)), dtype=int).tolist()
    longest = 0

    for i in range(len(prediction_tokens)):
        for j in range(len(target_tokens)):
            if prediction_tokens[i] != target_tokens[j]:
                continue
            elif i == 0 or j == 0:
                lengths[i][j] = 1
            else:
                lengths[i][j] = lengths[i - 1][j - 1] + 1

            longest = max(longest, lengths[i][j])

    return float(longest)


def memorization_score(prediction_tokens: List[int], target_tokens: List[int]) -> float:
    # See "Emergent and Predictable Memorization in Large Language Models"
    # https://arxiv.org/pdf/2304.11158.pdf
    correct = sum(
        pred == target for pred, target in zip(prediction_tokens, target_tokens)
    )
    correct_avg = correct / len(target_tokens)

    return float(correct_avg)


def cos_sim(prediction: np.ndarray, targets: np.ndarray):
    pred_norm = np.linalg.norm(prediction, axis=-1, keepdims=True)
    targets_norm = np.linalg.norm(targets, axis=-1, keepdims=True)

    normalized_preds = prediction / pred_norm
    normalized_targets = targets / targets_norm

    return np.einsum("ij,ij->i", normalized_preds, normalized_targets)
