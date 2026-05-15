# Copyright (c) Meta Platforms, Inc. and affiliates.

from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

import torch


def f1(prediction: str, targets: List[str]) -> float:
    def _f1(pred_tokens: List[str], gt_tokens: List[str]) -> float:
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(gt_tokens)
        return (2 * precision * recall) / (precision + recall)

    return max(_f1(prediction.split(), target.split()) for target in targets)


def exact_match(prediction: str, targets: List[str]) -> float:
    return max(float(prediction == target) for target in targets)


def one_character_exact_match(prediction: str, targets: List[str]) -> float:
    return max(float(prediction.strip()[0] == target.strip()[0]) for target in targets)


def cosine_based_match(
    prediction: torch.Tensor, choices: torch.Tensor, tgt_idx: int
) -> float:
    cosine_sim = torch.nn.CosineSimilarity(dim=-1)
    pred_idx = cosine_sim(prediction, choices).argmax()
    return float(pred_idx.item() == tgt_idx)


def exact_match_f1(prediction: str, targets: List[str]) -> Tuple[float, float]:
    return (exact_match(prediction, targets), f1(prediction, targets))


def bleu(prediction: str, targets: List[str], lang=None, **kwargs: Any) -> float:
    import sacrebleu  # type: ignore

    if not lang:
        return sacrebleu.corpus_bleu([prediction], [targets], **kwargs).score

    tokenizer = "13a"
    if lang.startswith("zho"):
        tokenizer = "zh"
    elif lang.startswith("jpn"):
        tokenizer = "ja-mecab"
    elif lang.startswith("kor"):
        tokenizer = "ko-mecab"

    return sacrebleu.corpus_bleu(
        [prediction], [targets], tokenize=tokenizer, **kwargs
    ).score


def sentence_bleu(prediction: str, targets: List[str], **kwargs: Any) -> float:
    import sacrebleu  # type: ignore

    return sacrebleu.sentence_bleu(prediction, targets, **kwargs).score


def rouge_score(
    prediction: str,
    targets: List[str],
    types: Sequence[str] = ("rouge3", "rougeL", "rougeLsum"),
    **kwargs: Any,
) -> Dict[str, float]:
    from rouge_score import rouge_scorer  # type: ignore

    split_summaries: bool = kwargs.pop("split_summaries", True)
    scorer = rouge_scorer.RougeScorer(types, split_summaries=split_summaries, **kwargs)
    if hasattr(scorer, "score_multi"):
        scores = scorer.score_multi(targets, prediction)  # type: ignore
    else:
        assert len(targets) == 1, len(targets)
        scores = scorer.score(targets[0], prediction)
    avg_fmeasures = {s: scores[s].fmeasure for s in types}
    return avg_fmeasures


def ngram_score(prediction: str, source: str) -> Dict[str, float]:
    from lcm.evaluation.utils.segment_alignment import get_all_ngrams

    src_ngrams = get_all_ngrams(source, 1, 4)
    tgt_ngrams = get_all_ngrams(prediction, 1, 4)
    result = {}
    result["ngram_overlap"] = len(set(tgt_ngrams).intersection(src_ngrams)) / max(1, len(set(src_ngrams)))  # fmt: skip
    result["repetition_4"] = len(tgt_ngrams) / max(1, len(set(tgt_ngrams))) - 1  # fmt: skip
    return result
