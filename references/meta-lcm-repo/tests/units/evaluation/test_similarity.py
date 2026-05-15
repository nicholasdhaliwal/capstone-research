# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

import math
import unittest
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch

from lcm.evaluation.metrics import (
    edit_distance,
    longest_common_substring,
    memorization_score,
    nltk_sentence_bleu,
)
from lcm.evaluation.metrics.similarity import mse_constrative_accuracy


class WhitespaceTokenizer:
    def __call__(self, s: str) -> List[str]:
        return s.split()


class StubTokenizer:
    def __init__(self) -> None:
        self.tok_to_id: Dict[str, int] = {}

    def __call__(self, s: str) -> List[int]:
        toks = s.split()
        tokenized = []
        for tok in toks:
            if tok not in self.tok_to_id:
                self.tok_to_id[tok] = len(self.tok_to_id)
            tokenized.append(self.tok_to_id[tok])
        return tokenized


def get_examples() -> List[Tuple[str, str]]:
    return [
        (
            "John Doe and he lives in the United Kingdom .",
            "Jane Doe and she lives in the United States .",
        ),
        (
            "the ratio of the radius of a circle to its",
            "a famous decimal that never enters a repeating pattern .",
        ),
        (
            "Billy Bob . They are on trial for tax fraud",
            "Billy Bob . Are they really on trial for tax",
        ),
        (
            "Billy Bob . They are on trial for tax fraud",
            "Billy Bob . They are on trial for tax fraud",
        ),
    ]


class TestSimilarity(unittest.TestCase):
    def run_scenarios(
        self,
        score_fn: Callable[..., Any],
        expected: List[float],
    ) -> None:
        for i, (q, g) in enumerate(get_examples()):
            # Ensure it works with str tokens and token ids (List[int])
            for tok_cls in [WhitespaceTokenizer, StubTokenizer]:
                with self.subTest(tokenizer=tok_cls.__name__, example_id=i):
                    tok = tok_cls()
                    self.assertAlmostEqual(
                        expected[i],
                        score_fn(tok(q), tok(g)),  # type: ignore
                        places=4,  # type: ignore
                    )

    def test_bleu(self) -> None:
        self.run_scenarios(
            score_fn=nltk_sentence_bleu,
            expected=[0.3247, 0.0211, 0.3799, 1.0],
        )

    def test_edit_distance(self) -> None:
        self.run_scenarios(
            score_fn=edit_distance,
            expected=[3.0, 9.0, 4.0, 0.0],
        )

    def test_memorization_score(self) -> None:
        self.run_scenarios(
            score_fn=memorization_score,
            expected=[0.7, 0.1, 0.3, 1.0],
        )

    def test_longest_common_substring(self) -> None:
        self.run_scenarios(
            score_fn=longest_common_substring,
            expected=[4.0, 1.0, 4.0, 10.0],
        )

    def test_mse_constrative_accuracy(self) -> None:
        q, g = get_examples()[-1]
        expected = 0.9
        tok = StubTokenizer()
        tokq = tok(q)
        tokg = tok(g)

        embed = torch.nn.Embedding(len(tok.tok_to_id), 5).requires_grad_(False)

        # Repeat the text to run the constrative against the first sentence as distractors
        embedq = embed(torch.tensor(tokq)).repeat(2, 1, 1)
        embedg = embed(torch.tensor(tokg)).repeat(2, 1, 1)

        acc = mse_constrative_accuracy(embedq, embedg)
        assert len(acc) == 20  # batch_size x seq_len

        mean_acc = np.asarray(acc).reshape(2, 10).mean(axis=1)[0]

        assert math.isclose(expected, mean_acc, abs_tol=0.01)


if __name__ == "__main__":
    unittest.main()
