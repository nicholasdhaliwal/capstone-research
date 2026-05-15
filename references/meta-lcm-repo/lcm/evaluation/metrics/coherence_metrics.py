# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#


# - [ ] contrastive perplexity
# - [ ] self-NLI
# - [ ] the momentum metric

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from fairseq2.typing import CPU, DataType, Device
from tqdm.auto import trange
from transformers import XLNetTokenizer

from lcm.datasets.sentence_splitting import ResplitSentenceSplitter
from lcm.evaluation.api import Scorer
from lcm.evaluation.metrics.sentence_fluency import PerplexityScorer
from lcm.evaluation.metrics.utils import (
    create_context_prediction_pairs,
    divide_chunks_as,
)
from lcm.evaluation.utils.hf import infer_cache_dir

try:
    from sgnlp.models.coherence_momentum import (
        CoherenceMomentumConfig,
        CoherenceMomentumModel,
        CoherenceMomentumPreprocessor,
    )

    # The installation is optional, because sgnlp has conflcting dependency and should be installed manually.
except ImportError:
    CoherenceMomentumModel, CoherenceMomentumConfig, CoherenceMomentumPreprocessor = (
        None,
        None,
        None,
    )


class MomentumCoherenceProcessor(Scorer):
    """
    A class for applying a momentum text coherence classifier (Jwalapuram et al, 2022) to multi-sentence texts.
    The model was trained to assign a high score to original news articles, and a low score, to ones with permuted sentences.
    Model: https://huggingface.co/aisingapore/coherence-momentum.
    Code: https://github.com/aisingapore/sgnlp.
    Note: its "preprocessor" downloads an xlnet-base-cased tokenizer.
    """

    def __init__(
        self,
        model_name: str = "aisingapore/coherence-momentum",
        outputs: str = "coherence_momentum",
        apply_sigmoid: bool = True,
        sigmoid_temperature: float = 3.0,
        class_id: int = 0,
        device: Device = CPU,
        dtype: Optional[DataType] = None,
        batch_size: int = 16,
        min_sents_to_split: int = 15,  # according to the paper, it was 10
        sents_split_size: int = 10,  # according to the paper
        split_stride: int = 5,  # the docs with less than 4 sentences were excluded
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            outputs=outputs,
            device=device,
            dtype=dtype,
            **kwargs,
        )
        if CoherenceMomentumModel is None:
            raise ValueError(
                "Please install the sgnlp package. Use no dependencies, if possible, e.g. `pip install sgnlp --no-dependencies`."
            )

        self.class_id = class_id
        self.batch_size = batch_size

        self.apply_sigmoid = apply_sigmoid
        self.sigmoid_temperature = sigmoid_temperature

        self.min_sents_to_split = min_sents_to_split
        self.sents_split_size = sents_split_size
        self.split_stride = split_stride

    def init_model(self):
        cache_dir = self.kwargs.get("cache_dir", infer_cache_dir())
        token = os.environ.get("HF_AUTH_TOKEN", None)
        offload_folder = self.kwargs.get("offload_folder", None)

        self.model_config = CoherenceMomentumConfig.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
            token=token,
        )

        # load the tokenizer "manually", so that we could feed it the right cache directory
        # rationale: the same was done in the sgnlp CoherenceMomentumPreprocessor anyway, see
        # https://github.com/aisingapore/sgnlp/blob/main/sgnlp/models/coherence_momentum/preprocess.py#L12
        parent_model_name = f"xlnet-{self.model_config.model_size}-cased"
        tokenizer = XLNetTokenizer.from_pretrained(
            parent_model_name,
            cache_dir=cache_dir,
            token=token,
        )
        self.preprocessor = CoherenceMomentumPreprocessor(
            self.model_config.model_size, self.model_config.max_len, tokenizer=tokenizer
        )

        self.model = CoherenceMomentumModel.from_pretrained(
            self.model_name,
            config=self.model_config,
            cache_dir=cache_dir,
            token=token,
            offload_folder=offload_folder,
        ).to(self.device)

        self.splitter = ResplitSentenceSplitter()

    def score_texts(
        self,
        texts: Sequence[str],
        references: Optional[Sequence[str]] = None,
        show_progress: bool = False,
    ) -> np.ndarray:  # type: ignore
        """
        Output an array of coherence values (one for each text).
        The coherence values can be any real numbers and are guaranteed to be comparable only for similar texts (i.e. are not calibrated).
        """
        # Step 1: split the texts into chunks of manageable size (up to about 600 tokens)
        input_chunks = []
        chunks_per_doc = []
        for doc in texts:
            sents = self.splitter(doc)
            if len(sents) <= self.min_sents_to_split:
                # use the full doc
                input_chunks.append(sents)
                chunks_per_doc.append(1)
            else:
                # split the doc into blocks of 10 sents, with stride of 5
                chunks = [
                    sents[start : start + self.sents_split_size]
                    for start in range(
                        0,
                        len(sents) - self.split_stride,
                        self.sents_split_size - self.split_stride,
                    )
                ]
                input_chunks.extend(chunks)
                chunks_per_doc.append(len(chunks))

        # Step 2: score the chunks
        per_chunk_results = self._score_texts(input_chunks, show_progress=show_progress)

        # Step 3: average per-chunk results into document scores, and optionally rescale from 0 to 1
        per_doc_results = []

        prev_start = 0
        for doc_len in chunks_per_doc:
            start = prev_start
            end = prev_start + doc_len
            per_doc_results.append(per_chunk_results[start:end].mean())
            prev_start = end

        if self.apply_sigmoid:
            per_doc_results = 1 / (
                1 + np.exp(-np.array(per_doc_results) / self.sigmoid_temperature)
            )

        return np.array(per_doc_results)

    def _score_texts(
        self, input_chunks: Sequence[Sequence[str]], show_progress: bool = True
    ) -> np.ndarray:
        results = []
        range_fn = trange if show_progress else range
        for i in range_fn(0, len(input_chunks), self.batch_size):  # type: ignore
            # TODO: maybe, insert tokenization spaces, because it seems the training data for this model has be preprocessed this way.
            batch = self.preprocessor(
                [" ".join(chunk) for chunk in input_chunks[i : i + self.batch_size]]
            )
            with torch.inference_mode():
                out = self.model.get_main_score(
                    batch["tokenized_texts"].to(self.model.device)
                )  # this could be any real number
                scores = out.cpu().numpy()
            results.append(scores)
        per_chunk_results = np.concatenate(results)
        return per_chunk_results

    def score_next_sentences(
        self,
        gt_docs: Sequence[Sequence[str]],
        pred_docs: Sequence[Sequence[str]],
        max_ctx_len: int = 10,
        show_progress: bool = False,
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Given paired lists of ground truth context documents and predicted continuation documents,
        produce a list of per-sentence scores (differences in coherence where the last sentence in a ground truth prefix
        is replaced with a predicted sentence) and trivial (unit) weights.
        """
        gt_suffixes = [
            gt_doc[-len(pred_doc) :] for gt_doc, pred_doc in zip(gt_docs, pred_docs)
        ]
        # a list of prefixes of the ground truth documents with a last sentence replaced by a prediction
        predicted_prefixes = [
            list(ctx_sents) + [pred_sent]
            for ctx_sents, pred_sent in create_context_prediction_pairs(
                gt_docs, pred_docs, max_ctx_len=max_ctx_len
            )
        ]
        # a list of prefixes of the ground truth documents with a last sentence as is
        gt_prefixes = [
            list(ctx_sents) + [pred_sent]
            for ctx_sents, pred_sent in create_context_prediction_pairs(
                gt_docs, gt_suffixes, max_ctx_len=max_ctx_len
            )
        ]
        scores_pred = self._score_texts(predicted_prefixes, show_progress=show_progress)
        scores_gt = self._score_texts(gt_prefixes, show_progress=show_progress)

        differences_flat = scores_pred - scores_gt
        weights_flat = np.ones_like(differences_flat)

        scores = divide_chunks_as(differences_flat, pred_docs)
        weights = divide_chunks_as(weights_flat, pred_docs)
        return scores, weights


class PrefixContrastiveSentencePerplexityScorer(PerplexityScorer):
    """
    A scorer for checking, for a given text, how much each next sentence is conditioned on the previous ones.
    Concretely, it measures, how on average the perplexity of each sentence decreases when the n preceding sentences are given.
    In other words, it implements a particular way of measuring mutual information between the sentences.
    The perplexities are normalized in the end per-token, so the output is the average per-token inter-sentential mutual information.
    """

    def init_model(self):
        super().init_model()
        self.splitter = ResplitSentenceSplitter()

    def score_texts(
        self,
        texts: Sequence[str],
        references: Optional[Sequence[str]] = None,
        show_progress: bool = False,
        # this token is used to mark the beginning and end of the evaluated sentence.
        # we do not use a space, because it is sometimes glued to the next word
        bos_token: Optional[str] = "\n",
        eos_token: Optional[str] = "\n",
        # this is the most typical token to separate the sentences of the context
        sep_token: str = " ",
        num_context_sentences: int = 3,
        exclude_first_sentence: bool = True,
    ) -> np.ndarray:  # type: ignore
        """
        Compute per-token inter-sentence mutual informations, averaged for each document.
        The output can be any real number (typically small); positive numbers indicate more coherent texts.
        """

        # Step 1: create small context + suffix pairs based on all the documents
        doc_lens = []
        all_suffixes = []
        all_prefixes = []
        all_contrastive_scores: List[float] = []
        all_sent_lens: List[float] = []
        for doc in texts:
            sents = self.splitter(doc)
            doc_lens.append(len(sents))
            suffixes = [bos_token + sent + eos_token for sent in sents]  # type: ignore
            prefixes = [
                sep_token.join(sents[max(0, i - num_context_sentences) : i])
                for i in range(len(sents))
            ]
            all_suffixes.extend(suffixes)
            all_prefixes.extend(prefixes)

        # Step 2: score the suffixes with and without contexts
        range_fn = trange if show_progress else range
        for i in range_fn(0, len(all_suffixes), self.batch_size):  # type: ignore
            suffixes = all_suffixes[i : i + self.batch_size]
            prefixes = all_prefixes[i : i + self.batch_size]

            loss_reductions, suffix_lens = self._score_prefix_suffix_pairs(
                prefixes, suffixes
            )

            all_contrastive_scores.extend(loss_reductions)
            all_sent_lens.extend(suffix_lens)

        # Step 3: aggregate the chunks into per-document scores
        per_doc_scores = []
        prev_start = 0
        for doc_len in doc_lens:
            start = prev_start + int(exclude_first_sentence)
            end = prev_start + doc_len
            total_loss_decrease = sum(all_contrastive_scores[start:end])
            total_tokens = sum(all_sent_lens[start:end])
            per_doc_scores.append(total_loss_decrease / max(total_tokens, 1))
            prev_start = end

        return np.array(per_doc_scores)

    def _get_shifted_token_losses(self, inputs, out):
        shift_logits = out.logits[..., :-1, :].contiguous()
        shift_labels = inputs.input_ids[..., 1:].contiguous()

        flat_logits = shift_logits.view(-1, self.model.config.vocab_size)
        flat_labels = shift_labels.view(-1)
        loss = torch.nn.functional.cross_entropy(
            flat_logits,
            target=flat_labels,
            ignore_index=self.tokenizer.pad_token_id,
            reduction="none",
        ).view(shift_labels.shape)
        return loss

    def _score_prefix_suffix_pairs(
        self, prefixes: Sequence[str], suffixes: Sequence[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """For each prefix and suffix, compute their total mutual information and suffix length."""
        suffix_inputs = self.tokenizer(
            suffixes, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)
        total_inputs = self.tokenizer(
            [p + s for p, s in zip(prefixes, suffixes)],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        # lengths of the prefixes to exclude from the conditional entropy computation
        prefix_lens = [
            sum(mask)
            for mask in self.tokenizer(prefixes, padding=True)["attention_mask"]
        ]
        # lengths of the prefixes that do participate in the conditional entropy computation
        suffix_lens = [
            total_inputs.attention_mask[i].sum().item() - pl - 1
            for i, pl in enumerate(prefix_lens)
        ]

        with torch.inference_mode():
            suffix_out = self.model(**suffix_inputs)
            total_out = self.model(**total_inputs)

            loss_wo_context = self._get_shifted_token_losses(suffix_inputs, suffix_out)
            loss_with_context = self._get_shifted_token_losses(total_inputs, total_out)
            # exclude the target elements that were truncated in the with-context batch from the loss computation
            for i, suffix_len in enumerate(suffix_lens):
                loss_wo_context[i, suffix_len:] = 0
            # exclude the context elements from the loss computation
            for i, prefix_len in enumerate(prefix_lens):
                loss_with_context[i, :prefix_len] = 0

            diffs = loss_wo_context.sum(axis=1) - loss_with_context.sum(axis=1)
        return diffs.cpu().numpy(), np.array(suffix_lens)

    def score_next_sentences(
        self,
        gt_docs: Sequence[Sequence[str]],
        pred_docs: Sequence[Sequence[str]],
        max_ctx_len: int = 10,
        show_progress: bool = False,
        # this token is used to mark the beginning and end of the evaluated sentence.
        # we do not use a space, because it is sometimes glued to the next word
        bos_token: str = "\n",
        eos_token: str = "\n",
        # this is the most typical token to separate the sentences of the context
        sep_token: str = " ",
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Given paired lists of ground truth context documents and predicted continuation documents,
        produce a list of per-sentence scores (mutual informations) and their weights (numbers of tokens).
        The average per-token mutual information in a document (or a corpus) equals the mean of scores weighted by the weights.
        """
        context_pairs = list(
            create_context_prediction_pairs(gt_docs, pred_docs, max_ctx_len=max_ctx_len)
        )
        scores_flat: List[float] = []
        weights_flat: List[float] = []

        range_fn = trange if show_progress else range
        for i in range_fn(0, len(context_pairs), self.batch_size):  # type: ignore
            batch = context_pairs[i : i + self.batch_size]
            prefixes = [sep_token.join(prefix_sents) for prefix_sents, _ in batch]
            suffixes = [bos_token + suffix_sent + eos_token for _, suffix_sent in batch]

            loss_reductions, suffix_lens = self._score_prefix_suffix_pairs(
                prefixes, suffixes
            )

            scores_flat.extend(
                [
                    total_diff / num_tokens
                    for total_diff, num_tokens in zip(loss_reductions, suffix_lens)
                ]
            )
            weights_flat.extend(suffix_lens)

        scores = divide_chunks_as(scores_flat, pred_docs)
        weights = divide_chunks_as(weights_flat, pred_docs)
        return scores, weights
