# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#
#
# Wrappers for different Llama models


from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, cast

import numpy as np
import torch
from tqdm import tqdm
from transformers import GenerationConfig

from lcm.evaluation.api import Prediction, Prompts
from lcm.evaluation.predictors.huggingface import (
    HuggingfacePredictor,
    HuggingfacePredictorConfig,
)
from lcm.evaluation.utils.distributed import get_global_rank, rank_zero_warn
from lcm.utils.common import batched

LLAMA_STOP_STRINGS = [
    "</s>",
    "Let me know if you'd like me to make any further changes!",
]


@dataclass
class HFLlamaPredictorConfig(HuggingfacePredictorConfig):
    """
    Config to load Llama models via transformers API. It contains some additional
    parameters to control the generation process (see
    https://huggingface.co/docs/transformers/en/main_classes/text_generation).
    """

    system_prompt: Optional[str] = None

    @classmethod
    def predictor_class(cls):
        return HFLlamaPredictor


class HFLlamaPredictor(HuggingfacePredictor):
    def __init__(self, model: Any, tokenizer: Any, config: HFLlamaPredictorConfig):  # type: ignore
        super().__init__(model, tokenizer=tokenizer, config=config)
        self.stop_token_ids = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

    @staticmethod
    def from_config(config: HFLlamaPredictorConfig, **kwargs) -> "HFLlamaPredictor":  # type: ignore
        predictor = HuggingfacePredictor.from_config(config, **kwargs)
        return HFLlamaPredictor(predictor.model, predictor.tokenizer, config)

    def apply_chat_template(
        self, prompts: List[str], max_prompt_len: Optional[int] = None
    ) -> Tuple[Dict[str, Any], List[str]]:
        sys_prompt = getattr(self.config, "system_prompt", None)
        sys_msg = {"role": "system", "content": sys_prompt} if sys_prompt else None
        chats = []
        for prompt in prompts:
            chat = [sys_msg] if sys_msg is not None else []
            chat.append({"role": "user", "content": prompt})
            chats.append(chat)

        texts = self.tokenizer.apply_chat_template(
            chats, add_generation_prompt=True, tokenize=False
        )
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_prompt_len,
        )
        verbose_texts = self.tokenizer.batch_decode(
            inputs.input_ids, skip_special_tokens=True
        )
        return inputs, verbose_texts

    @torch.no_grad()
    def __call__(  # type: ignore
        self,
        prompts: Prompts,
        max_prompt_len: Optional[int] = None,
        max_gen_len: Optional[int] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        echo: bool = True,
        return_logprobs: bool = False,
        show_progress: bool = False,
        **kwargs,
    ) -> Sequence[Prediction]:
        if not all(isinstance(p, str) for p in prompts):  # type: ignore
            raise NotImplementedError(
                "Llama predictor only supports text prompts for now"
            )
        prompts_s = cast(List[str], prompts)
        kwargs["do_sample"] = temperature != 0
        if kwargs["do_sample"]:
            kwargs.update({"top_p": top_p, "top_k": top_k, "temperature": temperature})
        max_gen_len = int(max_gen_len or 0)
        min_gen_len = int(kwargs.get("min_gen_len", 1))
        config = GenerationConfig(
            # generate at least 1 token to avoid warning when max_gen_len = 0
            max_new_tokens=max(min_gen_len, max_gen_len),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.stop_token_ids,
            return_dict_in_generate=True,
            use_cache=True,
            stop_strings=LLAMA_STOP_STRINGS,
            repetition_penalty=self.config.repetition_penalty,  # type: ignore
            encoder_repetition_penalty=self.config.encoder_repetition_penalty,  # type: ignore
            no_repeat_ngram_size=self.config.no_repeat_ngram_size,  # type: ignore
            **kwargs,
        )

        order = np.argsort(np.array([len(prompt) for prompt in prompts_s]))
        iterator: Iterator[Tuple[int, str]] = iter([(i, prompts_s[i]) for i in order])

        disable_progress = not show_progress or get_global_rank() != 0
        progress = tqdm(total=len(prompts), disable=disable_progress, leave=False)
        predictions: List[Prediction] = [None for _ in range(len(prompts_s))]  # type: ignore

        for batch in batched(iterator, self.config.generator_batch_size):
            prompt_indices, prompt_texts = zip(*batch)
            inputs, verbose = self.apply_chat_template(prompt_texts, max_prompt_len)  # type: ignore

            input_ids = inputs.input_ids.to(self.model.device)  # type: ignore
            mask = inputs.attention_mask.to(self.model.device)  # type: ignore
            if "max_gen_len_ratio" in kwargs and kwargs["max_gen_len_ratio"]:
                config.max_new_tokens = max(min_gen_len, int(input_ids.shape[1] * kwargs["max_gen_len_ratio"]))  # fmt: skip

            outputs = self.model.generate(
                input_ids, config, attention_mask=mask, tokenizer=self.tokenizer
            )
            logprobs: Optional[List[List[float]]] = None
            if return_logprobs:
                prompt_outputs = self.model(input_ids, attention_mask=mask)
                scores = prompt_outputs.logits.log_softmax(dim=-1)[:, :-1, :]
                prefill = scores.gather(2, input_ids[:, 1:, None]).squeeze(-1)
                generated = self.model.compute_transition_scores(
                    outputs.sequences, outputs.scores, normalize_logits=True
                )
                logprobs = [p + g for p, g in zip(prefill.tolist(), generated.tolist())]

            output_texts = self.tokenizer.batch_decode(
                outputs.sequences, skip_special_tokens=True
            )
            for idx, prompt_idx in enumerate(prompt_indices):
                response = output_texts[idx][len(verbose[idx]) :]

                try:
                    encoded = self.tokenizer(response, return_offsets_mapping=True)
                    tokens = [response[s:e] for s, e in encoded["offset_mapping"]]
                    response_offsets = [s for s, _ in encoded["offset_mapping"]]
                except NotImplementedError:
                    rank_zero_warn(
                        f"offset_mapping not supported in {self.config.tokenizer_name}"
                    )
                    encoded = self.tokenizer(response, return_offsets_mapping=False)
                    tokens = encoded["input_ids"]
                    response_offsets = None
                predictions[prompt_idx] = Prediction(
                    text=response,
                    tokens=tokens,
                    text_offsets=response_offsets,
                    logprobs=logprobs[idx] if logprobs else None,
                )
            progress.update(len(batch))
        return predictions
