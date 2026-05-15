# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, cast

import hydra
import numpy as np
import torch
from tqdm import tqdm
from transformers import GenerationConfig

from lcm.evaluation.api import Prediction, Predictor, PredictorConfig, Prompts
from lcm.evaluation.utils.distributed import (
    get_global_rank,
    init_torch_distributed,
    rank_zero_warn,
)
from lcm.evaluation.utils.hf import infer_cache_dir, infer_hf_device_memory
from lcm.utils.common import batched, torch_type


@dataclass
class HuggingfacePredictorConfig(PredictorConfig):
    model_name: str = ""
    """Used to set the param `pretrained_model_name_or_path` of the AutoModel"""

    revision: str = "main"
    """Used to set the param `revision` of the AutoModel"""

    use_auth_token: Optional[str] = None
    """User authentication token to access gated model"""

    tokenizer_name: str = ""
    """Model name to load the tokenizer. This can be different from the AutoModel name"""

    tokenizer_revision: str = ""
    """Revision for the tokenizer"""

    generator_batch_size: int = 32
    """Size of the internal batch to run the generator. By default this equals to the
    batch size of the task"""

    model_parallel_size: int = 1
    """Use to allocate the memory for the model"""

    cache_dir: Optional[str] = None
    """Use to cache the downloaded HF model in local directory"""

    model_class: str = "AutoModelForCausalLM"
    """The actual model class to load"""

    tokenizer_class: str = "PreTrainedTokenizerFast"
    """The actual tokenizer class to load"""

    repetition_penalty: float = 1.0

    encoder_repetition_penalty: float = 1.0

    encoder_no_repeat_ngram_size: Optional[int] = None

    no_repeat_ngram_size: int = 0

    def __post_init__(self):
        if not self.tokenizer_name:
            self.tokenizer_name = self.model_name
        if not self.tokenizer_revision:
            self.tokenizer_revision = self.revision
        self.model_class = "transformers." + self.model_class
        self.tokenizer_class = "transformers." + self.tokenizer_class

    @classmethod
    def predictor_class(cls):
        return HuggingfacePredictor


class HuggingfacePredictor(Predictor):
    def __init__(
        self, model: Any, tokenizer: Any, config: HuggingfacePredictorConfig
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.config = config

    @staticmethod
    def from_config(
        config: HuggingfacePredictorConfig,  # type: ignore
        **kwargs,  # type: ignore
    ) -> "HuggingfacePredictor":  # type: ignore
        assert isinstance(config, HuggingfacePredictorConfig)
        max_memory = None
        port = kwargs.get("port", None)
        if torch.cuda.is_available():
            init_torch_distributed(port=port)
            max_memory = infer_hf_device_memory(config.model_parallel_size)
        dtype = torch_type(kwargs.get("dtype", None))

        if config.model_parallel_size > 1:
            device = "auto"
        else:
            default_device = "cuda" if torch.cuda.is_available() else "cpu"
            device = torch.device(kwargs.get("device", default_device))  # type: ignore

        model_cls = hydra.utils.get_class(config.model_class)
        tokenizer_cls = hydra.utils.get_class(config.tokenizer_class)
        cache_dir = config.cache_dir or infer_cache_dir()
        if config.use_auth_token:
            model = model_cls.from_pretrained(  # type: ignore
                pretrained_model_name_or_path=config.model_name,
                revision=config.revision,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map=device,
                cache_dir=cache_dir,
                max_memory=max_memory,
                token=config.use_auth_token,
                offload_folder="offload",
                offload_state_dict=True,
            )
            tokenizer = tokenizer_cls.from_pretrained(  # type: ignore
                pretrained_model_name_or_path=config.tokenizer_name,
                padding_side="left",
                truncation_side="left",
                trust_remote_code=True,
                cache_dir=config.cache_dir,
                token=config.use_auth_token,
                clean_up_tokenization_spaces=False,
            )
        else:
            model = model_cls.from_pretrained(  # type: ignore
                pretrained_model_name_or_path=config.model_name,
                revision=config.revision,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map=device,
                cache_dir=config.cache_dir,
                max_memory=max_memory,
                offload_folder="offload",
                offload_state_dict=True,
            )
            tokenizer = tokenizer_cls.from_pretrained(  # type: ignore
                pretrained_model_name_or_path=config.tokenizer_name,
                padding_side="left",
                truncation_side="left",
                trust_remote_code=True,
                cache_dir=config.cache_dir,
                clean_up_tokenization_spaces=False,
            )
        if tokenizer.pad_token_id is None:
            config_eos_token_id = model.config.eos_token_id
            if isinstance(config_eos_token_id, list):
                config_eos_token_id = config_eos_token_id[0]
            tokenizer.pad_token_id = (
                model.config.pad_token_id
                or tokenizer.eos_token_id
                or config_eos_token_id
            )

        return HuggingfacePredictor(model, tokenizer, config)

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
            raise NotImplementedError("Huggingface predictor only support text prompts")
        prompts_s = cast(List[str], prompts)
        kwargs["do_sample"] = temperature != 0
        if kwargs["do_sample"]:
            kwargs.update({"top_p": top_p, "top_k": top_k, "temperature": temperature})
        max_gen_len = int(max_gen_len or 0)
        min_gen_len = int(kwargs.get("min_gen_len", 1))
        config = GenerationConfig(
            max_new_tokens=max(min_gen_len, max_gen_len),
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=return_logprobs,
            use_cache=True,
            repetition_penalty=self.config.repetition_penalty,  # type: ignore
            encoder_repetition_penalty=self.config.encoder_repetition_penalty,  # type: ignore
            no_repeat_ngram_size=self.config.no_repeat_ngram_size,  # type: ignore
            encoder_no_repeat_ngram_size=self.config.encoder_no_repeat_ngram_size,  # type: ignore
            **kwargs,
        )
        order = np.argsort(np.array([len(prompt) for prompt in prompts_s]))
        iterator = iter([(i, prompts_s[i]) for i in order])

        disable_progress = not show_progress or get_global_rank() != 0
        progress = tqdm(total=len(prompts), disable=disable_progress, leave=False)
        predictions: List[Prediction] = [None for _ in range(len(prompts_s))]  # type: ignore

        for batch in batched(iterator, self.config.generator_batch_size):
            prompt_indices, prompt_texts = zip(*batch)
            tokenized_inputs = self.tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_prompt_len,
            )
            input_ids = tokenized_inputs.input_ids.to(self.model.device)  # type: ignore
            mask = tokenized_inputs.attention_mask.to(self.model.device)  # type: ignore
            if "max_gen_len_ratio" in kwargs and kwargs["max_gen_len_ratio"]:
                config.max_new_tokens = max(min_gen_len, int(input_ids.shape[1] * kwargs["max_gen_len_ratio"]))  # fmt: skip

            outputs = self.model.generate(input_ids, config, attention_mask=mask)
            logprobs: Optional[List[List[float]]] = None
            if return_logprobs:
                prompt_outputs = self.model(input_ids, attention_mask=mask)
                scores = prompt_outputs.logits.log_softmax(dim=-1)[:, :-1, :]
                prefill = scores.gather(2, input_ids[:, 1:, None]).squeeze(-1)
                generated = self.model.compute_transition_scores(
                    outputs.sequences, outputs.scores, normalize_logits=True
                )
                logprobs = [p + g for p, g in zip(prefill.tolist(), generated.tolist())]

            # The model is able to learn when to stop
            # In this case the output sequence will not have leading pad tokens
            if outputs.sequences.size(1) < input_ids.size(1):
                start_indices = [0] * len(mask)
            else:
                start_indices = (input_ids.size(1) - echo * mask.sum(dim=-1)).tolist()

            sequences = outputs.sequences.tolist()
            for idx, prompt_idx in enumerate(prompt_indices):
                start, end = start_indices[idx], input_ids.size(1) + max_gen_len
                text = self.tokenizer.decode(
                    sequences[idx][start:end], skip_special_tokens=True
                )

                # Some models repeat the prompt before the actual response
                if text.startswith(prompt_texts[idx]):
                    text = text[len(prompt_texts[idx]) :]
                text = text.strip()
                try:
                    encoded = self.tokenizer(text, return_offsets_mapping=True)
                    tokens = [text[s:e] for s, e in encoded["offset_mapping"]]
                    text_offsets = [s for s, _ in encoded["offset_mapping"]]
                except NotImplementedError:
                    rank_zero_warn(
                        f"offset_mapping is not supported in {self.config.tokenizer_name}"
                    )
                    encoded = self.tokenizer(text, return_offsets_mapping=False)
                    tokens = encoded["input_ids"]
                    text_offsets = None
                predictions[prompt_idx] = Prediction(
                    text=text,
                    tokens=tokens,
                    text_offsets=text_offsets,
                    logprobs=logprobs[idx][start : end - 1] if logprobs else None,
                )
            progress.update(len(batch))
        return predictions
