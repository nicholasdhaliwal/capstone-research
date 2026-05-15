# Copyright (c) Meta Platforms, Inc. and affiliates.
#
#


from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from fairseq2.typing import CPU, DataType, Device
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from lcm.evaluation.api import Scorer
from lcm.evaluation.utils.hf import infer_offline_model_dir
from lcm.utils.common import batched


class SeahorseScorer(Scorer):
    def __init__(
        self,
        model_name: str = "google/seahorse-large-q5",
        inputs: Tuple[str, ...] = ("prediction", "inputs"),
        bad_token_id: int = 497,
        good_token_id: int = 333,
        device: Device = CPU,
        dtype: Optional[DataType] = None,
        batch_size: int = 1,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            inputs=inputs,
            device=device,
            dtype=dtype,
            **kwargs,
        )
        self.batch_size = batch_size
        self.bad_token_id = bad_token_id
        self.good_token_id = good_token_id

    @staticmethod
    def prompt_seahorse(article: str, summary: str):
        return f"premise: {article} hypothesis: {summary}"

    @classmethod
    def default_outputs(cls, model_name: str) -> Tuple[str, ...]:
        question_id = model_name[-1]
        return (f"seahorse-q{question_id}",)

    def init_model(self):
        # For Seahorse, we do not use HF cache, as this does not save
        # all SPM model.
        model_root = infer_offline_model_dir(dtype=self.dtype)
        if model_root:
            model_name = str(Path(model_root).joinpath(self.model_name))
        else:
            model_name = self.model_name
        offload_folder = self.kwargs.get("offload_folder", None)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            clean_up_tokenization_spaces=False,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            offload_folder=offload_folder,
        ).to(self.device)

    def score_texts(
        self,
        texts: Sequence[str],
        references: Optional[Sequence[str]] = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        assert references, "Seahorse require references text (source documents)"
        batch_size = self.batch_size or len(texts)

        results: List[np.ndarray] = []

        pairs_iter = batched(zip(references, texts), batch_size)
        if show_progress:
            pairs_iter = tqdm(pairs_iter)
        for pairs_batch in pairs_iter:
            prompts = list(map(lambda x: self.prompt_seahorse(*x), pairs_batch))

            inputs = self.tokenizer(
                prompts, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)
            prefix = torch.tensor([[self.tokenizer.pad_token_id]] * len(prompts)).to(self.device)  # fmt: skip
            with torch.inference_mode():
                outputs = self.model(**inputs, decoder_input_ids=prefix)
                logits = outputs.logits[:, 0]
                norm_logits = torch.nn.functional.softmax(logits, dim=1)
                results.append(norm_logits[:, 333].cpu().numpy())
        return np.concatenate(results)
