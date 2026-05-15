# Copyright (c) Meta Platforms, Inc. and affiliates.
#

import logging
import textwrap
from pathlib import Path
from typing import Optional

from lcm.datasets.configs import (
    ColumnsNames,
    JSONDatasetConfig,
)
from lcm.evaluation.api import (
    PREDICTION_TEXT_COLUMN,
    Example,
)
from lcm.evaluation.metrics import rouge_score
from lcm.evaluation.tasks import register_task
from lcm.evaluation.tasks.base import GenerationTaskConfig
from lcm.evaluation.utils.common import evaluate
from lcm.evaluation.utils.data_utils import as_py

FORMS = ["", "inverse_"]

logger = logging.getLogger("lcm.evaluation.tasks.slum WARMUP")

PROMPT_TEMPLATE: str = textwrap.dedent(
    """\
    {% for x in few_shot -%}
    Summarize this article in one sentence.

    Article: {{ x["text"] }}
    Summary: {{ x["_target_text_column"] }}

    {% endfor -%}
    Summarize this article in one sentence.

    Article: {{ text }}
    Summary:"""
)

LANGUAGES = [
    "oromo",
    "french",
    "amharic",
    "arabic",
    "azerbaijani",
    "bengali",
    "burmese",
    "chinese_simplified",
    "chinese_traditional",
    "welsh",
    "english",
    "kirundi",
    "gujarati",
    "hausa",
    "hindi",
    "igbo",
    "indonesian",
    "japanese",
    "korean",
    "kyrgyz",
    "marathi",
    "spanish",
    "scottish_gaelic",
    "nepali",
    "pashto",
    "persian",
    "pidgin",
    "portuguese",
    "punjabi",
    "russian",
    "serbian_cyrillic",
    "serbian_latin",
    "sinhala",
    "somali",
    "swahili",
    "tamil",
    "telugu",
    "thai",
    "tigrinya",
    "turkish",
    "ukrainian",
    "urdu",
    "uzbek",
    "vietnamese",
    "yoruba",
]


# map XLSUM supported languages to NLLB language code
xlsum_languages_to_code = {
    "oromo": "gaz_Latn",  # West Central Oromo
    "french": "fra_Latn",
    "amharic": "amh_Ethi",
    "arabic": "arb_Arab",  # Modern Standard Arabic
    "azerbaijani": "azj_Latn",  # North Azerbaijani
    "bengali": "ben_Beng",
    "burmese": "mya_Mymr",
    "chinese_simplified": "zho_Hans",
    "chinese_traditional": "zho_Hant",  # note so many issues with zho_Hant and yue_Hant with NLLB
    "welsh": "cym_Latn",
    "english": "eng_Latn",
    "kirundi": "run_Latn",  # Rundi
    "gujarati": "guj_Gujr",
    "hausa": "hau_Latn",
    "hindi": "hin_Deva",
    "igbo": "ibo_Latn",
    "indonesian": "ind_Latn",
    "japanese": "jpn_Jpan",
    "korean": "kor_Hang",
    "kyrgyz": "kir_Cyrl",
    "marathi": "mar_Deva",
    "spanish": "spa_Latn",
    "scottish_gaelic": "gla_Latn",
    "nepali": "npi_Deva",
    "pashto": "pbt_Arab",  # Southern Pashto
    "persian": "pes_Arab",  # Western Persian
    "portuguese": "por_Latn",
    "punjabi": "pan_Guru",  # Eastern Panjabi
    "russian": "rus_Cyrl",
    "serbian_cyrillic": "srp_Cyrl",
    "sinhala": "sin_Sinh",
    "somali": "som_Latn",
    "swahili": "swh_Latn",
    "tamil": "tam_Taml",
    "telugu": "tel_Telu",
    "thai": "tha_Thai",
    "tigrinya": "tir_Ethi",
    "turkish": "tur_Latn",
    "ukrainian": "ukr_Cyrl",
    "urdu": "urd_Arab",
    "vietnamese": "vie_Latn",
    "yoruba": "yor_Latn",
    # THE LANGUAGES BELOW ARE NOT SUPPORTED AND ARE EXPECTED TO PERFORM POORLY
    "pidgin": "eng_Latn",  # West African Pidgin English is not supported by NLLB / SONAR
    "serbian_latin": "srp_Cyrl",  # Only Serbian in Cyrillic script is supported by NLLB / SONAR
    "uzbek": "uzn_Latn",  # Only Uzbek in Latin script is supported by NLLB / SONAR, but XLSUM Uzbek is Cyrillic
}

SPLITS = ["test", "sampled_test", "validation", "sampled_validation"]


def postprocess(x: Example) -> Example:
    print("postprocessing", list(x))
    prediction_texts = x[PREDICTION_TEXT_COLUMN][0]
    if isinstance(prediction_texts, str):  # type: ignore
        prediction_text = prediction_texts.split(".")[0]
    else:
        prediction_text = prediction_texts[0]
    targets = as_py(x[ColumnsNames.target_text_column.value])
    if isinstance(targets, str):
        targets = [targets]
    return {
        "prediction": prediction_text,
        "targets": targets,
    }


@register_task("xlsum_llm.{lang}.{split}", {"split": SPLITS, "lang": LANGUAGES})
def get_task_config_llm(
    dataset: JSONDatasetConfig,
    dataset_dir: str,
    lang: str,
    split: str,
    num_shots: int = 0,
    max_gen_len: int = 512,
    max_gen_len_ratio: Optional[float] = None,
    max_prompt_len: int = 4096,
) -> GenerationTaskConfig:
    file_path = f"{dataset_dir}/xlsum/{lang}/{split}.jsonl"

    # In case the user specifies the directory that point directly to the task dir
    if not Path(file_path).exists():
        file_path = f"{dataset_dir}/{lang}/{split}.jsonl"

    assert Path(file_path).exists(), f"{file_path} not found."

    dataset.file_path = file_path

    dataset.prompt_template = PROMPT_TEMPLATE
    dataset.target_text_column = "summary"
    return GenerationTaskConfig(
        dataset=dataset,
        num_few_shot=num_shots,
        few_shot_file=f"{dataset_dir}/xlsum/{lang}/validation.jsonl",
        postprocess_fn=postprocess,
        metric_fns=[
            evaluate(
                rouge_score,
                inputs=("prediction", "targets"),
                outputs=("rouge2", "rougeL", "rougeLsum"),
                types=("rouge2", "rougeL", "rougeLsum"),
            )
        ],
        max_gen_len=max_gen_len,
        max_gen_len_ratio=max_gen_len_ratio,
        max_prompt_len=max_prompt_len,
    )
