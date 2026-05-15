# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#


import codecs
import re
import typing as tp
from functools import lru_cache

import spacy
import torch
from sacremoses import MosesDetokenizer, MosesPunctNormalizer
from stopes.pipelines.monolingual.utils.sentence_split import get_split_algo
from stopes.utils.language_codes import language_code_to_short_code


def remove_emojis(text: str) -> str:
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f1e0-\U0001f1ff"  # flags (iOS)
        "\U00002702-\U000027b0"
        "\U000024c2-\U0001f251"
        "\U0001f900-\U0001f9ff"  # Supplemental Symbols and Pictographs
        "\U0001f700-\U0001f77f"  # Alchemical Symbols
        "\U0001f780-\U0001f7ff"  # Geometric Shapes Extended
        "\U0001f800-\U0001f8ff"  # Supplemental Arrows-C
        "\U0001fa00-\U0001fa6f"  # Chess Symbols
        "\U0001fa70-\U0001faff"  # Symbols and Pictographs Extended-A
        "\U0001f6c0-\U0001f6cf"  # Miscellaneous Symbols and Pictographs (part)
        "\U0001f6d0-\U0001f6d5"  # Miscellaneous Symbols and Pictographs (part)
        "\U0001f6f0-\U0001f6fa"  # Miscellaneous Symbols and Pictographs (part)
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)


def batched(inputs: tp.Iterable, batch_size=10000) -> tp.Iterable:
    batch = []
    for line in inputs:
        batch.append(line)
        if len(batch) == batch_size:
            yield batch
            batch = []
    yield batch


def filter_empty_string(text):
    return not any(char.isalnum() for char in text)


def remove_non_printable_chars(string):
    return re.sub(r"[^\x20-\x7E]", "", string)


def deescape_special_chars(string):
    return codecs.decode(string, "unicode_escape")


def resplit(text: str, max_length: int, sep: str) -> tp.List[str]:
    words = text.split(sep)
    result = []
    current_piece = ""

    for i, word in enumerate(words[:-1]):
        # Append separator back to each word except the last
        word += sep
        if len(current_piece) + len(word) <= max_length:
            current_piece += word
        else:
            if current_piece:
                result.append(current_piece)
            current_piece = word

    # Handle the last word separately to avoid adding an extra separator
    last_word = words[-1]
    if len(current_piece) + len(last_word) <= max_length:
        current_piece += last_word
    else:
        if current_piece:
            result.append(current_piece)
        current_piece = last_word

    if current_piece:
        result.append(current_piece)

    return result


@lru_cache
def get_moses_normalizers(lang):
    moses_lang = language_code_to_short_code(lang, try_replacing_with_macro=True)
    mpn = MosesPunctNormalizer(lang=moses_lang)
    mpn.substitutions = [(re.compile(r), sub) for r, sub in mpn.substitutions]
    md = MosesDetokenizer(lang=moses_lang)
    return mpn, md


@lru_cache
def get_splitter(lang: str, model_name: str = None):
    moses_lang = language_code_to_short_code(lang, try_replacing_with_macro=True)
    if model_name is None:
        model_name = (
            f"{moses_lang}_core_web_sm"
            if moses_lang == "en"
            else f"{moses_lang}_core_news_sm"
        )
    try:
        if torch.cuda.is_available():
            spacy.require_gpu()
        spacy_nlp = spacy.load(model_name, enable=["sentencizer"])
        spacy_nlp.add_pipe("sentencizer")

        def spacy_splitter(text):
            for batch in batched(text, batch_size=999_000):
                for sent in spacy_nlp("".join(batch)).sents:
                    yield str(sent)

        return spacy_splitter
    except ModuleNotFoundError:
        print(
            f"Spacy splitter not found for {lang}, switching to stopes implementation"
        )
        return get_split_algo(lang[:3], "default")


class ResplitSentenceSplitter:
    def __init__(
        self,
        fallback_separators=(".", "!", "?", "...", "\n", ";", ",", ":", ">", " "),
    ):
        self.fallback_separators = fallback_separators

    def __call__(
        self, document: str, lang: str = "eng_Latn", max_length: int = 200
    ) -> tp.List[str]:
        mpn, md = get_moses_normalizers(lang)
        # XXX: two below are not various language friendly
        # document = deescape_special_chars(document)
        # document = remove_non_printable_chars(document)
        document = remove_emojis(document)

        raw_sentences = get_splitter(lang)(document)
        for separator in self.fallback_separators or []:
            raw_sentences = [
                subchunk.strip()
                for sent in raw_sentences
                for subchunk in resplit(sent, max_length=max_length, sep=separator)
            ]

        return [
            mpn.normalize(md.detokenize(sent.strip().split()))
            for sent in raw_sentences
            if len(sent) > 1 and not filter_empty_string(sent)
        ]
