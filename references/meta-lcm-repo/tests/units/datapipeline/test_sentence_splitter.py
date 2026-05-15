# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#


from lcm.datasets.sentence_splitting import (
    ResplitSentenceSplitter,
    deescape_special_chars,
    filter_empty_string,
    remove_emojis,
    remove_non_printable_chars,
    resplit,
)


def test_remove_emojis():
    assert remove_emojis("Hello 😊, 🤣 how are you? 🤔") == "Hello ,  how are you? "


def test_filter_empty_string():
    assert not filter_empty_string("This is a long sentence with multiple words.")
    assert filter_empty_string("     ")


def test_remove_non_printable_chars():
    assert (
        remove_non_printable_chars("Hello\nWorld. This is a test sentence.")
        == "HelloWorld. This is a test sentence."
    )


def test_deescape_special_chars():
    assert (
        deescape_special_chars("Hello\\nWorld. This is a test\\nsentence.")
        == "Hello\nWorld. This is a test\nsentence."
    )


def test_resplit():
    assert resplit(
        "This is a long sentence that should be split into multiple parts.",
        max_length=20,
        sep=" ",
    ) == [
        "This is a long ",
        "sentence that ",
        "should be split ",
        "into multiple parts.",
    ]


def test_ResplitSentenceSplitter():
    splitter = ResplitSentenceSplitter()
    assert splitter(
        "This is a long sentence. It should be split into two parts.", "eng", 200
    ) == ["This is a long sentence.", "It should be split into two parts."]

    assert splitter(
        "This is a long sentence?It should be split into two parts.", "eng", 50
    ) == ["This is a long sentence?", "It should be split into two parts."]
