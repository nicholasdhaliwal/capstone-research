#  Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
#

"""
LCM: Modular and Extensible Reasoning in an Embedding Space
Code base for training different LCM models.
"""

from fairseq2 import setup_extensions
from fairseq2.assets import default_asset_store

__version__ = "0.1.0.dev0"


def setup_fairseq2() -> None:
    default_asset_store.add_package_metadata_provider("lcm.cards")


# This call activates setup_fairseq2 and potentially other extensions,
setup_extensions()
