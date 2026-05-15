# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

from typing import Dict, Optional, final

from fairseq2.nn.incremental_state import IncrementalState, IncrementalStateBag
from fairseq2.nn.transformer import FullAttentionState
from torch import Tensor
from torch.nn import Module


@final
class LCMIncrementalStateBag(IncrementalStateBag):  # type: ignore
    """Holds the module states during incremental decoding."""

    _module_states: Dict[Module, FullAttentionState]  # type: ignore

    def __init__(
        self, max_num_steps: int, *, capacity_increment: Optional[int] = 16
    ) -> None:
        super().__init__(
            max_num_steps=max_num_steps, capacity_increment=capacity_increment
        )

    def reorder(self, new_order: Tensor) -> None:
        """Reorder the module states.

        See :meth:`IncrementalState.reorder` for more information.
        """
        # FIXME Deal with reordering diffusion state bags here
        for state in self._module_states.values():
            state.reorder(new_order)

    def set_state(self, m: Module, state: IncrementalState) -> None:
        """Set the state of ``m``.
        :param m: The module.
        :param state: The state to store.
        There is no current call to `set_state` when the bag
        is frozen, but it's implemented here for completeness
        """
        super().set_state(m, state)
