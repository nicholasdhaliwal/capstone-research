# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

import math
from typing import Optional

import torch
from fairseq2.nn.projection import Linear
from fairseq2.typing import DataType, Device
from torch import Tensor
from torch.nn import Module

from lcm.nn.initialization import parse_activation_fn


class DiTTimestepEncoder(Module):
    """
    Embeds scalar timesteps into vector representations.
    Based on DiT's `TimestepEmbedder`
    https://github.com/facebookresearch/DiT/blob/main/models.py
    """

    def __init__(
        self,
        embedding_dim: int,
        frequency_embedding_size: int = 256,
        activation_fn_name: str = "silu",
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ):
        super().__init__()

        self.dtype = dtype

        self.device = device

        self.embedding_dim = embedding_dim

        self.frequency_embedding_size = frequency_embedding_size

        self.fc1 = Linear(
            frequency_embedding_size,
            embedding_dim,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.nonlin = parse_activation_fn(activation_fn_name)
        self.fc2 = Linear(
            embedding_dim,
            embedding_dim,
            bias=True,
            device=device,
            dtype=dtype,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        torch.nn.init.normal_(self.fc1.weight, std=0.02)
        torch.nn.init.normal_(self.fc2.weight, std=0.02)

        if self.fc1.bias is not None:
            torch.nn.init.zeros_(self.fc1.bias)

        if self.fc2.bias is not None:
            torch.nn.init.zeros_(self.fc2.bias)

    @staticmethod
    def sinusoidal_timestep_embedding(
        timestep, frequency_embedding_size, max_period=10000
    ):
        """
        Create sinusoidal timestep embeddings.
        :param timestep: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param frequency_embedding_size: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.

        Based on DiT's `TimestepEmbedder`
        https://github.com/facebookresearch/DiT/blob/main/models.py
        """
        half = frequency_embedding_size // 2

        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=timestep.device)

        args = timestep[:, None].float() * freqs[None]

        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if frequency_embedding_size % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )

        return embedding

    def forward(self, timesteps: Tensor) -> Tensor:
        initial_size = timesteps.size()

        flat_timesteps = timesteps.view(-1, 1)

        t_freq = self.sinusoidal_timestep_embedding(
            flat_timesteps, self.frequency_embedding_size
        ).to(self.dtype)

        t_emb = self.fc1(t_freq)

        if self.nonlin is not None:
            t_emb = self.nonlin(t_emb)

        t_emb = self.fc2(t_emb)

        return t_emb.view(*initial_size, self.embedding_dim)
