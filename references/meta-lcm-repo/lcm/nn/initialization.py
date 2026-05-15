# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
#

import math
from functools import partial
from typing import Literal, Optional

import torch
from fairseq2.nn.projection import Linear
from fairseq2.nn.transformer import TransformerNormOrder
from torch.nn import Module

SUPPORTED_INIT_TYPES = Literal[
    "xavier",
    "sonar",
    "zero",
    "trunc_normal",
    "kaiming_uniform",
    "none",
]


SONAR_STD = 0.006
# Most SONAR embeddings have a distribution with the mean close to 0 and std close to 0.006
# Initializing embedding-like parameters (e.g. end-of-text vector) from a similar distribution is recommended,
# to minimize their disruption of the model training


def get_init_fn(style: str = "xavier", sonar_std: float = SONAR_STD):
    if style == "xavier":
        return init_linear_xavier

    if style == "kaiming_uniform":
        return init_linear_kaiming_uniform

    if style == "sonar":
        return partial(init_linear_to_sonar, sonar_std=sonar_std)

    if style == "zero":
        return init_linear_zero

    if style == "trunc_normal":
        return init_linear_trunc_normal

    if style == "none":
        return None

    else:
        raise ValueError(f"Could not recognize initialization function {style}")


def init_linear_to_sonar(layer: Linear, sonar_std: float) -> None:
    """
    Initialize the post-lcm in such a way, that if it is fed layer-normed
    lcm outputs (with zero mean and unit variance), its outputs have zero
    mean and the variance of SONAR embeddings.
    """
    if layer.bias is not None:
        torch.nn.init.zeros_(layer.bias)

    std = sonar_std * (3 / layer.input_dim) ** 0.5

    torch.nn.init.uniform_(layer.weight, a=-std, b=std)


def init_linear_xavier(layer: Linear) -> None:
    torch.nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        torch.nn.init.zeros_(layer.bias)


def init_linear_zero(layer: Linear) -> None:
    torch.nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        torch.nn.init.zeros_(layer.bias)


def init_linear_trunc_normal(layer: Linear) -> None:
    torch.nn.init.trunc_normal_(layer.weight, std=1e-3)
    if layer.bias is not None:
        torch.nn.init.zeros_(layer.bias)


def init_linear_kaiming_uniform(layer: Linear) -> None:
    torch.nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))

    if layer.bias is not None:
        fan_in = layer.weight.size(1)

        m = 1
        if layer.weight.ndim > 2:
            for s in layer.weight.shape[2:]:
                m *= s

        fan_in *= m

        # We do not calculate the true standard deviation of the uniform
        # distribution (i.e. multiply with sqrt(3)). See
        # https://github.com/pytorch/pytorch/issues/57109#issuecomment-828847575.
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0

        torch.nn.init.uniform_(layer.bias, -bound, bound)


def parse_norm_order(var: str) -> TransformerNormOrder:
    norm_order: TransformerNormOrder
    if var == "pre":
        norm_order = TransformerNormOrder.PRE
    elif var == "post":
        norm_order = TransformerNormOrder.POST
    elif var == "normformer":
        norm_order = TransformerNormOrder.PRE_WITH_NORMFORMER
    else:
        raise ValueError(f"Unknown normalization order {var}")

    return norm_order


def parse_activation_fn(var: str = None) -> Optional[Module]:
    if var is None:
        return None

    activ_fn: Module

    if var == "relu":
        activ_fn = torch.nn.ReLU()
    elif var == "tanh":
        activ_fn = torch.nn.Tanh()
    elif var == "elu":
        activ_fn = torch.nn.ELU()
    elif var == "leaky_relu":
        activ_fn = torch.nn.LeakyReLU()
    elif var == "prelu":
        activ_fn = torch.nn.PReLU()
    elif var == "selu":
        activ_fn = torch.nn.SELU()
    elif var == "gelu":
        activ_fn = torch.nn.GELU()
    elif var == "silu":
        activ_fn = torch.nn.SiLU()
    elif var == "softsign":
        activ_fn = torch.nn.Softsign()
    elif var == "sigmoid":
        activ_fn = torch.nn.Sigmoid()
    elif var == "hardsigmoid":
        activ_fn = torch.nn.Hardsigmoid()
    else:
        raise ValueError(f"Unknown activation function {var}")

    return activ_fn
