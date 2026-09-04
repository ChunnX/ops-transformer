# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Generate resident BF16 inputs before validation launch loops."""

from __future__ import annotations


def make_tensor(shape, distribution, device, generator):
    import torch

    result = torch.empty(shape, dtype=torch.float32)
    if distribution["kind"] == "uniform":
        result.uniform_(distribution["low"], distribution["high"], generator=generator)
    else:
        result.normal_(distribution["mean"], distribution["std"], generator=generator)
    return result.to(dtype=torch.bfloat16, device=device)
