#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Scenario dispatcher registry for FIA performance harness.

Each scenario module exposes:
    build(case: dict) -> tuple[tuple, dict]
        Returns (positional_args=(query, key, value), kwargs={...}) ready for
        torch_npu.npu_fused_infer_attention_score(*pos, **kw).
        Allocates NPU tensors. Must be called inside an NPU device context.

    validate(case: dict) -> None
        Static, no-NPU validator. Raises ValueError on illegal combinations.
        Used by profile_grid.py to prune the cartesian product.

The dispatcher is a plain dict mapping scenario name → module.
"""

from . import no_quant
from . import mla
from . import antiquant
from . import full_quant
from . import contiguous
from . import mask


REGISTRY = {
    "no_quant": no_quant,
    "mla_decode": mla,
    "mla_prefill_split": mla,
    "mla_prefill_combine": mla,
    "antiquant": antiquant,
    "full_quant": full_quant,
    "contiguous": contiguous,
    "mask": mask,
}


def build(case):
    scenario = case.get("scenario", "no_quant")
    if scenario not in REGISTRY:
        raise ValueError(f"unknown scenario {scenario!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[scenario].build(case)


def validate(case):
    scenario = case.get("scenario", "no_quant")
    if scenario not in REGISTRY:
        raise ValueError(f"unknown scenario {scenario!r}")
    REGISTRY[scenario].validate(case)
