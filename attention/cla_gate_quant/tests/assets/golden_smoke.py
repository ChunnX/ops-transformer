#!/usr/bin/env python3
"""Standalone TTK golden plugin for the fixed CLA-gate smoke case."""
from pathlib import Path
import importlib.util

_IMPL = Path(__file__).with_name("golden.py")
_SPEC = importlib.util.spec_from_file_location("cla_gate_full_golden", _IMPL)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

__golden__ = {
    "kernel": {
        "cla_gate_quant": "cla_gate_smoke_golden",
    }
}


def cla_gate_smoke_golden(
    global_out,
    local_out,
    global_gate_logits,
    local_gate_logits,
    *,
    dst_type=36,
    scale_alg=1,
    **kwargs,
):
    """Compute sigmoid-gate merge, then row-wise and col-wise MXFP8."""
    return _MODULE.cla_gate_quant_golden(
        global_out,
        local_out,
        global_gate_logits,
        local_gate_logits,
        dst_type=dst_type,
        scale_alg=scale_alg,
        **kwargs,
    )
