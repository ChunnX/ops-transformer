#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backward-compat shim for the original perf_model.py.

The implementation has been split into the perf.model package. This module
re-exports the public API so existing imports (`from perf_model import ...`)
continue to work unchanged. New code should prefer the package paths:

    from perf.model.hardware import AscendHWSpec
    from perf.model.matmul_detailed import estimate_matmul_detailed
"""

from perf.model.hardware import (
    AscendHWSpec, CacheSpec,
    ASCEND_910B, ASCEND_950, CHIP_REGISTRY,
)
from perf.model.quant import (
    QuantMode, resolve_quant_mode, dtype_size,
    is_full_quant, is_int8_quant, is_mxfp8,
)
from perf.model.timing_types import MatMulTiming, VectorOpTiming
from perf.model.operator_params import OperatorParams
from perf.model.block_spec import MatmulBlockSpec, check_cache_fit, derive_stepK
from perf.model.matmul_legacy import estimate_matmul
from perf.model.matmul_detailed import estimate_matmul_detailed
from perf.model.matmul_split_kn import (
    SplitKNSpec, estimate_matmul_split_kn, estimate_mm2_split_k_cross_core,
)
from perf.model.matmul_split_m import SplitMSpec, estimate_matmul_split_m
from perf.model.matmul_k_outer import estimate_matmul_k_outer
from perf.model.vector_ops import estimate_vector_op
from perf.model.orchestration import (
    get_default_block_specs,
    estimate_all_stages,
    estimate_all_stages_split_m,
    estimate_all_stages_k_outer,
)
