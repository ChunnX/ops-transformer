#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-step kernel time estimation.

When the input batch exceeds stepBatchSize, the kernel processes data in
ceil(T/step) sequential steps. This module computes the aggregate time,
applying a small empirical overlap factor to the steady-state steps.
"""

from ..constants import PIPELINE_OVERLAP_FACTOR
from ..model.hardware import AscendHWSpec, ASCEND_910B
from ..model.operator_params import OperatorParams
from .dag import build_pipeline_dag


def estimate_kernel_time(
    params: OperatorParams,
    tiling,
    hw: AscendHWSpec = ASCEND_910B,
) -> float:
    """Estimate total kernel execution time in microseconds.

    Accounts for multi-step execution when T > stepBatchSize.
    """
    # Local helper to avoid importing from tiling.config (avoid cycle)
    def _ceil_div(a: int, b: int) -> int:
        if b == 0:
            return 0
        return (a + b - 1) // b

    T = params.T
    step = tiling.step_batch_size
    num_steps = _ceil_div(T, step)

    result = build_pipeline_dag(params, tiling, hw)
    single_step_us = result.total_us

    if num_steps <= 1:
        return single_step_us

    # Multi-step: first step + (num_steps-1) * steady_state
    # Steady state overlaps some init overhead (~5% from pipelining).
    # See perf.constants.PIPELINE_OVERLAP_FACTOR for details.
    steady_state_us = single_step_us * PIPELINE_OVERLAP_FACTOR
    return single_step_us + (num_steps - 1) * steady_state_us
