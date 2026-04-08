#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`estimate` mode: kernel duration estimate."""

from ...model.hardware import AscendHWSpec
from ...model.operator_params import OperatorParams
from ...pipeline.dag import build_pipeline_dag
from ...pipeline.kernel_time import estimate_kernel_time
from ...tiling.config import TilingConfig
from ..formatting import print_header


def run(params: OperatorParams, tiling: TilingConfig, hw: AscendHWSpec):
    """Mode 3: Kernel duration estimate."""
    print_header("KERNEL DURATION ESTIMATE")

    total_us = estimate_kernel_time(params, tiling, hw)
    T = params.T
    step = tiling.step_batch_size
    num_steps = (T + step - 1) // step

    print(f"Parameters: T={T}, He={params.He}, N={params.N}")
    print(f"stepBatchSize={step}, num_steps={num_steps}")
    print()
    print(f"Estimated kernel time: {total_us:.2f} us ({total_us / 1000:.3f} ms)")

    if num_steps > 1:
        result = build_pipeline_dag(params, tiling, hw)
        print(f"  Single step: {result.total_us:.2f} us")
        print(f"  Total ({num_steps} steps): {total_us:.2f} us")
