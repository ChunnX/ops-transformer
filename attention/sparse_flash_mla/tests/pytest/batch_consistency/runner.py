#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Consistency runner: wraps the existing ``call_npu`` with deterministic level 3.

Design rules:

* **Isolation** – the runner reads the real old deterministic level via
  ``torch_npu.npu._get_deterministic_level()``, sets level 3 before each NPU
  invocation, and restores the *real* old value in a ``finally`` block.
  If ``set_deterministic_level(3)`` fails, a ``RuntimeError`` is raised –
  the runner never silently continues with a wrong level.
* **Legacy contract** – ``call_npu`` still returns ``(output, softmax_lse)``;
  the runner wraps it into :class:`RunResult` but never changes the call
  signature or return type of the original function.
* **No pollution** – the runner imports ``call_npu`` from the existing
  ``batch.sparse_flash_mla_process`` module; no existing file is modified.
"""

import os
import sys
from typing import Any, Dict, Optional, Tuple

import torch_npu

# Import the existing call_npu without modifying it
_BATCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BATCH_DIR not in sys.path:
    sys.path.insert(0, _BATCH_DIR)
from batch import sparse_flash_mla_process  # noqa: E402

from .model import RunResult
from . import schema


_DETERMINISTIC_LEVEL_BATCH_CONSISTENCY = 3


class DeterministicLevelError(RuntimeError):
    """Raised when setting or reading the deterministic level fails."""


class BatchConsistencyUnsupportedError(RuntimeError):
    """Raised when the installed torch_npu/runtime rejects level 3."""


def _is_level3_unsupported(error: RuntimeError) -> bool:
    message = str(error).lower()
    return (
        "set_deterministic_level" in message
        and "only supports configuring 0/1/2" in message
    )


def _get_current_deterministic_level() -> int:
    """Read the current deterministic level via the real NPU API.

    Raises :class:`DeterministicLevelError` if the API is unavailable.
    """
    get_fn = getattr(torch_npu.npu, "_get_deterministic_level", None)
    if get_fn is None:
        raise DeterministicLevelError(
            "torch_npu.npu._get_deterministic_level is not available; "
            "cannot read current deterministic level"
        )
    return int(get_fn())


def _set_deterministic_strict(level: int) -> None:
    """Set deterministic level, raising on failure."""
    torch_npu.npu.set_deterministic_level(level)
    # Verify the level was actually set
    actual = _get_current_deterministic_level()
    if actual != level:
        raise DeterministicLevelError(
            f"set_deterministic_level({level}) did not take effect; "
            f"actual level is {actual}"
        )


def run_with_batch_consistency(
    input_data: Dict[str, Any],
    validate: bool = True,
) -> RunResult:
    """Run the main SMLA op on NPU with deterministic level 3.

    This function:

    1. Optionally validates ``input_data`` schema.
    2. Reads the current deterministic level (real old value).
    3. Sets ``deterministic_level = 3`` (batch consistency), raising on failure.
    4. Calls the existing ``sparse_flash_mla_process.call_npu``.
    5. Restores the *real* previous deterministic level in ``finally``.
    6. Returns a :class:`RunResult`.
    """
    if validate:
        schema.validate_schema(input_data)

    prev_level = _get_current_deterministic_level()
    try:
        try:
            _set_deterministic_strict(_DETERMINISTIC_LEVEL_BATCH_CONSISTENCY)
            output, softmax_lse = sparse_flash_mla_process.call_npu(input_data)
        except RuntimeError as error:
            if _is_level3_unsupported(error):
                raise BatchConsistencyUnsupportedError(
                    "The installed torch_npu/CANN runtime accepts level 3 lazily but "
                    "rejects it on the first NPU operation (only levels 0/1/2 are supported). "
                    "Batch-consistency NPU validation is unavailable on this environment."
                ) from error
            raise
    finally:
        # Enter the restoration path even if setting level 3 partially succeeds
        # and its verification raises.
        try:
            current_level = _get_current_deterministic_level()
        except Exception:
            current_level = None
        if current_level != prev_level:
            _set_deterministic_strict(prev_level)

    return RunResult(output=output, softmax_lse=softmax_lse, metadata=None)


def run_baseline(
    input_data: Dict[str, Any],
    validate: bool = True,
) -> Tuple[RunResult, Any]:
    """Run the baseline case and return (RunResult, baseline_origins)."""
    if validate:
        schema.validate_schema(input_data)

    result = run_with_batch_consistency(input_data, validate=False)
    origins = schema.build_baseline_origins(input_data)
    return result, origins


def _get_layout_q(input_data: Dict[str, Any]) -> str:
    layout_q = input_data["input"]["layout_q"]
    if isinstance(layout_q, (list, tuple)):
        layout_q = layout_q[0]
    return layout_q


def run_mode1(
    input_data: Dict[str, Any],
    order: Optional[list] = None,
    seed: Optional[int] = None,
    validate: bool = True,
) -> Tuple[Any, RunResult, RunResult, list]:
    """Run Mode 1: batch reorder consistency test.

    Returns (consistency_case, baseline_result, derived_result, comparison).
    """
    from .transform import transform_mode1_reorder
    from .compare import compare_consistency
    from .schema import get_q_lengths

    if validate:
        schema.validate_schema(input_data)

    # Baseline
    baseline_result, baseline_origins = run_baseline(input_data, validate=False)
    baseline_q_lengths = get_q_lengths(input_data)
    layout_q = _get_layout_q(input_data)

    # Transform
    case = transform_mode1_reorder(input_data, order=order, seed=seed)

    # Derived
    derived_result = run_with_batch_consistency(case.input_data, validate=False)

    # Compare
    compare_lse = input_data["params"].get("return_softmax_lse", False)
    comparison = compare_consistency(
        baseline_output=baseline_result.output,
        baseline_origins=baseline_origins,
        derived_output=derived_result.output,
        derived_origins=case.output_origins,
        layout_q=layout_q,
        baseline_q_lengths=baseline_q_lengths,
        compare_lse=compare_lse,
        baseline_lse=baseline_result.softmax_lse if compare_lse else None,
        derived_lse=derived_result.softmax_lse if compare_lse else None,
    )

    return case, baseline_result, derived_result, comparison


def run_mode2(
    input_data: Dict[str, Any],
    groups: Optional[list] = None,
    seed: Optional[int] = None,
    validate: bool = True,
) -> Tuple[list, RunResult, list, list, Any]:
    """Run Mode 2: batch split consistency test.

    Returns (cases, baseline_result, derived_results, sub_comparisons, merged_comparison).
    """
    from .transform import transform_mode2_split
    from .compare import compare_consistency, merge_split_outputs, merge_split_lse
    from .schema import get_q_lengths

    if validate:
        schema.validate_schema(input_data)

    # Baseline
    baseline_result, baseline_origins = run_baseline(input_data, validate=False)
    baseline_q_lengths = get_q_lengths(input_data)
    layout_q = _get_layout_q(input_data)
    compare_lse = input_data["params"].get("return_softmax_lse", False)

    # Transform
    cases = transform_mode2_split(input_data, groups=groups, seed=seed)

    # Derived: run each sub-case
    derived_results = []
    for case in cases:
        result = run_with_batch_consistency(case.input_data, validate=False)
        derived_results.append((result, case))

    # Merge sub-case outputs
    sub_outputs = [(r.output, c.output_origins, c.transform_meta) for r, c in derived_results]
    merged_output, merged_origins = merge_split_outputs(sub_outputs, layout_q, baseline_q_lengths)

    # Merge sub-case LSE if needed
    merged_lse = None
    if compare_lse:
        sub_lses = [(r.softmax_lse, c.output_origins) for r, c in derived_results]
        merged_lse, _ = merge_split_lse(sub_lses, layout_q, baseline_q_lengths)

    # Compare merged output (and LSE) with baseline
    merged_comparison = compare_consistency(
        baseline_output=baseline_result.output,
        baseline_origins=baseline_origins,
        derived_output=merged_output,
        derived_origins=merged_origins,
        layout_q=layout_q,
        baseline_q_lengths=baseline_q_lengths,
        compare_lse=compare_lse,
        baseline_lse=baseline_result.softmax_lse if compare_lse else None,
        derived_lse=merged_lse,
    )

    return cases, baseline_result, derived_results, [], merged_comparison
