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

"""Small pytest-facing adapter around the shared consistency runner."""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import pytest
import torch

from .adapter import CaseAdapter
from .runner import BatchConsistencyUnsupportedError, ConsistencyRunner
from .schema import get_q_lengths
from .transform import InvalidTransformError, groups_from_split_sizes


SUPPORTED_MODES = ("reorder", "split", "token-split", "shape-change")


def parse_int_list(value: str, name: str) -> List[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError(f"{name} must be a comma-separated integer list") from error
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"{name} must contain positive integers")
    return values


def parse_shape_change(value: str) -> Tuple[int, int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("shape change must be common_tokens,derived_extra_tokens") from error
    if len(values) != 2 or values[0] <= 0 or values[1] < 0:
        raise ValueError(
            "shape change requires positive common_tokens and non-negative derived_extra_tokens"
        )
    return values[0], values[1]


def collect_case_matrix(prefix: str, default_dir: str) -> List[Tuple[str, str]]:
    if os.environ.get(f"{prefix}_CONSISTENCY_TEST", "0") != "1":
        return []
    selected = os.environ.get(
        f"{prefix}_CONSISTENCY_MODES", ",".join(SUPPORTED_MODES)
    )
    modes = [value.strip() for value in selected.split(",") if value.strip()]
    unknown = sorted(set(modes) - set(SUPPORTED_MODES))
    if unknown:
        raise ValueError(f"unsupported batch-consistency modes: {unknown}")

    single_case = os.environ.get(f"{prefix}_CONSISTENCY_CASE", "").strip()
    if not single_case:
        single_case = os.environ.get("QSAS_TESTCASE_PATH", "").strip()
    if single_case:
        paths = [single_case] if Path(single_case).is_file() else []
    else:
        case_dir_value = os.environ.get(f"{prefix}_PT_DIR", "")
        if prefix == "SMLA" and not case_dir_value:
            case_dir_value = os.environ.get("SMLA_PT_LOAD_PATH", "")
        case_dir = Path(case_dir_value or default_dir)
        paths = [str(path) for path in sorted(case_dir.glob("*.pt"))] if case_dir.is_dir() else []
    return [(path, mode) for path in paths for mode in modes]


def balanced_split(total: int) -> List[int]:
    if total < 2:
        return []
    first = total // 2
    return [first, total - first]


def run_selected_mode(
    runner: ConsistencyRunner,
    data: Dict[str, Any],
    mode: str,
    prefix: str,
) -> Dict[str, Any]:
    seed = int(os.environ.get(f"{prefix}_CONSISTENCY_SEED", "20260716"))
    if mode == "reorder":
        return runner.run_mode1(data, None, seed)

    if mode == "split":
        batch_size = CaseAdapter(data).get_batch_size()
        configured = os.environ.get(f"{prefix}_CONSISTENCY_BATCH_SPLIT", "").strip()
        sizes = parse_int_list(configured, "batch split") if configured else balanced_split(batch_size)
        if not sizes:
            pytest.skip("NOT_APPLICABLE: split requires at least two batches")
        return runner.run_mode2(data, groups_from_split_sizes(batch_size, sizes), seed)

    batch_id = int(os.environ.get(f"{prefix}_CONSISTENCY_MODE_BATCH", "0"))
    valid_lengths = get_q_lengths(data, valid_only=True)
    if batch_id < 0 or batch_id >= len(valid_lengths):
        raise ValueError(f"mode batch {batch_id} is outside [0, {len(valid_lengths)})")
    token_count = valid_lengths[batch_id]
    if mode == "token-split":
        configured = os.environ.get(f"{prefix}_CONSISTENCY_TOKEN_SPLIT", "").strip()
        sizes = parse_int_list(configured, "token split") if configured else balanced_split(token_count)
        if not sizes:
            pytest.skip("NOT_APPLICABLE: token-split requires at least two valid Q tokens")
        return runner.run_mode3(data, batch_id, sizes)

    configured = os.environ.get(f"{prefix}_CONSISTENCY_SHAPE_CHANGE", "").strip()
    if configured:
        common_tokens, derived_extra = parse_shape_change(configured)
    else:
        common_tokens = max(1, token_count - 1)
        derived_extra = token_count - common_tokens + 1
    return runner.run_mode4(data, batch_id, common_tokens, derived_extra)


def run_pytest_case(
    case_path: str,
    mode: str,
    prefix: str,
    operator: str,
    executor: Callable[[Dict[str, Any]], Any],
    precision_compare: Callable[[Any, Any], Any],
    set_device: Callable[[], None],
) -> None:
    data = torch.load(case_path, map_location="cpu", weights_only=False)
    detected = CaseAdapter(data).name
    if detected != operator:
        pytest.skip(f"NOT_APPLICABLE: PT belongs to {detected}, expected {operator}")
    runner = ConsistencyRunner(executor, precision_compare, set_device)
    try:
        report = run_selected_mode(runner, data, mode, prefix)
    except InvalidTransformError as error:
        pytest.skip(f"INVALID_TRANSFORM: {error}")
    except BatchConsistencyUnsupportedError as error:
        pytest.skip(str(error))
    print(report)
    if report.get("pass") is not True:
        pytest.fail(f"batch consistency failed: {report}")
