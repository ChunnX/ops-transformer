# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import importlib.util
from pathlib import Path

import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "mqfa_ci_pt_analysis",
    Path(__file__).resolve().parents[1] / "tools/analyze_ci_pt.py",
)
analysis = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analysis)


def test_nonfinite_errors_are_not_hidden():
    a = torch.tensor([float("inf"), float("-inf"), float("nan"), 0.0, 1.0])
    b = torch.tensor([float("inf"), float("-inf"), float("nan"), float("inf"), 2.0])
    metrics, _, failed = analysis.tensor_metrics(a, b, 0.0, 0.0)
    assert metrics["bad_nonfinite"] == 2
    assert metrics["failed_elements"] == 3
    assert failed.tolist() == [False, False, True, True, True]


def test_empty_and_shape_mismatch():
    metrics, _, _ = analysis.tensor_metrics(torch.empty(0), torch.empty(0), 0.0, 0.0)
    assert metrics["failed_elements"] == 0
    assert metrics["max_abs"] == 0.0
    with pytest.raises(ValueError, match="Shape mismatch"):
        analysis.tensor_metrics(torch.empty(0), torch.empty(1), 0.0, 0.0)


def test_fingerprint_tracks_inputs_not_test_options():
    inputs = {
        "q": torch.tensor([[1.0, 2.0]], dtype=torch.float16),
        "sinks": torch.tensor([400.0]),
    }
    original = analysis.input_fingerprint(inputs)
    assert analysis.input_fingerprint({**inputs, "require_fd": True}) == original
    assert (
        analysis.input_fingerprint({**inputs, "sinks": torch.tensor([401.0])})
        != original
    )


def test_row_error_positions():
    a = torch.zeros(2, 3, 4)
    b = a.clone()
    b[1, 2, 3] = 7.0
    _, delta, failed = analysis.tensor_metrics(a, b, 0.0, 0.0)
    rows = analysis.worst_rows(a, delta, failed, 1)
    assert [int(axis) for axis in rows[0]["row"]] == [1, 2]
    assert rows[0]["failed_elements"] == 1
