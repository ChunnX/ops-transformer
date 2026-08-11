# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import torch

from validation.precision import analyze_accuracy, precision_standard


def test_bfloat16_opbase_thresholds_are_fixed():
    standard = precision_standard(torch.bfloat16)
    assert standard["rtol"] == 2 ** -6
    assert standard["atol"] == 2 ** -6
    assert standard["required_matched_ratio"] == 0.99
    assert standard["max_abs_error_limit"] == 1.0


def test_exactly_one_percent_mismatch_passes_when_hard_limit_passes():
    expected = torch.zeros((100, 1), dtype=torch.bfloat16)
    actual = expected.clone()
    actual[0, 0] = 0.5
    report = analyze_accuracy(actual, expected, global_table=[[100]])
    assert report["passed"]
    assert report["matched_ratio"] == 0.99
    assert report["by_local_expert"][0]["mismatches"] == 1


def test_ratio_and_hard_limit_failures_are_reported_separately():
    expected = torch.zeros((100, 1), dtype=torch.bfloat16)
    actual = expected.clone()
    actual[:2, 0] = 2.0
    report = analyze_accuracy(actual, expected, global_table=[[100]])
    assert not report["passed"]
    assert report["failure_reasons"] == ["FAIL_MATCHED_RATIO", "FAIL_MAX_ABS_ERROR"]
    assert report["hard_limit_violation_count"] == 2
