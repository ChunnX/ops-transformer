# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import torch

from validation.precision import _quantiles, analyze_accuracy, precision_standard


def test_bfloat16_opbase_thresholds_include_per_element_ulp_parameters():
    standard = precision_standard(torch.bfloat16)
    assert standard["rtol"] == 2 ** -6
    assert standard["atol"] == 2 ** -6
    assert standard["required_matched_ratio"] == 0.99
    assert standard["max_abs_error_limit_fixed"] == 1.0
    assert standard["max_abs_error_limit_scope"] == "per_element"
    assert standard["ulp_mantissa_bits"] == 7
    assert standard["ulp_min_exponent"] == -126


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


def test_bfloat16_hard_limit_scales_with_each_golden_magnitude():
    expected = torch.full((100, 1), 1024.0, dtype=torch.bfloat16)
    actual = expected.clone()
    actual[0, 0] = 1280.0
    report = analyze_accuracy(actual, expected, global_table=[[100]])

    assert report["passed"]
    assert report["matched_ratio"] == 0.99
    assert report["max_abs_error"] == 256.0
    assert report["hard_limit_violation_count"] == 0
    assert report["top_mismatches"][0]["ulp_at_abs_expected"] == 8.0
    assert report["top_mismatches"][0]["hard_limit"] == 256.0

    actual[0, 0] = 1536.0
    report = analyze_accuracy(actual, expected, global_table=[[100]])
    assert not report["passed"]
    assert report["failure_reasons"] == ["FAIL_MAX_ABS_ERROR"]
    assert report["hard_limit_violation_count"] == 1
    assert report["maximum_hard_error_ratio"] == 2.0


def test_large_diagnostic_quantiles_use_deterministic_sampling_only_for_percentiles():
    values = torch.arange(100, dtype=torch.float64)
    statistics = _quantiles(values, max_quantile_samples=10)
    assert statistics["quantiles_approximate"]
    assert statistics["total_elements"] == 100
    assert statistics["finite_elements"] == 100
    assert statistics["quantile_sample_elements"] == 10
    assert statistics["mean"] == 49.5
    assert statistics["max"] == 99.0


def test_small_diagnostic_quantiles_remain_exact():
    values = torch.tensor([0.0, 1.0, 2.0, 3.0, float("inf")])
    statistics = _quantiles(values, max_quantile_samples=10)
    assert not statistics["quantiles_approximate"]
    assert statistics["finite_elements"] == 4
    assert statistics["quantile_sample_elements"] == 4
    assert statistics["median"] == 1.5
