# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import torch

from validation.precision import (
    _quantiles,
    analyze_accuracy,
    precision_standard,
    write_accuracy_artifacts,
)


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


def test_export_report_accepts_source_rank_by_local_expert_route_table():
    expected = torch.zeros((6, 2), dtype=torch.bfloat16)
    actual = expected.clone()
    report = analyze_accuracy(
        actual,
        expected,
        global_table=[[1, 2], [2, 1]],
    )

    assert report["passed"]
    assert [item["elements"] for item in report["by_local_expert"]] == [6, 6]
    assert [item["elements"] for item in report["by_source_rank"]] == [6, 6]


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


def test_nan_is_an_explicit_failure_with_actionable_location(tmp_path):
    expected = torch.zeros((4, 3), dtype=torch.bfloat16)
    actual = expected.clone()
    actual[2, 1] = float("nan")
    report = analyze_accuracy(actual, expected, global_table=[[4]])

    assert not report["passed"]
    assert "FAIL_ACTUAL_NONFINITE" in report["failure_reasons"]
    assert report["actual_nonfinite"]["total_nonfinite"] == 1
    assert report["actual_nonfinite"]["first_locations"][0] == {
        "flat_index": 7,
        "row": 2,
        "column": 1,
        "local_expert": 0,
        "source_rank": 0,
        "kind": "nan",
    }

    write_accuracy_artifacts(tmp_path, report)
    rows = (tmp_path / "nonfinite_values.csv").read_text(encoding="utf-8")
    assert "actual,7,2,1,0,0,nan" in rows


def test_mismatch_report_groups_contiguous_failing_rows_and_route_boundaries():
    expected = torch.zeros((8, 2), dtype=torch.bfloat16)
    actual = expected.clone()
    actual[1:3] = 2.0
    actual[5] = 3.0

    report = analyze_accuracy(actual, expected, global_table=[[4], [4]])
    row_runs = report["mismatch_row_runs"]

    assert row_runs["rows_with_mismatch"] == 3
    assert row_runs["first_row"] == 1
    assert row_runs["last_row"] == 5
    assert row_runs["total_runs"] == 2
    assert row_runs["runs"][0]["row_start"] == 1
    assert row_runs["runs"][0]["row_end"] == 3
    assert row_runs["runs"][0]["start_source_rank"] == 0
    assert row_runs["runs"][1]["row_start"] == 5
    assert row_runs["runs"][1]["start_source_rank"] == 1


def test_accuracy_artifacts_include_row_expert_and_source_breakdowns(tmp_path):
    expected = torch.zeros((8, 2), dtype=torch.bfloat16)
    actual = expected.clone()
    actual[1:3] = 2.0
    report = analyze_accuracy(actual, expected, global_table=[[4], [4]])

    write_accuracy_artifacts(tmp_path, report)

    assert "1,3,2" in (tmp_path / "mismatch_row_runs.csv").read_text()
    assert "local_expert,elements,mismatches" in (
        tmp_path / "by_local_expert.csv"
    ).read_text()
    assert "source_rank,elements,mismatches" in (
        tmp_path / "by_source_rank.csv"
    ).read_text()


def test_oversized_failure_dump_keeps_compact_failing_rows(tmp_path):
    expected = torch.zeros((8, 4), dtype=torch.bfloat16)
    actual = expected.clone()
    actual[3, 2] = 4.0
    report = analyze_accuracy(actual, expected, global_table=[[8]])

    write_accuracy_artifacts(
        tmp_path,
        report,
        dump_payload={
            "case": {"id": "compact"},
            "rank": 0,
            "actual": actual,
            "expected": expected,
            "large_input": torch.zeros(1024, dtype=torch.bfloat16),
            "local_table": torch.tensor([8], dtype=torch.int64),
        },
        max_dump_bytes=1,
    )

    assert not (tmp_path / "failure_dump.pt").exists()
    compact_path = tmp_path / "failure_dump_compact.pt"
    assert compact_path.exists()
    compact = torch.load(compact_path, weights_only=False)
    assert 3 in compact["selected_output_rows"]
    assert compact["actual_selected_rows"].shape[1] == 4
    assert compact["expected_selected_rows"].shape[1] == 4
    assert torch.equal(compact["local_table"], torch.tensor([8]))
