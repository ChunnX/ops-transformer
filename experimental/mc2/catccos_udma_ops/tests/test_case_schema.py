# Copyright (c) 2026 Huawei Technologies Co., Ltd.

from pathlib import Path

import pytest

from validation.case_schema import (
    CaseManifestError, load_case_manifest, performance_workload_digest,
    read_resolved_cases, select_cases, write_resolved_cases,
)


CASES = Path(__file__).parents[1] / "validation" / "cases" / "alltoallv_grouped_mat_mul.yaml"
GROUPED_CASES = (
    Path(__file__).parents[1]
    / "validation"
    / "cases"
    / "grouped_mat_mul_all_to_allv.yaml"
)


def test_default_manifest_resolves_and_selects_modes():
    cases = load_case_manifest(CASES)
    assert len(cases) == 5
    assert len(select_cases(cases, mode="accuracy")) == 5
    assert len(select_cases(cases, mode="performance")) == 4
    assert all(case["output_all_to_allv"] for case in cases)
    assert select_cases(cases, mode="accuracy", include_tags=["normal"])[0]["id"].endswith("normal")


def test_unknown_case_is_rejected():
    with pytest.raises(CaseManifestError, match="unknown case"):
        select_cases(load_case_manifest(CASES), mode="accuracy", case_ids=["missing"])


def test_grouped_mat_mul_all_to_allv_manifest_uses_stricter_expert_limit():
    cases = load_case_manifest(GROUPED_CASES)
    assert len(cases) == 5
    assert all(case["operator"] == "grouped_mat_mul_all_to_allv" for case in cases)
    assert all(not case["output_all_to_allv"] for case in cases)
    assert len(select_cases(cases, mode="performance")) == 4


def test_gmm_only_round_trip_and_baseline_separation(tmp_path):
    case = load_case_manifest(GROUPED_CASES)[-1]
    fused_digest = performance_workload_digest(case)
    case["performance"]["gmm_only"] = False
    assert performance_workload_digest(case) == fused_digest
    case["performance"]["gmm_only"] = True
    path = tmp_path / "resolved_case.json"
    write_resolved_cases(path, [case])
    restored = read_resolved_cases(path)[0]
    assert restored["performance"]["gmm_only"] is True
    assert restored["shape"] == case["shape"]
    assert performance_workload_digest(restored) != fused_digest


@pytest.mark.parametrize("value", ["true", 1, None])
def test_gmm_only_flag_requires_boolean(tmp_path, value):
    case = load_case_manifest(GROUPED_CASES)[0]
    case["performance"]["gmm_only"] = value
    path = tmp_path / "resolved_case.json"
    write_resolved_cases(path, [case])
    with pytest.raises(CaseManifestError, match="gmm_only must be a boolean"):
        read_resolved_cases(path)


def test_gmm_only_rejects_undispatched_inputs(tmp_path):
    case = load_case_manifest(CASES)[0]
    case["performance"]["gmm_only"] = True
    path = tmp_path / "resolved_case.json"
    write_resolved_cases(path, [case])
    with pytest.raises(CaseManifestError, match="expert-packed"):
        read_resolved_cases(path)
