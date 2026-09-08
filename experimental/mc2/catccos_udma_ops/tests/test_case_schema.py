# Copyright (c) 2026 Huawei Technologies Co., Ltd.

from pathlib import Path

import pytest

from validation.case_schema import CaseManifestError, load_case_manifest, select_cases


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
