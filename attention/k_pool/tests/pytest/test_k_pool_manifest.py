from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from k_pool_golden import validate_inputs
from k_pool_test_utils import (
    make_invalid_manifest_inputs,
    make_manifest_inputs,
    run_pair,
)


CASE_ROOT_ENV = "KPOOL_CASE_ROOT"


def load_cases(name: str) -> list[dict]:
    root_text = os.environ.get(CASE_ROOT_ENV)
    if not root_text:
        pytest.skip(f"{CASE_ROOT_ENV} is not set")
    path = Path(root_text) / name
    if not path.is_file():
        pytest.skip(f"manifest does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("manifest", [
    "smoke_manifest.json",
    "typical_manifest.json",
    "full_manifest.json",
])
def test_normal_manifest(manifest: str):
    cases = load_cases(manifest)
    run_count = 0
    reject_count = 0
    for case in cases:
        if case["expected"] != "run":
            reject_count += 1
            continue
        inputs = make_manifest_inputs(case)
        run_pair(*inputs, cmp_ratio=int(case["cmp_ratio"]))
        run_count += 1
    assert run_count > 0
    if manifest == "full_manifest.json":
        assert reject_count == 30


def test_performance_manifest_contract_only():
    cases = load_cases("performance_manifest.json")
    assert 12 <= len(cases) <= 18
    assert {case["layout"] for case in cases} == {"BSH", "TH"}
    assert {int(case["cmp_ratio"]) for case in cases} == {2, 4, 8, 16, 32, 64, 128}


def test_invalid_manifest_contracts():
    cases = load_cases("full_manifest.json")
    invalid_cases = [case for case in cases if case["expected"] == "reject"]
    tested = 0
    deferred = 0
    for case in invalid_cases:
        try:
            args, kwargs = make_invalid_manifest_inputs(case)
        except NotImplementedError:
            deferred += 1
            continue
        try:
            validate_inputs(*args, **kwargs)
        except (TypeError, ValueError, NotImplementedError):
            pass
        else:
            pytest.fail(
                f"{case['case_id']} did not reject reason={case['invalid_reason']}"
            )
        tested += 1
    assert tested == 26
    assert deferred == 4
