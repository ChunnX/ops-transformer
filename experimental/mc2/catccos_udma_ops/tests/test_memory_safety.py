# Copyright (c) 2026 Huawei Technologies Co., Ltd.

from pathlib import Path

import torch

from validation.memory_safety import (
    GuardedTensor,
    compare_fingerprints,
    nonfinite_summary,
    sampled_fingerprint,
)


PROJECT_ROOT = Path(__file__).parents[1]


def test_workspace_poison_excludes_sync_tail_and_is_registered():
    runtime_source = (
        PROJECT_ROOT
        / "csrc" / "all_to_allv_grouped_mat_mul_udma" / "ascend950"
        / "udma_runtime.cpp"
    ).read_text(encoding="utf-8")
    host_source = (
        PROJECT_ROOT
        / "csrc" / "all_to_allv_grouped_mat_mul_udma" / "ascend950"
        / "all_to_allv_grouped_mat_mul_udma.cpp"
    ).read_text(encoding="utf-8")
    assert "aclrtMemset(" in runtime_source
    assert "kKernelWorkspaceDataBytes" in runtime_source
    assert 'm.def("_poison_udma_workspace' in host_source


def test_nan_red_zones_detect_a_corrupted_suffix():
    guarded = GuardedTensor.allocate(
        (4, 8), dtype=torch.bfloat16, device="cpu", name="x",
        guard_elements=16, guard_kind="nan",
    )
    guarded.tensor.zero_()
    assert guarded.check()["passed"]
    guarded.storage[-1] = 0
    report = guarded.check()
    assert not report["passed"]
    assert report["suffix_corrupted_elements"] == 1


def test_exact_output_canary_and_payload_nan_are_independent():
    guarded = GuardedTensor.allocate(
        (2, 3), dtype=torch.bfloat16, device="cpu", name="out",
        guard_elements=4, guard_kind="exact", guard_value=-12352.0,
    )
    guarded.fill_payload(float("nan"))
    assert guarded.check()["passed"]
    assert torch.isnan(guarded.tensor).all()


def test_sampled_fingerprint_detects_sampled_input_mutation():
    tensor = torch.arange(32, dtype=torch.int64)
    before = sampled_fingerprint(tensor, max_elements=32)
    tensor[7] = 99
    after = sampled_fingerprint(tensor, max_elements=32)
    assert not compare_fingerprints(before, after)["passed"]


def test_nonfinite_summary_reports_locations_and_kinds():
    tensor = torch.zeros((2, 3), dtype=torch.float32)
    tensor[0, 2] = float("nan")
    tensor[1, 1] = float("inf")
    report = nonfinite_summary(tensor)
    assert report["total_nonfinite"] == 2
    assert not report["passed"]
    assert report["first_locations"] == [
        {"flat_index": 2, "row": 0, "column": 2, "kind": "nan"},
        {"flat_index": 4, "row": 1, "column": 1, "kind": "positive_inf"},
    ]
