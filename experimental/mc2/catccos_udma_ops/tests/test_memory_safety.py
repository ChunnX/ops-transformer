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


def test_mssanitizer_uses_single_launch_path_without_nan_poison():
    launcher = (
        PROJECT_ROOT / "scripts" / "run_mssanitizer.sh"
    ).read_text(encoding="utf-8")
    filter_position = launcher.index('command+=(--kernel-name="${kernel_name}")')
    worker_position = launcher.index('python3 "${PROJECT_DIR}/validation/run_accuracy.py"')
    assert "MSSANITIZER_KERNEL_NAME:-Grouped" in launcher
    assert filter_position < worker_position
    assert "run_memory_safety.sh" not in launcher
    assert "--mssanitizer" in launcher
    assert "MSSANITIZER_ROUTE_INPUT:-tensor" in launcher

    controller = (
        PROJECT_ROOT / "validation" / "run_accuracy.py"
    ).read_text(encoding="utf-8")
    worker = (
        PROJECT_ROOT / "validation" / "accuracy_worker.py"
    ).read_text(encoding="utf-8")
    assert "--mssanitizer requires exactly one selected case" in controller
    assert "if not mssanitizer_enabled:" in worker
    assert '"target_launches_per_rank": 1 if mssanitizer_enabled else None' in worker
    assert '"caller_provided_out": False if mssanitizer_enabled else None' in worker


def test_mssanitizer_wheel_enables_supported_ascend950_debug_information():
    root_cmake = (PROJECT_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    setup_source = (PROJECT_ROOT / "setup.py").read_text(encoding="utf-8")
    assert "option(CATCCOS_UDMA_MSSANITIZER" in root_cmake
    assert '"-g -DCATCCOS_UDMA_MSSANITIZER=1 -DASCENDC_DUMP=1"' in root_cmake
    assert 'CATCCOS_UDMA_ASCEND_SANITIZER_LINK_OPTIONS' not in root_cmake
    assert '"-g --cce-enable-sanitizer"' not in root_cmake
    assert "--cce-fatobj-link" not in root_cmake
    assert 'cmake_boolean_from_env("CATCCOS_UDMA_MSSANITIZER")' in setup_source
    for relative in (
        "csrc/all_to_allv_grouped_mat_mul_udma/ascend950/CMakeLists.txt",
        "csrc/grouped_mat_mul_all_to_allv_udma/ascend950/CMakeLists.txt",
    ):
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "${CATCCOS_UDMA_ASCEND_FLAGS}" in source


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
