# Copyright (c) 2026 Huawei Technologies Co., Ltd.

from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]
ASCEND950_ROOT = (
    PROJECT_ROOT
    / "csrc"
    / "all_to_allv_grouped_mat_mul_udma"
    / "ascend950"
)


def test_operator_passes_regular_x_directly_to_kernel_arguments():
    source = (ASCEND950_ROOT / "all_to_allv_grouped_mat_mul_udma.cpp").read_text(
        encoding="utf-8"
    )
    assert "runtime.StageInput" not in source
    assert "runtime.IsSymmetric" not in source
    assert "reinterpret_cast<uint8_t *>(x.data_ptr())" in source


def test_runtime_does_not_implement_input_d2d_staging():
    source = (ASCEND950_ROOT / "udma_runtime.cpp").read_text(encoding="utf-8")
    header = (ASCEND950_ROOT / "udma_runtime.h").read_text(encoding="utf-8")
    for forbidden in ("aclrtMemcpyAsync", "StageInput", "staging_", "stagingBytes_"):
        assert forbidden not in source
        assert forbidden not in header


def test_performance_worker_profiles_regular_input():
    source = (PROJECT_ROOT / "validation" / "performance_worker.py").read_text(
        encoding="utf-8"
    )
    assert "to_symmetric" not in source
    assert "x, weight, local_table, global_table" in source
