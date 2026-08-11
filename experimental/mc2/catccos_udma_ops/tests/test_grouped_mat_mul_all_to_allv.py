# Copyright (c) 2026 Huawei Technologies Co., Ltd.

from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]
ASCEND950_ROOT = (
    PROJECT_ROOT / "csrc" / "grouped_mat_mul_all_to_allv_udma" / "ascend950"
)


def test_kernel_keeps_inputs_and_output_in_regular_npu_memory():
    source = (ASCEND950_ROOT / "grouped_mat_mul_all_to_allv_udma.cpp").read_text(
        encoding="utf-8"
    )
    assert 'reinterpret_cast<uint8_t *>(x.data_ptr())' in source
    assert 'reinterpret_cast<uint8_t *>(weight.data_ptr())' in source
    assert ": at::empty({m64, n64}, x.options())" in source
    assert "runtime.IsSymmetric(output.data_ptr())" not in source
    assert "AllocateSymmetricTensor(x, {m64, n64}" not in source
    assert "aclrtMemcpyAsync" not in source


def test_validation_uses_regular_reusable_output():
    accuracy = (PROJECT_ROOT / "validation" / "accuracy_worker.py").read_text(
        encoding="utf-8"
    )
    performance = (PROJECT_ROOT / "validation" / "performance_worker.py").read_text(
        encoding="utf-8"
    )
    assert "reusable_out = torch.empty((m, n)" in accuracy
    assert "reusable_out = udma.empty_symmetric" not in accuracy
    assert "out = udma.empty_symmetric" not in performance


def test_kernel_configuration_matches_bf16_udma_get_example():
    source = (
        ASCEND950_ROOT / "grouped_mat_mul_all_to_allv_udma_device.h"
    ).read_text(encoding="utf-8")
    assert "MoeConstraints<1, 8, 32>" in source
    assert "CommEpilogue::Block::BlockCommSchedulerReduceScatterAllToAllV" in source
    assert "CopyDirect::Get" in source
    assert "CopyTransport::Mte" in source
    assert "200UL * 1024 * 1024 * sizeof(uint16_t)" in source


def test_python_facade_dispatches_both_allocation_modes():
    source = (PROJECT_ROOT / "ops_transformer_catccos_udma" / "ops.py").read_text(
        encoding="utf-8"
    )
    assert "grouped_mat_mul_all_to_allv_udma(" in source
    assert "grouped_mat_mul_all_to_allv_udma_out(" in source
