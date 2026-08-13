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
    assert "at::empty({outputRows64, n64}, x.options())" in source
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


def test_kernel_applies_upstream_non_comm_core_deadlock_fix():
    source = (
        PROJECT_ROOT
        / "csrc"
        / "catccos_overlay"
        / "include"
        / "catccos"
        / "dgemm"
        / "kernel"
        / "grouped_matmul_alltoallv_tla.hpp"
    ).read_text(encoding="utf-8")
    assert "if (isCommCore)" in source
    assert "aclshmem_ptr(params.syncMmadFinish, srcRankIdx)" in source
    assert source.index("if (isCommCore)") < source.index(
        "aclshmem_ptr(params.syncMmadFinish, srcRankIdx)"
    )


def test_python_facade_dispatches_both_allocation_modes():
    source = (PROJECT_ROOT / "ops_transformer_catccos_udma" / "ops.py").read_text(
        encoding="utf-8"
    )
    assert "grouped_mat_mul_all_to_allv_udma(" in source
    assert "grouped_mat_mul_all_to_allv_udma_out(" in source


def test_unbalanced_route_uses_distinct_input_and_output_row_counts():
    host = (ASCEND950_ROOT / "grouped_mat_mul_all_to_allv_udma.cpp").read_text(
        encoding="utf-8"
    )
    kernel = (
        PROJECT_ROOT
        / "csrc"
        / "catccos_overlay"
        / "include"
        / "catccos"
        / "dgemm"
        / "kernel"
        / "grouped_matmul_alltoallv_tla.hpp"
    ).read_text(encoding="utf-8")
    assert "const int64_t inputRows64 = x.size(0);" in host
    assert "outputRows64 = localTokensPerExpert.sum().item<int64_t>();" in host
    assert "outputRows64 = output.size(0);" in host
    assert "static_cast<uint32_t>(outputRows64)" in host
    assert "uint32_t outputRows;" in kernel
    assert "LayoutTagA layoutTagA{args.gemmShape.m(), args.gemmShape.k()}" in kernel
    assert "LayoutTagC layoutTagC{args.outputRows, args.gemmShape.n()}" in kernel
    assert "output.size(0) == x.size(0)" not in host


def test_reusable_out_path_does_not_add_route_sum_synchronization():
    host = (ASCEND950_ROOT / "grouped_mat_mul_all_to_allv_udma.cpp").read_text(
        encoding="utf-8"
    )
    out_branch = host[host.index("if (outputOptional.has_value())"):
                      host.index("} else {", host.index("if (outputOptional.has_value())"))]
    assert "outputRows64 = output.size(0);" in out_branch
    assert ".sum().item" not in out_branch
