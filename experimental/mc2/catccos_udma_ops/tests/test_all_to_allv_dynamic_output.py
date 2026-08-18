# Copyright (c) 2026 Huawei Technologies Co., Ltd.

from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]
HOST = (
    PROJECT_ROOT
    / "csrc"
    / "all_to_allv_grouped_mat_mul_udma"
    / "ascend950"
    / "all_to_allv_grouped_mat_mul_udma.cpp"
)
KERNEL = (
    PROJECT_ROOT
    / "csrc"
    / "catccos_overlay"
    / "include"
    / "catccos"
    / "dgemm"
    / "kernel"
    / "ascend950_alltoallv_grouped_matmul.hpp"
)


def test_host_allocates_and_validates_exact_output_rows():
    source = HOST.read_text(encoding="utf-8")
    assert "at::empty({receivedRows64, n64}, x.options())" in source
    assert "output.size(0) == receivedRows64" in source
    assert "allToAllVOutput.size(0) == receivedRows64" in source
    assert "at::zeros({worldSize * m64, n64}" not in source
    assert "int expert_num, int received_rows" in source


def test_kernel_layout_uses_exact_output_rows():
    source = KERNEL.read_text(encoding="utf-8")
    assert "uint32_t outputRows;" in source
    assert "LayoutTagC tagC{args.outputRows, args.gemmShape.n()};" in source
    assert "args.gemmShape.m() * args.rankSize" not in source


def test_meta_shape_uses_received_rows():
    source = HOST.read_text(encoding="utf-8")
    assert "return at::empty({receivedRows, weight.size(2)}, x.options());" in source
    assert "output.size(0) == receivedRows" in source
