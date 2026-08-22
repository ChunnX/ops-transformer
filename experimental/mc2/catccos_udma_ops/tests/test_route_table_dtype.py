# Copyright (c) 2026 Huawei Technologies Co., Ltd.

from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]


def test_host_bindings_require_int64_route_tables():
    sources = [
        PROJECT_ROOT
        / "csrc"
        / "all_to_allv_grouped_mat_mul_udma"
        / "ascend950"
        / "all_to_allv_grouped_mat_mul_udma.cpp",
        PROJECT_ROOT
        / "csrc"
        / "grouped_mat_mul_all_to_allv_udma"
        / "ascend950"
        / "grouped_mat_mul_all_to_allv_udma.cpp",
    ]
    for path in sources:
        source = path.read_text(encoding="utf-8")
        assert '"local_tokens_per_expert", at::kLong' in source
        assert '"global_tokens_per_local_expert", at::kLong' in source
        assert '"local_tokens_per_expert", at::kInt' not in source
        assert '"global_tokens_per_local_expert", at::kInt' not in source


def test_device_problem_shape_reads_int64_storage():
    source = (
        PROJECT_ROOT
        / "csrc"
        / "catccos_overlay"
        / "include"
        / "catccos"
        / "dgemm"
        / "alltoallv_allgather_problem_shape.hpp"
    ).read_text(encoding="utf-8")
    assert "reinterpret_cast<__gm__ int64_t *>" in source
    assert "__gm__ int64_t *ptrLocalTokensPerExpert_" in source
    assert "__gm__ int64_t *ptrGlobalTokensPerLocalExpert_" in source


def test_route_helper_constructs_int64_tensors():
    source = (
        PROJECT_ROOT / "ops_transformer_catccos_udma" / "routing.py"
    ).read_text(encoding="utf-8")
    assert source.count("dtype=torch.int64") == 2
    assert "dtype=torch.int32" not in source


def test_public_ops_normalize_route_tables_before_dispatch():
    source = (
        PROJECT_ROOT / "ops_transformer_catccos_udma" / "ops.py"
    ).read_text(encoding="utf-8")
    assert "route = route.reshape(-1).contiguous()" in source
    assert "local_tokens_per_expert: RouteInput" in source
    assert "global_tokens_per_local_expert: RouteInput" in source
    assert "global_tokens_per_local_expert.view(-1)" not in source
    assert "torch.tensor(host_values, dtype=torch.int64, device=x.device)" in source
