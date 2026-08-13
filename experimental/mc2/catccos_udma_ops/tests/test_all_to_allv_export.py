# Copyright (c) 2026 Huawei Technologies Co., Ltd.

from pathlib import Path
from types import SimpleNamespace

import sys
import types


PROJECT_ROOT = Path(__file__).parents[1]


def _load_ops(monkeypatch, calls, world_size=2):
    fake_torch = types.ModuleType("torch")
    fake_torch.Tensor = object
    fake_torch.empty = lambda shape, **kwargs: SimpleNamespace(shape=shape, **kwargs)

    class FakeOps:
        def __getattr__(self, name):
            def invoke(*args):
                calls.append((name, args))
                if "export" in name:
                    return args[-2:] if "out_export" in name else ("output", args[-1])
                return "output"
            return invoke

    fake_torch.ops = SimpleNamespace(ops_transformer_catccos_udma=FakeOps())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    runtime = types.ModuleType("ops_transformer_catccos_udma.runtime")
    runtime.get_udma_config = lambda: SimpleNamespace(world_size=world_size)
    monkeypatch.setitem(sys.modules, "ops_transformer_catccos_udma.runtime", runtime)
    namespace = {"__name__": "ops_transformer_catccos_udma.ops", "__package__": "ops_transformer_catccos_udma"}
    exec(
        (PROJECT_ROOT / "ops_transformer_catccos_udma" / "ops.py").read_text(encoding="utf-8"),
        namespace,
    )
    return namespace["all_to_allv_grouped_mat_mul"]


def test_optional_flag_selects_export_schema(monkeypatch):
    calls = []
    operator = _load_ops(monkeypatch, calls)
    route = SimpleNamespace(sum=lambda: SimpleNamespace(item=lambda: 3))
    x = SimpleNamespace(shape=(4, 8), dtype="bf16", device="npu")
    export = SimpleNamespace(shape=(3, 8))
    result = operator(
        x, "weight", "local", route,
        ep_size=2, expert_num=4,
        output_all_to_allv=True, all_to_allv_out=export,
    )
    assert calls[-1][0] == "all_to_allv_grouped_mat_mul_udma_export"
    assert result == ("output", export)


def test_default_path_keeps_original_schema(monkeypatch):
    calls = []
    operator = _load_ops(monkeypatch, calls)
    route = SimpleNamespace(sum=lambda: SimpleNamespace(item=lambda: 3))
    x = SimpleNamespace(shape=(4, 8), dtype="bf16", device="npu")
    assert operator(x, "weight", "local", route, ep_size=2, expert_num=4) == "output"
    assert calls[-1][0] == "all_to_allv_grouped_mat_mul_udma"


def test_python_facade_exposes_optional_all_to_allv_output():
    source = (PROJECT_ROOT / "ops_transformer_catccos_udma" / "ops.py").read_text(
        encoding="utf-8"
    )
    assert "output_all_to_allv: bool = False" in source
    assert "all_to_allv_out: Optional[torch.Tensor] = None" in source
    assert "all_to_allv_grouped_mat_mul_udma_export(" in source
    assert "all_to_allv_grouped_mat_mul_udma_out_export(" in source


def test_export_uses_compile_time_kernel_variant_and_ordinary_output():
    device = (
        PROJECT_ROOT
        / "csrc"
        / "all_to_allv_grouped_mat_mul_udma"
        / "ascend950"
        / "all_to_allv_grouped_mat_mul_udma_device.h"
    ).read_text(encoding="utf-8")
    kernel = (
        PROJECT_ROOT
        / "csrc"
        / "catccos_overlay"
        / "include"
        / "catccos"
        / "dgemm"
        / "kernel"
        / "ascend950_alltoallv_grouped_matmul.hpp"
    ).read_text(encoding="utf-8")
    assert "bool ExportAllToAllV = false" in device
    assert "AllToAllVGroupedMatMulUdmaDefaultExportConfig" in device
    assert "if constexpr (EXPORT_ALLTOALLV)" in kernel
    assert "gmOutput.SetGlobalBuffer(params.ptrAllToAllVOutput)" in kernel
    assert "shmem_ptr(params.ptrAllToAllVOutput" not in kernel


def test_export_order_is_local_expert_then_source_rank():
    kernel = (
        PROJECT_ROOT
        / "csrc"
        / "catccos_overlay"
        / "include"
        / "catccos"
        / "dgemm"
        / "kernel"
        / "ascend950_alltoallv_grouped_matmul.hpp"
    ).read_text(encoding="utf-8")
    local_expert_loop = "for (uint32_t localExpertIdx = 0;"
    source_rank_loop = "for (uint32_t srcRankIdx = 0;"
    assert kernel.index(local_expert_loop) < kernel.index(
        source_rank_loop, kernel.index(local_expert_loop)
    )
