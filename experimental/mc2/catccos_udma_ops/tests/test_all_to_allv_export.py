# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import struct
from pathlib import Path
from types import SimpleNamespace

import sys
import types


PROJECT_ROOT = Path(__file__).parents[1]


def _load_ops(
    monkeypatch,
    calls,
    world_size=2,
    operator_name="all_to_allv_grouped_mat_mul",
):
    class FakeTensor:
        def __init__(self, values=None, *, shape=None, dtype="int64", device=None):
            self.values = list(values) if values is not None else None
            self.shape = tuple(shape) if shape is not None else (len(self.values),)
            self.dtype = dtype
            self.device = device

        def reshape(self, *_shape):
            return self

        def contiguous(self):
            return self

        def numel(self):
            result = 1
            for dimension in self.shape:
                result *= dimension
            return result

        def sum(self):
            return SimpleNamespace(item=lambda: sum(self.values))

        def to(self, *, device):
            self.device = device
            return self

    fake_torch = types.ModuleType("torch")
    fake_torch.Tensor = FakeTensor
    fake_torch.int64 = "int64"
    fake_torch.uint8 = "uint8"
    fake_torch.is_tensor = lambda value: isinstance(value, FakeTensor)
    fake_torch.tensor = lambda values, **kwargs: FakeTensor(values, **kwargs)
    fake_torch.frombuffer = lambda buffer, **kwargs: FakeTensor(
        buffer, **kwargs
    )
    fake_torch.empty = lambda shape, **kwargs: FakeTensor(shape=shape, **kwargs)

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
    return namespace[operator_name], FakeTensor


def test_optional_flag_selects_export_schema(monkeypatch):
    calls = []
    operator, tensor_type = _load_ops(monkeypatch, calls)
    device = SimpleNamespace(type="npu")
    local = tensor_type([1, 1, 1, 1], device=device)
    route = tensor_type([1, 1, 1, 0], device=device)
    x = SimpleNamespace(shape=(4, 8), dtype="bf16", device=device)
    export = SimpleNamespace(shape=(3, 8))
    result = operator(
        x, "weight", local, route,
        ep_size=2, expert_num=4,
        received_rows=3,
        output_all_to_allv=True, all_to_allv_out=export,
    )
    assert calls[-1][0] == "all_to_allv_grouped_mat_mul_udma_export"
    assert result == ("output", export)


def test_default_path_keeps_original_schema(monkeypatch):
    calls = []
    operator, _ = _load_ops(monkeypatch, calls)
    device = SimpleNamespace(type="npu")
    x = SimpleNamespace(shape=(4, 8), dtype="bf16", device=device)
    assert operator(
        x, "weight", [1, 1, 1, 1], [1, 1, 1, 0],
        ep_size=2, expert_num=4,
    ) == "output"
    assert calls[-1][0] == "all_to_allv_grouped_mat_mul_udma"
    assert calls[-1][1][-1] == 3


def test_explicit_received_rows_avoids_device_sum(monkeypatch):
    calls = []
    operator, tensor_type = _load_ops(monkeypatch, calls)
    device = SimpleNamespace(type="npu")
    local = tensor_type([1, 1, 1, 1], device=device)
    route = tensor_type([1, 1, 1, 0], device=device)
    route.sum = lambda: (_ for _ in ()).throw(AssertionError("unexpected device sum"))
    x = SimpleNamespace(shape=(4, 8), dtype="bf16", device=device)
    operator(
        x, "weight", local, route,
        ep_size=2, expert_num=4, received_rows=3,
    )
    assert calls[-1][1][-1] == 3


def test_host_routes_allocate_exact_export_rows(monkeypatch):
    calls = []
    operator, _ = _load_ops(monkeypatch, calls)
    device = SimpleNamespace(type="npu")
    x = SimpleNamespace(shape=(4, 8), dtype="bf16", device=device)
    output, exported = operator(
        x, "weight", [1, 1, 1, 1], [1, 1, 1, 0],
        ep_size=2, expert_num=4, output_all_to_allv=True,
    )
    assert output == "output"
    assert exported.shape == (3, 8)
    assert calls[-1][1][-3] == 3


def test_host_global_route_supplies_packed_route_metadata(monkeypatch):
    calls = []
    operator, _ = _load_ops(monkeypatch, calls)
    device = SimpleNamespace(type="npu")
    x = SimpleNamespace(shape=(10, 8), dtype="bf16", device=device)
    operator(
        x,
        "weight",
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        ep_size=2,
        expert_num=4,
        output_all_to_allv=True,
    )

    metadata = calls[-1][1][-2]
    packed = bytes(metadata.values)
    assert metadata.dtype == "uint8"
    assert metadata.device is device
    assert len(packed) == 12880
    assert struct.unpack_from("<IIII", packed, 0) == (2, 2, 4, 0)
    assert struct.unpack_from("<8Q", packed, 16)[:2] == (3, 7)
    assert struct.unpack_from("<64Q", packed, 80)[:2] == (4, 6)
    assert struct.unpack_from("<QQIHH", packed, 592) == (0, 0, 1, 0, 0)
    assert struct.unpack_from("<QQIHH", packed, 616) == (0, 1, 3, 1, 0)
    assert struct.unpack_from("<QQIHH", packed, 640) == (1, 4, 2, 0, 1)
    assert struct.unpack_from("<QQIHH", packed, 664) == (3, 6, 4, 1, 1)


def test_device_global_route_keeps_null_metadata_fallback(monkeypatch):
    calls = []
    operator, tensor_type = _load_ops(monkeypatch, calls)
    device = SimpleNamespace(type="npu")
    local = tensor_type([1, 1, 1, 1], device=device)
    route = tensor_type([1, 1, 1, 0], device=device)
    x = SimpleNamespace(shape=(4, 8), dtype="bf16", device=device)
    operator(
        x,
        "weight",
        local,
        route,
        ep_size=2,
        expert_num=4,
        received_rows=3,
        output_all_to_allv=True,
    )
    assert calls[-1][1][-2] is None


def test_grouped_mat_mul_all_to_allv_accepts_host_route_lists(monkeypatch):
    calls = []
    operator, _ = _load_ops(
        monkeypatch,
        calls,
        operator_name="grouped_mat_mul_all_to_allv",
    )
    device = SimpleNamespace(type="npu")
    x = SimpleNamespace(shape=(10, 8), dtype="bf16", device=device)
    weight = SimpleNamespace(shape=(2, 8, 16))
    result = operator(
        x,
        weight,
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        ep_size=2,
        expert_num=4,
    )

    op_name, arguments = calls[-1]
    assert result == "output"
    assert op_name == "grouped_mat_mul_all_to_allv_udma_out"
    assert arguments[2].dtype == "int64"
    assert arguments[2].device is device
    assert arguments[3].dtype == "int64"
    assert arguments[3].device is device
    assert arguments[-1].shape == (10, 16)


def test_grouped_mat_mul_all_to_allv_device_routes_keep_raw_allocation_path(
    monkeypatch,
):
    calls = []
    operator, tensor_type = _load_ops(
        monkeypatch,
        calls,
        operator_name="grouped_mat_mul_all_to_allv",
    )
    device = SimpleNamespace(type="npu")
    local = tensor_type([1, 2, 3, 4], device=device)
    route = tensor_type([1, 2, 3, 4], device=device)
    local.sum = lambda: (_ for _ in ()).throw(
        AssertionError("unexpected Python device sum")
    )
    x = SimpleNamespace(shape=(10, 8), dtype="bf16", device=device)
    weight = SimpleNamespace(shape=(2, 8, 16))
    operator(
        x,
        weight,
        local,
        route,
        ep_size=2,
        expert_num=4,
    )
    assert calls[-1][0] == "grouped_mat_mul_all_to_allv_udma"


def test_python_facade_exposes_optional_all_to_allv_output():
    source = (PROJECT_ROOT / "ops_transformer_catccos_udma" / "ops.py").read_text(
        encoding="utf-8"
    )
    assert "output_all_to_allv: bool = False" in source
    assert "received_rows: Optional[int] = None" in source
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


def test_export_overlaps_previous_stage_with_current_communication():
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
    assert "class AllToAllVExportPipeline<true, BlockExport>" in kernel
    assert "WORKSPACE_STAGES >= 2" in kernel
    assert "uint32_t exportIdx = commIdx - 1;" in kernel
    assert "bool isSelfCopyWorker" in kernel
    assert "blockIdx += selfCopyCores" in kernel
    assert "exportPipeline.GetBlockExport()" in kernel
    assert "uint32_t exportIdx = commLoops - 1;" in kernel


def test_strong_ready_put_quiets_inline_without_deferred_state():
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
    block_dir = (
        PROJECT_ROOT
        / "csrc"
        / "catccos_overlay"
        / "include"
        / "catccos"
        / "comm"
        / "block"
    )
    wrapper = (block_dir / "comm_block_mte_udma.hpp").read_text(encoding="utf-8")
    remote = (block_dir / "comm_block_remote_copy.hpp").read_text(encoding="utf-8")

    assert "blockRemoteCopy.PutRemoteContiguousWithSignalAndQuiet(" in kernel
    assert "remoteComm.PutContiguousWithSignalAndQuiet(" in wrapper

    inline_method = remote.split(
        "void PutContiguousWithSignalAndQuiet(", 1
    )[1].split("private:", 1)[0]
    assert "tileRemoteCopy.PutContiguousWithSignal(" in inline_method
    assert "tileRemoteCopy.Finalize(rankIdx);" in inline_method
    assert "hasPendingCopy" not in inline_method



def test_delivery_route_metadata_and_export_worker_pool_are_optional():
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
    metadata = (
        PROJECT_ROOT
        / "csrc"
        / "catccos_overlay"
        / "include"
        / "catccos"
        / "dgemm"
        / "moe_route_metadata.hpp"
    ).read_text(encoding="utf-8")
    assert "__gm__ MoeRouteMetadata const *routeMetadata" in kernel
    assert "params.routeMetadata != nullptr" in kernel
    assert "params.routeMetadata == nullptr" in kernel
    assert "params.exportCoreCount" in kernel
    assert "struct MoeRouteSegment" in metadata


def test_host_binding_forwards_optional_route_metadata_to_kernel():
    host = (
        PROJECT_ROOT
        / "csrc"
        / "all_to_allv_grouped_mat_mul_udma"
        / "ascend950"
        / "all_to_allv_grouped_mat_mul_udma.cpp"
    ).read_text(encoding="utf-8")
    assert "Tensor? route_metadata" in host
    assert 'CheckNpuTensor(routeMetadata, "route_metadata", at::kByte)' in host
    assert "routeMetadataOptional.value().data_ptr()" in host
    assert "unexpected MoeRouteMetadata ABI" in host


def test_workspace_reuse_handshake_closes_final_stages_without_unsigned_subtraction():
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
    assert "commLoops - WORKSPACE_STAGES" not in kernel
    aic_body = kernel[
        kernel.index("void operator()<AscendC::AIC>"):
        kernel.index("void operator()<AscendC::AIV>")
    ]
    assert aic_body.count("CrossCoreSetFlag<0x2, PIPE_FIX>") == 1
    assert "uint32_t finalStageCount = Min(commLoops, WORKSPACE_STAGES);" in kernel
    assert "uint32_t firstFinalComm = commLoops - finalStageCount;" in kernel
    assert "flagAicFinishStore[stageIdx]);" in kernel
    final_drain = kernel.index("uint32_t finalStageCount")
    final_barrier = kernel.index(
        "CrossCoreBarrier<0x0, PIPE_MTE3>();", final_drain
    )
    assert final_drain < final_barrier
