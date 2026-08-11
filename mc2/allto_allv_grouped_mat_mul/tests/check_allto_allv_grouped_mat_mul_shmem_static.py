#!/usr/bin/env python3
"""Offline contract checks for the Ascend950 SHMEM migration."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_jsonl(relative: str) -> list[dict]:
    path = ROOT / relative
    require(path.suffix == ".jsonl", f"case file must be JSONL: {path}")
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def check_public_abi() -> None:
    header = read("op_api/aclnn_allto_allv_grouped_mat_mul_v2.h")
    flat = normalized(header)
    require('#include "aclnn_util.h"' in header,
            "public V2 header must include the ACLNN_API definition")
    signature = (
        "aclnnAlltoAllvGroupedMatMulV2GetWorkspaceSize( const aclTensor *gmmX, "
        "const aclTensor *gmmWeight, void *shmemContext, int64_t epWorldSize, "
        "const aclIntArray *sendCounts, const aclIntArray *recvCounts, aclTensor *gmmY, "
        "uint64_t *workspaceSize, aclOpExecutor **executor)"
    )
    require(signature in flat, "public V2 ABI does not match the approved SHMEM signature")
    require(re.search(r"\b(?:char|aclChar|std::string)\s*\*?\s*group\b", flat) is None,
            "public V2 header still contains group parameter")
    require("commMode" not in header and "comm_mode" not in header, "public V2 header still contains commMode")

    impl = read("op_api/aclnn_allto_allv_grouped_mat_mul_v2.cpp")
    require("aclnnInnerAlltoAllvGroupedMatMulShmemGetWorkspaceSize" in impl, "V2 does not dispatch to SHMEM op")
    require("CreateShmemContextTensor" in impl and "ACL_UINT8" in impl, "context is not wrapped as UINT8 tensor")
    require("NpuArch::DAV_3510" in impl, "V2 lacks Ascend950 platform guard")
    require("GetStorageFormat() != op::Format::FORMAT_ND" in impl,
            "V2 lacks the source ND storage-format guard")
    require(impl.count("MC2Aclnn::IsTensorContiguous(") == 3,
            "V2 must reject non-contiguous x/weight/y before SHMEM dispatch")

    fallback = read("op_graph/fallback_allto_allv_grouped_mat_mul.cpp")
    require("EXEC_OPAPI_CMD(aclnnAlltoAllvGroupedMatMulV2" not in fallback,
            "legacy HCCL fallback must not call the ABI-incompatible SHMEM V2 interface")
    require("EXEC_OPAPI_CMD(aclnnAlltoAllvGroupedMatMul," in fallback,
            "legacy A3 HCCL fallback must route through the V1 interface")
    fallback_call_begin = fallback.index("EXEC_OPAPI_CMD(aclnnAlltoAllvGroupedMatMul,")
    fallback_call_end = fallback.index(");", fallback_call_begin)
    require("commMode" not in fallback[fallback_call_begin:fallback_call_end],
            "legacy A3 fallback must use the V1 argument list without commMode")


def check_registration_and_binary() -> None:
    shmem_def = read("op_host/allto_allv_grouped_mat_mul_shmem_def.cpp")
    hccl_def = read("op_host/allto_allv_grouped_mat_mul_def.cpp")
    require('AddConfig("ascend950"' in shmem_def, "SHMEM op is not registered for ascend950")
    shmem_code = re.sub(r"//.*", "", shmem_def)
    require(".HcclGroup(" not in shmem_code, "SHMEM op accidentally registers an HCCL group")
    require(".AutoContiguous()" not in shmem_code,
            "SHMEM op must not replace the symmetric gmm_x address with an implicit contiguous copy")
    require('AddConfig("ascend910_93"' in hccl_def, "A3 HCCL registration is missing")
    require('AddConfig("ascend950"' not in hccl_def, "legacy HCCL op is still registered for ascend950")

    config_dir = ROOT / "op_host/config/ascend950"
    require(not (config_dir / "allto_allv_grouped_mat_mul_binary.json").exists(), "legacy A5 HCCL binary remains")
    config = json.loads((config_dir / "allto_allv_grouped_mat_mul_shmem_binary.json").read_text("utf-8"))
    require(config["op_type"] == "AlltoAllvGroupedMatMulShmem", "wrong SHMEM op_type")
    item = config["op_list"][0]
    require([x["name"] for x in item["inputs"]] == ["gmm_x", "gmm_weight", "shmem_context"],
            "binary input order differs from kernel/internal op")
    require([x["dtype"] for x in item["inputs"]] == ["bfloat16", "bfloat16", "uint8"],
            "binary input dtypes differ from source contract")
    require([x["name"] for x in item["outputs"]] == ["gmm_y"], "unexpected SHMEM outputs")


def check_tiling_and_kernel() -> None:
    tiling_header = read("op_kernel/allto_allv_grouped_mat_mul_shmem_tiling.h")
    for token in (
        "MAX_EP_WORLD_SIZE = 8",
        "MAX_LOCAL_EXPERTS = 64",
        "WORKSPACE_STAGES = 2",
        "WORKSPACE_DATA_BYTES = 400ULL * 1024ULL * 1024ULL",
        "WORKSPACE_BYTES = WORKSPACE_DATA_BYTES + 4096ULL",
        "sizeof(AlltoAllvGroupedMatMulShmemTilingData) <= 8192",
    ):
        require(token in tiling_header, f"missing tiling contract token: {token}")

    tiling = read("op_host/op_tiling/arch35/allto_allv_grouped_mat_mul_shmem_tiling.cpp")
    for token in (
        "SOURCE_M0_SMALL = 128",
        "SOURCE_N0_LARGE = 256",
        "SOURCE_M0_LARGE = 256",
        "SOURCE_N0_SMALL = 128",
        "SOURCE_K0 = 256",
        "SOURCE_COMM_INTERVAL = 10",
        "SOURCE_COMM_TILE_M = 32",
        "SOURCE_COMM_BLOCK_M = 64",
        "SHMEM_HEAP_LIMIT_BYTES = 1004ULL * 1024ULL * 1024ULL",
        "Per-peer send/recv totals must fit the source uint32 scheduler",
    ):
        require(token in tiling, f"missing source tiling invariant: {token}")
    require("uint32_t aicCoreNum, ::TCubeTiling &tiling)" in normalized(tiling),
            "host tiling must use the global TCubeTiling type stored in kernel tiling data")
    require(tiling.count("GetStorageFormat() != ge::FORMAT_ND") == 4,
            "tiling must enforce ND for x/weight/context/y")

    cmake = read("op_host/CMakeLists.txt")
    require("ENV{SHMEM_HOME_PATH}" in cmake and "ENV{SHMEM_HOME}" in cmake,
            "CMake must support source and HCCL-integrated SHMEM SDK layouts")
    require("SHMEM_HOME_PATH (source SDK layout) or SHMEM_HOME (HCCL-integrated layout)" in cmake,
            "Ascend950 configuration must fail clearly when SHMEM headers are unavailable")
    a5_begin = cmake.index('if("${CONDITION_UNIT}" STREQUAL "ascend950")')
    shmem_dependency = cmake.index("if(DEFINED SHMEM_HOME_PATH")
    a3_begin = cmake.index('elseif("${CONDITION_UNIT}" STREQUAL "ascend910_93")')
    require(a5_begin < shmem_dependency < a3_begin,
            "SHMEM dependency discovery must remain scoped to Ascend950 and must not break A3 builds")

    entry = read("op_kernel/allto_allv_grouped_mat_mul_shmem.cpp")
    kernel = read("op_kernel/allto_allv_grouped_mat_mul_shmem.h")
    require("allto_allv_grouped_mat_mul_shmem" in entry, "dedicated kernel entry is missing")
    require("KERNEL_TYPE_MIX_AIC_1_2" in entry, "kernel is not configured for one AIC/two AIV")
    require("SetSyncBaseAddr" in entry and "shmem_my_pe" in entry and "shmem_n_pes" in entry,
            "kernel does not consume the SHMEM context/topology")
    require("aclshmemx_udma_put_nbi<uint8_t, PIPE_S>" in kernel and
            "if (rows == 1 || (srcStride == columns && dstStride == columns))" in kernel and
            "for (uint32_t row = 0; row < rows; ++row)" in kernel,
            "kernel lacks source UDMA contiguous/row-wise copy paths")
    submit_begin = kernel.index("inline void SubmitUdma")
    submit_end = kernel.index("__gm__ Element *x_", submit_begin)
    submit_udma = kernel[submit_begin:submit_end]
    require("shmem_ptr(" not in submit_udma and "aclshmem_ptr(" not in submit_udma,
            "UDMA destination must be the local symmetric address plus peer, not a translated remote pointer")
    require("DataCopyPad" in kernel, "kernel lacks source local MTE copy path")
    require("WaitFlag<HardEvent::MTE3_MTE2>" in kernel and "WaitFlag<HardEvent::MTE2_MTE3>" in kernel,
            "local MTE double buffer lacks source dependency/reuse events")
    require("stage * copyTileElements_ * BF16_BYTES" in kernel,
            "local UB ping-pong offset must be expressed in bytes")
    require("Hccl" not in entry + kernel and "HCCL" not in entry + kernel, "SHMEM kernel contains HCCL code")


def check_pta_bridge() -> None:
    bridge = read("examples/arch35/pta/allto_allv_grouped_mat_mul_shmem_torch.cpp")
    for token in (
        "TORCH_LIBRARY(allto_allv_grouped_mat_mul_shmem_pta, m)",
        'm.def("init(int rank, int world_size, int heap_bytes, str ip_port) -> int")',
        'm.def("run(Tensor x, Tensor weight, int[] send_counts, int[] recv_counts) -> Tensor")',
        'm.def("barrier() -> int")',
        'm.def("finalize() -> int")',
        "aclshmemx_init_attr",
        "shmemx_get_ffts_config",
        "aclshmem_malloc",
        "aclshmem_calloc",
        "aclrtMemcpyAsync",
        "aclnnAlltoAllvGroupedMatMulV2GetWorkspaceSize",
        "aclnnAlltoAllvGroupedMatMulV2(",
        "aclrtSynchronizeStream",
        "aclshmem_barrier_all",
        "shmem_free",
    ):
        require(token in bridge, f"PTA bridge missing contract token: {token}")
    require("init_process_group" not in bridge and "Hccl" not in bridge and "HCCL" not in bridge,
            "PTA bridge must not initialize or call HCCL")


def check_pta_build() -> None:
    cmake = read("examples/arch35/pta/CMakeLists.txt")
    for token in (
        "find_package(Python3 REQUIRED COMPONENTS Interpreter Development)",
        "find_package(Torch REQUIRED)",
        "cxx_std_17",
        "SHMEM_HOME_PATH",
        "SHMEM_HOME",
        "CUST_OPAPI_LIBRARY",
        "CUST_OPAPI_INCLUDE_DIR",
        "torch_npu",
        "ascendcl",
        "shmem_utils",
        "LIBRARY_OUTPUT_DIRECTORY",
    ):
        require(token in cmake, f"PTA CMake missing contract token: {token}")

    build = read("examples/arch35/pta/build_pta.sh")
    for token in (
        "set -euo pipefail",
        "ASCEND_HOME_PATH",
        "SHMEM_HOME_PATH",
        "SHMEM_HOME",
        "CUST_OPAPI_LIBRARY",
        "PYTHON_BIN",
        "torch.utils.cmake_prefix_path",
        "import torch_npu",
        "cmake -S",
        "cmake --build",
    ):
        require(token in build, f"PTA build script missing contract token: {token}")
    require("liballto_allv_grouped_mat_mul_shmem_pta.so" in build,
            "PTA build script must print and verify the expected shared library")


def check_cases() -> None:
    precision = load_jsonl("tests/precision/allto_allv_grouped_mat_mul_shmem_precision_cases.jsonl")
    perf = load_jsonl("tests/performance/allto_allv_grouped_mat_mul_shmem_perf_cases.jsonl")
    require(len(precision) == 36, f"expected 36 legal precision cases, got {len(precision)}")
    require(len(perf) == 10, f"expected 10 performance cases, got {len(perf)}")
    names = [case["case_id"] for case in precision]
    require(len(names) == len(set(names)), "duplicate precision case ids")
    eps: set[int] = set()
    for case in precision:
        specs = {item["name"]: item for item in case["inputs"]}
        x_shape = specs["gmm_x"]["shape"]
        w_shape = specs["gmm_weight"]["shape"]
        ep = specs["ep_world_size"]["value"]
        experts = specs["local_experts"]["value"]
        eps.add(ep)
        require(specs["gmm_x"]["dtype"] == specs["gmm_weight"]["dtype"] == "bfloat16", "non-BF16 case")
        require(len(x_shape) == 2 and len(w_shape) == 3 and x_shape[1] == w_shape[1], "invalid M/K/N shape")
        require(1 <= ep <= 8 and 1 <= experts <= 64, "case exceeds source EP/E limits")
        require(case["route_seed"] == 20260810, "route seed drift")
    require(eps == set(range(1, 9)), f"EP coverage is incomplete: {sorted(eps)}")
    precision_names = set(names)
    require(all(case["case_id"] in precision_names for case in perf), "performance case not in precision set")


def main() -> int:
    checks = [
        check_public_abi,
        check_registration_and_binary,
        check_tiling_and_kernel,
        check_pta_bridge,
        check_pta_build,
        check_cases,
    ]
    for check in checks:
        check()
        print(f"PASS {check.__name__}")
    print(f"PASS all static SHMEM migration checks ({len(checks)} groups)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
