# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import json
from pathlib import Path

from validation.runtime_log_collection import (
    collect_runtime_logs,
    runtime_log_roots,
    snapshot_runtime_logs,
)

PROJECT_ROOT = Path(__file__).parents[1]


def test_runtime_log_collection_keeps_only_new_and_appended_bytes(tmp_path):
    shmem = tmp_path / "shmem"
    plog = tmp_path / "plog"
    shmem.mkdir()
    plog.mkdir()
    existing = shmem / "aclshmem_rank0.log"
    existing.write_text("old line\n", encoding="utf-8")
    roots = runtime_log_roots(
        {"HOME": str(tmp_path / "unused")},
        shmem_log_dirs=[str(shmem)],
        plog_dirs=[str(plog)],
    )
    before = snapshot_runtime_logs(roots)

    with existing.open("a", encoding="utf-8") as stream:
        stream.write("new shmem line\n")
    new_plog = plog / "plog-123.log"
    new_plog.write_text("new plog line\n", encoding="utf-8")

    mode_output = tmp_path / "mode"
    result = collect_runtime_logs(before, roots, mode_output / "runtime_logs")
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert result["captured_file_count"] == 2
    assert result["error_count"] == 0
    by_source = {Path(item["source"]).name: item for item in manifest["files"]}
    appended = by_source[existing.name]
    created = by_source[new_plog.name]
    assert appended["reason"] == "appended"
    assert Path(appended["destination"]).read_text(encoding="utf-8") == (
        "new shmem line\n"
    )
    assert created["reason"] == "created"
    assert Path(created["destination"]).read_text(encoding="utf-8") == (
        "new plog line\n"
    )


def test_udma_tiles_use_public_completion_without_private_qp_access():
    tile = (
        PROJECT_ROOT / "csrc" / "catccos_overlay" / "include" / "catccos"
        / "comm" / "tile" / "tile_remote_copy_udma_2d.hpp"
    ).read_text(encoding="utf-8")
    assert "aclshmemx_udma_put_signal_nbi" in tile
    assert "aclshmemx_udma_quiet" in tile
    for private_symbol in (
        "aclshmemi_udma_poll_cq", "aclshmemi_udma_qp_info_fetch",
        "aclshmemi_udma_active_table", "aclshmemi_udma_qp_table_t",
        "aclshmemi_udma_wq_ctx_t", "wqe_cnt", "cqe_cnt",
    ):
        assert private_symbol not in tile
    overlay = PROJECT_ROOT / "csrc" / "catccos_overlay" / "include" / "catccos"
    for header in overlay.rglob("*.hpp"):
        source = header.read_text(encoding="utf-8")
        assert "FinalizeWithStatus" not in source, header
        assert "FinalizeBlockLoopWithStatus" not in source, header


def test_mssanitizer_build_enables_only_ready_wait_timeout_diagnostics():
    root_cmake = (PROJECT_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
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

    mssanitizer_branch = root_cmake.split(
        "if(CATCCOS_UDMA_MSSANITIZER)", 1
    )[1].split("endif()", 1)[0]
    assert "-DCATCCOS_UDMA_MSSANITIZER=1" in mssanitizer_branch
    assert "-DASCENDC_DUMP=1" in mssanitizer_branch
    assert "-DCATCCOS_UDMA_DEVICE_DIAGNOSTICS=1" not in mssanitizer_branch

    assert "READY_WAIT_TIMEOUT_CYCLES = 21000000000LL" in kernel
    assert '"observed=0x%x expected=0x%x "' in kernel
    assert "observed_hi=" not in kernel
    assert "expected_hi=" not in kernel
    assert "dcci_cacheline((__gm__ uint8_t *)readyAddress)" in kernel
    assert "AscendC::GetSystemCycle() - startCycle" in kernel
    assert "ascendc_assert(" in kernel
    assert "AscendC::printf(" not in kernel
    assert "trap();" not in kernel


def test_entry_barrier_precedes_reusable_signal_reset():
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

    entry_barrier = kernel.index("aclshmemx_barrier_all_vec();")
    reset = kernel.index("syncMmadFinish, 0, ACLSHMEM_SIGNAL_SET", entry_barrier)
    ready_reset = kernel.index("readyWords + wordIdx, 0", reset)
    initialization_barrier = kernel.index("aclshmemx_barrier_all_vec();", reset)
    assert entry_barrier < reset < ready_reset < initialization_barrier


def test_ready_protocol_uses_reset_round_values_without_launch_epoch():
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
    host = (
        PROJECT_ROOT
        / "csrc"
        / "all_to_allv_grouped_mat_mul_udma"
        / "ascend950"
        / "all_to_allv_grouped_mat_mul_udma.cpp"
    ).read_text(encoding="utf-8")
    runtime_header = (
        PROJECT_ROOT
        / "csrc"
        / "all_to_allv_grouped_mat_mul_udma"
        / "ascend950"
        / "udma_runtime.h"
    ).read_text(encoding="utf-8")
    runtime_source = (
        PROJECT_ROOT
        / "csrc"
        / "all_to_allv_grouped_mat_mul_udma"
        / "ascend950"
        / "udma_runtime.cpp"
    ).read_text(encoding="utf-8")

    assert "launchEpoch" not in kernel
    assert "launchEpoch" not in host
    assert "NextAllToAllVGroupedMatMulEpoch" not in runtime_header
    assert "NextAllToAllVGroupedMatMulEpoch" not in runtime_source
    assert "allToAllVGroupedMatMulEpoch_" not in runtime_header
    assert "static_cast<uint64_t>(commIdx + 1)" in kernel
    assert "stageIdx * READY_SLOT_RANKS + rankIdx" in kernel
    assert "stageIdx * READY_SLOT_RANKS + srcRankIdx" in kernel
