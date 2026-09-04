# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import json
from pathlib import Path

from validation.common import sync_completion_mode_from_env
from validation.run_sync_diagnosis import (
    _collect_runtime_evidence,
    classify_mode_results,
)
from validation.runtime_log_collection import (
    collect_runtime_logs,
    runtime_log_roots,
    snapshot_runtime_logs,
)


PROJECT_ROOT = Path(__file__).parents[1]


def _results(
    baseline, barrier_only, generation_after_quiet, strong_ready, cache_safe
):
    return [
        {
            "mode": "baseline",
            "summary_available": True,
            "passed": baseline,
        },
        {
            "mode": "barrier_only",
            "summary_available": True,
            "passed": barrier_only,
        },
        {
            "mode": "generation_after_quiet",
            "summary_available": True,
            "passed": generation_after_quiet,
        },
        {
            "mode": "strong_ready",
            "summary_available": True,
            "passed": strong_ready,
        },
        {
            "mode": "strong_ready_cache_safe",
            "summary_available": True,
            "passed": cache_safe,
        },
    ]


def test_barrier_only_pass_is_classified_as_invocation_lifecycle():
    decision = classify_mode_results(_results(False, True, True, True, True))
    assert (
        decision["classification"]
        == "cross_invocation_lifecycle_or_reset_race"
    )
    assert decision["conclusive"]


def test_earlier_conclusive_pair_does_not_require_later_modes():
    results = _results(False, True, False, False, False)[:2]
    decision = classify_mode_results(results)
    assert (
        decision["classification"]
        == "cross_invocation_lifecycle_or_reset_race"
    )
    assert decision["conclusive"]


def test_generation_after_quiet_is_classified_as_aggregate_counter():
    decision = classify_mode_results(_results(False, False, True, True, True))
    assert decision["classification"] == "legacy_aggregate_counter_protocol"
    assert decision["conclusive"]


def test_generation_conclusion_does_not_require_wwn_modes():
    results = _results(False, False, True, False, False)[:3]
    decision = classify_mode_results(results)
    assert decision["classification"] == "legacy_aggregate_counter_protocol"
    assert decision["conclusive"]


def test_strong_ready_pass_is_classified_as_publication_order():
    decision = classify_mode_results(_results(False, False, False, True, True))
    assert (
        decision["classification"]
        == "cross_engine_payload_ready_publication"
    )
    assert decision["conclusive"]


def test_only_cache_safe_pass_is_classified_as_visibility():
    decision = classify_mode_results(_results(False, False, False, False, True))
    assert decision["classification"] == "receiver_cache_visibility"
    assert decision["conclusive"]


def test_non_monotonic_strong_only_pass_is_not_conclusive():
    decision = classify_mode_results(_results(False, False, False, True, False))
    assert decision["classification"] == "non_monotonic_mode_result"
    assert not decision["conclusive"]


def test_missing_mode_is_an_infrastructure_failure():
    results = _results(False, False, False, False, False)[:-1]
    decision = classify_mode_results(results)
    assert decision["classification"] == "infrastructure_or_runtime_failure"
    assert not decision["conclusive"]


def test_runtime_evidence_invalidates_an_accuracy_matrix():
    results = _results(False, False, False, True, True)
    results[0]["runtime_evidence"] = {
        "detected": True,
        "counts": {"udma_cq_timeout": 1},
    }
    decision = classify_mode_results(results)
    assert decision["classification"] == "runtime_or_transport_error_observed"
    assert not decision["conclusive"]


def test_checked_quiet_cq_failure_is_conclusive_even_without_summary():
    results = _results(False, False, False, False, False)
    results[2]["summary_available"] = False
    results[2]["runtime_evidence"] = {
        "detected": True,
        "counts": {"udma_quiet_cq_failure": 1},
    }
    decision = classify_mode_results(results)
    assert decision["classification"] == "udma_quiet_cq_failure"
    assert decision["conclusive"]


def test_checked_quiet_debug_abort_timeout_is_also_conclusive():
    results = _results(False, False, False, False, False)
    results[2]["summary_available"] = False
    results[2]["runtime_evidence"] = {
        "detected": True,
        "counts": {"udma_cq_timeout": 1},
    }
    decision = classify_mode_results(results)
    assert decision["classification"] == "udma_quiet_cq_failure"
    assert decision["conclusive"]


def test_runtime_evidence_extracts_cq_and_shmem_failures(tmp_path):
    log = tmp_path / "torchrun_ep8.log"
    log.write_text(
        "[CATCCOS_UDMA_SYNC_DIAG] quiet_poll_failure status=0xff\n"
        "poll cq timeout while waiting for completion\n"
        "HcommChannelCreate failed for dst pe 7, ret = 9\n",
        encoding="utf-8",
    )
    evidence = _collect_runtime_evidence(tmp_path)
    assert evidence["detected"]
    assert evidence["counts"]["udma_quiet_cq_failure"] == 1
    assert evidence["counts"]["udma_cq_timeout"] == 1
    assert evidence["counts"]["udma_channel_failure"] == 1


def test_runtime_evidence_extracts_shmem_init_timeout(tmp_path):
    log = tmp_path / "torchrun_ep8.log"
    log.write_text(
        "Wait hcomm UDMA channel ready timeout, channelNum = 7\n"
        "aclshmemx_init_attr failed: -7\n",
        encoding="utf-8",
    )
    evidence = _collect_runtime_evidence(tmp_path)
    assert evidence["detected"]
    assert evidence["counts"]["udma_channel_failure"] == 1
    assert evidence["counts"]["shmem_initialization_failure"] == 1


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


def test_accuracy_launcher_collects_runtime_logs_without_double_collection():
    accuracy = (PROJECT_ROOT / "validation" / "run_accuracy.py").read_text(
        encoding="utf-8"
    )
    diagnosis = (
        PROJECT_ROOT / "validation" / "run_sync_diagnosis.py"
    ).read_text(encoding="utf-8")
    assert '"--no-runtime-log-collection", action="store_true"' in accuracy
    assert "log_snapshot = snapshot_runtime_logs(log_roots)" in accuracy
    assert '"runtime_logs": runtime_log_results' in accuracy
    assert '"--no-runtime-log-collection",' in diagnosis


def test_kernel_and_runner_keep_the_five_modes_isolated():
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
    worker = (PROJECT_ROOT / "validation" / "accuracy_worker.py").read_text(
        encoding="utf-8"
    )
    host = (
        PROJECT_ROOT
        / "csrc"
        / "all_to_allv_grouped_mat_mul_udma"
        / "ascend950"
        / "all_to_allv_grouped_mat_mul_udma.cpp"
    ).read_text(encoding="utf-8")
    runner = (PROJECT_ROOT / "validation" / "run_sync_diagnosis.py").read_text(
        encoding="utf-8"
    )
    assert "COMPLETION_MODE_BARRIER_ONLY" in kernel
    assert "COMPLETION_MODE_GENERATION_AFTER_QUIET" in kernel
    assert "COMPLETION_MODE_STRONG_READY_CACHE_SAFE" in kernel
    assert 'value == "barrier_only"' in host
    assert 'value == "generation_after_quiet"' in host
    assert '"barrier_only",' in runner
    assert '"generation_after_quiet",' in runner
    assert "syncCommReady" in kernel
    assert "aclshmemi_signal_wait_until_eq<uint64_t>" in kernel
    assert "quiet_poll_failure" not in kernel
    assert "CATCCOS_UDMA_A2AV_GMM_READY_TIMEOUT" in kernel
    assert 'caller_repeats = (' in worker
    assert '"all_to_allv_export"' in worker


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
        "elseif(CATCCOS_UDMA_MSSANITIZER)", 1
    )[1].split("else()", 1)[0]
    assert "-DCATCCOS_UDMA_READY_WAIT_TIMEOUT_DIAGNOSTICS=1" in mssanitizer_branch
    assert "-DCATCCOS_UDMA_DEVICE_DIAGNOSTICS=0" in mssanitizer_branch
    assert "-DASCENDC_DUMP=1" in mssanitizer_branch
    assert "-DCATCCOS_UDMA_DEVICE_DIAGNOSTICS=1" not in mssanitizer_branch

    assert "READY_WAIT_TIMEOUT_CYCLES = 45000000000LL" in kernel
    assert '"observed=0x%x expected=0x%x "' in kernel
    assert "observed_hi=" not in kernel
    assert "expected_hi=" not in kernel
    assert "dcci_cacheline((__gm__ uint8_t *)readyAddress)" in kernel
    assert "AscendC::GetSystemCycle() - startCycle" in kernel
    assert "ascendc_assert(" in kernel
    assert "AscendC::printf(" not in kernel
    assert "trap();" not in kernel


def test_production_default_uses_strong_ready_without_l2_bypass(monkeypatch):
    monkeypatch.delenv("CATCCOS_UDMA_SYNC_DIAGNOSTIC_MODE", raising=False)
    assert sync_completion_mode_from_env() == "production"

    host = (
        PROJECT_ROOT
        / "csrc"
        / "all_to_allv_grouped_mat_mul_udma"
        / "ascend950"
        / "all_to_allv_grouped_mat_mul_udma.cpp"
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

    assert 'raw == nullptr ? "production" : raw' in host
    assert (
        "return outputAllToAllV ? CompletionMode::kStrongReady\n"
        "                               : CompletionMode::kBaseline;"
    ) in host
    assert kernel.count("SetL2CacheHint") == 2
    cursor = 0
    for _ in range(2):
        hint = kernel.index("SetL2CacheHint", cursor)
        assert "COMPLETION_MODE_STRONG_READY_CACHE_SAFE" in kernel[
            max(0, hint - 256):hint
        ]
        cursor = hint + 1


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

    entry_condition = kernel.index("if (useEntryBarrier)")
    entry_barrier = kernel.index("aclshmemx_barrier_all_vec();", entry_condition)
    reset = kernel.index("syncMmadFinish, 0, ACLSHMEM_SIGNAL_SET", entry_barrier)
    ready_reset = kernel.index("readyWords + readyWordIdx, 0", reset)
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
    assert "uint32_t readyGeneration = commIdx + 1U;" in kernel
    assert (
        "uint64_t readyGeneration =\n"
        "                                        static_cast<uint64_t>(commIdx + 1);"
    ) in kernel
