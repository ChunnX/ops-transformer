# Copyright (c) 2026 Huawei Technologies Co., Ltd.

from pathlib import Path

import torch

from validation.device_precision import analyze_accuracy_on_device
from validation.precision import analyze_accuracy
from validation.run_gmm_a2av_root_cause import (
    DIAGNOSTIC_MODES,
    classify_results,
)


PROJECT_ROOT = Path(__file__).parents[1]


def _results(**overrides):
    values = {mode: False for mode in DIAGNOSTIC_MODES}
    values.update(overrides)
    return [
        {"mode": mode, "passed": values[mode], "cq_failure_count": 0}
        for mode in DIAGNOSTIC_MODES
    ]


def test_safe_modes_run_before_the_original_baseline():
    assert DIAGNOSTIC_MODES == (
        "full_barrier",
        "full_barrier_mte3",
        "producer_barrier",
        "full_barrier_official_block",
        "baseline",
    )


def test_baseline_pass_means_the_failure_did_not_reproduce():
    decision = classify_results(_results(baseline=True))
    assert decision["classification"] == "baseline_did_not_reproduce"
    assert not decision["conclusive"]


def test_producer_barrier_isolates_producer_visibility():
    decision = classify_results(_results(producer_barrier=True))
    assert decision["classification"] == (
        "producer_staging_visibility_or_ready_order"
    )
    assert decision["conclusive"]


def test_full_barrier_isolates_stage_recycle():
    decision = classify_results(_results(full_barrier=True))
    assert decision["classification"] == "premature_symmetric_stage_recycle"
    assert decision["conclusive"]


def test_mte3_only_pass_isolates_pipe_s_publication():
    decision = classify_results(_results(full_barrier_mte3=True))
    assert decision["classification"] == (
        "pipe_s_wqe_publication_topology_sensitivity"
    )
    assert decision["conclusive"]


def test_official_block_only_pass_isolates_custom_udma_get_path():
    decision = classify_results(_results(
        full_barrier_official_block=True
    ))
    assert decision["classification"] == (
        "custom_udma_get_or_deferred_quiet_path"
    )
    assert decision["conclusive"]


def test_cq_status_overrides_accuracy_results():
    results = _results(full_barrier=True)
    results[0]["cq_failure_count"] = 1
    decision = classify_results(results)
    assert decision["classification"] == "udma_get_cq_or_transport_failure"
    assert decision["conclusive"]


def test_baseline_timeout_can_use_already_completed_safe_modes():
    results = _results(producer_barrier=True)[:-1]
    decision = classify_results(results, timed_out_mode="baseline")
    assert decision["classification"] == (
        "producer_staging_visibility_or_ready_order"
    )
    assert decision["conclusive"]


def test_strengthened_mode_timeout_is_not_overclassified():
    decision = classify_results(
        _results()[:1], timed_out_mode="full_barrier"
    )
    assert decision["classification"] == "diagnostic_protocol_timeout"
    assert not decision["conclusive"]


def test_worker_generates_inputs_and_golden_once_before_mode_loop():
    worker = (
        PROJECT_ROOT / "validation" / "gmm_a2av_root_cause_worker.py"
    ).read_text(encoding="utf-8")
    loop = worker.index("for mode in args.mode:")
    assert worker.index("x = _make_tensor(") < loop
    assert worker.index(
        "expected = _grouped_mat_mul_all_to_allv_golden("
    ) < loop
    assert worker.count("_grouped_mat_mul_all_to_allv_golden(") == 1
    assert '"input_generation_count": 1' in worker
    assert '"golden_generation_count": 1' in worker


def test_worker_uses_a_regular_diagnostic_tensor_and_private_raw_op():
    worker = (
        PROJECT_ROOT / "validation" / "gmm_a2av_root_cause_worker.py"
    ).read_text(encoding="utf-8")
    assert "DIAGNOSTIC_TENSOR_WORDS = 131072" in worker
    assert "diagnostic_status = torch.zeros(" in worker
    assert "._grouped_mat_mul_all_to_allv_diagnostic(" in worker
    assert "_parse_diagnostic_tensor(" in worker
    assert 'launch_dir / "diagnostic.json"' in worker
    assert "AscendC::printf" not in worker


def test_worker_isolates_outputs_and_aligns_each_collective_launch():
    worker = (
        PROJECT_ROOT / "validation" / "gmm_a2av_root_cause_worker.py"
    ).read_text(encoding="utf-8")
    launch_loop = worker[worker.index(
        "for launch_index in range(args.launches_per_mode):"
    ):worker.index("mutation_checks = {")]
    assert 'diagnostic_out.fill_(float("nan"))' in launch_loop
    assert launch_loop.count("dist.barrier()") == 3
    assert '"allocation": "caller_provided_per_launch"' in launch_loop
    assert "resident_tensors_unchanged" in launch_loop


def test_device_comparator_matches_regular_opbase_pass_fail_decisions():
    expected = torch.zeros((100, 2), dtype=torch.bfloat16)
    variants = [expected.clone()]
    one_percent = expected.clone()
    one_percent[0] = 0.5
    variants.append(one_percent)
    hard_failure = expected.clone()
    hard_failure[:2] = 2.0
    variants.append(hard_failure)
    nonfinite = expected.clone()
    nonfinite[3, 1] = float("nan")
    variants.append(nonfinite)
    segments = [{
        "local_expert": 0,
        "global_expert": 0,
        "source_rank": 0,
        "row_start": 0,
        "row_end": 100,
        "rows": 100,
    }]
    for actual in variants:
        regular = analyze_accuracy(
            actual, expected, segments_override=segments, topk=8
        )
        device = analyze_accuracy_on_device(
            actual, expected, segments,
            topk=8, chunk_elements=17, include_breakdown=True,
        )
        assert device["passed"] == regular["passed"]
        assert device["failure_reasons"] == regular["failure_reasons"]
        assert device["mixed_mismatch_count"] == regular["mismatch_count"]
        assert device["matched_ratio"] == regular["matched_ratio"]
        assert device["hard_limit_violation_count"] == (
            regular["hard_limit_violation_count"]
        )


def test_runner_uses_one_persistent_torchrun_and_one_optional_build():
    runner = (
        PROJECT_ROOT / "validation" / "run_gmm_a2av_root_cause.py"
    ).read_text(encoding="utf-8")
    assert runner.count('"torch.distributed.run"') == 1
    assert '"single_torchrun": True' in runner
    assert '"golden_generation_count_per_rank": 1' in runner
    assert '"baseline_runs_last": True' in runner
    assert '"--build-install", action="store_true"' in runner
    assert 'build_environment["CATCCOS_UDMA_GMM_A2AV_ROOT_CAUSE"] = "1"' in runner


def test_kernel_exposes_the_five_disjoint_runtime_modes():
    kernel = (
        PROJECT_ROOT / "csrc" / "catccos_overlay" / "include" /
        "catccos" / "dgemm" / "kernel" /
        "grouped_matmul_alltoallv_tla.hpp"
    ).read_text(encoding="utf-8")
    assert "GMM_A2AV_DIAGNOSTIC_BASELINE = 1" in kernel
    assert "GMM_A2AV_DIAGNOSTIC_PRODUCER_BARRIER = 2" in kernel
    assert "GMM_A2AV_DIAGNOSTIC_FULL_BARRIER = 3" in kernel
    assert "GMM_A2AV_DIAGNOSTIC_FULL_BARRIER_MTE3 = 4" in kernel
    assert (
        "GMM_A2AV_DIAGNOSTIC_FULL_BARRIER_OFFICIAL_BLOCK = 5"
        in kernel
    )
    assert "UsesProducerBarrier(params.diagnosticMode)" in kernel
    assert "UsesRecycleBarrier(params.diagnosticMode)" in kernel
    assert "AscendC::SyncAll<true>()" in kernel
    assert "aclshmemx_barrier_all_vec()" in kernel
    assert "ptrDiagnostic" in kernel
    assert "RecordTargetOrderComm(" in kernel


def test_host_registers_private_diagnostic_op_and_mte3_variants():
    host = (
        PROJECT_ROOT / "csrc" / "grouped_mat_mul_all_to_allv_udma" /
        "ascend950" / "grouped_mat_mul_all_to_allv_udma.cpp"
    ).read_text(encoding="utf-8")
    device = (
        PROJECT_ROOT / "csrc" / "grouped_mat_mul_all_to_allv_udma" /
        "ascend950" / "grouped_mat_mul_all_to_allv_udma_device.h"
    ).read_text(encoding="utf-8")
    tile = (
        PROJECT_ROOT / "csrc" / "catccos_overlay" / "include" /
        "catccos" / "comm" / "tile" /
        "tile_remote_copy_udma_2d.hpp"
    ).read_text(encoding="utf-8")
    assert "_grouped_mat_mul_all_to_allv_diagnostic" in host
    assert "kDiagnosticTensorWords = 131072" in host
    assert "GroupedMatMulAllToAllVUdmaMte3Config" in host
    assert "GroupedMatMulAllToAllVUdmaM0_256Mte3Config" in host
    assert "GroupedMatMulAllToAllVUdmaOfficialBlockConfig" in host
    assert (
        "GroupedMatMulAllToAllVUdmaM0_256OfficialBlockConfig"
        in host
    )
    assert "TileRemoteCopyUdma2DWithScratch" in device
    scratch = tile[tile.index("class TileRemoteCopyUdma2DWithScratch"):]
    assert "static constexpr bool DEFER_QUIET = true" in scratch
    assert "uint32_t FinalizeWithStatus" in scratch


def test_official_block_mode_calls_the_official_tile_once():
    device = (
        PROJECT_ROOT / "csrc" / "grouped_mat_mul_all_to_allv_udma" /
        "ascend950" / "grouped_mat_mul_all_to_allv_udma_device.h"
    ).read_text(encoding="utf-8")
    kernel = (
        PROJECT_ROOT / "csrc" / "catccos_overlay" / "include" /
        "catccos" / "dgemm" / "kernel" /
        "grouped_matmul_alltoallv_tla.hpp"
    ).read_text(encoding="utf-8")
    assert "OfficialUdmaTileAdapter" in device
    assert "CopyTransport::Udma" in device
    assert "UseOfficialUdmaBlock" in device
    branch_start = kernel.index("if constexpr (OFFICIAL_UDMA_BLOCK)")
    branch_end = kernel.index("if (quietStatus == 0)", branch_start)
    official_branch = kernel[branch_start:branch_end]
    assert "blockRemoteCopy(" in official_branch
    assert "InitBlockLoop" not in official_branch.split("} else {")[0]
    assert "FinalizeBlockLoopWithStatus" not in official_branch.split(
        "} else {"
    )[0]


def test_root_cause_kernels_are_compiled_out_of_delivery_wheels():
    cmake = (PROJECT_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    setup = (PROJECT_ROOT / "setup.py").read_text(encoding="utf-8")
    assert "option(CATCCOS_UDMA_GMM_A2AV_ROOT_CAUSE" in cmake
    assert "-DCATCCOS_UDMA_GMM_A2AV_ROOT_CAUSE=0" in cmake
    assert 'cmake_boolean_from_env(\n                    "CATCCOS_UDMA_GMM_A2AV_ROOT_CAUSE"' in setup


def test_regular_accuracy_entrypoint_remains_available():
    regular = PROJECT_ROOT / "scripts" / "run_accuracy.sh"
    diagnosis = PROJECT_ROOT / "scripts" / "run_gmm_a2av_root_cause.sh"
    assert regular.is_file()
    assert diagnosis.is_file()
    assert "run_accuracy.py" in regular.read_text(encoding="utf-8")
    assert "run_gmm_a2av_root_cause.py" in diagnosis.read_text(
        encoding="utf-8"
    )
