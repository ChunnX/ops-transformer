# Copyright (c) 2026 Huawei Technologies Co., Ltd.

from pathlib import Path

from validation.run_gmm_a2av_root_cause import (
    DIAGNOSTIC_MODES,
    classify_results,
)


PROJECT_ROOT = Path(__file__).parents[1]


def _results(**overrides):
    values = {mode: False for mode in DIAGNOSTIC_MODES}
    values.update(overrides)
    return [
        {
            "mode": mode,
            "passed": values[mode],
            "cq_failure_count": 0,
        }
        for mode in DIAGNOSTIC_MODES
    ]


def _method_body(source, method, next_method):
    start = source.index("void {}".format(method))
    end = source.index("void {}".format(next_method), start)
    return source[start:end]


def test_safe_protocols_run_first_and_production_runs_last():
    assert DIAGNOSTIC_MODES == (
        "official_barrier_exact",
        "peer_generation_release_acquire",
        "peer_generation_acquire_sync",
        "peer_generation_release_fence",
        "peer_generation_baseline",
        "production",
    )


def test_peer_baseline_pass_is_epoch_clear_or_aba_protocol():
    decision = classify_results(_results(peer_generation_baseline=True))
    assert decision["classification"] == "legacy_clear_barrier_or_aba_protocol"
    assert decision["conclusive"]


def test_acquire_only_pass_selects_destination_acquire_sync():
    decision = classify_results(
        _results(peer_generation_acquire_sync=True)
    )
    assert decision["classification"] == "missing_destination_acquire_sync"
    assert decision["conclusive"]


def test_release_only_pass_selects_source_release_fence():
    decision = classify_results(
        _results(peer_generation_release_fence=True)
    )
    assert decision["classification"] == "missing_source_release_fence"
    assert decision["conclusive"]


def test_only_combined_mode_passes_requires_both_edges():
    decision = classify_results(
        _results(peer_generation_release_acquire=True)
    )
    assert decision["classification"] == "missing_release_and_acquire_edges"
    assert decision["conclusive"]


def test_official_only_pass_is_signal_protocol_gap():
    decision = classify_results(_results(official_barrier_exact=True))
    assert decision["classification"] == (
        "peer_signal_protocol_lacks_barrier_guarantee"
    )
    assert decision["conclusive"]


def test_both_isolated_edges_passing_is_timing_sensitive():
    decision = classify_results(_results(
        peer_generation_acquire_sync=True,
        peer_generation_release_fence=True,
    ))
    assert decision["classification"] == (
        "multiple_independent_ordering_mitigations"
    )
    assert not decision["conclusive"]


def test_cq_status_overrides_accuracy_results():
    results = _results(official_barrier_exact=True)
    results[0]["cq_failure_count"] = 1
    decision = classify_results(results)
    assert decision["classification"] == "udma_get_cq_or_transport_failure"
    assert decision["conclusive"]


def test_production_pass_means_failure_did_not_reproduce():
    decision = classify_results(_results(production=True))
    assert decision["classification"] == "production_did_not_reproduce"
    assert not decision["conclusive"]


def test_production_timeout_can_still_use_prior_matrix():
    results = _results(peer_generation_acquire_sync=True)[:-1]
    decision = classify_results(results, timed_out_mode="production")
    assert decision["classification"] == "missing_destination_acquire_sync"
    assert decision["conclusive"]


def test_diagnostic_timeout_is_not_overclassified_from_partial_matrix():
    results = _results(official_barrier_exact=True)[:2]
    results[1].update({"passed": False, "timed_out": True})
    decision = classify_results(
        results, timed_out_mode="peer_generation_release_acquire"
    )
    assert decision["classification"] == "diagnostic_protocol_timeout"
    assert not decision["conclusive"]
    assert decision["successful_modes"] == ["official_barrier_exact"]


def test_worker_generates_inputs_and_golden_before_the_mode_loop():
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


def test_runner_uses_one_persistent_torchrun_and_one_wheel():
    runner = (
        PROJECT_ROOT / "validation" / "run_gmm_a2av_root_cause.py"
    ).read_text(encoding="utf-8")
    assert runner.count('"torch.distributed.run"') == 1
    assert '"single_torchrun": True' in runner
    assert '"golden_generation_count_per_rank": 1' in runner
    assert '"--build-install", action="store_true"' in runner


def test_kernel_keeps_official_and_peer_protocols_disjoint():
    kernel = (
        PROJECT_ROOT / "csrc" / "catccos_overlay" / "include" /
        "catccos" / "dgemm" / "kernel" /
        "grouped_matmul_alltoallv_tla.hpp"
    ).read_text(encoding="utf-8")
    official = _method_body(
        kernel, "RunOfficialBarrierAiv", "RunPeerGenerationAiv"
    )
    peer = _method_body(
        kernel, "RunPeerGenerationAiv", "RunProductionAiv"
    )
    production = kernel[kernel.index("void RunProductionAiv"):]

    assert official.count("aclshmemx_barrier_all_vec()") >= 2
    assert "FinalizeBlockLoopWithStatus" in official
    assert "AscendC::SyncAll" not in official
    assert "aclshmem_fence" not in official

    assert "aclshmemx_barrier_all_vec" not in peer
    assert peer.count("AscendC::SyncAll<true>()") == 3
    assert "GmmA2avDiagnosticUsesAcquireSync" in peer
    assert "GmmA2avDiagnosticUsesReleaseFence" in peer
    assert "ProducerReadyEpochOffset" in peer
    assert "ProducerReadyRoundOffset" in peer
    assert "ConsumerDoneEpochOffset" in peer
    assert "ConsumerDoneRoundOffset" in peer

    assert "aclshmemx_barrier_all_vec()" in production
    assert "AscendC::SyncAll<true>()" in production


def test_kernel_and_host_expose_the_new_runtime_matrix():
    kernel = (
        PROJECT_ROOT / "csrc" / "catccos_overlay" / "include" /
        "catccos" / "dgemm" / "kernel" /
        "grouped_matmul_alltoallv_tla.hpp"
    ).read_text(encoding="utf-8")
    host = (
        PROJECT_ROOT / "csrc" / "grouped_mat_mul_all_to_allv_udma" /
        "ascend950" / "grouped_mat_mul_all_to_allv_udma.cpp"
    ).read_text(encoding="utf-8")
    for mode in DIAGNOSTIC_MODES:
        assert 'value == "{}"'.format(mode) in host
    assert "GMM_A2AV_DIAGNOSTIC_PRODUCTION = 0" in kernel
    assert "GMM_A2AV_DIAGNOSTIC_OFFICIAL_BARRIER_EXACT" in kernel
    assert "GMM_A2AV_DIAGNOSTIC_PEER_GENERATION_RELEASE_ACQUIRE" in kernel
    assert "GMM_A2AV_DIAGNOSTIC_SERIALIZED" not in kernel
    assert "syncDiagnosticStatus" in kernel
    assert "DIAGNOSTIC_STATUS_OFFSET_BYTES" in host
    assert "_grouped_mat_mul_all_to_allv_diagnostic_status" in host


def test_regular_accuracy_entrypoint_remains_unchanged_and_available():
    script = PROJECT_ROOT / "scripts" / "run_accuracy.sh"
    diagnosis = PROJECT_ROOT / "scripts" / "run_gmm_a2av_root_cause.sh"
    assert script.is_file()
    assert diagnosis.is_file()
    assert "run_accuracy.py" in script.read_text(encoding="utf-8")
    assert "run_gmm_a2av_root_cause.py" in diagnosis.read_text(
        encoding="utf-8"
    )
