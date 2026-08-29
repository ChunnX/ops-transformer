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


def test_production_runs_last_after_all_safe_discriminators():
    assert DIAGNOSTIC_MODES[-1] == "production"
    assert DIAGNOSTIC_MODES[:4] == (
        "serialized_default",
        "serialized_delay",
        "serialized_fix_bypass",
        "serialized_mte_republish",
    )


def test_udma_ack_isolated_pass_selects_ack_publication_fix():
    decision = classify_results(_results(rolling_udma_ack=True))
    assert decision["classification"] == (
        "consumer_ack_publication_or_stage_reuse"
    )
    assert decision["conclusive"]


def test_fix_bypass_isolated_pass_selects_staging_visibility():
    decision = classify_results(_results(rolling_fix_bypass=True))
    assert decision["classification"] == "fix_staging_l2_visibility"
    assert decision["conclusive"]


def test_only_combined_rolling_mode_passes_is_classified_as_combined():
    decision = classify_results(_results(rolling_bypass_udma_ack=True))
    assert decision["classification"] == (
        "combined_fix_visibility_and_ack_ordering"
    )
    assert decision["conclusive"]


def test_serialized_default_pass_is_stage_lifecycle():
    decision = classify_results(_results(serialized_default=True))
    assert decision["classification"] == (
        "rolling_stage_lifecycle_or_cross_rank_ordering"
    )
    assert decision["conclusive"]


def test_serialized_delay_only_pass_is_visibility_latency():
    decision = classify_results(_results(serialized_delay=True))
    assert decision["classification"] == "fix_to_udma_visibility_latency"
    assert decision["conclusive"]


def test_mte_republish_only_pass_is_fix_publication_path():
    decision = classify_results(_results(serialized_mte_republish=True))
    assert decision["classification"] == "fix_output_publication_path"
    assert decision["conclusive"]


def test_cq_status_overrides_accuracy_results():
    results = _results(serialized_default=True)
    results[0]["cq_failure_count"] = 1
    decision = classify_results(results)
    assert decision["classification"] == "udma_get_cq_or_transport_failure"
    assert decision["conclusive"]


def test_production_timeout_can_still_use_prior_matrix():
    results = _results(rolling_udma_ack=True)[:-1]
    decision = classify_results(results, timed_out_mode="production")
    assert decision["classification"] == (
        "consumer_ack_publication_or_stage_reuse"
    )
    assert decision["conclusive"]


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


def test_kernel_exposes_runtime_modes_and_nonprintf_status():
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
    assert "GMM_A2AV_DIAGNOSTIC_SERIALIZED_MTE_REPUBLISH" in kernel
    assert "aclshmemx_udma_put_signal_nbi<uint8_t, PIPE_S>" in kernel
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
