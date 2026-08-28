# Copyright (c) 2026 Huawei Technologies Co., Ltd.

from pathlib import Path

from validation.run_sync_diagnosis import classify_mode_results


PROJECT_ROOT = Path(__file__).parents[1]


def _results(baseline, strong_ready, cache_safe):
    return [
        {
            "mode": "baseline",
            "summary_available": True,
            "passed": baseline,
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


def test_strong_ready_pass_is_classified_as_publication_order():
    decision = classify_mode_results(_results(False, True, True))
    assert decision["classification"] == "completion_publication_order"
    assert decision["conclusive"]


def test_only_cache_safe_pass_is_classified_as_visibility():
    decision = classify_mode_results(_results(False, False, True))
    assert decision["classification"] == "receiver_cache_visibility"
    assert decision["conclusive"]


def test_non_monotonic_strong_only_pass_is_not_conclusive():
    decision = classify_mode_results(_results(False, True, False))
    assert decision["classification"] == "non_monotonic_mode_result"
    assert not decision["conclusive"]


def test_missing_mode_is_an_infrastructure_failure():
    results = _results(False, False, False)[:-1]
    decision = classify_mode_results(results)
    assert decision["classification"] == "infrastructure_or_runtime_failure"
    assert not decision["conclusive"]


def test_kernel_and_runner_keep_the_three_modes_isolated():
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
    tile = (
        PROJECT_ROOT
        / "csrc"
        / "catccos_overlay"
        / "include"
        / "catccos"
        / "comm"
        / "tile"
        / "tile_remote_copy_udma_2d.hpp"
    ).read_text(encoding="utf-8")
    worker = (PROJECT_ROOT / "validation" / "accuracy_worker.py").read_text(
        encoding="utf-8"
    )
    assert "COMPLETION_MODE_STRONG_READY_CACHE_SAFE" in kernel
    assert "syncCommReady" in kernel
    assert "aclshmemi_signal_wait_until_eq<uint64_t>" in kernel
    assert "aclshmemx_udma_put_signal_nbi" in tile
    assert 'caller_repeats = (' in worker
    assert '"all_to_allv_export"' in worker
