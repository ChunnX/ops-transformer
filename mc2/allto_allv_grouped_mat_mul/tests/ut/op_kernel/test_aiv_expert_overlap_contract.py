"""Source-level contracts for the AIV/AIC expert pipeline.

The device-only implementation is excluded from kernel-test host compilation,
so these tests protect the orchestration order directly. NPU precision and
profiler tests remain the behavioral acceptance tests.
"""

from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[3]
    / "op_kernel"
    / "allto_allv_grouped_mat_mul_aiv_mode.h"
)


def _function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    body_start = source.index("{", start)
    depth = 0
    for index in range(body_start, len(source)):
        character = source[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return source[body_start + 1 : index]
    raise AssertionError(f"unterminated function body: {signature}")


def _source() -> str:
    return SOURCE.read_text(encoding="utf-8")


def test_shared_mm_precedes_counted_cross_core_expert_waits() -> None:
    body = _function_body(_source(), "__aicore__ inline void ProcessAic(")

    shared_mm = body.index("RunSharedExpertGemm")
    expert_loop = body.index("for (uint32_t expertIdx")
    wait_ready = body.index("WaitAicExpertReady", expert_loop)
    routed_gemm = body.index("RunExpertGemm", expert_loop)

    assert shared_mm < expert_loop
    assert expert_loop < wait_ready < routed_gemm


def test_shared_mm_starts_after_local_payload_publish_but_before_peer_wait() -> None:
    source = _source()
    aiv_body = _function_body(source, "__aicore__ inline void ProcessAiv(")
    publish = aiv_body.index("kPublishSlotsOffset")
    shared_ready = aiv_body.index("SignalAicSharedExpertReady", publish)
    peer_wait = aiv_body.index("WaitPeerPhase", shared_ready)

    aic_body = _function_body(source, "__aicore__ inline void ProcessAic(")
    wait_local_payload = aic_body.index("WaitAicSharedExpertReady")
    shared_mm = aic_body.index("RunSharedExpertGemm")

    assert publish < shared_ready < peer_wait
    assert wait_local_payload < shared_mm


def test_aiv_signals_each_expert_after_the_collective_copy_barrier() -> None:
    body = _function_body(_source(), "__aicore__ inline void ProcessAiv(")
    expert_loop = body.index("for (uint32_t expertIdx")
    final_release = body.index("kReleaseSlotsOffset")
    copy_expert = body.index("CopyExpertFromSource", expert_loop)
    copy_barrier = body.index("SyncAll<true>", copy_expert)
    publish_ready = body.index("SignalAicExpertReady", copy_barrier)

    assert expert_loop < copy_expert < copy_barrier < publish_ready < final_release


def test_small_or_single_nonempty_workloads_use_one_shot_routed_ready() -> None:
    source = _source()
    assert "ShouldUseExpertOverlap" in source
    assert "kMinExpertOverlapTokens" in source
    assert "kMinExpertOverlapDimension" in source
    assert "kMinExpertOverlapMacs" not in source
    assert "totalTokens * elementsPerToken" not in source
    assert "uint64_t minimumTokens" not in source

    aic_body = _function_body(source, "__aicore__ inline void ProcessAic(")
    choose_aic = aic_body.index("ShouldUseExpertOverlap")
    fallback_wait = aic_body.index("WaitAicRoutedDataReady", choose_aic)
    expert_loop = aic_body.index("for (uint32_t expertIdx", fallback_wait)
    expert_wait = aic_body.index("WaitAicExpertReady", expert_loop)
    assert choose_aic < fallback_wait < expert_loop < expert_wait

    aiv_body = _function_body(source, "__aicore__ inline void ProcessAiv(")
    choose_aiv = aiv_body.index("ShouldUseExpertOverlap")
    expert_loop = aiv_body.index("for (uint32_t expertIdx", choose_aiv)
    per_expert_signal = aiv_body.index("SignalAicExpertReady", expert_loop)
    fallback_signal = aiv_body.index("SignalAicRoutedDataReady", per_expert_signal)
    assert choose_aiv < expert_loop < per_expert_signal < fallback_signal


def test_expert_ready_uses_megamoe_counted_cross_core_protocol() -> None:
    source = _source()
    signal = _function_body(
        source, "__aicore__ inline void SignalAicExpertReady("
    )
    wait = _function_body(
        source, "__aicore__ inline void WaitAicExpertReady("
    )

    assert "CrossCoreSetFlag<0x2, PIPE_MTE3>" in signal
    assert "CrossCoreWaitFlag<0x2>" in wait
    assert "ReadySequenceForExpert" in signal
    assert "ReadySequenceForExpert" in wait


def test_invocation_epoch_stays_in_the_hccl_peer_window() -> None:
    source = _source()
    load_epoch = _function_body(
        source, "__aicore__ inline int32_t LoadInvocationEpoch("
    )

    assert "context.Window(context.RankId())" in load_epoch
    assert "workspace" not in load_epoch
    assert "PublishInvocationEpoch" not in source
