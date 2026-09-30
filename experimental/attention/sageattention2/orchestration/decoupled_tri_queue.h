#pragma once

#include "../core/platform_traits.h"
#include "../core/numerical_types.h"
#include "../core/math_approximations.h"
#include "../memory/ring_buffer_layout.h"
#include "../memory/accompanying_v_cache.h"
#include "../memory/catlass_fractal_packer.h"
#include "../microkernel/dynamic_symmetric_quant.h"
#include "../microkernel/mxfp8_e4m3_encoder.h"
#include "../microkernel/online_reduction_softmax.h"
#include "../microkernel/delayed_fp16_gemm.h"
#include "../dispatch/split_kv_grid.h"
#include "../dispatch/snake_wavefront_balancer.h"
#include "credit_handshake.h"

namespace sage::orchestration {

using namespace sage::core;
using namespace sage::memory;
using namespace sage::microkernel;
using namespace sage::dispatch;

#if defined(__CCE__)
using AscendC::RoundMode;
using AscendC::HardEvent;
using memory::Sync;

struct WideOutputInfo {
    uint32_t ordinal;
    uint32_t b;
    uint32_t h;
    uint32_t qstart;
    bool first;
    bool last;
};

struct CubeFragmentInfo {
    uint64_t kb;
    uint32_t part;
};

template <uint32_t PHASE>
__aicore__ inline __attribute__((always_inline)) void ProcessWideCubeStep(
    FixedResource &r, CubeFragmentInfo *pending, uint64_t fragment, bool hasQk,
    GM_ADDR k8, GM_ADDR v8, GM_ADDR vsmx, MxScaleCache &mxCache, const SageParams &p,
    uint32_t count, bool prefetchPv = true)
{
    using namespace AscendC;
    constexpr uint32_t DP = 128;
    if (hasQk) {
        const auto &current = pending[fragment & 3];
        GemmQkSplit<WIDE_B_L1>(r, (fragment & 3) * HALF * N * 2, current.part == 0, PHASE);
        if (current.part + 2 < count * 2) {
            PrefetchWideK<WIDE_B_L1>(r, k8 + (current.kb + (current.part + 2) * N) * DP, PHASE);
        }
    }
    if (fragment >= 2 && prefetchPv) {
        const auto &next = pending[(fragment - 2) & 3];
        PrefetchWideV<WIDE_B_L1>(r, v8 + (next.kb + next.part * N) * DP,
                                 vsmx + (next.kb + next.part * N) / 32 * DP, PHASE);
    }
    if (fragment >= 3) {
        uint64_t previous = fragment - 3;
        const auto &old = pending[previous & 3];
        uint32_t group = previous / 2;
        EventHandshake::WaitFromBothAiv<PIPE_MTE1>(TokenSignal::P_READY);
        uint32_t parity = previous & 3;
        uint32_t pvOffset = WideVectorLayout::PV_UB + (group & 1) * WideVectorLayout::PV_SLOT_BYTES;
        if constexpr (PHASE == 0) {
            if (group >= 2) {
                EventHandshake::WaitFromBothAiv<PIPE_FIX>(TokenSignal::PV_RELEASE);
            }
            Gemm<float8_e4m3_t, Catlass::layout::RowMajor, true, WIDE_B_L1, true, false, true>(
                r, v8, nullptr, N, DP, pvOffset, parity, true,
                vsmx + (old.kb + old.part * N) / 32 * DP, &mxCache, p.kp, old.part * N, (1 - PHASE));
            EventHandshake::NotifyBothAiv<PIPE_FIX>(TokenSignal::PV_READY);
        } else {
            Gemm<float8_e4m3_t, Catlass::layout::RowMajor, true, WIDE_B_L1, false, true, true>(
                r, v8, nullptr, N, DP, pvOffset, parity, true,
                vsmx + (old.kb + old.part * N) / 32 * DP, &mxCache, p.kp, old.part * N, (1 - PHASE));
        }
    }
}

template <class T, bool CAUSAL = false>
__aicore__ inline __attribute__((always_inline)) void ExecuteWideStream(
    GM_ADDR q, GM_ADDR k8, GM_ADDR v8, GM_ADDR qs, GM_ADDR ks, GM_ADDR mxScale,
    GM_ADDR vMean, GM_ADDR out, GM_ADDR lse, const SageParams &p)
{
    using namespace AscendC;
    FixedResource r;
    MxScaleCache mxCache;
    uint32_t core = GetBlockIdx();
    if ASCEND_IS_AIV core /= 2;

    const uint64_t totalTasks = DieTaskEnd(p, uint64_t(p.b) * p.hq * (p.sq / M), core);
    if (core >= totalTasks) return;

    uint32_t subBlock = GetSubBlockIdx();
    GlobalTensor<T> qg, og;
    qg.SetGlobalBuffer((__gm__ T *)q);
    qg.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    og.SetGlobalBuffer((__gm__ T *)out);
    GlobalTensor<half> qsg, ksg;
    qsg.SetGlobalBuffer((__gm__ half *)qs);
    ksg.SetGlobalBuffer((__gm__ half *)ks);
    ksg.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);
    GlobalTensor<float> vmg, lg;
    vmg.SetGlobalBuffer((__gm__ float *)vMean);
    vmg.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);
    lg.SetGlobalBuffer((__gm__ float *)lse);

    auto accum = r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::ACCUM_UB);
    auto denom = r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::DEN_UB);
    auto qsu = r.ubBuf.template GetBufferByByte<half>(WideVectorLayout::QS_UB);
    auto vmean = r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::VM_UB);
    auto qo = r.ubBuf.template GetBufferByByte<T>(WideVectorLayout::P_UB);

    if ASCEND_IS_AIC {
        InitMxP(r);
        SetFlag<HardEvent::MTE1_MTE2>(EventHandshake::K_PREFETCH_EVENT_BASE);
        SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(EventHandshake::K_PREFETCH_EVENT_BASE + 1));
        SetFlag<HardEvent::MTE1_MTE2>(EventHandshake::V_PREFETCH_EVENT_BASE);
        SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(EventHandshake::V_PREFETCH_EVENT_BASE + 1));
        SetFlag<HardEvent::MTE1_MTE2>(EventHandshake::QK_EVENT);
        SetFlag<HardEvent::MTE1_MTE2>(EventHandshake::PV_EVENT);
        SetFlag<HardEvent::M_MTE1>(EventHandshake::QK_EVENT);
        SetFlag<HardEvent::M_MTE1>(EventHandshake::PV_EVENT);
        SetFlag<HardEvent::FIX_M>(EventHandshake::QK_EVENT);
        SetFlag<HardEvent::FIX_M>(EventHandshake::QK_SECOND_FIX_EVENT);
        SetFlag<HardEvent::FIX_M>(EventHandshake::PV_EVENT);
    } else {
        auto first = SnakeScheduler<CAUSAL>::Decode(p, core, 0);
        uint64_t firstScale = first.kb;
        uint64_t secondScale = SnakeScheduler<CAUSAL>::LookaheadScale(p, first, 1, totalTasks);
        SetFlag<HardEvent::V_MTE2>(EventHandshake::SCALE_EVENT_BASE);
        SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(EventHandshake::SCALE_EVENT_BASE + 1));
        for (uint32_t slot = 0; slot < 2; ++slot) {
            uint64_t offset = slot ? secondScale : firstScale;
            if (offset != ~uint64_t(0)) {
                event_t ev = static_cast<event_t>(EventHandshake::SCALE_EVENT_BASE + slot);
                WaitFlag<HardEvent::V_MTE2>(ev);
                DataCopy(r.ubBuf.template GetBufferByByte<half>(WideVectorLayout::KS_UB + slot * 1024), ksg[offset], 512);
                SetFlag<HardEvent::MTE2_V>(ev);
            }
        }
    }

    WideOutputInfo outputPending[3];
    CubeFragmentInfo cubePending[4];
    uint64_t fragment = 0, retired = 0;
    uint32_t ordinal = 0;

    for (uint64_t task = core; task < totalTasks; task += p.cores, ++ordinal) {
        auto current = SnakeScheduler<CAUSAL>::Decode(p, task, ordinal);
        if ASCEND_IS_AIC {
            PrefetchWideK<WIDE_B_L1>(r, k8 + current.kb * 128, 0);
            PrefetchWideK<WIDE_B_L1>(r, k8 + (current.kb + N) * 128, 1);
            for (uint32_t part = 0; part < current.count * 2; part += 2, fragment += 2) {
                cubePending[fragment & 3] = {current.kb, part};
                ProcessWideCubeStep<0>(r, cubePending, fragment, true, k8, v8, mxScale, mxCache, p, current.count);
                cubePending[(fragment + 1) & 3] = {current.kb, part + 1};
                ProcessWideCubeStep<1>(r, cubePending, fragment + 1, true, k8, v8, mxScale, mxCache, p, current.count);
            }
        } else {
            for (uint32_t step = 0; step < current.count; ++step, fragment += 2) {
                const uint64_t group = fragment / 2;
                bool deferPair = false;
                if (group >= 2 && retired <= group - 2) {
                    const auto &old = outputPending[retired % 3];
                    deferPair = (retired + 1 < group && old.ordinal == outputPending[(retired + 1) % 3].ordinal);
                    if (deferPair) {
                        EventHandshake::WaitFromCube<PIPE_V>(TokenSignal::PV_READY);
                        EventHandshake::WaitFromCube<PIPE_V>(TokenSignal::PV_READY);
                    } else {
                        if (old.last) {
                            EventHandshake::WaitFromCube<PIPE_V>(TokenSignal::PV_READY);
                            Sync<HardEvent::V_MTE2>();
                            if (p.smoothV) {
                                uint32_t kh = CalcKh(old.h, p.hq, p.hk);
                                DataCopy(vmean, vmg[(uint64_t(old.b) * p.hk + kh) * p.dp], p.dp);
                            } else {
                                Duplicate(vmean, 0.0f, 128);
                            }
                            Sync<HardEvent::MTE2_V>();
                            auto pv = r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::PV_UB + (retired & 1) * WideVectorLayout::PV_SLOT_BYTES);
                            auto alpha = r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::ALPHA_UB + (retired % 3) * 512);
                            auto scratch = r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::SUM_UB);
                            OutAccumulator::FinalizeWideFp32<T, 128, false>(
                                (__ubuf__ T *)qo.GetPhyAddr(), (__ubuf__ float *)accum.GetPhyAddr(),
                                (__ubuf__ float *)pv.GetPhyAddr(), (__ubuf__ float *)alpha.GetPhyAddr(),
                                (__ubuf__ float *)denom.GetPhyAddr(), (__ubuf__ float *)scratch.GetPhyAddr(),
                                (__ubuf__ float *)vmean.GetPhyAddr());
                            PipeBarrier<PIPE_V>();
                            uint32_t firstRow = old.qstart + subBlock * HALF;
                            uint32_t validRows = firstRow < p.sq ? (p.sq - firstRow < HALF ? p.sq - firstRow : HALF) : 0;
                            if (validRows)
                                WriteRow(og, ((uint64_t(old.b) * p.hq + old.h) * p.sq + firstRow) * p.d, qo, validRows * p.d);
                            if (p.returnLse) {
                                auto lseScratch = r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::P_UB);
                                auto maximaOld = r.ubBuf.template GetBufferByByte<half>(WideVectorLayout::MAX_UB + (old.ordinal & 1) * 256);
                                StoreTileLse(lg, denom, maximaOld, lseScratch, firstRow, validRows, old.b, old.h, p);
                            }
                        }
                        ++retired;
                    }
                }

                const bool first = step == 0, last = step + 1 == current.count;
                auto maxima = r.ubBuf.template GetBufferByByte<half>(WideVectorLayout::MAX_UB + (current.ordinal & 1) * 256);

                if (first) {
                    constexpr uint32_t Q_IN_UB = WideVectorLayout::SCORE_UB;
                    constexpr uint32_t Q_FP32_UB = Q_IN_UB + HALF * 128 * sizeof(T);
                    constexpr uint32_t Q_INT8_UB = Q_FP32_UB + HALF * 128 * sizeof(float);
                    constexpr uint32_t Q_RAW_SCALE = Q_INT8_UB + HALF * 128 * sizeof(int8_t);
                    constexpr uint32_t Q_INV_SCALE = Q_RAW_SCALE + HALF * sizeof(float);

                    auto input = r.ubBuf.template GetBufferByByte<T>(Q_IN_UB);
                    auto x = r.ubBuf.template GetBufferByByte<float>(Q_FP32_UB);
                    auto qOut = r.ubBuf.template GetBufferByByte<int8_t>(Q_INT8_UB);
                    auto scales = r.ubBuf.template GetBufferByByte<float>(Q_RAW_SCALE);
                    auto inverseScales = r.ubBuf.template GetBufferByByte<float>(Q_INV_SCALE);

                    Sync<HardEvent::MTE3_MTE2>();
                    Sync<HardEvent::V_MTE2>();
                    uint64_t qToken = current.qb + current.qstart + subBlock * HALF;
                    DataCopy(input, qg[qToken * 128], HALF * 128);
                    Sync<HardEvent::MTE2_V>();

                    Cast(x, input, RoundMode::CAST_NONE, HALF * 128);
                    PipeBarrier<PIPE_V>();

                    QuantRowsVF((__ubuf__ float *)x.GetPhyAddr(), (__ubuf__ float *)x.GetPhyAddr(),
                                (__ubuf__ float *)scales.GetPhyAddr(), (__ubuf__ int8_t *)qOut.GetPhyAddr(),
                                (__ubuf__ half *)qsu.GetPhyAddr(), p.qFactor,
                                (__ubuf__ float *)inverseScales.GetPhyAddr(), 128, 0, HALF, p.qg);

                    Sync<HardEvent::V_MTE3>();
                    auto ql1 = r.l1Buf.template GetBufferByByte<int8_t>(subBlock * HALF * 32);
                    DataCopyParams plane{HALF, 1, 128 / 32 - 1, 0};
                    #pragma unroll
                    for (uint32_t cb = 0; cb < 128 / 32; ++cb)
                        DataCopy(ql1[cb * M * 32], qOut[cb * 32], plane);

                    EventHandshake::NotifyCube<PIPE_MTE3>(TokenSignal::Q_READY);
                    Duplicate(maxima, half(SAGE_FP16_MIN), 128);
                    PipeBarrier<PIPE_V>();
                }

                auto scores = r.ubBuf.template GetBufferByByte<half>(WideVectorLayout::SCORE_UB + (group & 1) * 2 * HALF * N * 2);
                auto p8 = scores.template ReinterpretCast<float8_e4m3_t>();
                auto alpha = r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::ALPHA_UB + (group % 3) * 512);

                EventHandshake::WaitFromCube<PIPE_V>(TokenSignal::QK_READY);
                EventHandshake::WaitFromCube<PIPE_V>(TokenSignal::QK_READY);

                auto ksu = r.ubBuf.template GetBufferByByte<half>(WideVectorLayout::KS_UB + (group & 1) * 1024);
                event_t scaleEvent = static_cast<event_t>(EventHandshake::SCALE_EVENT_BASE + (group & 1));
                WaitFlag<HardEvent::MTE2_V>(scaleEvent);

                const int32_t qStartInSk = static_cast<int32_t>(current.qstart + subBlock * HALF);
                const int32_t kvStartInSk = static_cast<int32_t>(step * 512);
                const int32_t queryInGroup = qStartInSk - kvStartInSk;

                if constexpr (CAUSAL) {
                    if (queryInGroup < 512) {
                        SimdSoftmax::SoftmaxWideFirstVF<true>(
                            (__ubuf__ half *)scores.GetPhyAddr(), (__ubuf__ half *)scores.GetPhyAddr() + 64 * 16,
                            (__ubuf__ half *)maxima.GetPhyAddr(), (__ubuf__ float *)alpha.GetPhyAddr(),
                            (__ubuf__ float8_e4m3_t *)p8.GetPhyAddr(), (__ubuf__ half *)qsu.GetPhyAddr(),
                            (__ubuf__ half *)ksu.GetPhyAddr(),
                            (__ubuf__ float *)r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::PARTIAL_SUMS_UB).GetPhyAddr(),
                            queryInGroup);
                    } else {
                        SimdSoftmax::SoftmaxWideFirstVF<false>(
                            (__ubuf__ half *)scores.GetPhyAddr(), (__ubuf__ half *)scores.GetPhyAddr() + 64 * 16,
                            (__ubuf__ half *)maxima.GetPhyAddr(), (__ubuf__ float *)alpha.GetPhyAddr(),
                            (__ubuf__ float8_e4m3_t *)p8.GetPhyAddr(), (__ubuf__ half *)qsu.GetPhyAddr(),
                            (__ubuf__ half *)ksu.GetPhyAddr(),
                            (__ubuf__ float *)r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::PARTIAL_SUMS_UB).GetPhyAddr());
                    }
                } else {
                    SimdSoftmax::SoftmaxWideFirstVF<false>(
                        (__ubuf__ half *)scores.GetPhyAddr(), (__ubuf__ half *)scores.GetPhyAddr() + 64 * 16,
                        (__ubuf__ half *)maxima.GetPhyAddr(), (__ubuf__ float *)alpha.GetPhyAddr(),
                        (__ubuf__ float8_e4m3_t *)p8.GetPhyAddr(), (__ubuf__ half *)qsu.GetPhyAddr(),
                        (__ubuf__ half *)ksu.GetPhyAddr(),
                        (__ubuf__ float *)r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::PARTIAL_SUMS_UB).GetPhyAddr());
                }

                SetFlag<HardEvent::V_MTE2>(scaleEvent);
                uint64_t lookaheadOffset = SnakeScheduler<CAUSAL>::LookaheadScale(p, current, step + 2, totalTasks);
                if (lookaheadOffset != ~uint64_t(0)) {
                    WaitFlag<HardEvent::V_MTE2>(scaleEvent);
                    DataCopy(ksu, ksg[lookaheadOffset], 512);
                    SetFlag<HardEvent::MTE2_V>(scaleEvent);
                }

                Sync<HardEvent::V_MTE3>();
                auto pl1_0 = r.l1Buf.template GetBufferByByte<float8_e4m3_t>(fragment * M * N);
                DataCopyParams cp0{N / 32, HALF, 0, HALF};
                DataCopy(pl1_0[subBlock * HALF * 32], p8, cp0);
                EventHandshake::NotifyCube<PIPE_MTE3>(TokenSignal::P_READY);

                SimdSoftmax::SoftmaxWideSecondVF(
                    (__ubuf__ half *)scores.GetPhyAddr() + 64 * 32, (__ubuf__ half *)maxima.GetPhyAddr(),
                    (__ubuf__ float *)alpha.GetPhyAddr(), (__ubuf__ float8_e4m3_t *)p8.GetPhyAddr() + 64 * 32,
                    (__ubuf__ float *)r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::PARTIAL_SUMS_UB).GetPhyAddr());

                Sync<HardEvent::V_MTE3>();
                auto pl1_1 = r.l1Buf.template GetBufferByByte<float8_e4m3_t>((fragment + 1) * M * N);
                DataCopyParams cp1{N / 32, HALF, 0, HALF};
                DataCopy(pl1_1[subBlock * HALF * 32], p8[64 * 32], cp1);
                EventHandshake::NotifyCube<PIPE_MTE3>(TokenSignal::P_READY);

                outputPending[group % 3] = {current.ordinal, current.b, current.h, current.qstart, first, last};

                if (deferPair) {
                    const auto firstInfo = outputPending[retired % 3];
                    const auto secondInfo = outputPending[(retired + 1) % 3];
                    Sync<HardEvent::V_MTE2>();
                    if (p.smoothV) {
                        uint32_t kh = CalcKh(secondInfo.h, p.hq, p.hk);
                        DataCopy(vmean, vmg[(uint64_t(secondInfo.b) * p.hk + kh) * p.dp], p.dp);
                    } else {
                        Duplicate(vmean, 0.0f, 128);
                    }
                    Sync<HardEvent::MTE2_V>();
                    auto pv0 = r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::PV_UB + (retired & 1) * WideVectorLayout::PV_SLOT_BYTES);
                    auto pv1 = r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::PV_UB + ((retired + 1) & 1) * WideVectorLayout::PV_SLOT_BYTES);
                    auto a0 = r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::ALPHA_UB + (retired % 3) * 512);
                    auto a1 = r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::ALPHA_UB + ((retired + 1) % 3) * 512);
                    auto scratch = r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::SUM_UB);

                    if (firstInfo.first && secondInfo.last) {
                        OutAccumulator::UpdateWidePair<T, 128, true, true>(
                            (__ubuf__ T *)qo.GetPhyAddr(), (__ubuf__ float *)accum.GetPhyAddr(),
                            (__ubuf__ float *)pv0.GetPhyAddr(), (__ubuf__ float *)pv1.GetPhyAddr(),
                            (__ubuf__ float *)a0.GetPhyAddr(), (__ubuf__ float *)a1.GetPhyAddr(),
                            (__ubuf__ float *)denom.GetPhyAddr(), (__ubuf__ float *)scratch.GetPhyAddr(),
                            (__ubuf__ float *)vmean.GetPhyAddr());
                    } else if (firstInfo.first) {
                        OutAccumulator::UpdateWidePair<T, 128, true, false>(
                            (__ubuf__ T *)qo.GetPhyAddr(), (__ubuf__ float *)accum.GetPhyAddr(),
                            (__ubuf__ float *)pv0.GetPhyAddr(), (__ubuf__ float *)pv1.GetPhyAddr(),
                            (__ubuf__ float *)a0.GetPhyAddr(), (__ubuf__ float *)a1.GetPhyAddr(),
                            (__ubuf__ float *)denom.GetPhyAddr(), (__ubuf__ float *)scratch.GetPhyAddr(),
                            (__ubuf__ float *)vmean.GetPhyAddr());
                    } else if (secondInfo.last) {
                        OutAccumulator::UpdateWidePair<T, 128, false, true>(
                            (__ubuf__ T *)qo.GetPhyAddr(), (__ubuf__ float *)accum.GetPhyAddr(),
                            (__ubuf__ float *)pv0.GetPhyAddr(), (__ubuf__ float *)pv1.GetPhyAddr(),
                            (__ubuf__ float *)a0.GetPhyAddr(), (__ubuf__ float *)a1.GetPhyAddr(),
                            (__ubuf__ float *)denom.GetPhyAddr(), (__ubuf__ float *)scratch.GetPhyAddr(),
                            (__ubuf__ float *)vmean.GetPhyAddr());
                    } else {
                        OutAccumulator::UpdateWidePair<T, 128, false, false>(
                            (__ubuf__ T *)qo.GetPhyAddr(), (__ubuf__ float *)accum.GetPhyAddr(),
                            (__ubuf__ float *)pv0.GetPhyAddr(), (__ubuf__ float *)pv1.GetPhyAddr(),
                            (__ubuf__ float *)a0.GetPhyAddr(), (__ubuf__ float *)a1.GetPhyAddr(),
                            (__ubuf__ float *)denom.GetPhyAddr(), (__ubuf__ float *)scratch.GetPhyAddr(),
                            (__ubuf__ float *)vmean.GetPhyAddr());
                    }
                    if (secondInfo.last) {
                        PipeBarrier<PIPE_V>();
                        uint32_t firstRow = secondInfo.qstart + subBlock * HALF;
                        uint32_t validRows = firstRow < p.sq ? (p.sq - firstRow < HALF ? p.sq - firstRow : HALF) : 0;
                        if (validRows)
                            WriteRow(og, ((uint64_t(secondInfo.b) * p.hq + secondInfo.h) * p.sq + firstRow) * p.d, qo, validRows * p.d);
                        if (p.returnLse) {
                            auto lseScratch = r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::P_UB);
                            auto maximaOld = r.ubBuf.template GetBufferByByte<half>(WideVectorLayout::MAX_UB + (secondInfo.ordinal & 1) * 256);
                            StoreTileLse(lg, denom, maximaOld, lseScratch, firstRow, validRows, secondInfo.b, secondInfo.h, p);
                        }
                    }
                    retired += 2;
                }
            }
        }
    }

    if ASCEND_IS_AIC {
        ProcessWideCubeStep<0>(r, cubePending, fragment, false, k8, v8, mxScale, mxCache, p, 0);
        ProcessWideCubeStep<1>(r, cubePending, fragment + 1, false, k8, v8, mxScale, mxCache, p, 0);
        ProcessWideCubeStep<0>(r, cubePending, fragment + 2, false, k8, v8, mxScale, mxCache, p, 0, false);
    } else {
        const uint64_t groups = fragment / 2;
        while (retired < groups) {
            const auto old = outputPending[retired % 3];
            if (old.last) {
                EventHandshake::WaitFromCube<PIPE_V>(TokenSignal::PV_READY);
                Sync<HardEvent::V_MTE2>();
                if (p.smoothV) {
                    uint32_t kh = CalcKh(old.h, p.hq, p.hk);
                    DataCopy(vmean, vmg[(uint64_t(old.b) * p.hk + kh) * p.dp], p.dp);
                } else {
                    Duplicate(vmean, 0.0f, 128);
                }
                Sync<HardEvent::MTE2_V>();
                auto pv = r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::PV_UB + (retired & 1) * WideVectorLayout::PV_SLOT_BYTES);
                auto alpha = r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::ALPHA_UB + (retired % 3) * 512);
                auto scratch = r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::SUM_UB);
                OutAccumulator::FinalizeWideFp32<T, 128, false>(
                    (__ubuf__ T *)qo.GetPhyAddr(), (__ubuf__ float *)accum.GetPhyAddr(),
                    (__ubuf__ float *)pv.GetPhyAddr(), (__ubuf__ float *)alpha.GetPhyAddr(),
                    (__ubuf__ float *)denom.GetPhyAddr(), (__ubuf__ float *)scratch.GetPhyAddr(),
                    (__ubuf__ float *)vmean.GetPhyAddr());
                PipeBarrier<PIPE_V>();
                uint32_t firstRow = old.qstart + subBlock * HALF;
                uint32_t validRows = firstRow < p.sq ? (p.sq - firstRow < HALF ? p.sq - firstRow : HALF) : 0;
                if (validRows)
                    WriteRow(og, ((uint64_t(old.b) * p.hq + old.h) * p.sq + firstRow) * p.d, qo, validRows * p.d);
                if (p.returnLse) {
                    auto lseScratch = r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::P_UB);
                    auto maximaOld = r.ubBuf.template GetBufferByByte<half>(WideVectorLayout::MAX_UB + (old.ordinal & 1) * 256);
                    StoreTileLse(lg, denom, maximaOld, lseScratch, firstRow, validRows, old.b, old.h, p);
                }
            }
            ++retired;
        }
    }

    if ASCEND_IS_AIC {
        for (uint32_t i = 0; i < (fragment / 2 < 2 ? fragment / 2 : 2); ++i) {
            EventHandshake::WaitFromBothAiv<PIPE_FIX>(TokenSignal::PV_RELEASE);
        }
        WaitFlag<HardEvent::MTE1_MTE2>(4);
        WaitFlag<HardEvent::MTE1_MTE2>(5);
        WaitFlag<HardEvent::MTE1_MTE2>(6);
        WaitFlag<HardEvent::MTE1_MTE2>(7);
        WaitFlag<HardEvent::MTE1_MTE2>(EventHandshake::QK_EVENT);
        WaitFlag<HardEvent::MTE1_MTE2>(EventHandshake::PV_EVENT);
        WaitFlag<HardEvent::M_MTE1>(EventHandshake::QK_EVENT);
        WaitFlag<HardEvent::M_MTE1>(EventHandshake::PV_EVENT);
        WaitFlag<HardEvent::FIX_M>(EventHandshake::QK_EVENT);
        WaitFlag<HardEvent::FIX_M>(EventHandshake::QK_SECOND_FIX_EVENT);
        WaitFlag<HardEvent::FIX_M>(EventHandshake::PV_EVENT);
    }
    PipeBarrier<PIPE_ALL>();
}

#endif // __CCE__

} // namespace sage::orchestration

namespace sage {
#if defined(__CCE__)
using orchestration::ExecuteWideStream;
#endif
}
