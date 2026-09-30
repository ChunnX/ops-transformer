#pragma once

#include "platform_traits.h"
#include "sage_attention2_tiling_data.h"
#include "math_approximations.h"
#include "ring_buffer_layout.h"
#include "accompanying_v_cache.h"
#include "catlass_fractal_packer.h"
#include "dynamic_symmetric_quant.h"
#include "mxfp8_e4m3_encoder.h"
#include "online_reduction_softmax.h"
#include "delayed_fp16_gemm.h"
#include "split_kv_grid.h"
#include "snake_wavefront_balancer.h"
#include "credit_handshake.h"
#include "decoupled_tri_queue.h"

namespace sage::orchestration {

using namespace sage::core;
using namespace sage::memory;
using namespace sage::microkernel;
using namespace sage::dispatch;

#if defined(__CCE__)
using AscendC::RoundMode;
using AscendC::HardEvent;
using memory::Sync;

template <class T, uint32_t DP, bool CAUSAL = false>
class StandardAttentionKernel {
public:
    __aicore__ inline __attribute__((always_inline)) void Init(
        GM_ADDR q8, GM_ADDR k8, GM_ADDR v8, GM_ADDR qs, GM_ADDR ks, GM_ADDR mxScale,
        GM_ADDR vMean, GM_ADDR out, GM_ADDR lse, const SageParams &params)
    {
        using namespace AscendC;
        p_ = params;
        p_.dp = DP;
        q8_ = q8;
        k8_ = k8;
        v8_ = v8;
        vsmx_ = mxScale;
        out_ = out;
        lse_ = lse;
        subBlock_ = GetSubBlockIdx();

        qsg_.SetGlobalBuffer((__gm__ half *)qs);
        ksg_.SetGlobalBuffer((__gm__ half *)ks);
        vmg_.SetGlobalBuffer((__gm__ float *)vMean);
        og_.SetGlobalBuffer((__gm__ T *)out);
        lg_.SetGlobalBuffer((__gm__ float *)lse);

        accum_ = resource_.ubBuf.template GetBufferByByte<float>(ACCUM_UB);
        qo_ = resource_.ubBuf.template GetBufferByByte<T>(P_UB);
        qsu_ = resource_.ubBuf.template GetBufferByByte<half>(QS_UB);
        maxima_ = resource_.ubBuf.template GetBufferByByte<half>(MAX_UB);
        denom_ = resource_.ubBuf.template GetBufferByByte<float>(DEN_UB);
        vmean_ = resource_.ubBuf.template GetBufferByByte<float>(VM_UB);
    }

    __aicore__ inline __attribute__((always_inline)) void Process() {
        using namespace AscendC;
        auto &r = resource_;
        MxScaleCache mxCache;
        SageParams p = p_;
        GM_ADDR q8 = q8_;
        GM_ADDR k8 = k8_;
        GM_ADDR v8 = v8_;
        uint32_t core = GetBlockIdx();
        if ASCEND_IS_AIV core /= 2;

        if ASCEND_IS_AIC {
            InitMxP(r);
            SetFlag<HardEvent::MTE1_MTE2>(EventHandshake::QK_EVENT);
            SetFlag<HardEvent::MTE1_MTE2>(EventHandshake::PV_EVENT);
            SetFlag<HardEvent::M_MTE1>(EventHandshake::QK_EVENT);
            SetFlag<HardEvent::M_MTE1>(EventHandshake::PV_EVENT);
            SetFlag<HardEvent::FIX_M>(EventHandshake::QK_EVENT);
            SetFlag<HardEvent::FIX_M>(EventHandshake::PV_EVENT);
        }

        uint32_t tiles = (p.sq + M - 1) / M;
        const uint64_t total = uint64_t(p.b) * p.hq * tiles;
        const uint64_t taskEnd = DieTaskEnd(p, total, core);
        uint32_t ordinal = 0;

        for (uint64_t task = core; task < taskEnd; task += p.cores, ++ordinal) {
            auto taskInfo = SnakeScheduler<CAUSAL>::Decode(p, task, ordinal);
            uint32_t b = taskInfo.b, h = taskInfo.h;
            uint32_t qstart = taskInfo.qstart;
            uint64_t qb = taskInfo.qb, kb = taskInfo.kb;

            if ASCEND_IS_AIV {
                Sync<HardEvent::V_MTE2>();
                Duplicate(accum_, 0.0f, HALF * p.dp);
                DataCopy(qsu_, qsg_[qb + qstart + subBlock_ * HALF], HALF);
                Sync<HardEvent::MTE2_S>();
                Duplicate(maxima_, half(SAGE_FP16_MIN), 128);
                Duplicate(denom_, 0.0f, 64);
                PipeBarrier<PIPE_V>();
            }

            uint32_t limit = p.sk;
            if constexpr (CAUSAL) limit = (p.sk < qstart + M ? p.sk : qstart + M);
            uint32_t count = (limit + N - 1) / N;

            if ASCEND_IS_AIV {
                SetFlag<HardEvent::V_MTE2>(EventHandshake::SCALE_EVENT_BASE);
                SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(EventHandshake::SCALE_EVENT_BASE + 1));
                for (uint32_t slot = 0; slot < 2; ++slot) {
                    if (slot < count) {
                        event_t ev = static_cast<event_t>(EventHandshake::SCALE_EVENT_BASE + slot);
                        WaitFlag<HardEvent::V_MTE2>(ev);
                        DataCopy(r.ubBuf.template GetBufferByByte<half>(KS_UB + slot * N * 2), ksg_[kb + slot * N], N);
                        SetFlag<HardEvent::MTE2_V>(ev);
                    }
                }
            }

            for (uint32_t step = 0; step < count + LOOKAHEAD; ++step) {
                uint32_t start = step * N, parity = step % PIPE_SLOTS;
                uint32_t previous = (step + PIPE_SLOTS - LOOKAHEAD) % PIPE_SLOTS;
                if ASCEND_IS_AIC {
                    if (step < count) {
                        Gemm<int8_t, Catlass::layout::ColumnMajor, false, B_L1>(
                            r, q8 + (qb + qstart) * p.dp, k8 + (kb + start) * p.dp, p.dp, N,
                            SCORE_UB + parity * HALF * N * 2, 0, step == 0);
                        EventHandshake::NotifyBothAiv<PIPE_FIX>(TokenSignal::QK_READY);
                    }
                    if (step >= LOOKAHEAD) {
                        EventHandshake::WaitFromBothAiv<PIPE_MTE1>(TokenSignal::P_READY);
                        if (step >= LOOKAHEAD + PV_SLOTS) {
                            EventHandshake::WaitFromBothAiv<PIPE_FIX>(TokenSignal::PV_RELEASE);
                        }
                        Gemm<float8_e4m3_t, Catlass::layout::RowMajor, true, B_L1>(
                            r, v8, v8 + (kb + (step - LOOKAHEAD) * N) * p.dp, N, p.dp,
                            PV_UB + ((step - LOOKAHEAD) % PV_SLOTS) * PV_SLOT_BYTES, previous, true,
                            vsmx_ + (kb + (step - LOOKAHEAD) * N) / 32 * p.dp, &mxCache, p.kp, (step - LOOKAHEAD) * N);
                        EventHandshake::NotifyBothAiv<PIPE_FIX>(TokenSignal::PV_READY);
                    }
                } else {
                    if (step < count) {
                        auto scores = r.ubBuf.template GetBufferByByte<half>(SCORE_UB + parity * HALF * N * 2);
                        auto p8 = r.ubBuf.template GetBufferByByte<float8_e4m3_t>(P_UB + parity * P_SLOT_BYTES);
                        auto alpha = r.ubBuf.template GetBufferByByte<float>(ALPHA_UB + (step % ALPHA_SLOTS) * 512);
                        EventHandshake::WaitFromCube<PIPE_V>(TokenSignal::QK_READY);
                        auto ksu = r.ubBuf.template GetBufferByByte<half>(KS_UB + (step & 1) * N * 2);
                        event_t scaleEvent = static_cast<event_t>(EventHandshake::SCALE_EVENT_BASE + (step & 1));
                        WaitFlag<HardEvent::MTE2_V>(scaleEvent);
                        const bool isFull = (qstart + subBlock_ * HALF + HALF <= p.sq && start + N <= p.sk &&
                                             (!CAUSAL || start + N <= qstart + subBlock_ * HALF + 1));
                        const uint32_t qOffset = qstart + subBlock_ * HALF;
                        auto scratch = r.ubBuf.template GetBufferByByte<float>(SUM_UB);

                        if (isFull) {
                            SimdSoftmax::SoftmaxHalfVF<HALF, N, true, P_NZ_PAD_ROWS, CAUSAL>(
                                (__ubuf__ half *)scores.GetPhyAddr(), (__ubuf__ half *)maxima_.GetPhyAddr(),
                                (__ubuf__ float *)denom_.GetPhyAddr(), (__ubuf__ float *)alpha.GetPhyAddr(),
                                (__ubuf__ float8_e4m3_t *)p8.GetPhyAddr(), (__ubuf__ half *)qsu_.GetPhyAddr(),
                                (__ubuf__ half *)ksu.GetPhyAddr(), (__ubuf__ float *)scratch.GetPhyAddr(),
                                qOffset, p.sq, start, p.sk);
                        } else {
                            SimdSoftmax::SoftmaxHalfVF<HALF, N, false, P_NZ_PAD_ROWS, CAUSAL>(
                                (__ubuf__ half *)scores.GetPhyAddr(), (__ubuf__ half *)maxima_.GetPhyAddr(),
                                (__ubuf__ float *)denom_.GetPhyAddr(), (__ubuf__ float *)alpha.GetPhyAddr(),
                                (__ubuf__ float8_e4m3_t *)p8.GetPhyAddr(), (__ubuf__ half *)qsu_.GetPhyAddr(),
                                (__ubuf__ half *)ksu.GetPhyAddr(), (__ubuf__ float *)scratch.GetPhyAddr(),
                                qOffset, p.sq, start, p.sk);
                        }
                        SetFlag<HardEvent::V_MTE2>(scaleEvent);
                        if (step + 2 < count) {
                            WaitFlag<HardEvent::V_MTE2>(scaleEvent);
                            DataCopy(ksu, ksg_[kb + (step + 2) * N], N);
                            SetFlag<HardEvent::MTE2_V>(scaleEvent);
                        }
                        Sync<HardEvent::V_MTE3>();
                        auto pl1 = r.l1Buf.template GetBufferByByte<float8_e4m3_t>(P_L1 + parity * M * N);
                        DataCopyParams cp{N / 32, HALF, P_NZ_PAD_ROWS, HALF};
                        DataCopy(pl1[subBlock_ * HALF * 32], p8, cp);
                        EventHandshake::NotifyCube<PIPE_MTE3>(TokenSignal::P_READY);
                    }
                    if (step >= LOOKAHEAD) {
                        uint32_t index = step - LOOKAHEAD;
                        if (index & 1) {
                            EventHandshake::WaitFromCube<PIPE_V>(TokenSignal::PV_READY);
                            EventHandshake::WaitFromCube<PIPE_V>(TokenSignal::PV_READY);
                            if (p.accum == SagePvAccum::FP32_PLUS_FP16) {
                                auto pv0 = r.ubBuf.template GetBufferByByte<float>(PV_UB);
                                auto pv1 = r.ubBuf.template GetBufferByByte<float>(PV_UB + PV_SLOT_BYTES);
                                OutAccumulator::RoundPartialFp16((__ubuf__ float *)pv0.GetPhyAddr(), HALF * p.dp);
                                OutAccumulator::RoundPartialFp16((__ubuf__ float *)pv1.GetPhyAddr(), HALF * p.dp);
                            }
                            auto alpha0 = r.ubBuf.template GetBufferByByte<float>(ALPHA_UB + ((index - 1) % ALPHA_SLOTS) * 512);
                            auto alpha1 = r.ubBuf.template GetBufferByByte<float>(ALPHA_UB + (index % ALPHA_SLOTS) * 512);
                            OutAccumulator::AccumulateFp32Pair<DP>(
                                (__ubuf__ float *)accum_.GetPhyAddr(),
                                (__ubuf__ float *)r.ubBuf.template GetBufferByByte<float>(PV_UB).GetPhyAddr(),
                                (__ubuf__ float *)r.ubBuf.template GetBufferByByte<float>(PV_UB + PV_SLOT_BYTES).GetPhyAddr(),
                                (__ubuf__ float *)alpha0.GetPhyAddr(), (__ubuf__ float *)alpha1.GetPhyAddr());
                            EventHandshake::NotifyCube<PIPE_V>(TokenSignal::PV_RELEASE);
                            EventHandshake::NotifyCube<PIPE_V>(TokenSignal::PV_RELEASE);
                        } else if (index + 1 == count) {
                            EventHandshake::WaitFromCube<PIPE_V>(TokenSignal::PV_READY);
                            if (p.accum == SagePvAccum::FP32_PLUS_FP16) {
                                auto pv0 = r.ubBuf.template GetBufferByByte<float>(PV_UB);
                                OutAccumulator::RoundPartialFp16((__ubuf__ float *)pv0.GetPhyAddr(), HALF * p.dp);
                            }
                            auto alpha = r.ubBuf.template GetBufferByByte<float>(ALPHA_UB + (index % ALPHA_SLOTS) * 512);
                            OutAccumulator::AccumulateFp32<DP>(
                                (__ubuf__ float *)accum_.GetPhyAddr(),
                                (__ubuf__ float *)r.ubBuf.template GetBufferByByte<float>(PV_UB).GetPhyAddr(),
                                (__ubuf__ float *)alpha.GetPhyAddr());
                            EventHandshake::NotifyCube<PIPE_V>(TokenSignal::PV_RELEASE);
                        }
                    }
                }
            }
            if ASCEND_IS_AIC {
                for (uint32_t drain = 0; drain < (count < PV_SLOTS ? count : PV_SLOTS); ++drain) {
                    EventHandshake::WaitFromBothAiv<PIPE_FIX>(TokenSignal::PV_RELEASE);
                }
            }
            if ASCEND_IS_AIV {
                WaitFlag<HardEvent::V_MTE2>(EventHandshake::SCALE_EVENT_BASE);
                WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(EventHandshake::SCALE_EVENT_BASE + 1));
                Sync<HardEvent::V_MTE2>();
                if (p.smoothV) {
                    uint32_t kh = CalcKh(h, p.hq, p.hk);
                    DataCopy(vmean_, vmg_[(uint64_t(b) * p.hk + kh) * p.dp], p.dp);
                } else {
                    Duplicate(vmean_, 0.0f, 128);
                }
                Sync<HardEvent::MTE2_V>();
                OutAccumulator::NormalizeCast<T, DP>(
                    (__ubuf__ T *)qo_.GetPhyAddr(), (__ubuf__ float *)accum_.GetPhyAddr(),
                    (__ubuf__ float *)denom_.GetPhyAddr(),
                    (__ubuf__ float *)r.ubBuf.template GetBufferByByte<float>(SUM_UB).GetPhyAddr(),
                    (__ubuf__ float *)vmean_.GetPhyAddr());
                PipeBarrier<PIPE_V>();
                uint32_t first = qstart + subBlock_ * HALF;
                uint32_t validRows = first < p.sq ? (p.sq - first < HALF ? p.sq - first : HALF) : 0;
                if (p.d == p.dp) {
                    if (validRows)
                        WriteRow(og_, ((uint64_t(b) * p.hq + h) * p.sq + first) * p.d, qo_, validRows * p.d);
                } else {
                    for (uint32_t i = 0; i < validRows; ++i)
                        WriteRow(og_, ((uint64_t(b) * p.hq + h) * p.sq + first + i) * p.d, qo_[i * p.dp], p.d);
                }
                if (p.returnLse) {
                    auto scratch = r.ubBuf.template GetBufferByByte<float>(P_UB);
                    StoreTileLse(lg_, denom_, maxima_, scratch, first, validRows, b, h, p);
                }
            }
        }
        if ASCEND_IS_AIC {
            WaitFlag<HardEvent::MTE1_MTE2>(EventHandshake::QK_EVENT);
            WaitFlag<HardEvent::MTE1_MTE2>(EventHandshake::PV_EVENT);
            WaitFlag<HardEvent::M_MTE1>(EventHandshake::QK_EVENT);
            WaitFlag<HardEvent::M_MTE1>(EventHandshake::PV_EVENT);
            WaitFlag<HardEvent::FIX_M>(EventHandshake::QK_EVENT);
            WaitFlag<HardEvent::FIX_M>(EventHandshake::PV_EVENT);
        }
        PipeBarrier<PIPE_ALL>();
    }

private:
    FixedResource resource_;
    SageParams p_;
    GM_ADDR q8_;
    GM_ADDR k8_;
    GM_ADDR v8_;
    GM_ADDR vsmx_;
    GM_ADDR out_;
    GM_ADDR lse_;
    uint32_t subBlock_;
    AscendC::GlobalTensor<half> qsg_, ksg_;
    AscendC::GlobalTensor<float> vmg_, lg_;
    AscendC::GlobalTensor<T> og_;
    AscendC::LocalTensor<float> accum_, denom_, vmean_;
    AscendC::LocalTensor<T> qo_;
    AscendC::LocalTensor<half> qsu_, maxima_;
};

template <class T, bool WIDE>
class AttentionEngine {
public:
    __aicore__ static inline __attribute__((always_inline)) void Execute(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR lse,
        GM_ADDR workspace, const SageAttention2TilingData &tilingData)
    {
        static_assert(sage::is_supported_input_type_v<T>, "Unsupported input dtype for SageAttention2");
        const SageParams &params = tilingData.params;

        if ASCEND_IS_AIV {
            FastQuant<T, WIDE>::ProcessStats(
                k, v, workspace + tilingData.mean, workspace + tilingData.vMean, params);

            FastQuant<T, WIDE>::Quantize(
                q, k, v, workspace + tilingData.mean, workspace + tilingData.vMean,
                workspace + tilingData.vMxScale, workspace + tilingData.q8,
                workspace + tilingData.k8, workspace + tilingData.v8,
                workspace + tilingData.qScale, workspace + tilingData.kScale, params);
        }

        if constexpr (WIDE) {
            if (params.causal) {
                ExecuteWideStream<T, true>(
                    q, workspace + tilingData.k8, workspace + tilingData.v8,
                    workspace + tilingData.qScale, workspace + tilingData.kScale,
                    workspace + tilingData.vMxScale, workspace + tilingData.vMean,
                    out, lse, params);
            } else {
                ExecuteWideStream<T, false>(
                    q, workspace + tilingData.k8, workspace + tilingData.v8,
                    workspace + tilingData.qScale, workspace + tilingData.kScale,
                    workspace + tilingData.vMxScale, workspace + tilingData.vMean,
                    out, lse, params);
            }
        } else {
            if (params.dp == 128) {
                if (params.causal) {
                    StandardAttentionKernel<T, 128, true> kernel;
                    kernel.Init(workspace + tilingData.q8, workspace + tilingData.k8, workspace + tilingData.v8,
                                workspace + tilingData.qScale, workspace + tilingData.kScale,
                                workspace + tilingData.vMxScale, workspace + tilingData.vMean,
                                out, lse, params);
                    kernel.Process();
                } else {
                    StandardAttentionKernel<T, 128, false> kernel;
                    kernel.Init(workspace + tilingData.q8, workspace + tilingData.k8, workspace + tilingData.v8,
                                workspace + tilingData.qScale, workspace + tilingData.kScale,
                                workspace + tilingData.vMxScale, workspace + tilingData.vMean,
                                out, lse, params);
                    kernel.Process();
                }
            } else {
                if (params.causal) {
                    StandardAttentionKernel<T, 64, true> kernel;
                    kernel.Init(workspace + tilingData.q8, workspace + tilingData.k8, workspace + tilingData.v8,
                                workspace + tilingData.qScale, workspace + tilingData.kScale,
                                workspace + tilingData.vMxScale, workspace + tilingData.vMean,
                                out, lse, params);
                    kernel.Process();
                } else {
                    StandardAttentionKernel<T, 64, false> kernel;
                    kernel.Init(workspace + tilingData.q8, workspace + tilingData.k8, workspace + tilingData.v8,
                                workspace + tilingData.qScale, workspace + tilingData.kScale,
                                workspace + tilingData.vMxScale, workspace + tilingData.vMean,
                                out, lse, params);
                    kernel.Process();
                }
            }
        }
    }
};

#endif // __CCE__

} // namespace sage::orchestration

namespace sage {
#if defined(__CCE__)
using orchestration::AttentionEngine;
using orchestration::StandardAttentionKernel;
#endif
}
