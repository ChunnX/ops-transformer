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

namespace sage::orchestration {

#if defined(__CCE__)

struct WideFragmentSlot {
    uint64_t keyBase{0};
    uint32_t fragmentPart{0};
};

struct WideTileHistoryEntry {
    uint32_t tileOrdinal{0};
    uint32_t batchIdx{0};
    uint32_t headIdx{0};
    uint32_t qOffsetStart{0};
    bool isQueryFirst{false};
    bool isQueryLast{false};
};

class WideCubeStreamEngine {
public:
    __aicore__ inline explicit WideCubeStreamEngine(memory::FixedResource &res, microkernel::MxScaleCache &cache)
        : res_(res), mxCache_(cache) {}

    template <bool CAUSAL>
    __aicore__ inline void Execute(
        uint32_t core, uint64_t totalTasks, GM_ADDR k8, GM_ADDR v8, GM_ADDR mxScale, const sage::core::SageParams &p)
    {
        InitializeSyncFlags();
        uint64_t fragment = 0;
        uint32_t ordinal = 0;
        for (uint64_t task = core; task < totalTasks; task += p.cores, ++ordinal) {
            const auto current = dispatch::SnakeScheduler<CAUSAL>::Decode(p, task, ordinal);
            PrimeTileKeyPrefetch(k8 + current.kb * 128);
            const uint32_t partLimit = current.count * 2;
            for (uint32_t part = 0; part < partLimit; part += 2, fragment += 2) {
                pendingQueue_[fragment & 3] = {current.kb, part};
                ExecuteFragmentStep<0>(fragment, true, k8, v8, mxScale, current.count, p, true);
                pendingQueue_[(fragment + 1) & 3] = {current.kb, part + 1};
                ExecuteFragmentStep<1>(fragment + 1, true, k8, v8, mxScale, current.count, p, true);
            }
        }
        DrainTrailingFragments(fragment, k8, v8, mxScale, p);
        FinalizeSyncFlags(fragment);
    }

private:
    __aicore__ inline void PrimeTileKeyPrefetch(GM_ADDR keyBaseAddr) {
        memory::PrefetchWideK<sage::core::WIDE_B_L1>(res_, keyBaseAddr, 0);
        memory::PrefetchWideK<sage::core::WIDE_B_L1>(res_, keyBaseAddr + sage::core::N * 128, 1);
    }

    __aicore__ inline void InitializeSyncFlags() {
        microkernel::InitMxP(res_);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EventHandshake::K_PREFETCH_EVENT_BASE);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<event_t>(EventHandshake::K_PREFETCH_EVENT_BASE + 1));
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EventHandshake::V_PREFETCH_EVENT_BASE);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<event_t>(EventHandshake::V_PREFETCH_EVENT_BASE + 1));
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EventHandshake::QK_EVENT);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EventHandshake::PV_EVENT);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EventHandshake::QK_EVENT);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EventHandshake::PV_EVENT);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EventHandshake::QK_EVENT);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EventHandshake::QK_SECOND_FIX_EVENT);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EventHandshake::PV_EVENT);
    }

    template <uint32_t PHASE>
    __aicore__ inline void ExecutePvGemmStage(
        GM_ADDR v8, GM_ADDR scaleBase, uint32_t pvOffset, uint32_t parity,
        const WideFragmentSlot &slot, const sage::core::SageParams &p)
    {
        constexpr bool INIT_ACCUM = (PHASE == 0);
        constexpr bool IS_LAST = (PHASE != 0);
        constexpr uint32_t PREV_PARITY = (1 - PHASE);
        microkernel::Gemm<float8_e4m3_t, Catlass::layout::RowMajor, true, sage::core::WIDE_B_L1, INIT_ACCUM, IS_LAST, true>(
            res_, v8, nullptr, sage::core::N, 128, pvOffset, parity, true,
            scaleBase + (slot.keyBase + slot.fragmentPart * sage::core::N) / 32 * 128,
            &mxCache_, p.kp, slot.fragmentPart * sage::core::N, PREV_PARITY);
    }

    template <uint32_t PHASE>
    __aicore__ inline void ExecuteFragmentStep(
        uint64_t fragIdx, bool hasQk, GM_ADDR k8, GM_ADDR v8, GM_ADDR vsmx,
        uint32_t count, const sage::core::SageParams &p, bool prefetchPv)
    {
        constexpr uint32_t DP = 128;
        if (hasQk) {
            const auto &curSlot = pendingQueue_[fragIdx & 3];
            microkernel::GemmQkSplit<sage::core::WIDE_B_L1>(res_, (fragIdx & 3) * sage::core::HALF * sage::core::N * 2, curSlot.fragmentPart == 0, PHASE);
            if (curSlot.fragmentPart + 2 < count * 2) {
                memory::PrefetchWideK<sage::core::WIDE_B_L1>(res_, k8 + (curSlot.keyBase + (curSlot.fragmentPart + 2) * sage::core::N) * DP, PHASE);
            }
        }
        if (fragIdx >= 2 && prefetchPv) {
            const auto &vSlot = pendingQueue_[(fragIdx - 2) & 3];
            memory::PrefetchWideV<sage::core::WIDE_B_L1>(
                res_, v8 + (vSlot.keyBase + vSlot.fragmentPart * sage::core::N) * DP,
                vsmx + (vSlot.keyBase + vSlot.fragmentPart * sage::core::N) / 32 * DP, PHASE);
        }
        if (fragIdx >= 3) {
            const uint64_t prevIdx = fragIdx - 3;
            const auto &oldSlot = pendingQueue_[prevIdx & 3];
            const uint32_t grp = prevIdx / 2;
            EventHandshake::WaitFromBothAiv<PIPE_MTE1>(TokenSignal::P_READY);
            const uint32_t parity = prevIdx & 3;
            const uint32_t pvOffset = memory::WideVectorLayout::PV_UB + (grp & 1) * memory::WideVectorLayout::PV_SLOT_BYTES;
            if constexpr (PHASE == 0) {
                if (grp >= 2) {
                    EventHandshake::WaitFromBothAiv<PIPE_FIX>(TokenSignal::PV_RELEASE);
                }
                ExecutePvGemmStage<0>(v8, vsmx, pvOffset, parity, oldSlot, p);
                EventHandshake::NotifyBothAiv<PIPE_FIX>(TokenSignal::PV_READY);
            } else {
                ExecutePvGemmStage<1>(v8, vsmx, pvOffset, parity, oldSlot, p);
            }
        }
    }

    __aicore__ inline void DrainTrailingFragments(
        uint64_t fragment, GM_ADDR k8, GM_ADDR v8, GM_ADDR mxScale, const sage::core::SageParams &p)
    {
        ExecuteFragmentStep<0>(fragment, false, k8, v8, mxScale, 0, p, true);
        ExecuteFragmentStep<1>(fragment + 1, false, k8, v8, mxScale, 0, p, true);
        ExecuteFragmentStep<0>(fragment + 2, false, k8, v8, mxScale, 0, p, false);
    }

    __aicore__ inline void FinalizeSyncFlags(uint64_t fragment) {
        const uint32_t drainLimit = (fragment / 2 < 2 ? fragment / 2 : 2);
        for (uint32_t i = 0; i < drainLimit; ++i) {
            EventHandshake::WaitFromBothAiv<PIPE_FIX>(TokenSignal::PV_RELEASE);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<event_t>(EventHandshake::K_PREFETCH_EVENT_BASE));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<event_t>(EventHandshake::K_PREFETCH_EVENT_BASE + 1U));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<event_t>(EventHandshake::V_PREFETCH_EVENT_BASE));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<event_t>(EventHandshake::V_PREFETCH_EVENT_BASE + 1U));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EventHandshake::QK_EVENT);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EventHandshake::PV_EVENT);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EventHandshake::QK_EVENT);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EventHandshake::PV_EVENT);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EventHandshake::QK_EVENT);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EventHandshake::QK_SECOND_FIX_EVENT);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EventHandshake::PV_EVENT);
    }

    memory::FixedResource &res_;
    microkernel::MxScaleCache &mxCache_;
    WideFragmentSlot pendingQueue_[4];
};

template <class T, bool CAUSAL>
class WideVectorStreamEngine {
public:
    __aicore__ inline explicit WideVectorStreamEngine(memory::FixedResource &res) : res_(res) {}

    // AIV 宽路径只消费原始 Q、随路 scale（QS/KS）、V 均值与 O/LSE 写回；
    // K8/V8/MXFP8 scale 由 Cube 侧与量化段消费，不进入本签名。
    __aicore__ inline void Execute(
        uint32_t core, uint64_t totalTasks, GM_ADDR q,
        GM_ADDR qs, GM_ADDR ks, GM_ADDR vMean, GM_ADDR out, GM_ADDR lse, const sage::core::SageParams &p)
    {
        uint32_t subBlock = 0U;
        if ASCEND_IS_AIV {
            subBlock = static_cast<uint32_t>(AscendC::GetBlockIdx()) & 1U;
        }
        AscendC::GlobalTensor<T> qg, og;
        qg.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(q));
        qg.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        og.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out));
        // DG.4: write-once output streams through L2 (P-07 16K protection).
        og.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

        AscendC::GlobalTensor<half> qsg, ksg;
        qsg.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(qs));
        ksg.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(ks));
        ksg.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);

        AscendC::GlobalTensor<float> vmg, lg;
        vmg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(vMean));
        vmg.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
        lg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(lse));
        lg.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

        auto accum = res_.ubBuf.template GetBufferByByte<float>(memory::WideVectorLayout::ACCUM_UB);
        auto denom = res_.ubBuf.template GetBufferByByte<float>(memory::WideVectorLayout::DEN_UB);
        auto qsu = res_.ubBuf.template GetBufferByByte<half>(memory::WideVectorLayout::QS_UB);
        auto vmean = res_.ubBuf.template GetBufferByByte<float>(memory::WideVectorLayout::VM_UB);
        auto qo = res_.ubBuf.template GetBufferByByte<T>(memory::WideVectorLayout::P_UB);

        SeedScalePrefetch(core, totalTasks, ksg, p);

        uint64_t fragment = 0;
        uint64_t retired = 0;
        uint32_t ordinal = 0;

        for (uint64_t task = core; task < totalTasks; task += p.cores, ++ordinal) {
            const auto current = dispatch::SnakeScheduler<CAUSAL>::Decode(p, task, ordinal);
            for (uint32_t step = 0; step < current.count; ++step, fragment += 2) {
                const uint64_t group = fragment / 2;
                bool deferPair = false;
                RetirePendingStep(group, retired, deferPair, vmean, vmg, qo, accum, denom, og, lg, subBlock, p);

                const bool first = (step == 0);
                const bool last = (step + 1 == current.count);
                auto maxima = res_.ubBuf.template GetBufferByByte<half>(memory::WideVectorLayout::MAX_UB + (current.ordinal & 1) * 256);

                if (first) {
                    QuantizeQueryTile(current, subBlock, qg, qsu, maxima, p);
                }

                ExecuteDualStageSoftmax(current, step, group, fragment, subBlock, ksg, qsu, maxima, totalTasks, p);

                tileHistory_[group % 3] = {current.ordinal, current.b, current.h, current.qstart, first, last};

                if (deferPair) {
                    RetireDeferredPair<true>(retired, vmean, vmg, qo, accum, denom, og, lg, subBlock, p);
                }
            }
        }

        DrainRemainingOutputs(fragment / 2, retired, vmean, vmg, qo, accum, denom, og, lg, subBlock, p);
    }

private:
    __aicore__ inline void SeedScalePrefetch(
        uint32_t core, uint64_t totalTasks, AscendC::GlobalTensor<half> &ksg, const sage::core::SageParams &p)
    {
        const auto first = dispatch::SnakeScheduler<CAUSAL>::Decode(p, core, 0);
        const uint64_t firstScale = first.kb;
        const uint64_t secondScale = dispatch::SnakeScheduler<CAUSAL>::LookaheadScale(p, first, 1, totalTasks);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EventHandshake::SCALE_EVENT_BASE);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(static_cast<event_t>(EventHandshake::SCALE_EVENT_BASE + 1));

        if (firstScale != ~uint64_t(0)) {
            const auto ev0 = static_cast<event_t>(EventHandshake::SCALE_EVENT_BASE);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ev0);
            AscendC::DataCopy(res_.ubBuf.template GetBufferByByte<half>(memory::WideVectorLayout::KS_UB), ksg[firstScale], 512);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ev0);
        }
        if (secondScale != ~uint64_t(0)) {
            const auto ev1 = static_cast<event_t>(EventHandshake::SCALE_EVENT_BASE + 1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ev1);
            AscendC::DataCopy(res_.ubBuf.template GetBufferByByte<half>(memory::WideVectorLayout::KS_UB + 1024), ksg[secondScale], 512);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ev1);
        }
    }

    __aicore__ inline void QuantizeQueryTile(
        const dispatch::TileTaskInfo &current, uint32_t subBlock,
        AscendC::GlobalTensor<T> &qg, AscendC::LocalTensor<half> &qsu,
        AscendC::LocalTensor<half> &maxima, const sage::core::SageParams &p)
    {
        constexpr uint32_t Q_IN_UB = memory::WideVectorLayout::SCORE_UB;
        constexpr uint32_t Q_FP32_UB = Q_IN_UB + sage::core::HALF * 128 * sizeof(T);
        constexpr uint32_t Q_INT8_UB = Q_FP32_UB + sage::core::HALF * 128 * sizeof(float);
        constexpr uint32_t Q_RAW_SCALE = Q_INT8_UB + sage::core::HALF * 128 * sizeof(int8_t);
        constexpr uint32_t Q_INV_SCALE = Q_RAW_SCALE + sage::core::HALF * sizeof(float);

        auto input = res_.ubBuf.template GetBufferByByte<T>(Q_IN_UB);
        auto x = res_.ubBuf.template GetBufferByByte<float>(Q_FP32_UB);
        auto qOut = res_.ubBuf.template GetBufferByByte<int8_t>(Q_INT8_UB);
        auto scales = res_.ubBuf.template GetBufferByByte<float>(Q_RAW_SCALE);
        auto inverseScales = res_.ubBuf.template GetBufferByByte<float>(Q_INV_SCALE);

        memory::Sync<AscendC::HardEvent::MTE3_MTE2>();
        memory::Sync<AscendC::HardEvent::V_MTE2>();
        const uint64_t qToken = current.qb + current.qstart + subBlock * sage::core::HALF;
        AscendC::DataCopy(input, qg[qToken * 128], sage::core::HALF * 128);
        memory::Sync<AscendC::HardEvent::MTE2_V>();

        AscendC::Cast(x, input, AscendC::RoundMode::CAST_NONE, sage::core::HALF * 128);
        AscendC::PipeBarrier<PIPE_V>();

        auto pX = reinterpret_cast<__ubuf__ float *>(x.GetPhyAddr());
        auto pScales = reinterpret_cast<__ubuf__ float *>(scales.GetPhyAddr());
        auto pQOut = reinterpret_cast<__ubuf__ int8_t *>(qOut.GetPhyAddr());
        auto pQsu = reinterpret_cast<__ubuf__ half *>(qsu.GetPhyAddr());
        auto pInvScales = reinterpret_cast<__ubuf__ float *>(inverseScales.GetPhyAddr());

        microkernel::QuantRowsVF(pX, pX, pScales, pQOut, pQsu, p.qFactor, pInvScales, 128, 0, sage::core::HALF, p.qg);

        memory::Sync<AscendC::HardEvent::V_MTE3>();
        auto ql1 = res_.l1Buf.template GetBufferByByte<int8_t>(subBlock * sage::core::HALF * 32);
        AscendC::DataCopyParams plane{sage::core::HALF, 1, 128 / 32 - 1, 0};
        #pragma unroll
        for (uint32_t cb = 0; cb < 128 / 32; ++cb) {
            AscendC::DataCopy(ql1[cb * sage::core::M * 32], qOut[cb * 32], plane);
        }

        EventHandshake::NotifyCube<PIPE_MTE3>(TokenSignal::Q_READY);
        AscendC::Duplicate(maxima, half(sage::core::SAGE_FP16_MIN), 128);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ExecuteDualStageSoftmax(
        const dispatch::TileTaskInfo &current, uint32_t step, uint64_t group, uint64_t fragment,
        uint32_t subBlock, AscendC::GlobalTensor<half> &ksg, AscendC::LocalTensor<half> &qsu,
        AscendC::LocalTensor<half> &maxima, uint64_t totalTasks, const sage::core::SageParams &p)
    {
        auto scores = res_.ubBuf.template GetBufferByByte<half>(memory::WideVectorLayout::SCORE_UB + (group & 1) * 2 * sage::core::HALF * sage::core::N * 2);
        auto p8 = scores.template ReinterpretCast<float8_e4m3_t>();
        auto alpha = res_.ubBuf.template GetBufferByByte<float>(memory::WideVectorLayout::ALPHA_UB + (group % 3) * 512);

        EventHandshake::WaitFromCube<PIPE_V>(TokenSignal::QK_READY);
        EventHandshake::WaitFromCube<PIPE_V>(TokenSignal::QK_READY);

        auto ksu = res_.ubBuf.template GetBufferByByte<half>(memory::WideVectorLayout::KS_UB + (group & 1) * 1024);
        const auto scaleEvent = static_cast<event_t>(EventHandshake::SCALE_EVENT_BASE + (group & 1));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(scaleEvent);

        const int32_t qStartInSk = static_cast<int32_t>(current.qstart + subBlock * sage::core::HALF);
        const int32_t kvStartInSk = static_cast<int32_t>(step * 512);
        const int32_t queryInGroup = qStartInSk - kvStartInSk;
        auto partialSums = res_.ubBuf.template GetBufferByByte<float>(memory::WideVectorLayout::PARTIAL_SUMS_UB);

        auto pScores = reinterpret_cast<__ubuf__ half *>(scores.GetPhyAddr());
        auto pMax = reinterpret_cast<__ubuf__ half *>(maxima.GetPhyAddr());
        auto pAlpha = reinterpret_cast<__ubuf__ float *>(alpha.GetPhyAddr());
        auto pOut8 = reinterpret_cast<__ubuf__ float8_e4m3_t *>(p8.GetPhyAddr());
        auto pQs = reinterpret_cast<__ubuf__ half *>(qsu.GetPhyAddr());
        auto pKs = reinterpret_cast<__ubuf__ half *>(ksu.GetPhyAddr());
        auto pSums = reinterpret_cast<__ubuf__ float *>(partialSums.GetPhyAddr());

        if constexpr (CAUSAL) {
            if (queryInGroup < 512) {
                microkernel::SoftmaxWideFirstVF<true>(
                    pScores, pScores + 64 * 16, pMax, pAlpha, pOut8, pQs, pKs, pSums, queryInGroup);
            } else {
                microkernel::SoftmaxWideFirstVF<false>(
                    pScores, pScores + 64 * 16, pMax, pAlpha, pOut8, pQs, pKs, pSums);
            }
        } else {
            microkernel::SoftmaxWideFirstVF<false>(
                pScores, pScores + 64 * 16, pMax, pAlpha, pOut8, pQs, pKs, pSums);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(scaleEvent);
        const uint64_t lookaheadOffset = dispatch::SnakeScheduler<CAUSAL>::LookaheadScale(p, current, step + 2, totalTasks);
        if (lookaheadOffset != ~uint64_t(0)) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(scaleEvent);
            AscendC::DataCopy(ksu, ksg[lookaheadOffset], 512);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(scaleEvent);
        }

        memory::Sync<AscendC::HardEvent::V_MTE3>();
        auto pl1_0 = res_.l1Buf.template GetBufferByByte<float8_e4m3_t>(fragment * sage::core::M * sage::core::N);
        AscendC::DataCopyParams cp0{sage::core::N / 32, sage::core::HALF, 0, sage::core::HALF};
        AscendC::DataCopy(pl1_0[subBlock * sage::core::HALF * 32], p8, cp0);
        EventHandshake::NotifyCube<PIPE_MTE3>(TokenSignal::P_READY);

        microkernel::SoftmaxWideSecondVF(
            pScores + 64 * 32, pMax, pAlpha, pOut8 + 64 * 32, pSums);

        memory::Sync<AscendC::HardEvent::V_MTE3>();
        auto pl1_1 = res_.l1Buf.template GetBufferByByte<float8_e4m3_t>((fragment + 1) * sage::core::M * sage::core::N);
        AscendC::DataCopyParams cp1{sage::core::N / 32, sage::core::HALF, 0, sage::core::HALF};
        AscendC::DataCopy(pl1_1[subBlock * sage::core::HALF * 32], p8[64 * 32], cp1);
        EventHandshake::NotifyCube<PIPE_MTE3>(TokenSignal::P_READY);
    }

    __aicore__ inline void RetireSingleStep(
        uint64_t &retired,
        AscendC::LocalTensor<float> &vmean, AscendC::GlobalTensor<float> &vmg,
        AscendC::LocalTensor<T> &qo, AscendC::LocalTensor<float> &accum,
        AscendC::LocalTensor<float> &denom, AscendC::GlobalTensor<T> &og,
        AscendC::GlobalTensor<float> &lg, uint32_t subBlock, const sage::core::SageParams &p)
    {
        const auto oldTile = tileHistory_[retired % 3];
        EventHandshake::WaitFromCube<PIPE_V>(TokenSignal::PV_READY);
        auto pv = res_.ubBuf.template GetBufferByByte<float>(memory::WideVectorLayout::PV_UB + (retired & 1) * memory::WideVectorLayout::PV_SLOT_BYTES);
        auto alpha = res_.ubBuf.template GetBufferByByte<float>(memory::WideVectorLayout::ALPHA_UB + (retired % 3) * 512);
        auto scratch = res_.ubBuf.template GetBufferByByte<float>(memory::WideVectorLayout::SUM_UB);

        auto pQo = reinterpret_cast<__ubuf__ T *>(qo.GetPhyAddr());
        auto pAccum = reinterpret_cast<__ubuf__ float *>(accum.GetPhyAddr());
        auto pPv = reinterpret_cast<__ubuf__ float *>(pv.GetPhyAddr());
        auto pAlpha = reinterpret_cast<__ubuf__ float *>(alpha.GetPhyAddr());
        auto pDen = reinterpret_cast<__ubuf__ float *>(denom.GetPhyAddr());
        auto pScratch = reinterpret_cast<__ubuf__ float *>(scratch.GetPhyAddr());

        microkernel::OutAccumulator::FoldPartial(pPv, p);

        if (oldTile.isQueryLast) {
            memory::Sync<AscendC::HardEvent::V_MTE2>();
            if (p.smoothV) {
                const uint32_t kh = dispatch::CalcKh(oldTile.headIdx, p.hq, p.hk);
                AscendC::DataCopy(vmean, vmg[(uint64_t(oldTile.batchIdx) * p.hk + kh) * p.dp], p.dp);
            } else {
                AscendC::Duplicate(vmean, 0.0f, 128);
            }
            memory::Sync<AscendC::HardEvent::MTE2_V>();
            auto pVmean = reinterpret_cast<__ubuf__ float *>(vmean.GetPhyAddr());

            if (oldTile.isQueryFirst) {
                microkernel::OutAccumulator::FinalizeWideFp32<T, 128, true>(
                    pQo, pAccum, pPv, pAlpha, pDen, pScratch, pVmean);
            } else {
                microkernel::OutAccumulator::FinalizeWideFp32<T, 128, false>(
                    pQo, pAccum, pPv, pAlpha, pDen, pScratch, pVmean);
            }
            AscendC::PipeBarrier<PIPE_V>();
            const uint32_t firstRow = oldTile.qOffsetStart + subBlock * sage::core::HALF;
            const uint32_t validRows = firstRow < p.sq ? (p.sq - firstRow < sage::core::HALF ? p.sq - firstRow : sage::core::HALF) : 0;
            if (validRows) {
                memory::WriteRow(og, ((uint64_t(oldTile.batchIdx) * p.hq + oldTile.headIdx) * p.sq + firstRow) * p.d, qo, validRows * p.d);
            }
            if (p.returnLse) {
                auto lseScratch = res_.ubBuf.template GetBufferByByte<float>(memory::WideVectorLayout::P_UB);
                auto maximaOld = res_.ubBuf.template GetBufferByByte<half>(memory::WideVectorLayout::MAX_UB + (oldTile.tileOrdinal & 1) * 256);
                microkernel::StoreTileLse(lg, denom, maximaOld, lseScratch, firstRow, validRows, oldTile.batchIdx, oldTile.headIdx, p);
            }
        } else if (oldTile.isQueryFirst) {
            microkernel::OutAccumulator::InitializeWideFp32<128>(pAccum, pPv, pAlpha, pDen);
        } else {
            microkernel::OutAccumulator::AccumulateWideFp32<128>(pAccum, pPv, pAlpha, pDen);
        }
        EventHandshake::NotifyCube<PIPE_V>(TokenSignal::PV_RELEASE);
        ++retired;
    }

    __aicore__ inline void RetirePendingStep(
        uint64_t group, uint64_t &retired, bool &deferPair,
        AscendC::LocalTensor<float> &vmean, AscendC::GlobalTensor<float> &vmg,
        AscendC::LocalTensor<T> &qo, AscendC::LocalTensor<float> &accum,
        AscendC::LocalTensor<float> &denom, AscendC::GlobalTensor<T> &og,
        AscendC::GlobalTensor<float> &lg, uint32_t subBlock, const sage::core::SageParams &p)
    {
        deferPair = false;
        if (group >= 2 && retired <= group - 2) {
            const auto &oldTile = tileHistory_[retired % 3];
            deferPair = (retired + 1 < group && oldTile.tileOrdinal == tileHistory_[(retired + 1) % 3].tileOrdinal);
            if (deferPair) {
                EventHandshake::WaitFromCube<PIPE_V>(TokenSignal::PV_READY);
            } else {
                RetireSingleStep(retired, vmean, vmg, qo, accum, denom, og, lg, subBlock, p);
            }
        }
    }

    template <bool FIRST_READY = false>
    __aicore__ inline void RetireDeferredPair(
        uint64_t &retired,
        AscendC::LocalTensor<float> &vmean, AscendC::GlobalTensor<float> &vmg,
        AscendC::LocalTensor<T> &qo, AscendC::LocalTensor<float> &accum,
        AscendC::LocalTensor<float> &denom, AscendC::GlobalTensor<T> &og,
        AscendC::GlobalTensor<float> &lg, uint32_t subBlock, const sage::core::SageParams &p)
    {
        if constexpr (!FIRST_READY) {
            EventHandshake::WaitFromCube<PIPE_V>(TokenSignal::PV_READY);
        }
        EventHandshake::WaitFromCube<PIPE_V>(TokenSignal::PV_READY);

        const auto firstInfo = tileHistory_[retired % 3];
        const auto secondInfo = tileHistory_[(retired + 1) % 3];
        memory::Sync<AscendC::HardEvent::V_MTE2>();
        if (p.smoothV) {
            const uint32_t kh = dispatch::CalcKh(secondInfo.headIdx, p.hq, p.hk);
            AscendC::DataCopy(vmean, vmg[(uint64_t(secondInfo.batchIdx) * p.hk + kh) * p.dp], p.dp);
        } else {
            AscendC::Duplicate(vmean, 0.0f, 128);
        }
        memory::Sync<AscendC::HardEvent::MTE2_V>();
        auto pv0 = res_.ubBuf.template GetBufferByByte<float>(memory::WideVectorLayout::PV_UB + (retired & 1) * memory::WideVectorLayout::PV_SLOT_BYTES);
        auto pv1 = res_.ubBuf.template GetBufferByByte<float>(memory::WideVectorLayout::PV_UB + ((retired + 1) & 1) * memory::WideVectorLayout::PV_SLOT_BYTES);
        auto a0 = res_.ubBuf.template GetBufferByByte<float>(memory::WideVectorLayout::ALPHA_UB + (retired % 3) * 512);
        auto a1 = res_.ubBuf.template GetBufferByByte<float>(memory::WideVectorLayout::ALPHA_UB + ((retired + 1) % 3) * 512);
        auto scratch = res_.ubBuf.template GetBufferByByte<float>(memory::WideVectorLayout::SUM_UB);

        auto pQo = reinterpret_cast<__ubuf__ T *>(qo.GetPhyAddr());
        auto pAccum = reinterpret_cast<__ubuf__ float *>(accum.GetPhyAddr());
        auto pPv0 = reinterpret_cast<__ubuf__ float *>(pv0.GetPhyAddr());
        auto pPv1 = reinterpret_cast<__ubuf__ float *>(pv1.GetPhyAddr());
        auto pA0 = reinterpret_cast<__ubuf__ float *>(a0.GetPhyAddr());
        auto pA1 = reinterpret_cast<__ubuf__ float *>(a1.GetPhyAddr());
        auto pDen = reinterpret_cast<__ubuf__ float *>(denom.GetPhyAddr());
        auto pScratch = reinterpret_cast<__ubuf__ float *>(scratch.GetPhyAddr());
        auto pVmean = reinterpret_cast<__ubuf__ float *>(vmean.GetPhyAddr());

        microkernel::OutAccumulator::FoldPartial(pPv0, p);
        microkernel::OutAccumulator::FoldPartial(pPv1, p);
        const uint32_t dispatchTag = (firstInfo.isQueryFirst ? 2 : 0) | (secondInfo.isQueryLast ? 1 : 0);
        switch (dispatchTag) {
        case 3:
            microkernel::OutAccumulator::UpdateWidePair<T, 128, true, true>(
                pQo, pAccum, pPv0, pPv1, pA0, pA1, pDen, pScratch, pVmean);
            break;
        case 2:
            microkernel::OutAccumulator::UpdateWidePair<T, 128, true, false>(
                pQo, pAccum, pPv0, pPv1, pA0, pA1, pDen, pScratch, pVmean);
            break;
        case 1:
            microkernel::OutAccumulator::UpdateWidePair<T, 128, false, true>(
                pQo, pAccum, pPv0, pPv1, pA0, pA1, pDen, pScratch, pVmean);
            break;
        default:
            microkernel::OutAccumulator::UpdateWidePair<T, 128, false, false>(
                pQo, pAccum, pPv0, pPv1, pA0, pA1, pDen, pScratch, pVmean);
            break;
        }

        if (secondInfo.isQueryLast) {
            AscendC::PipeBarrier<PIPE_V>();
            const uint32_t firstRow = secondInfo.qOffsetStart + subBlock * sage::core::HALF;
            const uint32_t validRows = firstRow < p.sq ? (p.sq - firstRow < sage::core::HALF ? p.sq - firstRow : sage::core::HALF) : 0;
            if (validRows) {
                memory::WriteRow(og, ((uint64_t(secondInfo.batchIdx) * p.hq + secondInfo.headIdx) * p.sq + firstRow) * p.d, qo, validRows * p.d);
            }
            if (p.returnLse) {
                auto lseScratch = res_.ubBuf.template GetBufferByByte<float>(memory::WideVectorLayout::P_UB);
                auto maximaOld = res_.ubBuf.template GetBufferByByte<half>(memory::WideVectorLayout::MAX_UB + (secondInfo.tileOrdinal & 1) * 256);
                microkernel::StoreTileLse(lg, denom, maximaOld, lseScratch, firstRow, validRows, secondInfo.batchIdx, secondInfo.headIdx, p);
            }
        }
        EventHandshake::NotifyCube<PIPE_V>(TokenSignal::PV_RELEASE);
        EventHandshake::NotifyCube<PIPE_V>(TokenSignal::PV_RELEASE);
        retired += 2;
    }

    __aicore__ inline void DrainRemainingOutputs(
        uint64_t groups, uint64_t &retired,
        AscendC::LocalTensor<float> &vmean, AscendC::GlobalTensor<float> &vmg,
        AscendC::LocalTensor<T> &qo, AscendC::LocalTensor<float> &accum,
        AscendC::LocalTensor<float> &denom, AscendC::GlobalTensor<T> &og,
        AscendC::GlobalTensor<float> &lg, uint32_t subBlock, const sage::core::SageParams &p)
    {
        while (retired < groups) {
            const auto firstInfo = tileHistory_[retired % 3];
            if (retired + 1 < groups && firstInfo.tileOrdinal == tileHistory_[(retired + 1) % 3].tileOrdinal) {
                RetireDeferredPair<false>(retired, vmean, vmg, qo, accum, denom, og, lg, subBlock, p);
            } else {
                RetireSingleStep(retired, vmean, vmg, qo, accum, denom, og, lg, subBlock, p);
            }
        }
    }

    memory::FixedResource &res_;
    WideTileHistoryEntry tileHistory_[3];
};

template <class T, bool CAUSAL = false>
__aicore__ inline __attribute__((always_inline)) void ExecuteWideStream(
    GM_ADDR q, GM_ADDR k8, GM_ADDR v8, GM_ADDR qs, GM_ADDR ks, GM_ADDR mxScale,
    GM_ADDR vMean, GM_ADDR out, GM_ADDR lse, const sage::core::SageParams &p)
{
    memory::FixedResource r;
    microkernel::MxScaleCache mxCache;
    uint32_t core = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV {
        core /= 2;
    }

    const uint64_t totalTasks = dispatch::DieTaskEnd(p, uint64_t(p.b) * p.hq * (p.sq / sage::core::M), core);
    if (core >= totalTasks) {
        return;
    }

    if ASCEND_IS_AIC {
        WideCubeStreamEngine cubeEngine(r, mxCache);
        cubeEngine.Execute<CAUSAL>(core, totalTasks, k8, v8, mxScale, p);
    } else {
        WideVectorStreamEngine<T, CAUSAL> vectorEngine(r);
        vectorEngine.Execute(core, totalTasks, q, qs, ks, vMean, out, lse, p);
    }

    AscendC::PipeBarrier<PIPE_ALL>();
}

#endif // __CCE__

} // namespace sage::orchestration

namespace sage {
#if defined(__CCE__)
using orchestration::ExecuteWideStream;
#endif
}
