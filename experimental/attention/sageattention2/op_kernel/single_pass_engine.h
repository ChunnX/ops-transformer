#pragma once

#include "decoupled_tri_queue.h"

namespace sage::orchestration {

#if defined(__CCE__)

struct StandardTileTask {
    uint32_t b{0};
    uint32_t h{0};
    uint32_t qstart{0};
    uint64_t qb{0};
    uint64_t kb{0};
    uint32_t count{0};
};

struct StandardKernelAddresses {
    GM_ADDR q8{0};
    GM_ADDR k8{0};
    GM_ADDR v8{0};
    GM_ADDR vsmx{0};
};

template <bool CAUSAL>
__aicore__ inline StandardTileTask BuildTileTask(
    const dispatch::TileTaskInfo &info, const sage::core::SageParams &p)
{
    StandardTileTask task;
    task.b = info.b;
    task.h = info.h;
    task.qstart = info.qstart;
    task.qb = info.qb;
    task.kb = info.kb;
    uint32_t limit = p.sk;
    if constexpr (CAUSAL) {
        limit = (p.sk < task.qstart + sage::core::M ? p.sk : task.qstart + sage::core::M);
    }
    task.count = (limit + sage::core::N - 1) / sage::core::N;
    return task;
}

template <uint32_t DP>
class StandardCubeScheduler {
public:
    __aicore__ inline explicit StandardCubeScheduler(memory::FixedResource &res, microkernel::MxScaleCache &cache)
        : res_(res), mxCache_(cache) {}

    __aicore__ inline void PrimeHardwareEvents() {
        microkernel::InitMxP(res_);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EventHandshake::QK_EVENT);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EventHandshake::PV_EVENT);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EventHandshake::QK_EVENT);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EventHandshake::PV_EVENT);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EventHandshake::QK_EVENT);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EventHandshake::PV_EVENT);
    }

    __aicore__ inline void TeardownHardwareEvents() {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EventHandshake::QK_EVENT);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EventHandshake::PV_EVENT);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EventHandshake::QK_EVENT);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EventHandshake::PV_EVENT);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EventHandshake::QK_EVENT);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EventHandshake::PV_EVENT);
    }

    __aicore__ inline void DispatchTile(
        const StandardTileTask &task, const StandardKernelAddresses &addrs, const sage::core::SageParams &p)
    {
        const uint32_t totalSteps = task.count + sage::core::LOOKAHEAD;
        for (uint32_t step = 0; step < totalSteps; ++step) {
            if (step < task.count) {
                DispatchQkMatmul(step, task, addrs.q8, addrs.k8, p);
            }
            if (step >= sage::core::LOOKAHEAD) {
                DispatchPvMatmul(step, task, addrs.v8, addrs.vsmx, p);
            }
        }
        DrainPvTokens(task.count);
    }

private:
    __aicore__ inline void DispatchQkMatmul(
        uint32_t step, const StandardTileTask &task, GM_ADDR q8, GM_ADDR k8, const sage::core::SageParams &p)
    {
        const uint32_t parity = step % sage::core::PIPE_SLOTS;
        const uint32_t colOffset = step * sage::core::N;
        microkernel::Gemm<int8_t, Catlass::layout::ColumnMajor, false, sage::core::B_L1>(
            res_, q8 + (task.qb + task.qstart) * p.dp, k8 + (task.kb + colOffset) * p.dp,
            p.dp, sage::core::N, memory::SCORE_UB + parity * sage::core::HALF * sage::core::N * 2, 0, step == 0);
        EventHandshake::NotifyBothAiv<PIPE_FIX>(TokenSignal::QK_READY);
    }

    __aicore__ inline void DispatchPvMatmul(
        uint32_t step, const StandardTileTask &task, GM_ADDR v8, GM_ADDR vsmx, const sage::core::SageParams &p)
    {
        EventHandshake::WaitFromBothAiv<PIPE_MTE1>(TokenSignal::P_READY);
        if (step >= sage::core::LOOKAHEAD + sage::core::PV_SLOTS) {
            EventHandshake::WaitFromBothAiv<PIPE_FIX>(TokenSignal::PV_RELEASE);
        }
        const uint32_t pvIdx = step - sage::core::LOOKAHEAD;
        const uint32_t parity = (step + sage::core::PIPE_SLOTS - sage::core::LOOKAHEAD) % sage::core::PIPE_SLOTS;
        const uint32_t pvOffset = memory::PV_UB + (pvIdx % sage::core::PV_SLOTS) * memory::PV_SLOT_BYTES;
        const uint32_t colOffset = pvIdx * sage::core::N;
        microkernel::Gemm<float8_e4m3_t, Catlass::layout::RowMajor, true, sage::core::B_L1>(
            res_, v8, v8 + (task.kb + colOffset) * p.dp, sage::core::N, p.dp,
            pvOffset, parity, true,
            vsmx + (task.kb + colOffset) / 32 * p.dp, &mxCache_, p.kp, colOffset);
        EventHandshake::NotifyBothAiv<PIPE_FIX>(TokenSignal::PV_READY);
    }

    __aicore__ inline void DrainPvTokens(uint32_t count) {
        const uint32_t tokensToDrain = count < sage::core::PV_SLOTS ? count : sage::core::PV_SLOTS;
        for (uint32_t i = 0; i < tokensToDrain; ++i) {
            EventHandshake::WaitFromBothAiv<PIPE_FIX>(TokenSignal::PV_RELEASE);
        }
    }

    memory::FixedResource &res_;
    microkernel::MxScaleCache &mxCache_;
};

template <class T, uint32_t DP, bool CAUSAL>
class StandardVectorScheduler {
public:
    __aicore__ inline StandardVectorScheduler(
        memory::FixedResource &res, uint32_t subBlock,
        AscendC::LocalTensor<float> &accum, AscendC::LocalTensor<T> &qo,
        AscendC::LocalTensor<half> &qsu, AscendC::LocalTensor<half> &maxima,
        AscendC::LocalTensor<float> &denom, AscendC::LocalTensor<float> &vmean,
        AscendC::GlobalTensor<half> &qsg, AscendC::GlobalTensor<half> &ksg,
        AscendC::GlobalTensor<float> &vmg, AscendC::GlobalTensor<T> &og,
        AscendC::GlobalTensor<float> &lg)
        : res_(res), subBlock_(subBlock),
          accum_(accum), qo_(qo), qsu_(qsu), maxima_(maxima), denom_(denom), vmean_(vmean),
          qsg_(qsg), ksg_(ksg), vmg_(vmg), og_(og), lg_(lg) {}

    __aicore__ inline void PrepareTileState(const StandardTileTask &task, const sage::core::SageParams &p) {
        memory::Sync<AscendC::HardEvent::V_MTE2>();
        AscendC::Duplicate(accum_, 0.0f, sage::core::HALF * p.dp);
        AscendC::DataCopy(qsu_, qsg_[task.qb + task.qstart + subBlock_ * sage::core::HALF], sage::core::HALF);
        memory::Sync<AscendC::HardEvent::MTE2_S>();
        AscendC::Duplicate(maxima_, half(sage::core::SAGE_FP16_MIN), 128);
        AscendC::Duplicate(denom_, 0.0f, 64);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SeedScalePrefetch(const StandardTileTask &task) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EventHandshake::SCALE_EVENT_BASE);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(static_cast<event_t>(EventHandshake::SCALE_EVENT_BASE + 1));
        if (task.count > 0) {
            const auto ev0 = static_cast<event_t>(EventHandshake::SCALE_EVENT_BASE);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ev0);
            AscendC::DataCopy(res_.ubBuf.template GetBufferByByte<half>(memory::KS_UB), ksg_[task.kb], sage::core::N);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ev0);
        }
        if (task.count > 1) {
            const auto ev1 = static_cast<event_t>(EventHandshake::SCALE_EVENT_BASE + 1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ev1);
            AscendC::DataCopy(res_.ubBuf.template GetBufferByByte<half>(memory::KS_UB + sage::core::N * 2),
                              ksg_[task.kb + sage::core::N], sage::core::N);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ev1);
        }
    }

    __aicore__ inline void StepSoftmax(uint32_t step, const StandardTileTask &task, const sage::core::SageParams &p) {
        const uint32_t parity = step % sage::core::PIPE_SLOTS;
        const uint32_t colOffset = step * sage::core::N;
        auto scores = res_.ubBuf.template GetBufferByByte<half>(memory::SCORE_UB + parity * sage::core::HALF * sage::core::N * 2);
        auto p8 = res_.ubBuf.template GetBufferByByte<float8_e4m3_t>(memory::P_UB + parity * sage::core::P_SLOT_BYTES);
        auto alpha = res_.ubBuf.template GetBufferByByte<float>(memory::ALPHA_UB + (step % sage::core::ALPHA_SLOTS) * 512);

        EventHandshake::WaitFromCube<PIPE_V>(TokenSignal::QK_READY);
        auto ksu = res_.ubBuf.template GetBufferByByte<half>(memory::KS_UB + (step & 1) * sage::core::N * 2);
        const auto scaleEv = static_cast<event_t>(EventHandshake::SCALE_EVENT_BASE + (step & 1));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(scaleEv);

        const uint32_t qOffset = task.qstart + subBlock_ * sage::core::HALF;
        const bool isFull = (qOffset + sage::core::HALF <= p.sq && colOffset + sage::core::N <= p.sk &&
                             (!CAUSAL || colOffset + sage::core::N <= qOffset + 1));
        auto scratch = res_.ubBuf.template GetBufferByByte<float>(memory::SUM_UB);

        auto pScores = reinterpret_cast<__ubuf__ half *>(scores.GetPhyAddr());
        auto pMax = reinterpret_cast<__ubuf__ half *>(maxima_.GetPhyAddr());
        auto pDen = reinterpret_cast<__ubuf__ float *>(denom_.GetPhyAddr());
        auto pAlpha = reinterpret_cast<__ubuf__ float *>(alpha.GetPhyAddr());
        auto pOut8 = reinterpret_cast<__ubuf__ float8_e4m3_t *>(p8.GetPhyAddr());
        auto pQs = reinterpret_cast<__ubuf__ half *>(qsu_.GetPhyAddr());
        auto pKs = reinterpret_cast<__ubuf__ half *>(ksu.GetPhyAddr());
        auto pScratch = reinterpret_cast<__ubuf__ float *>(scratch.GetPhyAddr());

        if (isFull) {
            microkernel::SoftmaxHalfVF<sage::core::HALF, sage::core::N, true, sage::core::P_NZ_PAD_ROWS, CAUSAL>(
                pScores, pMax, pDen, pAlpha, pOut8, pQs, pKs, pScratch, qOffset, p.sq, colOffset, p.sk);
        } else {
            microkernel::SoftmaxHalfVF<sage::core::HALF, sage::core::N, false, sage::core::P_NZ_PAD_ROWS, CAUSAL>(
                pScores, pMax, pDen, pAlpha, pOut8, pQs, pKs, pScratch, qOffset, p.sq, colOffset, p.sk);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(scaleEv);
        if (step + 2 < task.count) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(scaleEv);
            AscendC::DataCopy(ksu, ksg_[task.kb + (step + 2) * sage::core::N], sage::core::N);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(scaleEv);
        }

        memory::Sync<AscendC::HardEvent::V_MTE3>();
        auto pl1 = res_.l1Buf.template GetBufferByByte<float8_e4m3_t>(sage::core::P_L1 + parity * sage::core::M * sage::core::N);
        AscendC::DataCopyParams cp{sage::core::N / 32, sage::core::HALF, sage::core::P_NZ_PAD_ROWS, sage::core::HALF};
        AscendC::DataCopy(pl1[subBlock_ * sage::core::HALF * 32], p8, cp);
        EventHandshake::NotifyCube<PIPE_MTE3>(TokenSignal::P_READY);
    }

    __aicore__ inline void StepAccumulate(uint32_t step, uint32_t totalCount, const sage::core::SageParams &p) {
        const uint32_t pvIndex = step - sage::core::LOOKAHEAD;
        if (pvIndex & 1) {
            EventHandshake::WaitFromCube<PIPE_V>(TokenSignal::PV_READY);
            EventHandshake::WaitFromCube<PIPE_V>(TokenSignal::PV_READY);
            auto pv0 = res_.ubBuf.template GetBufferByByte<float>(memory::PV_UB);
            auto pv1 = res_.ubBuf.template GetBufferByByte<float>(memory::PV_UB + memory::PV_SLOT_BYTES);
            microkernel::OutAccumulator::FoldPartial(reinterpret_cast<__ubuf__ float *>(pv0.GetPhyAddr()), p);
            microkernel::OutAccumulator::FoldPartial(reinterpret_cast<__ubuf__ float *>(pv1.GetPhyAddr()), p);
            auto alpha0 = res_.ubBuf.template GetBufferByByte<float>(memory::ALPHA_UB + ((pvIndex - 1) % sage::core::ALPHA_SLOTS) * 512);
            auto alpha1 = res_.ubBuf.template GetBufferByByte<float>(memory::ALPHA_UB + (pvIndex % sage::core::ALPHA_SLOTS) * 512);
            microkernel::OutAccumulator::AccumulateFp32Pair<DP>(
                reinterpret_cast<__ubuf__ float *>(accum_.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(pv0.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(pv1.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(alpha0.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(alpha1.GetPhyAddr()));
            EventHandshake::NotifyCube<PIPE_V>(TokenSignal::PV_RELEASE);
            EventHandshake::NotifyCube<PIPE_V>(TokenSignal::PV_RELEASE);
        } else if (pvIndex + 1 == totalCount) {
            EventHandshake::WaitFromCube<PIPE_V>(TokenSignal::PV_READY);
            auto pv0 = res_.ubBuf.template GetBufferByByte<float>(memory::PV_UB);
            microkernel::OutAccumulator::FoldPartial(reinterpret_cast<__ubuf__ float *>(pv0.GetPhyAddr()), p);
            auto alpha = res_.ubBuf.template GetBufferByByte<float>(memory::ALPHA_UB + (pvIndex % sage::core::ALPHA_SLOTS) * 512);
            microkernel::OutAccumulator::AccumulateFp32<DP>(
                reinterpret_cast<__ubuf__ float *>(accum_.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(pv0.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(alpha.GetPhyAddr()));
            EventHandshake::NotifyCube<PIPE_V>(TokenSignal::PV_RELEASE);
        }
    }

    __aicore__ inline void EpilogueTile(const StandardTileTask &task, const sage::core::SageParams &p) {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EventHandshake::SCALE_EVENT_BASE);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(static_cast<event_t>(EventHandshake::SCALE_EVENT_BASE + 1));
        memory::Sync<AscendC::HardEvent::V_MTE2>();
        if (p.smoothV) {
            const uint32_t kh = dispatch::CalcKh(task.h, p.hq, p.hk);
            AscendC::DataCopy(vmean_, vmg_[(uint64_t(task.b) * p.hk + kh) * p.dp], p.dp);
        } else {
            AscendC::Duplicate(vmean_, 0.0f, 128);
        }
        memory::Sync<AscendC::HardEvent::MTE2_V>();
        auto scratch = res_.ubBuf.template GetBufferByByte<float>(memory::SUM_UB);
        microkernel::OutAccumulator::NormalizeCast<T, DP>(
            reinterpret_cast<__ubuf__ T *>(qo_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(accum_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(denom_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(scratch.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(vmean_.GetPhyAddr()));
        AscendC::PipeBarrier<PIPE_V>();
        const uint32_t first = task.qstart + subBlock_ * sage::core::HALF;
        const uint32_t validRows = first < p.sq ? (p.sq - first < sage::core::HALF ? p.sq - first : sage::core::HALF) : 0;
        if (p.d == p.dp) {
            if (validRows) {
                memory::WriteRow(og_, ((uint64_t(task.b) * p.hq + task.h) * p.sq + first) * p.d, qo_, validRows * p.d);
            }
        } else {
            for (uint32_t i = 0; i < validRows; ++i) {
                memory::WriteRow(og_, ((uint64_t(task.b) * p.hq + task.h) * p.sq + first + i) * p.d, qo_[i * p.dp], p.d);
            }
        }
        // [Contract P0.3] StoreTileLse is the single LSE emission site shared with the
        // Wide Streaming path; it writes the natural-log LSE of the smoothed scores and
        // the Python layer restores the smooth-K term. See online_reduction_softmax.h.
        if (p.returnLse) {
            auto lseScratch = res_.ubBuf.template GetBufferByByte<float>(memory::P_UB);
            microkernel::StoreTileLse(lg_, denom_, maxima_, lseScratch, first, validRows, task.b, task.h, p);
        }
    }

private:
    memory::FixedResource &res_;
    uint32_t subBlock_;
    AscendC::LocalTensor<float> &accum_;
    AscendC::LocalTensor<T> &qo_;
    AscendC::LocalTensor<half> &qsu_;
    AscendC::LocalTensor<half> &maxima_;
    AscendC::LocalTensor<float> &denom_;
    AscendC::LocalTensor<float> &vmean_;
    AscendC::GlobalTensor<half> &qsg_;
    AscendC::GlobalTensor<half> &ksg_;
    AscendC::GlobalTensor<float> &vmg_;
    AscendC::GlobalTensor<T> &og_;
    AscendC::GlobalTensor<float> &lg_;
};

template <class T, uint32_t DP, bool CAUSAL = false>
class StandardAttentionKernel {
public:
    __aicore__ inline __attribute__((always_inline)) void Init(
        GM_ADDR q8, GM_ADDR k8, GM_ADDR v8, GM_ADDR qs, GM_ADDR ks, GM_ADDR mxScale,
        GM_ADDR vMean, GM_ADDR out, GM_ADDR lse, const sage::core::SageParams &params)
    {
        p_ = params;
        p_.dp = DP;
        addrs_ = StandardKernelAddresses{q8, k8, v8, mxScale};
        if ASCEND_IS_AIV {
            subBlock_ = static_cast<uint32_t>(AscendC::GetBlockIdx()) & 1U;
        } else {
            subBlock_ = 0U;
        }

        qsg_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(qs));
        ksg_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(ks));
        vmg_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(vMean));
        og_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out));
        lg_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(lse));
        // DG.4: write-once outputs stream through L2 instead of evicting the
        // K/quant working set (matters at P-07 16K context lengths).
        og_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        lg_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

        accum_ = resource_.ubBuf.template GetBufferByByte<float>(memory::ACCUM_UB);
        qo_ = resource_.ubBuf.template GetBufferByByte<T>(memory::P_UB);
        qsu_ = resource_.ubBuf.template GetBufferByByte<half>(memory::QS_UB);
        maxima_ = resource_.ubBuf.template GetBufferByByte<half>(memory::MAX_UB);
        denom_ = resource_.ubBuf.template GetBufferByByte<float>(memory::DEN_UB);
        vmean_ = resource_.ubBuf.template GetBufferByByte<float>(memory::VM_UB);
    }

    __aicore__ inline __attribute__((always_inline)) void Process() {
        const sage::core::SageParams p = p_;
        uint32_t core = AscendC::GetBlockIdx();
        if ASCEND_IS_AIV {
            core /= 2;
        }

        const uint32_t tiles = (p.sq + sage::core::M - 1) / sage::core::M;
        const uint64_t total = uint64_t(p.b) * p.hq * tiles;
        const uint64_t taskEnd = dispatch::DieTaskEnd(p, total, core);

        if ASCEND_IS_AIC {
            StandardCubeScheduler<DP> cubeEngine(resource_, mxCache_);
            cubeEngine.PrimeHardwareEvents();
            uint32_t ordinal = 0;
            for (uint64_t task = core; task < taskEnd; task += p.cores, ++ordinal) {
                const auto taskInfo = dispatch::SnakeScheduler<CAUSAL>::Decode(p, task, ordinal);
                const StandardTileTask tileTask = BuildTileTask<CAUSAL>(taskInfo, p);
                cubeEngine.DispatchTile(tileTask, addrs_, p);
            }
            cubeEngine.TeardownHardwareEvents();
        } else {
            StandardVectorScheduler<T, DP, CAUSAL> vecEngine(
                resource_, subBlock_, accum_, qo_, qsu_, maxima_, denom_, vmean_,
                qsg_, ksg_, vmg_, og_, lg_);
            uint32_t ordinal = 0;
            for (uint64_t task = core; task < taskEnd; task += p.cores, ++ordinal) {
                const auto taskInfo = dispatch::SnakeScheduler<CAUSAL>::Decode(p, task, ordinal);
                const StandardTileTask tileTask = BuildTileTask<CAUSAL>(taskInfo, p);
                vecEngine.PrepareTileState(tileTask, p);
                vecEngine.SeedScalePrefetch(tileTask);
                const uint32_t totalSteps = tileTask.count + sage::core::LOOKAHEAD;
                for (uint32_t step = 0; step < totalSteps; ++step) {
                    if (step < tileTask.count) {
                        vecEngine.StepSoftmax(step, tileTask, p);
                    }
                    if (step >= sage::core::LOOKAHEAD) {
                        vecEngine.StepAccumulate(step, tileTask.count, p);
                    }
                }
                vecEngine.EpilogueTile(tileTask, p);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    memory::FixedResource resource_;
    microkernel::MxScaleCache mxCache_;
    sage::core::SageParams p_;
    StandardKernelAddresses addrs_;
    uint32_t subBlock_{0};
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
    // DP128/CAUSAL come from the template tiling key, so each compiled binary
    // carries exactly one engine specialization (no runtime dispatch over dead
    // instances, minimal icache footprint).
    template <bool DP128, bool CAUSAL>
    __aicore__ static inline __attribute__((always_inline)) void Execute(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR lse,
        GM_ADDR workspace, const SageAttention2TilingData &tilingData)
    {
        static_assert(sage::is_supported_input_type_v<T>, "Unsupported input dtype for SageAttention2");
        const sage::core::SageParams &params = tilingData.params;

        if ASCEND_IS_AIV {
            microkernel::FastQuant<T, WIDE>::ProcessStats(
                k, v, workspace + tilingData.mean, workspace + tilingData.vMean, params);

            AscendC::SyncAll<true>();

            microkernel::FastQuant<T, WIDE>::Quantize(
                q, k, v, workspace + tilingData.mean, workspace + tilingData.vMean,
                workspace + tilingData.vMxScale, workspace + tilingData.q8,
                workspace + tilingData.k8, workspace + tilingData.v8,
                workspace + tilingData.qScale, workspace + tilingData.kScale, params);

            AscendC::PipeBarrier<PIPE_ALL>();
        }

        AscendC::SyncAll<false>();

        if constexpr (WIDE) {
            ExecuteWideStream<T, CAUSAL>(
                q, workspace + tilingData.k8, workspace + tilingData.v8,
                workspace + tilingData.kScale, workspace + tilingData.vMxScale,
                workspace + tilingData.vMean, out, lse, params);
        } else {
            InvokeStandardKernel<DP128 ? 128U : 64U, CAUSAL>(workspace, out, lse, params, tilingData);
        }
    }

private:
    template <uint32_t DP, bool CAUSAL>
    __aicore__ static inline __attribute__((always_inline)) void InvokeStandardKernel(
        GM_ADDR workspace, GM_ADDR out, GM_ADDR lse, const sage::core::SageParams &params, const SageAttention2TilingData &tilingData)
    {
        StandardAttentionKernel<T, DP, CAUSAL> kernel;
        kernel.Init(workspace + tilingData.q8, workspace + tilingData.k8, workspace + tilingData.v8,
                    workspace + tilingData.qScale, workspace + tilingData.kScale,
                    workspace + tilingData.vMxScale, workspace + tilingData.vMean,
                    out, lse, params);
        kernel.Process();
    }
};

#endif // __CCE__

} // namespace sage::orchestration
