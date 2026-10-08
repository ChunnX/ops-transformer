/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MHC_NORM_POST_PREONLY_KERNEL_H_
#define MHC_NORM_POST_PREONLY_KERNEL_H_

#include "mhc_norm_post_preonly_common.h"
#include "mhc_norm_post_preonly_kernel_regbase.h"

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING = false>
class KernelMhcNormPostPreonly {
public:
    __aicore__ inline KernelMhcNormPostPreonly() {}

    // =====================================================================
    //  Init: 解析 tiling, 设置 GM buffer, 分配 UB
    // =====================================================================
    __aicore__ inline void Init(GM_ADDR h_out_gm, GM_ADDR residual_gm, GM_ADDR h_post_gm, GM_ADDR h_res_gm,
                                GM_ADDR phi_gm, GM_ADDR alpha_gm, GM_ADDR bias_gm, GM_ADDR pre_mix_gm,
                                GM_ADDR gamma_0_gm, GM_ADDR gamma_1_gm, GM_ADDR gamma_2_gm, GM_ADDR h_in_prime_gm,
                                GM_ADDR x_2_out_gm, GM_ADDR h_in_f32_gm, GM_ADDR h_pre_gm, GM_ADDR workspace,
                                GM_ADDR tiling);

    // =====================================================================
    //  Process: gamma hoisting + token loop dispatch
    // =====================================================================
    __aicore__ inline void Process();

    // =================================================================
    //  各执行路径的方法，实现在独立头文件中
    // =================================================================
private:
    __aicore__ inline void ProcessTokenSingleAiv(uint32_t token);
    __aicore__ inline void ProcessTokenDualSiblingAiv(uint32_t token);
    __aicore__ inline void ComputeX2PostTile(uint32_t token, uint32_t tOff, uint32_t tD, float invRms0,
                                             const float (&postVals)[N_MAX]);
    __aicore__ inline void ComputeX2PostTileForLane(uint32_t token, uint32_t tOff, uint32_t tD, float invRms0,
                                                    const float (&postVals)[N_MAX]);
    __aicore__ inline void CacheX2Tile(uint32_t token, uint32_t tOff, uint32_t tD, bool writeOutput,
                                       bool quantizeInput = true);
    __aicore__ inline void CacheX2TileForLane(uint32_t token, uint32_t tOff, uint32_t tD, bool writeOutput,
                                              bool quantizeInput = true);
    __aicore__ inline float CacheHInAndReduceSumSq(uint32_t tOff, uint32_t tD, LocalTensor<float> hIn);
    __aicore__ inline void LoadHInCacheTile(uint32_t tOff, uint32_t tD, LocalTensor<float> hIn);
    __aicore__ inline void AccumulateHalfPartials(uint32_t tOff, uint32_t tD, float (&sumSqHalf)[2],
                                                  float (&dotHalf)[2][N_MAX], uint32_t halfBegin = 0,
                                                  uint32_t halfEnd = 2);

    // =================================================================
    //  共享计算模块
    // =================================================================


// ========================================================================
// mhc_norm_post_preonly_kernel_compute.h
// 共享计算模块：数据加载、RMSNorm、MHC Post、门控
// ========================================================================

// 数据加载

/// 加载一个token的h_post[N]和h_res[N*N]。
__aicore__ inline void LoadTokenWeights(uint32_t token, float (&postVals)[N_MAX], uint32_t headBegin = 0U,
                                        uint32_t headEnd = N_MAX);

/// 把一个h_out分块预取到输入队列；双槽允许下一分块MTE2与当前分块Vector重叠。
__aicore__ inline void PrefetchHOutTile(uint64_t gmOffset, uint32_t tD);

/// 按队列顺序取出已经预取的h_out分块并转换为FP32。
__aicore__ inline void ConsumeHOutTile(uint32_t tD);

/// 非流水调用点仍使用成对的预取与消费。
__aicore__ inline void LoadHOutAndCast(uint64_t gmOffset, uint32_t tD);

/// 提前搬运完整h_out；后续权重加载的全流水等待同时保证该数据对Vector可见。
__aicore__ inline void CacheHOut(uint64_t gmOffset);

/// 每个token一次搬运完整gamma0，与h_out和权重共用初始全流水等待。
__aicore__ inline void CacheGamma0();

/// R6归约完成后发起完整gamma1搬运，与invRms1标量计算重叠。
__aicore__ inline void StartCacheGamma1();

/// R7读取完整gamma1前回收MTE2到Vector事件。
__aicore__ inline void WaitCacheGamma1();

/// 第二遍R1/R2直接从x2缓存恢复h_out，避免再次从GM读取。
__aicore__ inline void LoadCachedHOut(uint32_t tOff, uint32_t tD);

/// 将gamma的[tOff, tOff+tD)分块加载到指定缓存。
__aicore__ inline void LoadGammaInto(LocalTensor<float> dst, GlobalTensor<float> &src, uint32_t tOff, uint32_t tD);

// RMSNorm基础计算

/// 使用二叉树累加模式执行高精度ReduceSum。
__aicore__ inline float ReduceSumPrecise(LocalTensor<float> src, uint32_t tD);

/// 使用ReduceSumPrecise计算平方和。
__aicore__ inline float ReduceSumSq(LocalTensor<float> src, uint32_t tD);

/// R1直接从低精度h_out缓存计算平方和，省去先恢复完整FP32 h_out再平方的中间UB写读。
__aicore__ inline float ReduceSumSqFromCache(uint32_t tOff, uint32_t tD);

/// 使用标量精度计算1/sqrt(sumSq * invCount + eps)。
__aicore__ inline __attribute__((always_inline)) float ComputeInvRms(float sumSq, float invCount);

/// 原地依次执行标量缩放和gamma缩放。
__aicore__ inline void ApplyScaleAndGamma(LocalTensor<float> buf, float scale, LocalTensor<float> gamma, uint32_t tD);

// MHC_Post计算

/// 批量加载低精度残差，在R2的VF内完成转换并计算全部N个head的x2。
__aicore__ inline void ComputeX2AllHeadsBatched(const float (&postVals)[N_MAX], uint64_t gmBaseOffset, uint32_t tOff,
                                                uint32_t tD);

// 门控计算

/// 由xHatVals计算gate[0..N-1]，其中包含invRmsPre缩放：
///   gate[i] = sigmoid(alpha0 * xHat[i] + bias[i]) + hcEps
__aicore__ inline void ComputeGateValues(float totalSumSqPre, float (&xHatVals)[N_MAX]);



// A5单AIV与双sibling AIV路径共用的输入输出辅助函数。

__aicore__ inline void QuantizeRoundTrip(LocalTensor<float> buf, uint32_t count);

__aicore__ inline void CastToIO(LocalTensor<T> dst, LocalTensor<float> src, uint32_t count);

__aicore__ inline void FenceMte3ToMte2();

__aicore__ inline void FenceMte3ToV();

__aicore__ inline void FenceVToMte2();

__aicore__ inline void LoadLowPrecisionTile(GlobalTensor<T> &src, uint64_t gmOffset, LocalTensor<float> dst,
                                            uint32_t count);

__aicore__ inline void WriteLowPrecisionTile(GlobalTensor<T> &dst, uint64_t gmOffset, LocalTensor<float> src,
                                             uint32_t count);

__aicore__ inline void WriteHInPrimeTile(LocalTensor<float> src, uint64_t gmOffset, uint32_t count);

__aicore__ inline void WriteX2OutTile(LocalTensor<float> src, uint64_t gmOffset, uint32_t count);

__aicore__ inline uint32_t GetTileD(uint32_t tile) const;


    // =================================================================
    //  成员变量
    // =================================================================
    TPipe pipe_;

    // UB缓存
    TQue<TPosition::VECIN, BUFFER_NUM_ONE> bufHOutBf16_;
    TBuf<TPosition::VECCALC> bufGamma0Fp32_;
    TBuf<TPosition::VECCALC> bufGamma2Fp32_;
    TBuf<TPosition::VECCALC> bufX2Fp32_;
    TBuf<TPosition::VECCALC> bufX2Cache_;
    TQue<TPosition::VECIN, 1> bufResidualBf16_;
    TBuf<TPosition::VECCALC> bufResFp32_;
    TBuf<TPosition::VECCALC> bufPostComb_;
    TQue<TPosition::VECIN, BUFFER_NUM_TWO> bufPhiFp32_;
    TQue<TPosition::VECOUT, BUFFER_NUM_ONE> bufHInPrimeBf16_;
    TBuf<TPosition::VECCALC> bufScalar_;

    // GM张量
    GlobalTensor<T> hOutGm_, residualGm_, hInPrimeGm_, x2OutGm_;
    GlobalTensor<float> hPostGm_, hResGm_, phiGm_;
    GlobalTensor<float> alpha0Gm_, biasGm_;
    GlobalTensor<float> gamma0Gm_, gamma1Gm_, gamma2Gm_;
    GlobalTensor<float> hInF32Gm_;
    GlobalTensor<float> partial1Gm_;

    // Tiling参数
    uint32_t totalTokens_ = 0, N_ = 0, D_ = 0, K_ = 0;
    uint32_t usedCoreNum_ = 0, rowsPerCore_ = 0, rowsPerCoreSp_ = 0, blockPivot_ = 0;
    uint32_t dimTile_ = 0, dimLoop_ = 0, lastDimTile_ = 0;
    uint32_t phiRows_ = 0, phiCols_ = 0;
    float normEps_ = 1e-6f, mhcPreNormEps_ = 1e-6f, hcEps_ = 1e-6f, invD_ = 0.0f, invK_ = 0.0f;
    uint32_t tokenStart_ = 0, tokenEnd_ = 0;
    float alpha0_ = 0.0f;
    float biasCache_[N_MAX] = {0.0f};
    // A5执行方案选择值，第一阶段只使用1
    uint32_t coresPerToken_ = 1;
    uint32_t laneId_ = 0;

    int32_t evtMte2toV_ = 0;
    int32_t evtVtoMte2_ = 0;
    int32_t evtMte3toV_ = 0;
    int32_t evtVtoMte3_ = 0;
    int32_t evtMte3toMte2_ = 0;
    int32_t evtMte2toS_ = 0;
    int32_t evtStoMte3_ = 0;

    __aicore__ inline void ProcessTokenSingleHasMid(uint32_t token);

    __aicore__ inline void ProcessTokenSingleNoMid(uint32_t token);

    __aicore__ inline void ProcessTokenDualHasMid(uint32_t token);

    __aicore__ inline void ProcessTokenDualNoMid(uint32_t token);

};

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void
KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::AccumulateHalfPartials(
    uint32_t tOff, uint32_t tD, float (&sumSqHalf)[2], float (&dotHalf)[2][N_MAX], uint32_t halfBegin, uint32_t halfEnd)
{
    LocalTensor<float> x2 = bufX2Fp32_.Get<float>();
    LocalTensor<float> reduceRows = bufResFp32_.Get<float>();
    LocalTensor<float> tmp = bufGamma2Fp32_.Get<float>();

    DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
    copyParams.blockLen = tD * sizeof(float);
    DataCopyPadExtParams<float> copyPad = {false, 0, 0, 0};

    // 尾块的物理行跨度仍为dimTile_，不能作为紧凑二维张量归约；
    // 尾块保持逐行归约，完整分块才使用五行批量归约。
    if (tD != dimTile_) {
        LocalTensor<float> out = reduceRows[N_MAX * dimTile_];
        for (uint32_t half = halfBegin; half < halfEnd; ++half) {
            const uint32_t h0 = half * 2;
            const uint32_t h1 = h0 + 1;
            const uint32_t x2h0 = DUAL_SIBLING ? 0U : h0;
            const uint32_t x2h1 = DUAL_SIBLING ? 1U : h1;

            MhcNormRegBase::Mul(out, x2[x2h0 * dimTile_], x2[x2h0 * dimTile_], tD);
            PipeBarrier<PIPE_V>();
            MhcNormRegBase::Mul(tmp, x2[x2h1 * dimTile_], x2[x2h1 * dimTile_], tD);
            PipeBarrier<PIPE_V>();
            MhcNormRegBase::Add(out, out, tmp, tD);
            PipeBarrier<PIPE_V>();
            sumSqHalf[half] += ReduceSumPrecise(out, tD);

            for (uint32_t o = 0; o < N_MAX; ++o) {
                LocalTensor<float> phi = bufPhiFp32_.AllocTensor<float>();
                uint64_t phiOffset = (uint64_t)o * phiCols_ + (uint64_t)h0 * D_ + tOff;
                DataCopyPad(phi, phiGm_[phiOffset], copyParams, copyPad);
                bufPhiFp32_.EnQue(phi);
                phi = bufPhiFp32_.DeQue<float>();
                MhcNormRegBase::Mul(out, x2[x2h0 * dimTile_], phi, tD);
                PipeBarrier<PIPE_V>();
                bufPhiFp32_.FreeTensor(phi);

                phi = bufPhiFp32_.AllocTensor<float>();
                phiOffset = (uint64_t)o * phiCols_ + (uint64_t)h1 * D_ + tOff;
                DataCopyPad(phi, phiGm_[phiOffset], copyParams, copyPad);
                bufPhiFp32_.EnQue(phi);
                phi = bufPhiFp32_.DeQue<float>();
                MhcNormRegBase::Mul(tmp, x2[x2h1 * dimTile_], phi, tD);
                PipeBarrier<PIPE_V>();
                bufPhiFp32_.FreeTensor(phi);

                MhcNormRegBase::Add(out, out, tmp, tD);
                PipeBarrier<PIPE_V>();
                dotHalf[half][o] += ReduceSumPrecise(out, tD);
            }
        }
        return;
    }

    for (uint32_t half = halfBegin; half < halfEnd; ++half) {
        const uint32_t h0 = half * 2;
        const uint32_t h1 = h0 + 1;
        const uint32_t x2h0 = DUAL_SIBLING ? 0U : h0;
        const uint32_t x2h1 = DUAL_SIBLING ? 1U : h1;

        // 五行依次保存{sumSqPre,dot0,dot1,dot2,dot3}，一次按行归约，
        // 避免每个标量结果都单独触发一次Vector到Scalar的等待。
        MhcNormRegBase::Mul(reduceRows, x2[x2h0 * dimTile_], x2[x2h0 * dimTile_], tD);
        PipeBarrier<PIPE_V>();
        MhcNormRegBase::Mul(tmp, x2[x2h1 * dimTile_], x2[x2h1 * dimTile_], tD);
        PipeBarrier<PIPE_V>();
        MhcNormRegBase::Add(reduceRows, reduceRows, tmp, tD);

        for (uint32_t o = 0; o < N_MAX; ++o) {
            LocalTensor<float> dotRow = reduceRows[(o + 1) * dimTile_];
            LocalTensor<float> phi = bufPhiFp32_.AllocTensor<float>();
            uint64_t phiOffset = (uint64_t)o * phiCols_ + (uint64_t)h0 * D_ + tOff;
            DataCopyPad(phi, phiGm_[phiOffset], copyParams, copyPad);
            bufPhiFp32_.EnQue(phi);
            phi = bufPhiFp32_.DeQue<float>();
            MhcNormRegBase::Mul(dotRow, x2[x2h0 * dimTile_], phi, tD);
            PipeBarrier<PIPE_V>();
            bufPhiFp32_.FreeTensor(phi);

            phi = bufPhiFp32_.AllocTensor<float>();
            phiOffset = (uint64_t)o * phiCols_ + (uint64_t)h1 * D_ + tOff;
            DataCopyPad(phi, phiGm_[phiOffset], copyParams, copyPad);
            bufPhiFp32_.EnQue(phi);
            phi = bufPhiFp32_.DeQue<float>();
            MhcNormRegBase::Mul(tmp, x2[x2h1 * dimTile_], phi, tD);
            PipeBarrier<PIPE_V>();
            bufPhiFp32_.FreeTensor(phi);

            MhcNormRegBase::Add(dotRow, dotRow, tmp, tD);
        }

        PipeBarrier<PIPE_V>();
        LocalTensor<float> scBuf = bufScalar_.Get<float>();
        uint32_t reduceShape[2] = {N_MAX + 1, tD};
        // 五行输入在归约后不再使用，允许接口复用源缓冲，避免额外申请UB。
        ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(scBuf, reduceRows, reduceShape, true);
        PipeBarrier<PIPE_V>();
        sumSqHalf[half] += scBuf.GetValue(0);
        for (uint32_t o = 0; o < N_MAX; ++o) {
            dotHalf[half][o] += scBuf.GetValue(o + 1);
        }
    }
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::ApplyScaleAndGamma(LocalTensor<float> buf, float scale, LocalTensor<float> gamma, uint32_t tD)
{
    // TODO：完成多VF融合和双缓冲后，执行Membase回退消融，复测改回高层接口的组合收益。
    MhcNormRegBase::ScaleAndMulGamma(buf, buf, gamma, scale, tD);
    PipeBarrier<PIPE_V>();
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::CacheGamma0()
{
    LocalTensor<float> gamma0 = bufGamma0Fp32_.Get<float>();
    DataCopyExtParams params = {1, 0, 0, 0, 0};
    params.blockLen = D_ * sizeof(float);
    DataCopyPadExtParams<float> pad = {false, 0, 0, 0};
    DataCopyPad(gamma0, gamma0Gm_, params, pad);
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline float KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32,
                                                         DUAL_SIBLING>::CacheHInAndReduceSumSq(uint32_t tOff,
                                                                                               uint32_t tD,
                                                                                               LocalTensor<float> hIn)
{
    LocalTensor<T> hInCache = bufX2Cache_.Get<T>()[tOff];
    CastToIO(hInCache, hIn, tD);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> sqBuf = bufGamma2Fp32_.Get<float>();
    MhcNormRegBase::CastB16ToFp32Square(sqBuf, hInCache, tD);
    PipeBarrier<PIPE_V>();
    return ReduceSumPrecise(sqBuf, tD);
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::CacheHOut(uint64_t gmOffset)
{
    LocalTensor<T> hOutCache = bufX2Cache_.Get<T>();
    DataCopy(hOutCache, hOutGm_[gmOffset], D_);
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::CacheX2Tile(
    uint32_t token, uint32_t tOff, uint32_t tD, bool writeOutput, bool quantizeInput)
{
    LocalTensor<float> x2 = bufX2Fp32_.Get<float>();
    LocalTensor<T> x2Cache = bufX2Cache_.Get<T>();
    const uint64_t outBase = (uint64_t)token * N_ * D_ + tOff;

    // R2已在成对VF尾部完成首次量化；R3后的第二次量化仍使用该路径。
    if (quantizeInput) {
        for (uint32_t n = 0; n < N_MAX; n += 2U) {
            MhcNormRegBase::CastFp32PairToB16(x2Cache[n * D_ + tOff], x2Cache[(n + 1U) * D_ + tOff],
                                                  x2[n * dimTile_], x2[(n + 1U) * dimTile_], tD);
        }
        PipeBarrier<PIPE_V>();
    }
    // RMS_mid第一遍直接从低精度缓存计算平方和；R3后必须恢复，
    // 保证后续R4/R5仍使用低精度往返后的x2。
    if (!USE_RMS_MID || writeOutput) {
        for (uint32_t n = 0; n < N_MAX; n += 2U) {
            MhcNormRegBase::CastB16PairToFp32(x2[n * dimTile_], x2[(n + 1U) * dimTile_],
                                                  x2Cache[n * D_ + tOff], x2Cache[(n + 1U) * D_ + tOff], tD);
        }
        PipeBarrier<PIPE_V>();
    }

    if (writeOutput) {
        // cacheTile已经是x2_out所需的最终低精度位值，直接写GM，
        // 避免先恢复FP32再重复量化和反复申请输出队列。
        SetFlag<HardEvent::V_MTE3>(evtVtoMte3_);
        WaitFlag<HardEvent::V_MTE3>(evtVtoMte3_);
        for (uint32_t n = 0; n < N_MAX; ++n) {
            DataCopy(x2OutGm_[outBase + (uint64_t)n * D_], x2Cache[n * D_ + tOff], tD);
        }
    }
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32,
                                                        DUAL_SIBLING>::CacheX2TileForLane(uint32_t token, uint32_t tOff,
                                                                                          uint32_t tD, bool writeOutput,
                                                                                          bool quantizeInput)
{
    LocalTensor<float> x2 = bufX2Fp32_.Get<float>();
    LocalTensor<T> x2Cache = bufX2Cache_.Get<T>();
    const uint64_t outBase = (uint64_t)token * N_ * D_ + tOff;
    const uint32_t headBegin = laneId_ * 2U;

    if (quantizeInput) {
        for (uint32_t n = 0; n < N_MAX; n += 2U) {
            MhcNormRegBase::CastFp32PairToB16(x2Cache[n * D_ + tOff], x2Cache[(n + 1U) * D_ + tOff],
                                                  x2[n * dimTile_], x2[(n + 1U) * dimTile_], tD);
        }
        PipeBarrier<PIPE_V>();
    }
    if (!USE_RMS_MID || writeOutput) {
        // 双路R4/R5只读取本lane负责的相邻两行，避免恢复另一lane的无效副本。
        MhcNormRegBase::CastB16PairToFp32(x2, x2[dimTile_],
                                              x2Cache[headBegin * D_ + tOff],
                                              x2Cache[(headBegin + 1U) * D_ + tOff], tD);
        PipeBarrier<PIPE_V>();
    }

    if (writeOutput) {
        SetFlag<HardEvent::V_MTE3>(evtVtoMte3_);
        WaitFlag<HardEvent::V_MTE3>(evtVtoMte3_);
        for (uint32_t n = headBegin; n < headBegin + 2U; ++n) {
            DataCopy(x2OutGm_[outBase + (uint64_t)n * D_], x2Cache[n * D_ + tOff], tD);
        }
    }
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::CastToIO(LocalTensor<T> dst, LocalTensor<float> src, uint32_t count)
{
    MhcNormRegBase::CastFp32ToB16(dst, src, count);
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::ComputeGateValues(float totalSumSqPre, float (&xHatVals)[N_MAX])
{
    // Host已固定校验N为4，使用编译期常量便于展开标量循环。
    constexpr uint32_t N = N_MAX;
    float invRmsPre = 1.0f / sqrt(totalSumSqPre * invK_ + mhcPreNormEps_);
    for (uint32_t o = 0; o < N; ++o)
        xHatVals[o] *= invRmsPre;

    LocalTensor<float> scBuf = bufScalar_.Get<float>();
    for (uint32_t i = 0; i < N; ++i)
        scBuf.SetValue(i, -(alpha0_ * xHatVals[i] + biasCache_[i]));
    LocalTensor<float> gateBuf = scBuf[EXP_DIV_ELEM];
    // TODO：后续单独评估固定四元素循环展开，本轮保持通用VF循环。
    MhcNormRegBase::SigmoidAddEps(gateBuf, scBuf, hcEps_, N);
    PipeBarrier<PIPE_V>();
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline __attribute__((always_inline)) float KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::ComputeInvRms(float sumSq, float invCount)
{
    return 1.0f / sqrt(sumSq * invCount + normEps_);
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::ComputeX2AllHeadsBatched(const float (&postVals)[N_MAX], uint64_t gmBaseOffset, uint32_t tOff,
                                                uint32_t tD)
{
    // Host已固定校验N为4，使用编译期常量便于展开标量循环。
    constexpr uint32_t N = N_MAX;
    LocalTensor<T> residual = bufResidualBf16_.AllocTensor<T>();
    for (uint32_t n = 0; n < N; ++n)
        DataCopy(residual[n * dimTile_], residualGm_[gmBaseOffset + (uint64_t)n * D_], tD);
    bufResidualBf16_.EnQue(residual);
    residual = bufResidualBf16_.DeQue<T>();
    LocalTensor<T> x2Cache = bufX2Cache_.Get<T>();
    LocalTensor<float> hOut = bufResFp32_.Get<float>()[N_MAX * dimTile_];
    LocalTensor<float> combWeights = bufPostComb_.Get<float>()[ALIGN8];

    for (uint32_t h = 0; h < N; h += 2U) {
        MhcNormRegBase::ComputePostHeadPair<T>(x2Cache[h * D_ + tOff], x2Cache[(h + 1U) * D_ + tOff], hOut,
                                                   residual, combWeights, dimTile_, h, postVals[h], postVals[h + 1U],
                                                   tD);
    }
    PipeBarrier<PIPE_V>();
    bufResidualBf16_.FreeTensor(residual);
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32,
                                                        DUAL_SIBLING>::ComputeX2PostTile(uint32_t token, uint32_t tOff,
                                                                                         uint32_t tD, float invRms0,
                                                                                         const float (&postVals)[N_MAX])
{
    LoadCachedHOut(tOff, tD);
    LocalTensor<float> hOut = bufResFp32_.Get<float>()[N_MAX * dimTile_];
    ApplyScaleAndGamma(hOut, invRms0, bufGamma0Fp32_.Get<float>()[tOff], tD);
    QuantizeRoundTrip(hOut, tD);

    ComputeX2AllHeadsBatched(postVals, (uint64_t)token * N_ * D_ + tOff, tOff, tD);
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32,
                                                        DUAL_SIBLING>::ComputeX2PostTileForLane(
    uint32_t token, uint32_t tOff, uint32_t tD, float invRms0, const float (&postVals)[N_MAX])
{
    LoadCachedHOut(tOff, tD);
    LocalTensor<float> hOut = bufResFp32_.Get<float>()[N_MAX * dimTile_];
    ApplyScaleAndGamma(hOut, invRms0, bufGamma0Fp32_.Get<float>()[tOff], tD);
    QuantizeRoundTrip(hOut, tD);

    const uint32_t headBegin = laneId_ * 2U;
    const uint64_t gmBaseOffset = (uint64_t)token * N_ * D_ + tOff;
    LocalTensor<T> residual = bufResidualBf16_.AllocTensor<T>();
    for (uint32_t n = 0; n < N_MAX; ++n) {
        DataCopy(residual[n * dimTile_], residualGm_[gmBaseOffset + (uint64_t)n * D_], tD);
    }
    bufResidualBf16_.EnQue(residual);
    residual = bufResidualBf16_.DeQue<T>();
    LocalTensor<T> x2Cache = bufX2Cache_.Get<T>();
    LocalTensor<float> combWeights = bufPostComb_.Get<float>()[ALIGN8];
    MhcNormRegBase::ComputePostHeadPair<T>(
        x2Cache[headBegin * D_ + tOff], x2Cache[(headBegin + 1U) * D_ + tOff], hOut, residual, combWeights,
        dimTile_, headBegin, postVals[headBegin], postVals[headBegin + 1U], tD);
    PipeBarrier<PIPE_V>();
    bufResidualBf16_.FreeTensor(residual);
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::ConsumeHOutTile(uint32_t tD)
{
    LocalTensor<T> hBf16 = bufHOutBf16_.DeQue<T>();
    LocalTensor<float> hOut = bufResFp32_.Get<float>()[N_MAX * dimTile_];
    MhcNormRegBase::CastB16ToFp32(hOut, hBf16, tD);
    PipeBarrier<PIPE_V>();
    bufHOutBf16_.FreeTensor(hBf16);
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::FenceMte3ToMte2()
{
    SetFlag<HardEvent::MTE3_MTE2>(evtMte3toMte2_);
    WaitFlag<HardEvent::MTE3_MTE2>(evtMte3toMte2_);
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::FenceMte3ToV()
{
    SetFlag<HardEvent::MTE3_V>(evtMte3toV_);
    WaitFlag<HardEvent::MTE3_V>(evtMte3toV_);
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::FenceVToMte2()
{
    SetFlag<HardEvent::V_MTE2>(evtVtoMte2_);
    WaitFlag<HardEvent::V_MTE2>(evtVtoMte2_);
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline uint32_t KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::GetTileD(uint32_t tile) const
{
    if (lastDimTile_ != 0 && tile == dimLoop_ - 1) {
        return lastDimTile_;
    }
    return dimTile_;
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::Init(GM_ADDR h_out_gm, GM_ADDR residual_gm, GM_ADDR h_post_gm, GM_ADDR h_res_gm,
                                GM_ADDR phi_gm, GM_ADDR alpha_gm, GM_ADDR bias_gm, GM_ADDR pre_mix_gm,
                                GM_ADDR gamma_0_gm, GM_ADDR gamma_1_gm, GM_ADDR gamma_2_gm, GM_ADDR h_in_prime_gm,
                                GM_ADDR x_2_out_gm, GM_ADDR h_in_f32_gm, GM_ADDR h_pre_gm, GM_ADDR workspace,
                                GM_ADDR tiling)
{
        GET_TILING_DATA(td, tiling);

        // pre_mix 为接口预留的可选输入，h_pre 为接口预留的可选输出，当前版本均不参与计算。
        (void)pre_mix_gm;
        (void)h_pre_gm;

        totalTokens_ = td.totalTokens;
        N_ = td.headCount;
        D_ = td.embedDim;
        usedCoreNum_ = td.usedCoreNum;
        rowsPerCore_ = td.rowsPerCore;
        rowsPerCoreSp_ = td.rowsPerCoreSp;
        blockPivot_ = td.blockPivot;
        dimTile_ = td.dimTile;
        dimLoop_ = td.dimLoop;
        lastDimTile_ = td.lastDimTile;
        phiRows_ = td.phiRows;
        phiCols_ = td.phiCols;
        normEps_ = td.normEps;
        mhcPreNormEps_ = td.mhcPreNormEps;
        hcEps_ = td.hcEps;
        K_ = N_ * D_;
        invD_ = td.invD;
        invK_ = td.invK;
        coresPerToken_ = td.coresPerToken;

        const uint32_t aivIdx = GetBlockIdx();
        // MIX_AIC_1_2在A5上按每个物理AI Core展开两个连续AIV。
        // 该映射与同平台算子保持一致，只在双sibling模板中使用。
        laneId_ = DUAL_SIBLING ? (aivIdx & 1U) : 0U;
        const uint32_t blockIdx = DUAL_SIBLING ? (aivIdx >> 1U) : aivIdx;
        if (blockIdx >= usedCoreNum_) {
            tokenStart_ = 0;
            tokenEnd_ = 0;
            return;
        }

        // 第一阶段按token分区启动相互独立的AIV，不使用block奇偶配对、相邻block配对或N轴切分。
        if (blockIdx < blockPivot_) {
            tokenStart_ = blockIdx * rowsPerCoreSp_;
            tokenEnd_ = tokenStart_ + rowsPerCoreSp_;
        } else {
            tokenStart_ = blockPivot_ * rowsPerCoreSp_ + (blockIdx - blockPivot_) * rowsPerCore_;
            tokenEnd_ = tokenStart_ + rowsPerCore_;
        }
        if (tokenEnd_ > totalTokens_)
            tokenEnd_ = totalTokens_;

        // GM张量
        hOutGm_.SetGlobalBuffer((__gm__ T *)h_out_gm, (uint64_t)totalTokens_ * D_);
        residualGm_.SetGlobalBuffer((__gm__ T *)residual_gm, (uint64_t)totalTokens_ * N_ * D_);
        hPostGm_.SetGlobalBuffer((__gm__ float *)h_post_gm, (uint64_t)totalTokens_ * N_);
        hResGm_.SetGlobalBuffer((__gm__ float *)h_res_gm, (uint64_t)totalTokens_ * N_ * N_);
        phiGm_.SetGlobalBuffer((__gm__ float *)phi_gm, (uint64_t)phiRows_ * phiCols_);
        alpha0Gm_.SetGlobalBuffer((__gm__ float *)alpha_gm, ALIGN8);
        biasGm_.SetGlobalBuffer((__gm__ float *)bias_gm, ALIGN8);
        gamma0Gm_.SetGlobalBuffer((__gm__ float *)gamma_0_gm, D_);
        gamma1Gm_.SetGlobalBuffer((__gm__ float *)gamma_1_gm, D_);
        if constexpr (USE_RMS_MID) {
            gamma2Gm_.SetGlobalBuffer((__gm__ float *)gamma_2_gm, K_);
        }
        hInPrimeGm_.SetGlobalBuffer((__gm__ T *)h_in_prime_gm, (uint64_t)totalTokens_ * D_);
        x2OutGm_.SetGlobalBuffer((__gm__ T *)x_2_out_gm, (uint64_t)totalTokens_ * N_ * D_);
        if (h_in_f32_gm != nullptr) {
            hInF32Gm_.SetGlobalBuffer((__gm__ float *)h_in_f32_gm, (uint64_t)totalTokens_ * D_);
        }
        if constexpr (DUAL_SIBLING) {
            partial1Gm_.SetGlobalBuffer((__gm__ float *)workspace, (uint64_t)totalTokens_ * 2U * PARTIAL_SLOT_ELEMS);
        }

        // UB空间分配
        pipe_.InitBuffer(bufHOutBf16_, BUFFER_NUM_ONE, dimTile_ * sizeof(T));
        pipe_.InitBuffer(bufGamma0Fp32_, D_ * sizeof(float));
        // 双sibling每条lane只恢复并归约自己负责的两个head，使用紧凑两行缓存。
        const uint32_t x2Fp32Rows = DUAL_SIBLING ? 2U : N_;
        pipe_.InitBuffer(bufX2Fp32_, x2Fp32Rows * dimTile_ * sizeof(float));
        // 完整gamma0复用初始同步；gamma1在D不超过5120时完整缓存。
        // D=7168时总UB为260448B，小于A5的256KiB上限并保留1696B余量。
        pipe_.InitBuffer(bufX2Cache_, K_ * sizeof(T));
        pipe_.InitBuffer(bufResidualBf16_, 1, N_ * dimTile_ * sizeof(T));
        // 前四行先保存residual，之后复用为R4/R5的五行归约输入；第五行在前半程保存h_out，
        // h_out失效后再复用为R6/R7的h_in和通用输出暂存。
        pipe_.InitBuffer(bufResFp32_, (N_ + 1) * dimTile_ * sizeof(float));
        pipe_.InitBuffer(bufPostComb_, POST_COMB_SIZE * sizeof(float));
        pipe_.InitBuffer(bufPhiFp32_, BUFFER_NUM_TWO, dimTile_ * sizeof(float));
        // TODO: 后续重新设计流水并单独验证输入、输出双缓冲收益。
        pipe_.InitBuffer(bufHInPrimeBf16_, BUFFER_NUM_ONE, dimTile_ * sizeof(T));
        pipe_.InitBuffer(bufScalar_, SCALAR_BUF_SIZE);
        const uint32_t gamma1CacheElems = D_ <= GAMMA1_CACHE_MAX_D ? D_ : dimTile_;
        pipe_.InitBuffer(bufGamma2Fp32_, gamma1CacheElems * sizeof(float));

        evtMte2toV_ = static_cast<int32_t>(pipe_.AllocEventID<HardEvent::MTE2_V>());
        evtVtoMte2_ = static_cast<int32_t>(pipe_.AllocEventID<HardEvent::V_MTE2>());
        evtMte3toV_ = static_cast<int32_t>(pipe_.AllocEventID<HardEvent::MTE3_V>());
        evtVtoMte3_ = static_cast<int32_t>(pipe_.AllocEventID<HardEvent::V_MTE3>());
        evtMte3toMte2_ = static_cast<int32_t>(pipe_.AllocEventID<HardEvent::MTE3_MTE2>());
        if constexpr (DUAL_SIBLING) {
            evtMte2toS_ = static_cast<int32_t>(pipe_.AllocEventID<HardEvent::MTE2_S>());
            evtStoMte3_ = static_cast<int32_t>(pipe_.AllocEventID<HardEvent::S_MTE3>());
        }
    }

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::LoadCachedHOut(uint32_t tOff, uint32_t tD)
{
    LocalTensor<T> hOutCache = bufX2Cache_.Get<T>()[tOff];
    LocalTensor<float> hOut = bufResFp32_.Get<float>()[N_MAX * dimTile_];
    MhcNormRegBase::CastB16ToFp32(hOut, hOutCache, tD);
    PipeBarrier<PIPE_V>();
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::LoadGammaInto(LocalTensor<float> dst, GlobalTensor<float> &src, uint32_t tOff, uint32_t tD)
{
    DataCopyExtParams gP = {1, 0, 0, 0, 0};
    gP.blockLen = tD * sizeof(float);
    DataCopyPadExtParams<float> gPad = {false, 0, 0, 0};
    DataCopyPad(dst, src[tOff], gP, gPad);
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32,
                                                        DUAL_SIBLING>::LoadHInCacheTile(uint32_t tOff, uint32_t tD,
                                                                                        LocalTensor<float> hIn)
{
    LocalTensor<T> hInCache = bufX2Cache_.Get<T>()[tOff];
    MhcNormRegBase::CastB16ToFp32(hIn, hInCache, tD);
    PipeBarrier<PIPE_V>();
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::LoadHOutAndCast(uint64_t gmOffset, uint32_t tD)
{
    PrefetchHOutTile(gmOffset, tD);
    ConsumeHOutTile(tD);
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::LoadLowPrecisionTile(GlobalTensor<T> &src, uint64_t gmOffset, LocalTensor<float> dst,
                                            uint32_t count)
{
    LocalTensor<T> tmp = bufHOutBf16_.AllocTensor<T>();
    DataCopy(tmp, src[gmOffset], count);
    bufHOutBf16_.EnQue(tmp);
    tmp = bufHOutBf16_.DeQue<T>();
    MhcNormRegBase::CastB16ToFp32(dst, tmp, count);
    PipeBarrier<PIPE_V>();
    bufHOutBf16_.FreeTensor(tmp);
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::LoadTokenWeights(uint32_t token, float (&postVals)[N_MAX], uint32_t headBegin,
                                        uint32_t headEnd)
{
    // Host已固定校验N为4，使用编译期常量便于展开标量循环。
    constexpr uint32_t N = N_MAX;
    LocalTensor<float> postCombUb = bufPostComb_.Get<float>();
    LocalTensor<float> hPostUb = postCombUb;
    LocalTensor<float> hResUb = postCombUb[ALIGN8]; // h_post[N] aligned to 8 elements
    {
        DataCopyExtParams pp = {1, 0, 0, 0, 0};
        pp.blockLen = N * sizeof(float);
        DataCopyPadExtParams<float> pad = {true, 0, 0, 0};
        DataCopyPad(hPostUb, hPostGm_[(uint64_t)token * N], pp, pad);

        DataCopyExtParams cp = {1, 0, 0, 0, 0};
        cp.blockLen = N * N * sizeof(float);
        DataCopyPadExtParams<float> cpad = {true, 0, 0, 0};
        DataCopyPad(hResUb, hResGm_[(uint64_t)token * N * N], cp, cpad);
    }
    PipeBarrier<PIPE_ALL>();
    // 双AIV每条lane只消费自己负责的两个post标量，避免无效UB到Scalar回读。
    for (uint32_t i = headBegin; i < headEnd; ++i)
        postVals[i] = hPostUb.GetValue(i);
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::PrefetchHOutTile(uint64_t gmOffset, uint32_t tD)
{
    LocalTensor<T> hBf16 = bufHOutBf16_.AllocTensor<T>();
    DataCopy(hBf16, hOutGm_[gmOffset], tD);
    bufHOutBf16_.EnQue(hBf16);
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::Process()
{
        if (tokenStart_ >= tokenEnd_) {
            return;
        }
        // 双AIV的follower在会合后退出，不参与gate计算，省去五次无效标量GM读取。
        if constexpr (!DUAL_SIBLING) {
            alpha0_ = alpha0Gm_.GetValue(0);
            for (uint32_t i = 0; i < N_MAX; ++i) {
                biasCache_[i] = biasGm_.GetValue(i);
            }
        } else if (laneId_ == 0U) {
            alpha0_ = alpha0Gm_.GetValue(0);
            for (uint32_t i = 0; i < N_MAX; ++i) {
                biasCache_[i] = biasGm_.GetValue(i);
            }
        }

        // 仅单分块路径提前加载gamma
        if (dimLoop_ == 1) {
            DataCopyExtParams gP = {1, 0, 0, 0, 0};
            gP.blockLen = dimTile_ * sizeof(float);
            DataCopyPadExtParams<float> gPad = {false, 0, 0, 0};
            DataCopyPad(bufGamma0Fp32_.Get<float>(), gamma0Gm_[0], gP, gPad);
            // gamma_2的形状为[N*D]，在RMSNorm_mid中按head加载
        }

        SetFlag<HardEvent::MTE2_V>(evtMte2toV_);
        WaitFlag<HardEvent::MTE2_V>(evtMte2toV_);

        for (uint32_t t = tokenStart_; t < tokenEnd_; ++t) {
            if constexpr (DUAL_SIBLING) {
                ProcessTokenDualSiblingAiv(t);
            } else {
                ProcessTokenSingleAiv(t);
            }
        }

        pipe_.ReleaseEventID<HardEvent::MTE2_V>(evtMte2toV_);
        pipe_.ReleaseEventID<HardEvent::V_MTE2>(evtVtoMte2_);
        pipe_.ReleaseEventID<HardEvent::MTE3_V>(evtMte3toV_);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(evtVtoMte3_);
        pipe_.ReleaseEventID<HardEvent::MTE3_MTE2>(evtMte3toMte2_);
        if constexpr (DUAL_SIBLING) {
            pipe_.ReleaseEventID<HardEvent::MTE2_S>(evtMte2toS_);
            pipe_.ReleaseEventID<HardEvent::S_MTE3>(evtStoMte3_);
        }
    }

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32,
                                                        DUAL_SIBLING>::ProcessTokenDualSiblingAiv(uint32_t token)
{
    if constexpr (USE_RMS_MID) {
        ProcessTokenDualHasMid(token);
    } else {
        ProcessTokenDualNoMid(token);
    }
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void
KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::ProcessTokenSingleAiv(uint32_t token)
{
    if constexpr (USE_RMS_MID) {
        ProcessTokenSingleHasMid(token);
    } else {
        ProcessTokenSingleNoMid(token);
    }
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::QuantizeRoundTrip(LocalTensor<float> buf, uint32_t count)
{
    LocalTensor<T> tmp = bufHInPrimeBf16_.AllocTensor<T>();
    CastToIO(tmp, buf, count);
    PipeBarrier<PIPE_V>();
    MhcNormRegBase::CastB16ToFp32(buf, tmp, count);
    PipeBarrier<PIPE_V>();
    bufHInPrimeBf16_.FreeTensor(tmp);
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline float KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::ReduceSumPrecise(LocalTensor<float> src, uint32_t tD)
{
    LocalTensor<float> scBuf = bufScalar_.Get<float>();
    constexpr uint32_t REDUCE_ALIGN = 16; // ReduceSum要求16个元素对齐
    uint32_t tDAligned = (tD + REDUCE_ALIGN - 1) / REDUCE_ALIGN * REDUCE_ALIGN;
    uint32_t srcShape[2] = {1, tDAligned};
    // 输入在归约后不再使用，允许归约过程复用输入缓冲，减少内部UB与标量调度开销。
    ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(scBuf, src, srcShape, true);
    PipeBarrier<PIPE_V>();
    return scBuf.GetValue(0);
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline float KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::ReduceSumSq(LocalTensor<float> src, uint32_t tD)
{
    // gamma2缓存与平方归约不同时存活，复用同一块UB作为平方临时区。
    LocalTensor<float> sqBuf = bufGamma2Fp32_.Get<float>();
    MhcNormRegBase::Mul(sqBuf, src, src, tD);
    PipeBarrier<PIPE_V>();
    return ReduceSumPrecise(sqBuf, tD);
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline float KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::ReduceSumSqFromCache(uint32_t tOff, uint32_t tD)
{
    LocalTensor<T> hOutCache = bufX2Cache_.Get<T>()[tOff];
    LocalTensor<float> sqBuf = bufGamma2Fp32_.Get<float>();
    MhcNormRegBase::CastB16ToFp32Square(sqBuf, hOutCache, tD);
    PipeBarrier<PIPE_V>();
    return ReduceSumPrecise(sqBuf, tD);
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::StartCacheGamma1()
{
    if (D_ > GAMMA1_CACHE_MAX_D) {
        return;
    }
    LocalTensor<float> gamma1 = bufGamma2Fp32_.Get<float>();
    DataCopyExtParams params = {1, 0, 0, 0, 0};
    params.blockLen = D_ * sizeof(float);
    DataCopyPadExtParams<float> pad = {false, 0, 0, 0};
    DataCopyPad(gamma1, gamma1Gm_, params, pad);
    SetFlag<HardEvent::MTE2_V>(evtMte2toV_);
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::WaitCacheGamma1()
{
    if (D_ > GAMMA1_CACHE_MAX_D) {
        return;
    }
    WaitFlag<HardEvent::MTE2_V>(evtMte2toV_);
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::WriteHInPrimeTile(LocalTensor<float> src, uint64_t gmOffset, uint32_t count)
{
    WriteLowPrecisionTile(hInPrimeGm_, gmOffset, src, count);
    if constexpr (RETURN_H_IN_F32) {
        SetFlag<HardEvent::V_MTE3>(evtVtoMte3_);
        WaitFlag<HardEvent::V_MTE3>(evtVtoMte3_);
        DataCopy(hInF32Gm_[gmOffset], src, count);
        FenceMte3ToV();
    }
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::WriteLowPrecisionTile(GlobalTensor<T> &dst, uint64_t gmOffset, LocalTensor<float> src,
                                             uint32_t count)
{
    LocalTensor<T> tmp = bufHInPrimeBf16_.AllocTensor<T>();
    CastToIO(tmp, src, count);
    bufHInPrimeBf16_.EnQue(tmp);
    tmp = bufHInPrimeBf16_.DeQue<T>();
    DataCopy(dst[gmOffset], tmp, count);
    bufHInPrimeBf16_.FreeTensor(tmp);
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::WriteX2OutTile(LocalTensor<float> src, uint64_t gmOffset, uint32_t count)
{
    WriteLowPrecisionTile(x2OutGm_, gmOffset, src, count);
}

#include "mhc_norm_post_preonly_has_mid.h"
#include "mhc_norm_post_preonly_no_mid.h"

#endif // MHC_NORM_POST_PREONLY_KERNEL_H_
