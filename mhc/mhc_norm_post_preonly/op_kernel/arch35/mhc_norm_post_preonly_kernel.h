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

#include "mhc_norm_post_preonly_kernel_support.h"

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING = false>
class KernelMhcNormPostPreonly {
public:
    __aicore__ inline KernelMhcNormPostPreonly() = default;

    __aicore__ inline void Init(GM_ADDR h_out_gm, GM_ADDR residual_gm, GM_ADDR h_post_gm, GM_ADDR h_res_gm,
                                GM_ADDR phi_gm, GM_ADDR alpha_gm, GM_ADDR bias_gm, GM_ADDR pre_mix_gm,
                                GM_ADDR gamma_0_gm, GM_ADDR gamma_1_gm, GM_ADDR gamma_2_gm, GM_ADDR h_in_prime_gm,
                                GM_ADDR x_2_out_gm, GM_ADDR h_in_f32_gm, GM_ADDR h_pre_gm, GM_ADDR workspace,
                                GM_ADDR tilingDataGm);

    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessToken(uint32_t token);
    __aicore__ inline void ProcessTokenSingleNoMid(uint32_t token);
    __aicore__ inline void ProcessTokenDualNoMid(uint32_t token);
    __aicore__ inline void ProcessTokenSingleHasMid(uint32_t token);
    __aicore__ inline void ProcessTokenDualHasMid(uint32_t token);

    // 加载一个token的h_post[N]和h_res[N*N]。
    __aicore__ inline void CopyInTokenWeights(uint32_t token, float (&postVals)[N], uint32_t nStart = 0U,
                                              uint32_t nEnd = N);

    // 提前搬运完整h_out；后续权重加载的全流水等待同时保证该数据对Vector可见。
    __aicore__ inline void CopyInHOut(uint64_t gmOffset);

    // 多分块路径每个token搬运完整gamma0；D=2560由Process在token循环外仅加载一次。
    __aicore__ inline void CopyInGamma0();

    // 提前发起完整gamma1搬运到bufGammaFp32_，使MTE2与门控融合的Vector计算重叠。
    // bufGammaFp32_在中间归一化完成后即空闲，门控融合的平方缓存bufScratchFp32_不受影响。
    __aicore__ inline void StartCopyInGamma1();

    // 输出归一化读取完整gamma1前回收MTE2到Vector事件。
    __aicore__ inline void WaitCopyInGamma1();

    __aicore__ inline void CopyInGammaTile(LocalTensor<float> dst, GlobalTensor<float> &src, uint32_t tOff, uint32_t tD);

    // 使用二叉树累加模式执行高精度ReduceSum。
    __aicore__ inline float ReduceSumPrecise(LocalTensor<float> src, uint32_t tD);

    __aicore__ inline float ReduceSumSq(LocalTensor<float> src, uint32_t tD);

    // 输入归一化直接从低精度h_out缓存计算平方和，省去先恢复完整FP32 h_out再平方的中间UB写读。
    __aicore__ inline float ReduceSumSqFromCache(uint32_t tOff, uint32_t tD);

    // 使用标量精度计算1/sqrt(sumSq * invCount + eps)。
    __aicore__ inline __attribute__((always_inline)) float ComputeInvRms(float sumSq, float invCount);

    // 批量加载低精度残差，在残差融合的VF内完成转换并计算全部N个head的x2。
    __aicore__ inline void ComputeX2AllHeadsBatched(const float (&postVals)[N], uint64_t gmBaseOffset,
                                                    uint32_t tOff, uint32_t tD);

    /// 由xHatVals计算gate[0..N-1]，其中包含invRmsPre缩放：
    ///   gate[i] = sigmoid(alpha0 * xHat[i] + bias[i]) + hcEps
    __aicore__ inline void ComputeGateValues(float totalSumSqPre, float (&xHatVals)[N]);

    __aicore__ inline void CastToIO(LocalTensor<T> dst, LocalTensor<float> src, uint32_t count);

    __aicore__ inline void FenceMte3ToV();

    __aicore__ inline void FenceVToMte2();

    __aicore__ inline void CopyOutLowPrecisionTile(GlobalTensor<T> &dst, uint64_t gmOffset, LocalTensor<float> src,
                                                    uint32_t count);

    __aicore__ inline void CopyOutHInPrimeTile(LocalTensor<float> src, uint64_t gmOffset, uint32_t count);

    // RETURN_H_IN_F32关闭时，直接在已消费完的x2低精度缓存上完成输出归一化量化，避免FP32中间区和二次转换。
    __aicore__ inline void ComputeAndCopyOutHInPrimeTile(LocalTensor<T> srcDst, LocalTensor<float> gamma, float scale,
                                                         uint64_t gmOffset, uint32_t count);

    // 四条路径共享的输出归一化末段；gamma1 已由调用方在 Gate 之前发起预取。
    __aicore__ inline void FinalizeHInPrime(uint32_t token, float sumSq1);

    __aicore__ inline uint32_t GetTileD(uint32_t tile) const;

    __aicore__ inline void ComputeX2PostTile(uint32_t token, uint32_t tOff, uint32_t tD, float invRms0,
                                             const float (&postVals)[N]);

    // 双AIV仍需读取全部残差行，但每个sibling AIV只计算自己负责的两个输出head。
    __aicore__ inline void ComputeX2PostTileForSiblingAiv(uint32_t token, uint32_t tOff, uint32_t tD, float invRms0,
                                                           const float (&postVals)[N]);

    // 从最终低精度x2恢复投影所需的FP32值，并写出低精度x2。
    __aicore__ inline void RestoreAndCopyOutX2Tile(uint32_t token, uint32_t tOff, uint32_t tD);

    __aicore__ inline void RestoreAndCopyOutX2TileForSiblingAiv(uint32_t token, uint32_t tOff, uint32_t tD);

    // 门控融合读取完当前x2分块后，复用首个head缓存保留低精度h_in。
    // 恢复FP32与平方在同一VF内完成，保留量化往返边界并减少UB中间写读。
    __aicore__ inline float ComputeHInCacheAndReduceSumSq(uint32_t tOff, uint32_t tD, LocalTensor<T> x2Cache,
                                                          LocalTensor<float> gateBuf);

    // half0/half1分别覆盖展平区间[0,2D)/[2D,4D)，顶层合并由schedule执行。
    // 长D尾块与完整tile各保留已调优的五行归约；普通短尾块的Phi预取按no-mid/has-mid编译期特化。
    // dual has-mid无尾块时跨tile延迟归约，保留同一共享阶段以避免复制DMA/Vector/event序列。
    __aicore__ inline void AccumulateHalfPartials(uint32_t tOff, uint32_t tD, float (&sumSqHalf)[2],
                                                  float (&dotHalf)[2][N], uint32_t halfBegin = 0,
                                                  uint32_t halfEnd = 2, bool firstPhiPrefetched = false);

    TPipe pipe_;

    TBuf<TPosition::VECCALC> bufGammaFp32_;
    TBuf<TPosition::VECCALC> bufScratchFp32_;
    TBuf<TPosition::VECCALC> bufX2Fp32_;
    TBuf<TPosition::VECCALC> bufActivationLowp_;
    TQue<TPosition::VECIN, 1> queResidualLowp_;
    TBuf<TPosition::VECCALC> bufResFp32_;
    TBuf<TPosition::VECCALC> bufPostComb_;
    TQue<TPosition::VECIN, 2> bufPhiFp32_;
    TQue<TPosition::VECOUT, BUFFER_NUM_ONE> bufHInPrimeBf16_;
    TBuf<TPosition::VECCALC> bufScalar_;

    GlobalTensor<T> hOutGm_, residualGm_, hInPrimeGm_, x2OutGm_;
    GlobalTensor<float> hPostGm_, hResGm_, phiGm_;
    GlobalTensor<float> alpha0Gm_, biasGm_;
    GlobalTensor<float> gamma0Gm_, gamma1Gm_, gamma2Gm_;
    GlobalTensor<float> hInF32Gm_;
    GlobalTensor<float> partial1Gm_;

    uint32_t T_ = 0, N_ = 0, D_ = 0, K_ = 0;
    uint32_t usedCoreNum_ = 0, baseTPerCore_ = 0, remainderT_ = 0;
    uint32_t dTile_ = 0, dLoop_ = 0, lastDTile_ = 0;
    uint32_t phiCols_ = 0;
    float normEps_ = 1e-6f, mhcPreNormEps_ = 1e-6f, hcEps_ = 1e-6f, invD_ = 0.0f, invK_ = 0.0f;
    uint32_t tStart_ = 0, tEnd_ = 0;
    float alpha0_ = 0.0f;
    float biasCache_[N] = {0.0f};
    uint32_t siblingAivIdx_ = 0;

    int32_t evtMte2toV_ = 0;
    int32_t evtVtoMte2_ = 0;
    int32_t evtMte3toV_ = 0;
    int32_t evtVtoMte3_ = 0;
    int32_t evtMte2toS_ = 0;
    int32_t evtStoMte3_ = 0;

};

#include "mhc_norm_post_preonly_kernel_impl.h"
#include "mhc_norm_post_preonly_kernel_schedule.h"

#endif // MHC_NORM_POST_PREONLY_KERNEL_H_
