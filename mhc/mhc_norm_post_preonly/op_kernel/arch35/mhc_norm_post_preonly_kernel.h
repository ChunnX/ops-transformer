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
    __aicore__ inline KernelMhcNormPostPreonly() {}

    __aicore__ inline void Init(GM_ADDR h_out_gm, GM_ADDR residual_gm, GM_ADDR h_post_gm, GM_ADDR h_res_gm,
                                GM_ADDR phi_gm, GM_ADDR alpha_gm, GM_ADDR bias_gm, GM_ADDR pre_mix_gm,
                                GM_ADDR gamma_0_gm, GM_ADDR gamma_1_gm, GM_ADDR gamma_2_gm, GM_ADDR h_in_prime_gm,
                                GM_ADDR x_2_out_gm, GM_ADDR h_in_f32_gm, GM_ADDR h_pre_gm, GM_ADDR workspace,
                                GM_ADDR tilingDataGm);

    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessTokenSingleAiv(uint32_t token);
    __aicore__ inline void ProcessTokenDualSiblingAiv(uint32_t token);

    // 加载一个token的h_post[N]和h_res[N*N]。
    __aicore__ inline void LoadTokenWeights(uint32_t token, float (&postVals)[N_MAX], uint32_t headBegin = 0U,
                                            uint32_t headEnd = N_MAX);

    // 提前搬运完整h_out；后续权重加载的全流水等待同时保证该数据对Vector可见。
    __aicore__ inline void CacheHOut(uint64_t gmOffset);

    // 多分块路径每个token搬运完整gamma0；D=2560由Process在token循环外仅加载一次。
    __aicore__ inline void CacheGamma0();

    // 提前发起完整gamma1搬运到bufGamma0Fp32_，使MTE2与门控融合的Vector计算重叠。
    // bufGamma0Fp32_在中间归一化完成后即空闲，门控融合的平方缓存bufGamma2Fp32_不受影响。
    __aicore__ inline void StartCacheGamma1();

    // 输出归一化读取完整gamma1前回收MTE2到Vector事件。
    __aicore__ inline void WaitCacheGamma1();

    // 输入归一化与残差融合阶段直接从x2缓存恢复h_out，避免再次从GM读取。
    // 将gamma的[tOff, tOff+tD)分块加载到指定缓存。
    __aicore__ inline void LoadGammaInto(LocalTensor<float> dst, GlobalTensor<float> &src, uint32_t tOff, uint32_t tD);

    // RMSNorm基础计算
    // 使用二叉树累加模式执行高精度ReduceSum。
    __aicore__ inline float ReduceSumPrecise(LocalTensor<float> src, uint32_t tD);

    __aicore__ inline float ReduceSumSq(LocalTensor<float> src, uint32_t tD);

    // 输入归一化直接从低精度h_out缓存计算平方和，省去先恢复完整FP32 h_out再平方的中间UB写读。
    __aicore__ inline float ReduceSumSqFromCache(uint32_t tOff, uint32_t tD);

    // 使用标量精度计算1/sqrt(sumSq * invCount + eps)。
    __aicore__ inline __attribute__((always_inline)) float ComputeInvRms(float sumSq, float invCount);

    // 原地依次执行标量缩放和gamma缩放。
    // MHC_Post计算
    // 批量加载低精度残差，在残差融合的VF内完成转换并计算全部N个head的x2。
    __aicore__ inline void ComputeX2AllHeadsBatched(const float (&postVals)[N_MAX], uint64_t gmBaseOffset,
                                                    uint32_t tOff, uint32_t tD);

    // 门控计算

    /// 由xHatVals计算gate[0..N-1]，其中包含invRmsPre缩放：
    ///   gate[i] = sigmoid(alpha0 * xHat[i] + bias[i]) + hcEps
    __aicore__ inline void ComputeGateValues(float totalSumSqPre, float (&xHatVals)[N_MAX]);

    __aicore__ inline void CastToIO(LocalTensor<T> dst, LocalTensor<float> src, uint32_t count);

    __aicore__ inline void FenceMte3ToV();

    __aicore__ inline void FenceVToMte2();

    __aicore__ inline void WriteLowPrecisionTile(GlobalTensor<T> &dst, uint64_t gmOffset, LocalTensor<float> src,
                                                 uint32_t count);

    __aicore__ inline void WriteHInPrimeTile(LocalTensor<float> src, uint64_t gmOffset, uint32_t count);

    // RETURN_H_IN_F32关闭时，直接在已消费完的x2低精度缓存上完成输出归一化量化，避免FP32中间区和二次转换。
    __aicore__ inline void WriteHInPrimeTileFromCache(LocalTensor<T> srcDst, LocalTensor<float> gamma, float scale,
                                                      uint64_t gmOffset, uint32_t count);

    __aicore__ inline uint32_t GetTileD(uint32_t tile) const;

    // 单AIV和双sibling AIV路径共用的分块计算。
    __aicore__ inline void ComputeX2PostTile(uint32_t token, uint32_t tOff, uint32_t tD, float invRms0,
                                             const float (&postVals)[N_MAX]);

    // 双AIV仍需读取全部残差行，但每个lane只计算自己负责的两个输出head。
    __aicore__ inline void ComputeX2PostTileForLane(uint32_t token, uint32_t tOff, uint32_t tD, float invRms0,
                                                    const float (&postVals)[N_MAX]);

    __aicore__ inline void CacheX2Tile(uint32_t token, uint32_t tOff, uint32_t tD, bool writeOutput);

    __aicore__ inline void CacheX2TileForLane(uint32_t token, uint32_t tOff, uint32_t tD, bool writeOutput);

    // 门控融合读取完当前x2分块后，复用首个head缓存保留低精度h_in。
    // 恢复FP32与平方在同一VF内完成，保留量化往返边界并减少UB中间写读。
    __aicore__ inline float ComputeHInCacheAndReduceSumSq(uint32_t tOff, uint32_t tD, LocalTensor<T> x2Cache,
                                                          LocalTensor<float> gateBuf);

    // 固定的平方和与投影归约树由单AIV和双sibling AIV路径共用。half0固定覆盖展平区间[0,2D)，
    // half1固定覆盖[2D,4D)，调用方只执行一次顶层合并partial0+partial1。
    __aicore__ inline void AccumulateHalfPartials(uint32_t tOff, uint32_t tD, float (&sumSqHalf)[2],
                                                  float (&dotHalf)[2][N_MAX], uint32_t halfBegin = 0,
                                                  uint32_t halfEnd = 2, bool firstPhiPrefetched = false);

    TPipe pipe_;

    // UB缓存
    TBuf<TPosition::VECCALC> bufGamma0Fp32_;
    TBuf<TPosition::VECCALC> bufGamma2Fp32_;
    TBuf<TPosition::VECCALC> bufX2Fp32_;
    TBuf<TPosition::VECCALC> bufX2Cache_;
    TQue<TPosition::VECIN, 1> bufResidualBf16_;
    TBuf<TPosition::VECCALC> bufResFp32_;
    TBuf<TPosition::VECCALC> bufPostComb_;
    TQue<TPosition::VECIN, 2> bufPhiFp32_;
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
    uint32_t T_ = 0, N_ = 0, D_ = 0, K_ = 0;
    uint32_t usedCoreNum_ = 0, baseTPerCore_ = 0, remainderT_ = 0;
    uint32_t dTile_ = 0, dLoop_ = 0, lastDTile_ = 0;
    uint32_t phiCols_ = 0;
    float normEps_ = 1e-6f, mhcPreNormEps_ = 1e-6f, hcEps_ = 1e-6f, invD_ = 0.0f, invK_ = 0.0f;
    uint32_t tStart_ = 0, tEnd_ = 0;
    float alpha0_ = 0.0f;
    float biasCache_[N_MAX] = {0.0f};
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
