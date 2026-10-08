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

};

#include "mhc_norm_post_preonly_kernel_impl.h"
#include "mhc_norm_post_preonly_kernel_schedule.h"

#endif // MHC_NORM_POST_PREONLY_KERNEL_H_
