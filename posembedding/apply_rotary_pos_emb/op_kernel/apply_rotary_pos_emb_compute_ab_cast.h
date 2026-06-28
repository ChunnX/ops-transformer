/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file apply_rotary_pos_emb_compute_ab_cast.h
 * \brief
 */
#ifndef APPLY_ROTARY_POS_EMB_COMPUTE_AB_CAST_H
#define APPLY_ROTARY_POS_EMB_COMPUTE_AB_CAST_H

#include "kernel_operator.h"
#include "apply_rotary_pos_emb_base.h"

namespace ApplyRotaryPosEmb {
using namespace AscendC;

template <typename T1, typename T2>
class ARPEComputeABCast : public ApplyRotaryPosEmbBase<T1> {
public:
    __aicore__ inline ARPEComputeABCast(){};
    __aicore__ inline void Init(
        GM_ADDR q, GM_ADDR k, GM_ADDR cos, GM_ADDR sin, GM_ADDR qOut, GM_ADDR kOut, GM_ADDR workspace,
        const ApplyRotaryPosEmbTilingData* tilingData);
    __aicore__ inline void Process(const ApplyRotaryPosEmbTilingData* tilingData);

private:
    __aicore__ inline void ProcessPreCore(const ApplyRotaryPosEmbTilingData* tilingData);
    __aicore__ inline void ProcessLastCore(const ApplyRotaryPosEmbTilingData* tilingData);
    __aicore__ inline void ProcessOneBatch(
        const int64_t coreBatchIndex, const int64_t batchB, const ApplyRotaryPosEmbTilingData* tilingData);

    __aicore__ inline void CopyInTensor(
        const int64_t coreBatchIndex, const int64_t preCBatchB, GlobalTensor<T1>& gm, const int64_t coreOffset,
        const int64_t cNum, const int64_t cdNum, const int64_t dim3, LocalTensor<T1>& qUb,
        const ApplyRotaryPosEmbTilingData* tilingData);
    __aicore__ inline void CopyOutTensor(
        const int64_t coreBatchIndex, const int64_t preCBatchB, GlobalTensor<T1>& gm, const int64_t coreOffset,
        const int64_t cNum, const int64_t cdNum, const int64_t dim3, LocalTensor<T1>& qOutUb,
        const ApplyRotaryPosEmbTilingData* tilingData);

    __aicore__ inline void ComputeRotary(
        const int64_t preCBatchB, const int64_t cNum, LocalTensor<T1>& qSize, LocalTensor<T1>& cosSize,
        LocalTensor<T1>& sinSize, LocalTensor<T1>& outUb,
        const ApplyRotaryPosEmbTilingData* tilingData);
    __aicore__ inline void ComputeRotaryNoCast(
        const int64_t preCBatchB, const int64_t cNum, LocalTensor<T1>& qSize, LocalTensor<T1>& cosSize,
        LocalTensor<T1>& sinSize, LocalTensor<T1>& outUb,
        const ApplyRotaryPosEmbTilingData* tilingData);
    __aicore__ inline void ComputeRotaryBF16(
        const int64_t preCBatchB, const int64_t cNum, LocalTensor<T1>& qSize, LocalTensor<T2>& cosC1,
        LocalTensor<T2>& sinC1, LocalTensor<T1>& outUb,
        const ApplyRotaryPosEmbTilingData* tilingData);

    // 【新增】Small/Large QC 分支函数，适配 xl 分离式 Q/K 处理
    __aicore__ inline void SmallQC(
        const int64_t preCBatchB, const int64_t cNum, LocalTensor<T2>& mul1Ub, LocalTensor<T2>& sinC1,
        LocalTensor<T2>& mul2Ub, LocalTensor<T2>& cosC1, const ApplyRotaryPosEmbTilingData* tilingData);
    __aicore__ inline void LargeQC(
        const int64_t preCBatchB, const int64_t cNum, LocalTensor<T2>& mul1Ub, LocalTensor<T2>& sinC1,
        LocalTensor<T2>& mul2Ub, LocalTensor<T2>& cosC1, const ApplyRotaryPosEmbTilingData* tilingData);
    __aicore__ inline void SmallQCFP16(
        const int64_t preCBatchB, const int64_t cNum, LocalTensor<T1>& mul1Ub, LocalTensor<T1>& outUb,
        LocalTensor<T1>& sinSize, LocalTensor<T1>& mul2Ub, LocalTensor<T1>& qSize, LocalTensor<T1>& cosSize,
        const ApplyRotaryPosEmbTilingData* tilingData);
    __aicore__ inline void LargeQCFP16(
        const int64_t preCBatchB, const int64_t cNum, LocalTensor<T1>& mul1Ub, LocalTensor<T1>& outUb,
        LocalTensor<T1>& sinSize, LocalTensor<T1>& mul2Ub, LocalTensor<T1>& qSize, LocalTensor<T1>& cosSize,
        const ApplyRotaryPosEmbTilingData* tilingData);

    constexpr static int32_t bufferNum = 1;
    constexpr static int32_t bufferNumdb = 2;

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, bufferNum> qInQueue;
    TQue<QuePosition::VECIN, bufferNum> cosInQueue;
    TQue<QuePosition::VECIN, bufferNum> sinInQueue;
    TQue<QuePosition::VECOUT, bufferNum> qOutQueue;
    TBuf<QuePosition::VECCALC> mul1;
    TBuf<QuePosition::VECCALC> mul2;
#if ORIG_DTYPE_QUERY != DT_FLOAT
    TBuf<QuePosition::VECCALC> cosC;
    TBuf<QuePosition::VECCALC> sinC;
#endif

    GlobalTensor<T1> qGm;
    GlobalTensor<T1> kGm;
    GlobalTensor<T1> cosGm;
    GlobalTensor<T1> sinGm;
    UnaryRepeatParams mulRepeatP;
    BinaryRepeatParams repeatParams;
    uint64_t blockIdx = 0;
};

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::Init(
    GM_ADDR q, GM_ADDR k, GM_ADDR cos, GM_ADDR sin, GM_ADDR qOut, GM_ADDR kOut, GM_ADDR workspace,
    const ApplyRotaryPosEmbTilingData* tilingData)
{
    blockIdx = GetBlockIdx();
    qGm.SetGlobalBuffer((__gm__ T1*)q);
    kGm.SetGlobalBuffer((__gm__ T1*)k);
    cosGm.SetGlobalBuffer((__gm__ T1*)cos);
    sinGm.SetGlobalBuffer((__gm__ T1*)sin);
    // 【泛化需求】qInQueue一次仅放Q或K，size按preCBatchB*singleN*lastDim分配
    pipe.InitBuffer(cosInQueue, bufferNumdb, tilingData->cosPart1Ub);
    pipe.InitBuffer(qInQueue, bufferNumdb, tilingData->qPart1Ub);
    pipe.InitBuffer(sinInQueue, bufferNumdb, tilingData->cosPart1Ub);
    pipe.InitBuffer(mul1, tilingData->q2q1Part1Ub);
    pipe.InitBuffer(qOutQueue, bufferNumdb, tilingData->qPart1Ub);
    pipe.InitBuffer(mul2, tilingData->q2q1Part1Ub);
#if ORIG_DTYPE_QUERY != DT_FLOAT
    pipe.InitBuffer(cosC, tilingData->sin1UbSize);
    pipe.InitBuffer(sinC, tilingData->sin1UbSize);
#endif

    mulRepeatP.dstBlkStride = 1;
    mulRepeatP.srcBlkStride = 1;
    mulRepeatP.dstRepStride = tilingData->dstRepSBr;
    mulRepeatP.srcRepStride = tilingData->dstRepSBr;

    repeatParams.dstBlkStride = 1;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1BlkStride = 1;
}

// 【新增】SmallQC：head（cNum）外层循环，preCBatchB 作为 Mul 的 repeat 数
// sinC1/cosC1 在 head 方向 broadcast（src1RepStride 控制）
// 恢复 chunk 循环处理 lastDim > mask 的情况
template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::SmallQC(
    const int64_t preCBatchB, const int64_t cNum, LocalTensor<T2>& mul1Ub, LocalTensor<T2>& sinC1,
    LocalTensor<T2>& mul2Ub, LocalTensor<T2>& cosC1, const ApplyRotaryPosEmbTilingData* tilingData)
{
    int64_t fullMask = tilingData->mask;
    int64_t fullChunks = (tilingData->lastDim + fullMask - 1) / fullMask;
    for (int64_t ii = 0; ii < cNum; ii++) {
        for (int32_t j = 0; j < fullChunks; j++) {
            int64_t offset = j * fullMask;
            int64_t curMask = (offset + fullMask > tilingData->lastDim) ? (tilingData->lastDim - offset) : fullMask;
            int64_t off = ii * tilingData->lastDim + offset;
            // mul1Ub 存 rotate_half(q)，in-place 乘 sinC1
            Mul<T2>(mul1Ub[off], mul1Ub[off], sinC1[offset], curMask, preCBatchB, repeatParams);
            // mul2Ub 存 q (FP32)，in-place 乘 cosC1
            Mul<T2>(mul2Ub[off], mul2Ub[off], cosC1[offset], curMask, preCBatchB, repeatParams);
        }
    }
}

// 【新增】LargeQC：batch 外层循环，cNum 作为 Mul 的 repeat 数
// sinC1/cosC1 在 batch 方向 stride（每个 batch 有自己的 cos/sin）
// 恢复 chunk 循环处理 lastDim > mask 的情况
template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::LargeQC(
    const int64_t preCBatchB, const int64_t cNum, LocalTensor<T2>& mul1Ub, LocalTensor<T2>& sinC1,
    LocalTensor<T2>& mul2Ub, LocalTensor<T2>& cosC1, const ApplyRotaryPosEmbTilingData* tilingData)
{
    int64_t fullMask = tilingData->mask;
    int64_t fullChunks = (tilingData->lastDim + fullMask - 1) / fullMask;
    int64_t batchStride = cNum * tilingData->lastDim;
    for (int64_t i = 0; i < preCBatchB; i++) {
        for (int32_t j = 0; j < fullChunks; j++) {
            int64_t offset = j * fullMask;
            int64_t curMask = (offset + fullMask > tilingData->lastDim) ? (tilingData->lastDim - offset) : fullMask;
            int64_t off = i * batchStride + offset;
            int64_t sOff = i * tilingData->lastDim + offset;
            // mul1Ub 存 rotate_half(q)，in-place 乘 sinC1
            Mul<T2>(mul1Ub[off], mul1Ub[off], sinC1[sOff], curMask, cNum, repeatParams);
            // mul2Ub 存 q (FP32)，in-place 乘 cosC1
            Mul<T2>(mul2Ub[off], mul2Ub[off], cosC1[sOff], curMask, cNum, repeatParams);
        }
    }
}

// 【新增】SmallQCFP16：head（cNum）外层循环，preCBatchB 作为 Mul 的 repeat 数
// 注意：NoCast 路径保持原样，不做 chunk 循环（tiling 保证 lastDim <= mask）
template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::SmallQCFP16(
    const int64_t preCBatchB, const int64_t cNum, LocalTensor<T1>& mul1Ub, LocalTensor<T1>& outUb,
    LocalTensor<T1>& sinSize, LocalTensor<T1>& mul2Ub, LocalTensor<T1>& qSize, LocalTensor<T1>& cosSize,
    const ApplyRotaryPosEmbTilingData* tilingData)
{
    for (int64_t ii = 0; ii < cNum; ii++) {
        int64_t off = ii * tilingData->lastDim;
        Mul<T1>(mul1Ub[off], outUb[off], sinSize, tilingData->mask, preCBatchB, repeatParams);
        Mul<T1>(mul2Ub[off], qSize[off], cosSize, tilingData->mask, preCBatchB, repeatParams);
    }
}

// 【新增】LargeQCFP16：batch 外层循环，cNum 作为 Mul 的 repeat 数
template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::LargeQCFP16(
    const int64_t preCBatchB, const int64_t cNum, LocalTensor<T1>& mul1Ub, LocalTensor<T1>& outUb,
    LocalTensor<T1>& sinSize, LocalTensor<T1>& mul2Ub, LocalTensor<T1>& qSize, LocalTensor<T1>& cosSize,
    const ApplyRotaryPosEmbTilingData* tilingData)
{
    int64_t batchStride = cNum * tilingData->lastDim;
    for (int64_t i = 0; i < preCBatchB; i++) {
        int64_t off = i * batchStride;
        int64_t sOff = i * tilingData->lastDim;
        Mul<T1>(mul1Ub[off], outUb[off], sinSize[sOff], tilingData->mask, cNum, repeatParams);
        Mul<T1>(mul2Ub[off], qSize[off], cosSize[sOff], tilingData->mask, cNum, repeatParams);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::ComputeRotaryNoCast(
    const int64_t preCBatchB, const int64_t cNum, LocalTensor<T1>& qSize, LocalTensor<T1>& cosSize,
    LocalTensor<T1>& sinSize, LocalTensor<T1>& outUb,
    const ApplyRotaryPosEmbTilingData* tilingData)
{
    LocalTensor<T1> mul2Ub = mul2.Get<T1>();
    LocalTensor<T1> mul1Ub = mul1.Get<T1>();
    // sin取反，与旋转后的q相乘即得到 rotate(q)*sin
    Muls(sinSize, sinSize, T1(-1.0), tilingData->halfNum, preCBatchB, mulRepeatP);

    // 【泛化需求】使用DataCopy旋转单张量，stride按lastDim
    DataCopyParams rotCopy;
    rotCopy.blockCount = preCBatchB * cNum;
    rotCopy.blockLen = tilingData->blockLenq2q1;
    rotCopy.srcStride = tilingData->dstRepSBr;
    rotCopy.dstStride = tilingData->dstRepSBr;
    DataCopy(outUb, qSize[tilingData->halfNum], rotCopy);
    DataCopy(outUb[tilingData->halfNum], qSize, rotCopy);

    // 【修改】按 cNum >= preCBatchB 选择 Large/Small 路径
    if (cNum >= preCBatchB) {
        // Large 路径：batch 外层循环，cNum 作为 Mul 的 repeat 数，cos/sin batch 方向 stride
        repeatParams.dstRepStride = tilingData->dstRepSBr;
        repeatParams.src0RepStride = tilingData->dstRepSBr;
        repeatParams.src1RepStride = 0;
        LargeQCFP16(preCBatchB, cNum, mul1Ub, outUb, sinSize, mul2Ub, qSize, cosSize, tilingData);
    } else {
        // Small 路径：head（cNum）外层循环，preCBatchB 作为 Mul 的 repeat 数，cos/sin head 方向 broadcast
        // repeat 方向是 batch，stride 应为 cNum * lastDim / oneBlockFp32 = cNum * dstRepSBr
        // （mulNum = singleN * lastDim / oneBlockFp32，其中 singleN = max(qcNum, kcNum)，不适合单独 Q/K 的 Small 路径）
        repeatParams.dstRepStride = cNum * tilingData->dstRepSBr;
        repeatParams.src0RepStride = cNum * tilingData->dstRepSBr;
        repeatParams.src1RepStride = tilingData->dstRepSBr;
        SmallQCFP16(preCBatchB, cNum, mul1Ub, outUb, sinSize, mul2Ub, qSize, cosSize, tilingData);
    }

    // 两个 Mul 都完成后释放 qSize
    qInQueue.FreeTensor(qSize);

    Add(outUb, mul1Ub, mul2Ub, preCBatchB * cNum * tilingData->lastDim);
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::ComputeRotaryBF16(
   const int64_t preCBatchB, const int64_t cNum, LocalTensor<T1>& qSize, LocalTensor<T2>& cosC1,
   LocalTensor<T2>& sinC1, LocalTensor<T1>& outUb,
   const ApplyRotaryPosEmbTilingData* tilingData)
{
#if ORIG_DTYPE_QUERY != DT_FLOAT
    // ============ RoPE (Rotary Position Embedding) 算法 ============
    // 核心公式: q_embed = q * cos + rotate_half(q) * sin   (* 为逐元素乘)
    //   rotate_half(q) = concat(-q[half:], q[:half])    记 q1=q[:half], q2=q[half:]
    // 展开为:
    //   out[:half] = q1 * cos[:half] - q2 * sin[:half]
    //   out[half:] = q2 * cos[half:] + q1 * sin[:half]
    // 实现技巧: sinC1 已由外部 Cast 并对前半取负 ⇒ sinC1=[-sin[:half], sin[half:]];
    //   先用 DataCopy 把 q 前后半交换得 [q2,q1] 到 mul1Ub, 再 *sinC1 即得 rotate_half(q)*sin=[-q2*sin,q1*sin]
    //   mul2Ub 先存 q (FP32), 再 in-place *cosC1 得 q*cos
    //   最后 Add(mul2Ub, mul1Ub, mul2Ub) 得最终结果, Cast 回 outUb
    // =================================================================
    // 【泛化需求】BF16按实际lastDim cast到FP32计算，UB中按lastDim排布
    LocalTensor<T2> mul2Ub = mul2.Get<T2>();   // FP32 工作区: 先存 q, 后存 q*cos (in-place), 再存最终 q_embed
    LocalTensor<T2> mul1Ub = mul1.Get<T2>();   // FP32 工作区: 先存 rotate_half(q), 后存 rotate_half(q)*sin (in-place)
    Cast(mul2Ub, qSize, RoundMode::CAST_NONE, preCBatchB * cNum * tilingData->lastDim);  // mul2Ub = q (BF16->FP32)
    // 释放 qSize
    qInQueue.FreeTensor(qSize);

    // 参考主线CTotaryBF16：用DataCopy做前后半交换（旋转），减少Mul次数
    DataCopyParams rotCopy;
    rotCopy.blockCount = preCBatchB * cNum;
    rotCopy.blockLen = tilingData->blockLenq2q1;
    rotCopy.srcStride = tilingData->blockLenq2q1;
    rotCopy.dstStride = tilingData->blockLenq2q1;

    // 旋转结果直接写入 mul1Ub，不再借用 outUb 的 FP32 视图
    DataCopy(mul1Ub, mul2Ub[tilingData->halfNum], rotCopy);         // 前半 = q[half:] = q2
    DataCopy(mul1Ub[tilingData->halfNum], mul2Ub, rotCopy);         // 后半 = q[:half] = q1  => mul1Ub=[q2,q1]

    // cosC1/sinC1已由外部Cast并取负，此处直接使用
    // 【修改】按 cNum >= preCBatchB 选择 LargeQC 或 SmallQC
    if (cNum >= preCBatchB) {
        // Large 路径：batch 外层循环，cNum 作为 Mul 的 repeat 数，cos/sin batch 方向 stride
        repeatParams.dstRepStride = tilingData->dstRepSBr;
        repeatParams.src0RepStride = tilingData->dstRepSBr;
        repeatParams.src1RepStride = 0;
        LargeQC(preCBatchB, cNum, mul1Ub, sinC1, mul2Ub, cosC1, tilingData);
    } else {
        // Small 路径：head（cNum）外层循环，preCBatchB 作为 Mul 的 repeat 数，cos/sin head 方向 broadcast
        // repeat 方向是 batch，stride 应为 cNum * lastDim / oneBlockFp32 = cNum * dstRepSBr
        // （mulNum = singleN * lastDim / oneBlockFp32，其中 singleN = max(qcNum, kcNum)，不适合单独 Q/K 的 Small 路径）
        repeatParams.dstRepStride = cNum * tilingData->dstRepSBr;
        repeatParams.src0RepStride = cNum * tilingData->dstRepSBr;
        repeatParams.src1RepStride = tilingData->dstRepSBr;
        SmallQC(preCBatchB, cNum, mul1Ub, sinC1, mul2Ub, cosC1, tilingData);
    }

    // q_embed = rotate_half(q)*sin + q*cos ; Add 批量化一次性合并所有 batch
    // mul2Ub 存 q*cos (in-place), mul1Ub 存 rotate_half(q)*sin (in-place), 结果写入 mul2Ub (in-place)
    Add(mul2Ub, mul1Ub, mul2Ub, preCBatchB * cNum * tilingData->lastDim);  // mul2Ub = mul1Ub(rot*sin) + mul2Ub(q*cos)
    Cast(outUb, mul2Ub, RoundMode::CAST_ROUND, preCBatchB * cNum * tilingData->lastDim);  // FP32 -> BF16 回写 outUb
#endif
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::ComputeRotary(
    const int64_t preCBatchB, const int64_t cNum, LocalTensor<T1>& qSize, LocalTensor<T1>& cosSize,
    LocalTensor<T1>& sinSize, LocalTensor<T1>& outUb,
    const ApplyRotaryPosEmbTilingData* tilingData)
{
#if ORIG_DTYPE_QUERY != DT_FLOAT
    ComputeRotaryBF16(preCBatchB, cNum, qSize, cosSize, sinSize, outUb, tilingData);
#else
    ComputeRotaryNoCast(preCBatchB, cNum, qSize, cosSize, sinSize, outUb, tilingData);
#endif
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::CopyInTensor(
    const int64_t coreBatchIndex, const int64_t preCBatchB, GlobalTensor<T1>& gm, const int64_t coreOffset,
    const int64_t cNum, const int64_t cdNum, const int64_t dim3, LocalTensor<T1>& qUb,
    const ApplyRotaryPosEmbTilingData* tilingData)
{
    int64_t gmBatchOffset = blockIdx * coreOffset + coreBatchIndex * tilingData->preCBatchB * cdNum;
    if (dim3 > tilingData->lastDim) { // 部分维度旋转处理
        // 【泛化需求】仅搬运当前tensor（Q或K）的旋转部分；UB内按lastDim布局
        int64_t oneBlock = tilingData->lastDim / tilingData->blockMoveQ;
        int64_t batchSize = cNum * dim3;
        int64_t ubBatchSize = cNum * tilingData->lastDim;
        if (cNum == 1) {
            // kcNum==1: blockCount=1天然成立，不需要head间stride，batch间stride即head间stride
            DataCopyParams copyIn;
            copyIn.blockCount = 1;
            copyIn.blockLen = tilingData->blockMoveQ;
            copyIn.srcStride = 0;
            copyIn.dstStride = 0;
            for (int i = 0; i < preCBatchB; i++) {
                DataCopy(qUb[i * ubBatchSize], gm[gmBatchOffset + i * batchSize], copyIn);
            }
        } else {
            DataCopyParams copyIn;
            copyIn.blockCount = cNum;
            copyIn.blockLen = tilingData->blockMoveQ;
            copyIn.srcStride = dim3 / oneBlock - tilingData->blockMoveQ;
            copyIn.dstStride = tilingData->lastDim / oneBlock - tilingData->blockMoveQ;
            for (int i = 0; i < preCBatchB; i++) {
                DataCopy(qUb[i * ubBatchSize], gm[gmBatchOffset + i * batchSize], copyIn);
            }
        }
    } else { // 全维度旋转处理
        // 【泛化需求】仅搬运当前tensor的preCBatchB*cdNum个元素
        DataCopy(qUb, gm[gmBatchOffset], preCBatchB * cdNum);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::CopyOutTensor(
    const int64_t coreBatchIndex, const int64_t preCBatchB, GlobalTensor<T1>& gm, const int64_t coreOffset,
    const int64_t cNum, const int64_t cdNum, const int64_t dim3, LocalTensor<T1>& qOutUb,
    const ApplyRotaryPosEmbTilingData* tilingData)
{
    int64_t gmBatchOffset = blockIdx * coreOffset + coreBatchIndex * tilingData->preCBatchB * cdNum;
    if (dim3 > tilingData->lastDim) { // 部分维度旋转处理
        // 【泛化需求】仅写回当前tensor的旋转部分；UB内按lastDim布局
        int64_t oneBlock = tilingData->lastDim / tilingData->blockMoveQ;
        int64_t batchSize = cNum * dim3;
        int64_t ubBatchSize = cNum * tilingData->lastDim;
        if (cNum == 1) {
            // kcNum==1: blockCount=1天然成立，不需要head间stride
            DataCopyParams copyOut;
            copyOut.blockCount = 1;
            copyOut.blockLen = tilingData->blockMoveQ;
            copyOut.srcStride = 0;
            copyOut.dstStride = 0;
            for (int i = 0; i < preCBatchB; i++) {
                DataCopy(gm[gmBatchOffset + i * batchSize], qOutUb[i * ubBatchSize], copyOut);
            }
        } else {
            DataCopyParams copyOut;
            copyOut.blockCount = cNum;
            copyOut.blockLen = tilingData->blockMoveQ;
            copyOut.srcStride = tilingData->lastDim / oneBlock - tilingData->blockMoveQ;
            copyOut.dstStride = dim3 / oneBlock - tilingData->blockMoveQ;
            for (int i = 0; i < preCBatchB; i++) {
                DataCopy(gm[gmBatchOffset + i * batchSize], qOutUb[i * ubBatchSize], copyOut);
            }
        }
    } else { // 全维度旋转处理
        // 【泛化需求】仅写回当前tensor的preCBatchB*cdNum个元素
        DataCopy(gm[gmBatchOffset], qOutUb, preCBatchB * cdNum);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::ProcessOneBatch(
    const int64_t coreBatchIndex, const int64_t batchB, const ApplyRotaryPosEmbTilingData* tilingData)
{
    LocalTensor<T1> cosUb = cosInQueue.AllocTensor<T1>();
    LocalTensor<T1> sinUb = sinInQueue.AllocTensor<T1>();
    // CopyIn cos/sin
    DataCopy(cosUb,
             cosGm[blockIdx * tilingData->cosCoreOffset + coreBatchIndex * tilingData->preCBatchB * tilingData->coscdNum],
             batchB * tilingData->coscdNum);
    DataCopy(sinUb,
             sinGm[blockIdx * tilingData->cosCoreOffset + coreBatchIndex * tilingData->preCBatchB * tilingData->coscdNum],
             batchB * tilingData->coscdNum);

    // CopyIn Q
    LocalTensor<T1> qUb = qInQueue.AllocTensor<T1>();
    CopyInTensor(coreBatchIndex, batchB, qGm, tilingData->qCoreOffset, tilingData->qcNum, tilingData->qcdNum,
                    tilingData->qDim3, qUb, tilingData);
    qInQueue.EnQue(qUb);
    cosInQueue.EnQue(cosUb);
    sinInQueue.EnQue(sinUb);
    LocalTensor<T1> cosSize = cosInQueue.DeQue<T1>();
    LocalTensor<T1> sinSize = sinInQueue.DeQue<T1>();
    LocalTensor<T1> qSize = qInQueue.DeQue<T1>();

#if ORIG_DTYPE_QUERY != DT_FLOAT
    // BF16: cos/sin Cast到FP32只做一次，Q/K复用
    LocalTensor<T2> cosC1 = cosC.Get<T2>();
    LocalTensor<T2> sinC1 = sinC.Get<T2>();
    Cast(cosC1, cosSize, RoundMode::CAST_NONE, batchB * tilingData->lastDim);
    Cast(sinC1, sinSize, RoundMode::CAST_NONE, batchB * tilingData->lastDim);
    Muls(sinC1, sinC1, T2(-1.0), tilingData->halfNum, batchB, mulRepeatP);
#endif

    // Compute Q
    LocalTensor<T1> qOutUb = qOutQueue.AllocTensor<T1>();
#if ORIG_DTYPE_QUERY != DT_FLOAT
    ComputeRotaryBF16(batchB, tilingData->qcNum, qSize, cosC1, sinC1, qOutUb, tilingData);
#else
    ComputeRotaryNoCast(batchB, tilingData->qcNum, qSize, cosSize, sinSize, qOutUb, tilingData);
#endif

    // CopyIn K
    LocalTensor<T1> kUb = qInQueue.AllocTensor<T1>();
    CopyInTensor(coreBatchIndex, batchB, kGm, tilingData->kCoreOffset, tilingData->kcNum, tilingData->kcdNum,
                    tilingData->kDim3, kUb, tilingData);
    qOutQueue.EnQue(qOutUb);

    // CopyOut Q
    LocalTensor<T1> qOutUbSize = qOutQueue.DeQue<T1>();
    CopyOutTensor(coreBatchIndex, batchB, qGm, tilingData->qCoreOffset, tilingData->qcNum, tilingData->qcdNum,
                    tilingData->qDim3, qOutUbSize, tilingData);
    qOutQueue.FreeTensor(qOutUbSize);
    qInQueue.EnQue(kUb);

    // Compute K
    LocalTensor<T1> kSize = qInQueue.DeQue<T1>();
    LocalTensor<T1> kOutUb = qOutQueue.AllocTensor<T1>();
#if ORIG_DTYPE_QUERY != DT_FLOAT
    ComputeRotaryBF16(batchB, tilingData->kcNum, kSize, cosC1, sinC1, kOutUb, tilingData);
#else
    ComputeRotaryNoCast(batchB, tilingData->kcNum, kSize, cosSize, sinSize, kOutUb, tilingData);
#endif
    cosInQueue.FreeTensor(cosSize);
    sinInQueue.FreeTensor(sinSize);
    qOutQueue.EnQue(kOutUb);

    // CopyOut K
    LocalTensor<T1> kOutUbSize = qOutQueue.DeQue<T1>();
    CopyOutTensor(coreBatchIndex, batchB, kGm, tilingData->kCoreOffset, tilingData->kcNum, tilingData->kcdNum,
                    tilingData->kDim3, kOutUbSize, tilingData);
    qOutQueue.FreeTensor(kOutUbSize);
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::ProcessPreCore(const ApplyRotaryPosEmbTilingData* tilingData)
{
    for (int64_t i = 0; i < tilingData->preCLTimes; i++) {
        ProcessOneBatch(i, tilingData->preCBatchB, tilingData);
    }
    if (tilingData->preCBatchL > 0) {
        ProcessOneBatch(tilingData->preCLTimes, tilingData->preCBatchL, tilingData);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::ProcessLastCore(const ApplyRotaryPosEmbTilingData* tilingData)
{
    for (int64_t i = 0; i < tilingData->lastCLTimes; i++) {
        ProcessOneBatch(i, tilingData->preCBatchB, tilingData);
    }
    if (tilingData->lastCBatchL > 0) {
        ProcessOneBatch(tilingData->lastCLTimes, tilingData->lastCBatchL, tilingData);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::Process(const ApplyRotaryPosEmbTilingData* tilingData)
{
    if (blockIdx >= tilingData->useCoreNum) {
        return;
    }

    if (blockIdx == tilingData->useCoreNum - 1) {
        ProcessLastCore(tilingData);
    } else {
        ProcessPreCore(tilingData);
    }
}

} // namespace ApplyRotaryPosEmb

#endif // APPLY_ROTARY_POS_EMB_COMPUTE_AB_CAST_H
