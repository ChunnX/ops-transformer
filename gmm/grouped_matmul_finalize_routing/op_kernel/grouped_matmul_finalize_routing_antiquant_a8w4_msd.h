/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_finalize_routing_antiquant_a8w4_msd.h
 * \brief
 */

#ifndef ASCENDC_GROUPED_MATMUL_FINALIZE_ROUTING_ANTIQUANT_A8W4_MSD_H
#define ASCENDC_GROUPED_MATMUL_FINALIZE_ROUTING_ANTIQUANT_A8W4_MSD_H

#include <cstdint>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "grouped_matmul_finalize_routing_utils.h"

namespace GroupedMatmulFinalizeRouting {
using namespace matmul;
using namespace AscendC;

constexpr uint32_t BUFFER_NUM_BSD = 1;
constexpr uint32_t MM_BASE_BLOCK_OFFSET = 32768; // baseM * baseN = 128 * 256
constexpr uint32_t UB_ALIGN_LEN = 32;

template <typename T>
__aicore__ inline void DataCopyPad2DA8W4(const LocalTensor<T> &dst, const GlobalTensor<T> &src,
                                         const DataCopy2DDimParams& copyDimParams) {
    DataCopyExtParams datacopyParams;
    datacopyParams.blockCount = copyDimParams.dim1;
    datacopyParams.blockLen = copyDimParams.dim0 * sizeof(T);
    datacopyParams.srcStride = (copyDimParams.srcDim0 - copyDimParams.dim0) * sizeof(T);
    // 32: int32 -> float16, 为防止跨行数据进入同一32B block，提前每行按偶数block对齐
    datacopyParams.dstStride = Ceil(copyDimParams.dim0 * sizeof(T), 32) % 2;

    DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
    DataCopyPad(dst, src, datacopyParams, padParams);
}

template <typename T>
__aicore__ inline void DataCopyPad2DA8W4ND(const LocalTensor<T> &dst, const GlobalTensor<T> &src,
                                           const DataCopy2DDimParams& copyDimParams) {
    DataCopyExtParams params;
    params.blockCount = copyDimParams.dim1;
    params.blockLen = copyDimParams.dim0 * sizeof(T);
    params.srcStride = (copyDimParams.srcDim0 - copyDimParams.dim0) * sizeof(T);
    params.dstStride = 0;

    DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
    DataCopyPad(dst, src, params, padParams);
    return;
}

template <typename T>
__aicore__ inline void DataCopyPad2DA8W4(const GlobalTensor<T> &dst, const LocalTensor<T> &src,
                                         const DataCopy2DDimParams& copyDimParams, uint32_t dstDim0) {
    DataCopyExtParams params;
    params.blockCount = copyDimParams.dim1;
    params.blockLen = copyDimParams.dim0 * sizeof(T);
    // 32: ub访问粒度为32B
    params.srcStride = (copyDimParams.srcDim0 - copyDimParams.dim0) * sizeof(T) / 32;
    params.dstStride = (dstDim0 - copyDimParams.dim0) * sizeof(T);
    DataCopyPad(dst, src, params);
}

template <class mmType>
class GMMA8W4MSDCompute {
public:
    using aT = MatmulType<TPosition::GM, CubeFormat::ND, int4b_t>;
    using bT = typename mmType::BT;
    using biasT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
    using cT = MatmulType<TPosition::GM, CubeFormat::ND, half>;
    using DTYPE_OUT = float;

public:
    __aicore__ inline GMMA8W4MSDCompute(typename mmType::MT &matmul) : mm(matmul) {}
    __aicore__ inline void Init(const MMInitParams& initParams, const GroupMatmulFRTilingData *tilingData, TPipe *tPipeIn);
    __aicore__ inline void Process();
private:
    __aicore__ inline void InitUbBuffer();
    __aicore__ inline void PreProcess();
    __aicore__ inline void InitOutputWithZeros(uint64_t offset, uint64_t size);
    __aicore__ inline void MMCompute(uint32_t groupIdx, MNConfig& mnConfig);
    __aicore__ inline void VectorCompute(uint32_t groupIdx, MNConfig& mnConfig);
    __aicore__ inline void ComputeDequantAndActivate(MNConfig& mnConfig, uint32_t curVecBaseM, uint32_t alignBaseN,
                                                     uint32_t curVecBaseN, uint32_t offsetM);
    __aicore__ inline void CastMulsAdds(uint32_t computeSize, uint32_t addStartAddr);
    __aicore__ inline void AfterMuls(uint32_t addStartAddr);
    __aicore__ inline void DataCopyScale(uint32_t curBaseN, uint32_t alignBaseN, uint64_t scaleOffset);
    __aicore__ inline void DataCopyOffset(uint32_t curBaseN, uint32_t alignBaseN, uint64_t scaleOffset);
    __aicore__ inline void DataCopyPerTokenScaleAndBrcb(MNConfig& mnConfig, uint32_t curBaseM, uint32_t alignBaseN,
                                                        uint32_t offsetM);
    __aicore__ inline void VectorAtomicProcess(uint32_t curVecBaseM, uint32_t curVecBaseN, uint32_t alignBaseN,
                                               uint32_t rowIndexOffset, uint64_t yGmOffset);
    __aicore__ inline void  VectorProcess(VectorAtomicParams& vecAParams, MNConfig& mnConfig, uint64_t mmOutOffset,
                                          LocalTensor<cT::T>& mmOutLocal);
    __aicore__ inline void CopyMMOutLocal(LocalTensor<cT::T> &mmOutTensor, uint32_t dstOffset, uint64_t srcAddr,
                            uint32_t curVecBaseM, uint32_t curVecBaseN);
    __aicore__ inline void DataCopyAndBrcbOfRowSum(MNConfig& mnConfig, uint32_t curBaseM, uint32_t alignBaseN,
                                                        uint32_t offsetM);
private:
    typename mmType::MT& mm;
    const uint32_t HALF_ALIGN = 16;
    GlobalTensor<int4b_t> xGm;
    GlobalTensor<int4b_t> weightGm;
    GlobalTensor<float> biasGm;  // for 8 * weight
    GlobalTensor<cT::T> mmOutGm;
    GlobalTensor<uint64_t> scaleGm;
    GlobalTensor<float> perTokenScaleGm;
    GlobalTensor<float> offsetGm;
    GlobalTensor<int64_t> groupTokensGm;
    GlobalTensor<float> logitsGm;
    GlobalTensor<bfloat16_t> residualGm;
    GlobalTensor<int64_t> tokenRanksGm;
    GlobalTensor<DTYPE_OUT> yGm;
    GlobalTensor<float> xRowSumGm;
    // define the que
    TQue<QuePosition::VECIN, 1> vecInQueue;
    TQue<QuePosition::VECOUT, 1> vecOutQueue;
    TQue<QuePosition::VECIN, 1> scaleInQueue;
    TQue<QuePosition::VECIN, 1> offsetInQueue;
    TQue<QuePosition::VECIN, 1> perTokenScaleInQueue;
    TQue<QuePosition::VECIN, 1> xRowSumInQueue;
    TBuf<TPosition::VECCALC> tmpBuff;
    LocalTensor<float> offsetInUb;
    LocalTensor<float> scaleInUb;
    LocalTensor<float> afterProcessTempBuffer;
    LocalTensor<float> floatCalBuffer;
    LocalTensor<float> afterProcessOutBuffer;
    LocalTensor<float> scaleCalBuffer;
    LocalTensor<float> offsetCalBuffer;
    LocalTensor<uint8_t> tempBuffer;
    LocalTensor<uint8_t> tempRowSumBuffer;
    uint32_t subBlockIdx;
    uint32_t coreIdx;
    uint32_t quantGroupSize;
    uint32_t cubeCount = 0;
    uint32_t vecCount = 0;
    uint32_t xRowSumCount = 0;
    uint32_t withOffset;
    TPipe *pipe;
    const GroupMatmulFRTilingData *tiling;
};

template <typename mmType>
__aicore__ inline void GMMA8W4MSDCompute<mmType>::Init(const MMInitParams& initParams,
    const GroupMatmulFRTilingData *tilingData, TPipe *tPipeIn)
{
    tiling = tilingData;
    xRowSumCount = tiling->totalInGroup;
    withOffset = tiling->withOffset;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ int4b_t *>(initParams.x));
    weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ int4b_t *>(initParams.weight));
    biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.bias));
    scaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(initParams.scale));
    perTokenScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.pertoken_scale));
    groupTokensGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(initParams.group_tokens));
    logitsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.logits));
    tokenRanksGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(initParams.token_ranks));
    residualGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(initParams.residual));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_OUT *>(initParams.y));
    if (withOffset == uint32_t(1)) {
        xRowSumGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.workspace));
        mmOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ cT::T *>(initParams.workspace +
                                xRowSumCount * sizeof(float))); // add xRowSumGm
        offsetGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.offset));
    } else {
        mmOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ cT::T *>(initParams.workspace));
    }
    quantGroupSize = tiling->k / tiling->quantGroupNum;  // 约束为整除关系
    subBlockIdx = GetSubBlockIdx();
    coreIdx = GetBlockIdx();
    if ASCEND_IS_AIV {
        if (GetTaskRation() != 0) {
            coreIdx /= GetTaskRation();
        }
    }
    pipe = tPipeIn;
    InitUbBuffer();
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDCompute<mmType>::InitUbBuffer()
{
    if ASCEND_IS_AIC {
        return;
    }
    pipe->InitBuffer(scaleInQueue, BUFFER_NUM_BSD, tiling->matmulTiling.baseN * sizeof(float)); // bias queue
    // 2: pertoken scale和logits般到一块buffer上
    pipe->InitBuffer(perTokenScaleInQueue, BUFFER_NUM_BSD, Ceil(tiling->vBaseM * sizeof(float) * 2,
        UB_ALIGN_LEN) * UB_ALIGN_LEN);
    pipe->InitBuffer(vecInQueue, BUFFER_NUM_BSD, tiling->ubCalSize * 2 * sizeof(cT::T));
    pipe->InitBuffer(vecOutQueue, BUFFER_NUM_BSD, tiling->ubCalSize * sizeof(DTYPE_OUT));
    pipe->InitBuffer(tmpBuff, tiling->ubRestBytes);
    uint32_t ubCalSizeFloat = tiling->ubCalSize * sizeof(float);
    // ub分配，依次划分中间结果，划分方式参考设计文档
    afterProcessTempBuffer = tmpBuff.GetWithOffset<float>(tiling->ubCalSize * 2, 0);
    uint32_t offset = ubCalSizeFloat * uint32_t(2);
    floatCalBuffer = tmpBuff.GetWithOffset<float>(tiling->ubCalSize, offset);
    if (withOffset == uint32_t(1)) {
        pipe->InitBuffer(offsetInQueue, BUFFER_NUM_BSD, tiling->matmulTiling.baseN * sizeof(float));
        pipe->InitBuffer(xRowSumInQueue, BUFFER_NUM_BSD, Ceil(tiling->vBaseM * sizeof(float), 32) * 32);
        offsetCalBuffer = tmpBuff.GetWithOffset<float>(tiling->ubCalSize, ubCalSizeFloat);
        tempBuffer = tmpBuff.GetWithOffset<uint8_t>(ubCalSizeFloat, 0);
        tempRowSumBuffer = tmpBuff.GetWithOffset<uint8_t>(ubCalSizeFloat, ubCalSizeFloat);
    } else {
        tempBuffer = tmpBuff.GetWithOffset<uint8_t>(2 * ubCalSizeFloat, 0);
    }
    offset += ubCalSizeFloat;
    afterProcessOutBuffer = tmpBuff.GetWithOffset<float>(tiling->ubCalSize, offset);
    scaleCalBuffer = tmpBuff.GetWithOffset<float>(tiling->ubCalSize, 0);
}

template <class P>
__aicore__ inline void GMMA8W4MSDCompute<P>::InitOutputWithZeros(uint64_t offset, uint64_t size) {
    uint64_t singelCount = Ceil(size, uint32_t(GetBlockNum() * GetTaskRation()));
    singelCount = Ceil(singelCount, 512) * 512;
    uint64_t baseOffset = GetBlockIdx() * singelCount;
    if (baseOffset >= size) {
        return;
    }
    if (baseOffset + singelCount > size) {
        singelCount = size - baseOffset;
    }
    baseOffset += offset;

    uint64_t times = (singelCount + UINT32_MAX - uint32_t(1)) / UINT32_MAX;
    for (uint32_t i = 0; i < times; i++) {
        InitOutput<DTYPE_OUT>(yGm[baseOffset + (i * UINT32_MAX)], singelCount - (i * UINT32_MAX), 0);
    }
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDCompute<mmType>::PreProcess() {
    uint64_t totalOutput = (static_cast<uint64_t>(tiling->n)) * tiling->sharedInputLen;
    uint64_t singeCount = Ceil(totalOutput, uint32_t(GetBlockNum() * GetTaskRation()));
    uint64_t baseOffset;
    uint64_t outOffset;
    uint64_t curCount = tiling->ubCalSize;

    if (tiling->sharedInputOffset > 0) {
        InitOutputWithZeros(0, (static_cast<uint64_t>(tiling->n)) * tiling->sharedInputOffset);
    }
    uint64_t tail = tiling->sharedInputOffset + tiling->sharedInputLen;
    if (tail < tiling->batch) {
        InitOutputWithZeros(tail * (static_cast<uint64_t>(tiling->n)), tiling->n * (tiling->batch - tail));
    }

    singeCount = Ceil(singeCount, tiling->ubCalSize) * tiling->ubCalSize;
    baseOffset = GetBlockIdx() * singeCount;
    if (baseOffset >= totalOutput) {
        return;
    }

    if (baseOffset + singeCount > totalOutput) {
        singeCount = totalOutput - baseOffset;
    }
    outOffset = baseOffset + tiling->n * tiling->sharedInputOffset;

    DataCopyPadExtParams<bfloat16_t> padParams;
    for (uint32_t offset = 0; offset < singeCount; offset += curCount) {
        if (unlikely(offset + curCount > singeCount)) {
            curCount = singeCount - offset;
        }
        auto residualLocalTensor = vecInQueue.AllocTensor<bfloat16_t>();
        // 32B对齐搬运可以简化参数，这里按不对齐处理
        DataCopyExtParams params{1, static_cast<uint32_t>(curCount * sizeof(bfloat16_t)), 1, 1, 0};
        DataCopyPad(residualLocalTensor, residualGm[baseOffset + offset], params, padParams);
        vecInQueue.EnQue(residualLocalTensor);
        residualLocalTensor = vecInQueue.DeQue<bfloat16_t>();

        Cast(floatCalBuffer, residualLocalTensor, AscendC::RoundMode::CAST_NONE, curCount);
        vecInQueue.FreeTensor(residualLocalTensor);
        PipeBarrier<PIPE_V>();
        LocalTensor<DTYPE_OUT> yLocal = vecOutQueue.AllocTensor<DTYPE_OUT>();
        Muls(yLocal, floatCalBuffer, tiling->residualScale, curCount);
        vecOutQueue.EnQue(yLocal);

        DataCopyExtParams paramsOut{1, static_cast<uint32_t>(curCount * sizeof(float)), 1, 1, 0};
        yLocal = vecOutQueue.DeQue<DTYPE_OUT>();
        DataCopyPad(yGm[outOffset + offset], yLocal, paramsOut);
        vecOutQueue.FreeTensor(yLocal);
    }
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDCompute<mmType>::Process()
{
    if ASCEND_IS_AIV {
        PreProcess();
        SyncAll();
    }
    MNConfig mnConfig;
    mnConfig.baseM = tiling->matmulTiling.baseM;
    mnConfig.baseN = tiling->matmulTiling.baseN;
    mnConfig.singleM = mnConfig.baseM;
    mnConfig.singleN = mnConfig.baseN;
    mnConfig.blockDimN = Ceil(tiling->n, mnConfig.singleN);
    if ASCEND_IS_AIC {
        SyncAll<false>();
    }
    for (uint32_t groupIdx = 0, preCount = 0; groupIdx < tiling->groupNum; ++groupIdx) {
        int32_t m = static_cast<int32_t>(groupTokensGm.GetValue(groupIdx));
        if (m <= 0) {
            continue;
        }
        mnConfig.m = static_cast<uint32_t>(m) * 2;      // 2: int8 has been split in 2 int4
        mnConfig.blockDimM = Ceil(mnConfig.m, mnConfig.singleM);
        mm.SetOrgShape(mnConfig.m, tiling->n, tiling->k);
        uint32_t curCount = preCount + mnConfig.blockDimN * mnConfig.blockDimM;
        uint32_t curBlock = coreIdx >= preCount ? coreIdx : coreIdx + tiling->coreNum;

        while (curBlock < curCount) {
            mnConfig.mIdx = (curBlock - preCount) / mnConfig.blockDimN;
            mnConfig.nIdx = (curBlock - preCount) % mnConfig.blockDimN;
            MMCompute(groupIdx, mnConfig);
            if ASCEND_IS_AIV {
                VectorCompute(groupIdx, mnConfig);
            }
            curBlock += tiling->coreNum;
        }
        preCount = curCount % tiling->coreNum;
        mnConfig.offsetM += mnConfig.m;
    }
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDCompute<mmType>::MMCompute(uint32_t groupIdx, MNConfig& mnConfig)
{
    mnConfig.workSpaceOffset = MM_BASE_BLOCK_OFFSET * \
                                   (coreIdx + (cubeCount % tiling->parallNum) * tiling->coreNum);
    if ASCEND_IS_AIC {
        uint32_t tailN = mnConfig.nIdx * mnConfig.singleN;
        uint32_t curSingleN = mnConfig.singleN;
        if (unlikely(mnConfig.nIdx == mnConfig.blockDimN - 1)) {
            curSingleN = tiling->n - tailN;
        }
        uint32_t curSingleM = mnConfig.singleM;
        if (unlikely(mnConfig.mIdx == mnConfig.blockDimM - 1)) {
            curSingleM = mnConfig.m - mnConfig.mIdx * mnConfig.singleM;
        }

        uint64_t xOffset = (static_cast<uint64_t>(mnConfig.offsetM) + mnConfig.mIdx * mnConfig.singleM) * tiling->k;
        uint64_t weightOffset;
        if constexpr (mmType::BT::format == CubeFormat::NZ) {
            weightOffset = (static_cast<uint64_t>(groupIdx)) * tiling->n * tiling->k + tailN * tiling->k;
        } else {
            weightOffset = (static_cast<uint64_t>(groupIdx)) * tiling->n * tiling->k + tailN;
        }
        if (cubeCount >= tiling->parallNum) {
            CrossCoreWaitFlag(SYNC_AIV_TO_AIC);
        }
        mm.SetSingleShape(curSingleM, curSingleN, quantGroupSize); // 8, 256, 512 --> 514us
        GlobalTensor<int4b_t> weightSlice;
        for (uint32_t loopK = 0; loopK < tiling->quantGroupNum; loopK++) {
            mm.SetTensorA(xGm[xOffset + loopK * quantGroupSize]);
            if constexpr (mmType::BT::format == CubeFormat::NZ) { 
                weightSlice = weightGm[weightOffset + loopK * quantGroupSize * 64];
            } else {
                weightSlice = weightGm[weightOffset + loopK * quantGroupSize * tiling->n];
            }
            if (mnConfig.blockDimM == 1) {
                weightSlice.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            }
            mm.SetTensorB(weightSlice);
            mm.SetQuantVector(scaleGm[groupIdx * tiling->n * tiling->quantGroupNum + loopK * tiling->n + tailN]);
            uint64_t worskspaceOffset = mnConfig.workSpaceOffset;
            mm.Iterate();
            mm.GetTensorC(mmOutGm[worskspaceOffset], loopK == 0 ? 0 : 1, true);
            worskspaceOffset += MM_BASE_BLOCK_OFFSET;
        }
        CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_TO_AIV);  // 2: mode为2, group内同步
    }
    cubeCount++;
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDCompute<mmType>::CopyMMOutLocal(LocalTensor<cT::T> &mmOutTensor, uint32_t dstOffset, uint64_t srcAddr,
                            uint32_t curVecBaseM, uint32_t curVecBaseN)
{
    DataCopy2DDimParams copyDimParams{static_cast<uint32_t>(curVecBaseM),
                                      static_cast<uint32_t>(curVecBaseN),
                                      static_cast<uint32_t>(curVecBaseN * static_cast<uint32_t>(2))};
    if constexpr (mmType::BT::format == CubeFormat::ND) {
        DataCopyPad2DA8W4ND(mmOutTensor[dstOffset],
            mmOutGm[srcAddr], copyDimParams);    // 2: 2 lines int4 to 1 line int8
    } else {
        DataCopyPad2DA8W4(mmOutTensor[dstOffset],
            mmOutGm[srcAddr], copyDimParams);    // 2: 2 lines int4 to 1 line int8
    }
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDCompute<mmType>::VectorAtomicProcess(uint32_t curVecBaseM, uint32_t curVecBaseN, uint32_t alignBaseN,
                                        uint32_t rowIndexOffset, uint64_t yGmOffset)
{
    LocalTensor<DTYPE_OUT> yLocal = vecOutQueue.DeQue<DTYPE_OUT>();

    SetAtomicAdd<float>();
    DataCopyExtParams paramsOut{1, static_cast<uint32_t>(curVecBaseN * sizeof(float)), 1, 1, 0};
    for (uint32_t i = 0; i < curVecBaseM; i++) {
        auto outRow = static_cast<uint64_t>(tokenRanksGm.GetValue(rowIndexOffset + i));
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        DataCopyPad(yGm[outRow * tiling->n + yGmOffset], yLocal[i * alignBaseN], paramsOut);
    }
    SetAtomicNone();

    vecOutQueue.FreeTensor(yLocal);
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDCompute<mmType>::VectorProcess(VectorAtomicParams& vecAParams, MNConfig& mnConfig,
    uint64_t mmOutOffset, LocalTensor<cT::T>& mmOutLocal)
{
    uint32_t targetAddr = vecAParams.curVecBaseM * vecAParams.alignBaseN;
    uint64_t lowBitAddr = mmOutOffset + (vecAParams.offsetM * 2UL + 1UL) * vecAParams.curVecBaseN;
    CopyMMOutLocal(mmOutLocal, targetAddr, lowBitAddr, vecAParams.curVecBaseM, vecAParams.curVecBaseN);
    PipeBarrier<PIPE_MTE2>();
    vecInQueue.EnQue(mmOutLocal);
    ComputeDequantAndActivate(mnConfig, vecAParams.curVecBaseM, vecAParams.alignBaseN, vecAParams.curVecBaseN,
                              vecAParams.offsetM);
    VectorAtomicProcess(vecAParams.curVecBaseM, vecAParams.curVecBaseN, vecAParams.alignBaseN,
                        vecAParams.yGmOffset0, vecAParams.yGmOffset1);
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDCompute<mmType>::VectorCompute(uint32_t groupIdx, MNConfig& mnConfig)
{
    uint32_t curCubeSingleN = (mnConfig.nIdx == mnConfig.blockDimN - 1) ? tiling->n - mnConfig.nIdx * mnConfig.singleN : mnConfig.singleN;
    uint32_t curCubeSingleM = mnConfig.singleM / 2;
    uint32_t mGlobalOffset = mnConfig.offsetM / 2 + mnConfig.mIdx * curCubeSingleM; // 2: 2 lines int4 to 1 line int8
    uint64_t outOffset = mGlobalOffset * tiling->n + mnConfig.nIdx * mnConfig.singleN;
     // 2: 2 lines int4 to 1 line int8
    if (mnConfig.mIdx == mnConfig.blockDimM - 1) {
        curCubeSingleM = mnConfig.m / 2 - mnConfig.mIdx * curCubeSingleM;
    }
    uint32_t vecBaseM = tiling->ubCalSize / (Ceil(mnConfig.baseN, uint32_t(8)) * 8);  //  8: num int32_t in 32B ub block  32*256/256
    vecBaseM = vecBaseM < curCubeSingleM ? vecBaseM : curCubeSingleM;
    uint32_t curVecBaseN = mnConfig.baseN;
    uint64_t scaleOffset = (static_cast<uint64_t>(groupIdx)) * tiling->n + mnConfig.nIdx * mnConfig.singleN;
    uint64_t offsetOffset = scaleOffset;
    uint32_t taskRation = GetTaskRation();
    CrossCoreWaitFlag(SYNC_AIC_TO_AIV);
    for (uint32_t offsetN = 0; offsetN < curCubeSingleN; offsetN += mnConfig.baseN) {
        curVecBaseN = unlikely(offsetN + mnConfig.baseN >= curCubeSingleN) ? curCubeSingleN - offsetN : curVecBaseN;

        uint32_t alignBaseN = Ceil(curVecBaseN, uint32_t(8)) * 8;  //  8: num int32_t in 32B ub block
        DataCopyScale(curVecBaseN, alignBaseN, scaleOffset + offsetN);
        if (withOffset == uint32_t(1)) {
            DataCopyOffset(curVecBaseN, alignBaseN, offsetOffset + offsetN);
        }
        uint32_t curVecBaseM = vecBaseM;
        uint64_t mmOutOffset = mnConfig.workSpaceOffset + offsetN * mnConfig.baseM;
        for (uint32_t offsetM = 0; offsetM < curCubeSingleM; offsetM += vecBaseM, vecCount++) {
            if (taskRation != 0UL && vecCount % taskRation != subBlockIdx) {
                continue;
            }
            curVecBaseM = unlikely(offsetM + vecBaseM >= curCubeSingleM) ? curCubeSingleM - offsetM : curVecBaseM;

            LocalTensor<cT::T> mmOutLocal = vecInQueue.AllocTensor<cT::T>();
            uint64_t highBitAddr = mmOutOffset + offsetM * 2UL * curVecBaseN;
            CopyMMOutLocal(mmOutLocal, 0, highBitAddr, curVecBaseM, curVecBaseN);

            if constexpr (mmType::BT::format == CubeFormat::ND) {
                alignBaseN = (alignBaseN + HALF_ALIGN - uint32_t(1)) / HALF_ALIGN * HALF_ALIGN;
            }
            VectorAtomicParams vecAParams{curVecBaseM, curVecBaseN, alignBaseN, offsetM, mGlobalOffset,
                mGlobalOffset + offsetM, mnConfig.nIdx * mnConfig.singleN + offsetN};
            VectorProcess(vecAParams, mnConfig, mmOutOffset, mmOutLocal);
        }
        scaleInQueue.FreeTensor(scaleInUb);
        if (withOffset == uint32_t(1)) {
            offsetInQueue.FreeTensor(offsetInUb);
        }
    }
    CrossCoreSetFlag<2, PIPE_MTE2>(SYNC_AIV_TO_AIC);  // 2: mode为2, group内同步
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDCompute<mmType>::CastMulsAdds( uint32_t computeSize, uint32_t addStartAddr)
{
    LocalTensor<cT::T> mmOutInUb = vecInQueue.DeQue<cT::T>();
    uint32_t castSize;
    if constexpr (mmType::BT::format == CubeFormat::ND) {
        castSize = computeSize + (computeSize + HALF_ALIGN - uint32_t(1)) / HALF_ALIGN * HALF_ALIGN;
    } else {
        castSize = computeSize * uint32_t(2);
    }
    Cast(afterProcessTempBuffer, mmOutInUb, RoundMode::CAST_NONE, castSize);
    PipeBarrier<PIPE_V>();
    vecInQueue.FreeTensor(mmOutInUb);
    const float RIGHT_MOVE = 16.0f;         // right move int4 to int8
    Muls(floatCalBuffer, afterProcessTempBuffer, RIGHT_MOVE, computeSize);
    PipeBarrier<PIPE_V>();
    Add(afterProcessOutBuffer, afterProcessTempBuffer[addStartAddr], floatCalBuffer, computeSize);
    PipeBarrier<PIPE_V>();
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDCompute<mmType>::AfterMuls(uint32_t addStartAddr)
{
    PipeBarrier<PIPE_V>();
    LocalTensor<DTYPE_OUT> yLocalInUb = vecOutQueue.AllocTensor<DTYPE_OUT>();
    if (withOffset == uint32_t(1)) {
        Mul(yLocalInUb, offsetCalBuffer, afterProcessOutBuffer, addStartAddr);
    } else {
        Mul(yLocalInUb, floatCalBuffer, afterProcessOutBuffer, addStartAddr);
    }
    PipeBarrier<PIPE_V>();
    vecOutQueue.EnQue(yLocalInUb);
}


template <typename mmType>
__aicore__ inline void GMMA8W4MSDCompute<mmType>::ComputeDequantAndActivate(MNConfig& mnConfig, 
    uint32_t curVecBaseM, uint32_t alignBaseN, uint32_t curVecBaseN, uint32_t offsetM)
{
    uint32_t computeSize = curVecBaseM * alignBaseN;
    uint32_t addStartAddr;
    if constexpr (mmType::BT::format == CubeFormat::ND) {
        addStartAddr = (computeSize + HALF_ALIGN - uint32_t(1)) / HALF_ALIGN * HALF_ALIGN;
    } else {
        addStartAddr = computeSize;
    }
    CastMulsAdds(computeSize, addStartAddr);
    uint32_t loop = alignBaseN / uint32_t(64); // 256B为64个float，alignBaseN需约束为64倍数
    uint8_t blkStride = static_cast<uint8_t>(alignBaseN * sizeof(float) / uint32_t(32));  //32: 单位32B
    BinaryRepeatParams param(1, 1, 1, blkStride, blkStride, 0);
    uint64_t mask = 64;
    uint64_t last = alignBaseN % uint64_t(64);
    for (uint32_t i = 0; i < loop; i++) {
        uint32_t offset = i * uint32_t(64); // 每次64个元素
        Add(floatCalBuffer[offset], afterProcessOutBuffer[offset], scaleInUb[offset], mask, curVecBaseM, param);
    }
    PipeBarrier<PIPE_V>();
    if (unlikely(last > uint64_t(0))) {
        uint32_t offset = loop * uint32_t(64);
        Add(floatCalBuffer[offset], afterProcessOutBuffer[offset], scaleInUb[offset], last, curVecBaseM, param);
    }
    PipeBarrier<PIPE_V>();
    if (withOffset == uint32_t(1)) {
        DataCopyAndBrcbOfRowSum(mnConfig, curVecBaseM, alignBaseN, offsetM);
        PipeBarrier<PIPE_V>();

        for (uint32_t i = 0; i < loop; i++) {
            uint32_t offset = i * uint32_t(64); // 每次64个元素
            Mul(scaleCalBuffer[offset], afterProcessOutBuffer[offset], offsetInUb[offset], mask, curVecBaseM, param);
        }
        PipeBarrier<PIPE_V>();
        if (unlikely(last > uint64_t(0))) {
            uint32_t offset = loop * uint32_t(64);
            Mul(scaleCalBuffer[offset], afterProcessOutBuffer[offset], offsetInUb[offset], last, curVecBaseM, param);
        }
        PipeBarrier<PIPE_V>();

        Add(offsetCalBuffer, scaleCalBuffer, floatCalBuffer, computeSize);
        PipeBarrier<PIPE_V>();
    }
    DataCopyPerTokenScaleAndBrcb(mnConfig, curVecBaseM, alignBaseN, offsetM);
    AfterMuls(addStartAddr);
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDCompute<mmType>::DataCopyScale(uint32_t curBaseN, uint32_t alignBaseN, uint64_t scaleOffset)
{
    // GM拷贝scale
    DataCopyPadExtParams<float> padParams;
    DataCopyExtParams scaleParams{1, static_cast<uint32_t>(curBaseN * sizeof(float)), 1, 1, 0};
    LocalTensor<float> scaleLocal = scaleInQueue.AllocTensor<float>();
    DataCopyPad(scaleLocal, biasGm[scaleOffset], scaleParams, padParams);
    scaleInQueue.EnQue(scaleLocal);
    scaleInUb = scaleInQueue.DeQue<float>();
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDCompute<mmType>::DataCopyOffset(uint32_t curBaseN, uint32_t alignBaseN, uint64_t offsetOffset)
{
    DataCopyExtParams offsetParams{1, static_cast<uint32_t>(curBaseN * sizeof(float)), 1, 1, 0};
    DataCopyPadExtParams<float> offsetPadParams;
    LocalTensor<float> offsetLocal = offsetInQueue.AllocTensor<float>();
    DataCopyPad(offsetLocal, offsetGm[offsetOffset], offsetParams, offsetPadParams);
    offsetInQueue.EnQue(offsetLocal);
    offsetInUb = offsetInQueue.DeQue<float>();
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDCompute<mmType>::DataCopyPerTokenScaleAndBrcb(MNConfig& mnConfig,
        uint32_t curBaseM, uint32_t alignBaseN, uint32_t offsetM)
{
    uint64_t perTokenScaleOffset = mnConfig.offsetM / 2 + mnConfig.mIdx * mnConfig.singleM / 2 + offsetM; //2: m方向两行合并为1行
    uint32_t alignBaseM = Ceil(curBaseM, uint32_t(8)) * 8;  //  8: num int32_t in 32B ub block
    // GM拷贝per token scale
    DataCopyPadExtParams<float> padParams;
    DataCopyExtParams perTokenScaleParams{1, static_cast<uint32_t>(curBaseM * sizeof(float)), 0, 0, 0};
    LocalTensor<float> perTokenScaleLocal = perTokenScaleInQueue.AllocTensor<float>();
    DataCopyPad(perTokenScaleLocal, perTokenScaleGm[perTokenScaleOffset], perTokenScaleParams, padParams);

    DataCopyPad(perTokenScaleLocal[alignBaseM], logitsGm[perTokenScaleOffset], perTokenScaleParams, padParams);

    perTokenScaleInQueue.EnQue(perTokenScaleLocal);

    perTokenScaleLocal = perTokenScaleInQueue.DeQue<float>();
    auto scaleTmp = perTokenScaleLocal;

    Mul(scaleCalBuffer, perTokenScaleLocal, perTokenScaleLocal[alignBaseM], curBaseM);
    PipeBarrier<PIPE_V>();
    scaleTmp = scaleCalBuffer;

    const uint32_t broadCastDst[2] = {curBaseM, alignBaseN};
    const uint32_t broadCastSrc[2] = {curBaseM, 1};
    BroadCast<float, 2, 1>(afterProcessOutBuffer, scaleTmp, broadCastDst, broadCastSrc, tempBuffer);
    perTokenScaleInQueue.FreeTensor(perTokenScaleLocal);
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDCompute<mmType>::DataCopyAndBrcbOfRowSum(MNConfig& mnConfig,
        uint32_t curBaseM, uint32_t alignBaseN, uint32_t offsetM)
{
    const uint64_t xRowSumOffset = mnConfig.offsetM / 2 + mnConfig.mIdx * mnConfig.singleM / 2 + offsetM; //2: m方向两行合并为1行

    DataCopyPadExtParams<float> padParams;
    DataCopyExtParams xRowSumParams{1, static_cast<uint32_t>(curBaseM * sizeof(float)), 0, 0, 0};
    LocalTensor<float> xRowSumLocal = xRowSumInQueue.AllocTensor<float>();
    DataCopyPad(xRowSumLocal, xRowSumGm[xRowSumOffset], xRowSumParams, padParams);

    xRowSumInQueue.EnQue(xRowSumLocal);
    xRowSumLocal = xRowSumInQueue.DeQue<float>();

    const uint32_t xRowSumBroadCastDst[2] = {curBaseM, alignBaseN};
    const uint32_t xRowSumBroadCastSrc[2] = {curBaseM, 1};
    BroadCast<float, 2, 1>(afterProcessOutBuffer, xRowSumLocal, xRowSumBroadCastDst, xRowSumBroadCastSrc, tempRowSumBuffer);
    xRowSumInQueue.FreeTensor(xRowSumLocal);
}
} // namespace GROUPED_MATMUL
#endif