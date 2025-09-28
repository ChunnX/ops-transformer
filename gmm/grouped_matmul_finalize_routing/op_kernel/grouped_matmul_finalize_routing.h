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
 * \file grouped_matmul_finalize_routing.h
 * \brief
 */

#ifndef __GROUPED_MATMUL_FINALIZE_ROUTING_KERNEL_H_
#define __GROUPED_MATMUL_FINALIZE_ROUTING_KERNEL_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "grouped_matmul_finalize_routing_utils.h"

namespace GroupedMatmulFinalizeRouting {

using namespace matmul;
using namespace AscendC;

using aT = MatmulType<TPosition::GM, CubeFormat::ND, int8_t>;
using bT = MatmulType<TPosition::GM, CubeFormat::NZ, int8_t>;
using BiasT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
using cT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;

using MT = matmul::MatmulImpl<aT, bT, cT, BiasT, CFG_MDL>;

constexpr uint32_t BROADCAST_DIM = 2;
constexpr uint32_t BUFFER_NUM = 2;

template <bool combine_, class ROW_INDEX_DTYPE_, class TILING_TYPE_, class SCALE_TYPE_, bool transpose_ = false>
struct Param {
    static const bool combine = combine_;
    using ROW_INDEX_DTYPE = ROW_INDEX_DTYPE_;
    using TILING_TYPE = TILING_TYPE_;
    using SCALE_TYPE = SCALE_TYPE_;
    static const bool transpose = transpose_;
};


template <typename T>
__aicore__ inline void DataCopyPad2D(const LocalTensor<T> dst, const GlobalTensor<T> src,
                                     const DataCopy2DDimParams& dimParams) {
    DataCopyExtParams params;
    params.blockCount = dimParams.dim1;
    params.blockLen = dimParams.dim0 * sizeof(T);
    params.srcStride = (dimParams.srcDim0 - dimParams.dim0) * sizeof(T);
    // 32: int32 -> float16, 为防止跨行数据进入同一32B block，提前每行按偶数block对齐
    params.dstStride = Ceil(dimParams.dim0 * sizeof(T), 32) % 2;

    DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
    DataCopyPad(dst, src, params, padParams);
}

template <typename T>
__aicore__ inline void DataCopyPad2D(const GlobalTensor<T> dst, const LocalTensor<T> src,
                                     const DataCopy2DDimParams& dimParams, uint32_t dstDim0) {
    DataCopyExtParams params;
    params.blockCount = dimParams.dim1;
    params.blockLen = dimParams.dim0 * sizeof(T);
    // 32: ub访问粒度为32B
    params.srcStride = (dimParams.srcDim0 - dimParams.dim0) * sizeof(T) / 32;
    params.dstStride = (dstDim0 - dimParams.dim0) * sizeof(T);
    DataCopyPad(dst, src, params);
}

template <class P>
class QuantGroupMatmul {
    using DTYPE_OUT = std::conditional_t<P::combine, float, bfloat16_t>;

public:
    __aicore__ inline QuantGroupMatmul(MT &matmul) : mm(matmul) {}
    __aicore__ inline void Init(const MMInitParams& initParams, const typename P::TILING_TYPE *tilingData, TPipe *tPipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void PreProcess();
    __aicore__ inline void InitUbBuffer();
    __aicore__ inline void InitOutputWithZeros(uint64_t offset, uint64_t size);
    __aicore__ inline void MMCompute(uint32_t groupIdx, MNConfig& mnConfig);
    __aicore__ inline void VectorCompute(uint32_t groupIdx, MNConfig& mnConfig);
    __aicore__ inline void ComputeDequantAndActivate(MNConfig& mnConfig, const VectorAtomicParams& vecAParams);
    __aicore__ inline void ComputeDequantProcess(uint32_t computeSize);
    __aicore__ inline void DataCopyScale(uint32_t curBaseN, uint32_t alignBaseN, uint64_t scaleOffset);
    __aicore__ inline void DataCopyBias(uint32_t curBaseN, uint32_t alignBaseN, uint64_t scaleOffset);
    __aicore__ inline void DataCopyPerTokenScaleAndBrcb(MNConfig& mnConfig, uint32_t curBaseM, uint32_t alignBaseN,
                                                        uint32_t offsetM);
    __aicore__ inline void VectorAtomicProcess(const VectorAtomicParams& vecAParams);
private:
    MT& mm;
    GlobalTensor<int8_t> xGm;
    GlobalTensor<int8_t> weightGm;
    GlobalTensor<bfloat16_t> biasGm;
    GlobalTensor<int32_t> mmOutGm;
    GlobalTensor<typename P::SCALE_TYPE> scaleGm;
    GlobalTensor<float> perTokenScaleGm;
    GlobalTensor<int64_t> groupTokensGm;
    GlobalTensor<float> logitsGm;
    GlobalTensor<bfloat16_t> residualGm;
    GlobalTensor<typename P::ROW_INDEX_DTYPE> tokenRanksGm;
    GlobalTensor<DTYPE_OUT> yGm;
    // define the que
    TQue<QuePosition::VECIN, 1> vecInQueue;
    TQue<QuePosition::VECOUT, 1> vecOutQueue;
    TQue<QuePosition::VECIN, 1> scaleInQueue;
    TQue<QuePosition::VECIN, 1> biasInQueue;
    TQue<QuePosition::VECIN, 1> perTokenScaleInQueue;
    TBuf<TPosition::VECCALC> tmpBuff;
    LocalTensor<typename P::SCALE_TYPE> scaleInUb;
    LocalTensor<bfloat16_t> biasInUb;
    LocalTensor<float> dequantMiddleResult;
    LocalTensor<uint8_t> sharedTmpLocal;
    LocalTensor<float> biasTmp;
    LocalTensor<float> mulsResultLocal;
    LocalTensor<float> pertokenBrcbLocal;
    LocalTensor<float> biasCalcLocal;
    LocalTensor<float> perTokenScaleInUb;
    uint32_t subBlockIdx;
    uint32_t coreIdx;
    uint32_t cubeCount = 0;
    uint32_t hasPertokenScale;
    uint32_t hasBias;
    TPipe *pipe;
    const typename P::TILING_TYPE *tiling;
};

template <class P>
__aicore__ inline void QuantGroupMatmul<P>::Init(const MMInitParams& initParams, const typename P::TILING_TYPE *tilingData,
                                                 TPipe *tPipeIn)
{
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(initParams.x));
    weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(initParams.weight));
    biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(initParams.bias));  // unused
    mmOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(initParams.workspace));
    scaleGm.SetGlobalBuffer(reinterpret_cast<__gm__  typename P::SCALE_TYPE*>(initParams.scale));
    perTokenScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.pertoken_scale));
    groupTokensGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(initParams.group_tokens));
    logitsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.logits));
    tokenRanksGm.SetGlobalBuffer(reinterpret_cast<__gm__ typename P::ROW_INDEX_DTYPE *>(initParams.token_ranks));
    residualGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(initParams.residual));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_OUT *>(initParams.y));
    tiling = tilingData;
    hasPertokenScale = tiling->hasPertokenScale;
    hasBias = tiling->hasBias;
    subBlockIdx = GetSubBlockIdx();
    coreIdx = GetBlockIdx();
    if ASCEND_IS_AIV {
        coreIdx /= GetTaskRation();
    }
    pipe = tPipeIn;
    InitUbBuffer();
}


template <class P>
__aicore__ inline void QuantGroupMatmul<P>::InitUbBuffer()
{
    if ASCEND_IS_AIC {
        return;
    }
    pipe->InitBuffer(scaleInQueue, BUFFER_NUM, tiling->matmulTiling.baseN * sizeof(float));
    pipe->InitBuffer(biasInQueue, BUFFER_NUM, tiling->matmulTiling.baseN * sizeof(bfloat16_t));
    
    if (P::combine && tiling->scatterAdd) {
        // 2: pertoken scale和logits般到一块buffer上
        uint32_t perTokenScalebufferNum = (hasPertokenScale != 0) ? 2 : 1;
        pipe->InitBuffer(perTokenScaleInQueue, BUFFER_NUM, Ceil(tiling->vBaseM * sizeof(float) * perTokenScalebufferNum, 32) * 32);
    } else {
        pipe->InitBuffer(perTokenScaleInQueue, BUFFER_NUM, Ceil(tiling->vBaseM * sizeof(float), 32) * 32);
    }
    pipe->InitBuffer(vecInQueue, BUFFER_NUM, tiling->ubCalSize * sizeof(cT::T));
    pipe->InitBuffer(vecOutQueue, BUFFER_NUM, tiling->ubCalSize * sizeof(DTYPE_OUT));
    pipe->InitBuffer(tmpBuff, tiling->ubRestBytes);
    uint32_t ubCalSizeFloat = tiling->ubCalSize * sizeof(float);
    // ub分配，依次划分中间结果
    dequantMiddleResult = tmpBuff.GetWithOffset<float>(tiling->ubCalSize, 0);
    pertokenBrcbLocal = tmpBuff.GetWithOffset<float>(tiling->ubCalSize, ubCalSizeFloat);
    // 2: 偏移两份，前面为反量化输出和pertoken scale
    mulsResultLocal = tmpBuff.GetWithOffset<float>(tiling->ubCalSize, 2 * ubCalSizeFloat);
    // 3: 再上面的基础上再偏移一份
    biasCalcLocal = tmpBuff.GetWithOffset<float>(tiling->ubCalSize, 3 * ubCalSizeFloat);
    // api需要的临时空间，复用中间结果的空间
    // 2: ub临时空间总共4份，高层api分配两份
    sharedTmpLocal = tmpBuff.GetWithOffset<uint8_t>(2 * ubCalSizeFloat, 4 * ubCalSizeFloat);
}

template <class P>
__aicore__ inline void QuantGroupMatmul<P>::InitOutputWithZeros(uint64_t offset, uint64_t size) {
    uint64_t singeCount = Ceil(size, uint32_t(GetBlockNum() * GetTaskRation()));
    singeCount = Ceil(singeCount, 512) * 512;
    uint64_t baseOffset = GetBlockIdx() * singeCount;
    if (baseOffset >= size) {
        return;
    }
    if (baseOffset + singeCount > size) {
        singeCount = size - baseOffset;
    }
    baseOffset += offset;
    InitOutput<DTYPE_OUT>(yGm[baseOffset], singeCount, 0);
}

template <class P>
__aicore__ inline void QuantGroupMatmul<P>::PreProcess() {
    if (tiling->sharedInputOffset > 0) {
        InitOutputWithZeros(0, tiling->n * tiling->sharedInputOffset);
    }
    uint64_t tail = tiling->sharedInputOffset + tiling->sharedInputLen;
    if (tail < tiling->batch) {
        InitOutputWithZeros(tail * tiling->n, tiling->n * (tiling->batch - tail));
    }
    uint64_t totalOutput = static_cast<uint64_t>(tiling->n) * tiling->sharedInputLen;
    uint64_t singeCount = Ceil(totalOutput, uint32_t(GetBlockNum() * GetTaskRation()));
    singeCount = Ceil(singeCount, tiling->ubCalSize) * tiling->ubCalSize;
    uint64_t baseOffset = GetBlockIdx() * singeCount;
    if (baseOffset >= totalOutput) {
        return;
    }
    if (baseOffset + singeCount > totalOutput) {
        singeCount = totalOutput - baseOffset;
    }
    uint64_t outOffset = baseOffset + static_cast<uint64_t>(tiling->n) * tiling->sharedInputOffset;
    uint64_t curCount = tiling->ubCalSize;
    DataCopyPadExtParams<bfloat16_t> padParams;
    for (uint32_t offset = 0; offset < singeCount; offset += curCount) {
        if (unlikely(offset + curCount > singeCount)) {
            curCount = singeCount - offset;
        }
        auto residualLocal = vecInQueue.AllocTensor<bfloat16_t>();
        // 32B对齐搬运可以简化参数，这里按不对齐处理
        DataCopyExtParams params{1, static_cast<uint32_t>(curCount * sizeof(bfloat16_t)), 1, 1, 0};
        DataCopyPad(residualLocal, residualGm[baseOffset + offset], params, padParams);
        vecInQueue.EnQue(residualLocal);
        residualLocal = vecInQueue.DeQue<bfloat16_t>();

        Cast(dequantMiddleResult, residualLocal, AscendC::RoundMode::CAST_NONE, curCount);
        vecInQueue.FreeTensor(residualLocal);
        PipeBarrier<PIPE_V>();
        LocalTensor<DTYPE_OUT> yLocal = vecOutQueue.AllocTensor<DTYPE_OUT>();
        Muls(yLocal, dequantMiddleResult, tiling->residualScale, curCount);
        vecOutQueue.EnQue(yLocal);

        DataCopyExtParams paramsOut{1, static_cast<uint32_t>(curCount * sizeof(float)), 1, 1, 0};
        yLocal = vecOutQueue.DeQue<DTYPE_OUT>();
        DataCopyPad(yGm[outOffset + offset], yLocal, paramsOut);
        vecOutQueue.FreeTensor(yLocal);
    }
}

template <class P>
__aicore__ inline void QuantGroupMatmul<P>::Process()
{
    if ASCEND_IS_AIV {
        if (P::combine && tiling->scatterAdd) {
            PreProcess();
            SyncAll();
        }
    }
    MNConfig mnConfig;
    mnConfig.baseM = tiling->matmulTiling.baseM;
    mnConfig.baseN = tiling->matmulTiling.baseN;
    mnConfig.singleM = mnConfig.baseM;
    mnConfig.singleN = mnConfig.baseN;
    mnConfig.blockDimN = Ceil(tiling->n, mnConfig.singleN);
    for (uint32_t groupIdx = 0, preCount = 0; groupIdx < tiling->groupNum; ++groupIdx) {
        uint32_t m = static_cast<uint32_t>(groupTokensGm.GetValue(groupIdx));
        if (m <= 0) {
            continue;
        }
        mnConfig.m = static_cast<uint32_t>(m);
        mnConfig.blockDimM = Ceil(mnConfig.m, mnConfig.singleM);
        uint32_t curCount = preCount + mnConfig.blockDimN * mnConfig.blockDimM;
        uint32_t curBlock = coreIdx >= preCount ? coreIdx : coreIdx + tiling->coreNum;

        while (curBlock < curCount) {
            mnConfig.mIdx = (curBlock - preCount) / mnConfig.blockDimN;
            mnConfig.nIdx = (curBlock - preCount) % mnConfig.blockDimN;
            MMCompute(groupIdx, mnConfig);
            VectorCompute(groupIdx, mnConfig);
            curBlock += tiling->coreNum;
        }
        preCount = curCount % tiling->coreNum;
        mnConfig.offsetM += mnConfig.m;
    }
}

template <class P>
__aicore__ inline void QuantGroupMatmul<P>::MMCompute(uint32_t groupIdx, MNConfig& mnConfig)
{
    uint32_t tailN = mnConfig.nIdx * mnConfig.singleN;
    uint32_t curSingleN = mnConfig.singleN;
    if (mnConfig.nIdx == mnConfig.blockDimN - 1) {
        curSingleN = tiling->n - tailN;
    }
    uint32_t curSingleM = mnConfig.singleM;
    if (mnConfig.mIdx == mnConfig.blockDimM - 1) {
        curSingleM = mnConfig.m - mnConfig.mIdx * mnConfig.singleM;
    }
    uint64_t xOffset = (static_cast<uint64_t>(mnConfig.offsetM) + mnConfig.mIdx * mnConfig.singleM) * tiling->k;
    uint64_t weightOffset = static_cast<uint64_t>(groupIdx) * tiling->n * tiling->k + tailN * tiling->k;  // for no transpose nz weight
    mnConfig.workSpaceOffset =
        mnConfig.singleN * mnConfig.singleM * (coreIdx + (cubeCount % tiling->parallNum) * tiling->coreNum);
    if ASCEND_IS_AIC {
        if (cubeCount >= tiling->parallNum) {
            CrossCoreWaitFlag(SYNC_AIV_TO_AIC);
        }
        mm.SetOrgShape(mnConfig.m, tiling->n, tiling->k);
        mm.SetSingleShape(curSingleM, curSingleN, tiling->k);
        mm.SetTensorA(xGm[xOffset]);
        auto weightSlice = weightGm[weightOffset];
        if (mnConfig.blockDimM == 1) {
            weightSlice.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        mm.SetTensorB(weightSlice);
        uint64_t worskspaceOffset = mnConfig.workSpaceOffset;
        while (mm.Iterate()) {
            mm.GetTensorC(mmOutGm[worskspaceOffset], 0, true);
            CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_TO_AIV);  // 2: mode为2, group内同步
            worskspaceOffset += (mnConfig.baseM * mnConfig.baseN);
        }
    }
    cubeCount++;
}

template <class P>
__aicore__ inline void QuantGroupMatmul<P>::VectorAtomicProcess(const VectorAtomicParams& vecAParams)
{
    LocalTensor<DTYPE_OUT> yLocal = vecOutQueue.DeQue<DTYPE_OUT>();
    if constexpr (P::combine) {
        if (tiling->scatterAdd) {
            SetAtomicAdd<float>();
        }

        DataCopyExtParams paramsOut{1, static_cast<uint32_t>(vecAParams.curVecBaseN * sizeof(float)), 1, 1, 0};
        for (uint32_t i = 0; i < vecAParams.curVecBaseM; i++) {
                auto outRow = static_cast<uint64_t>(
                    tokenRanksGm.GetValue(vecAParams.mGlobalOffset + vecAParams.offsetM + i));
                DataCopyPad(yGm[outRow * tiling->n + vecAParams.yGmOffset0],
                            yLocal[i * vecAParams.alignBaseN], paramsOut);
        }

        if (tiling->scatterAdd) {
            SetAtomicNone();
        }
    } else {
        DataCopy2DDimParams dimParams{vecAParams.curVecBaseM, vecAParams.curVecBaseN, vecAParams.alignBaseN};
        DataCopyPad2D(yGm[vecAParams.yGmOffset1], yLocal, dimParams, tiling->n);
    }
    vecOutQueue.FreeTensor(yLocal);
}

template <class P>
__aicore__ inline void QuantGroupMatmul<P>::VectorCompute(uint32_t groupIdx, MNConfig& mnConfig)
{
    if ASCEND_IS_AIC {
        return;
    }
    uint32_t curCubeSingleN = (mnConfig.nIdx == mnConfig.blockDimN - 1)
                               ? (tiling->n - mnConfig.nIdx * mnConfig.singleN) : mnConfig.singleN;
    uint32_t curCubeSingleM = (mnConfig.mIdx == mnConfig.blockDimM - 1)
                               ? (mnConfig.m - mnConfig.mIdx * mnConfig.singleM) : mnConfig.singleM;
    uint64_t mGlobalOffset = mnConfig.offsetM + mnConfig.mIdx * mnConfig.singleM;
    uint64_t outOffset = mGlobalOffset * tiling->n + mnConfig.nIdx * mnConfig.singleN;
    uint32_t vecBaseM = tiling->ubCalSize / (Ceil(mnConfig.baseN, uint32_t(8)) * 8);
    vecBaseM = vecBaseM  < curCubeSingleM ? vecBaseM : curCubeSingleM;  //  8: num int32_t in 32B ub block
    uint32_t curVecBaseN = mnConfig.baseN;
    uint64_t scaleOffset = groupIdx * tiling->n + mnConfig.nIdx * mnConfig.singleN;
    uint32_t taskRation = GetTaskRation();
    for (uint32_t offsetN = 0, vecCount = 0; offsetN < curCubeSingleN; offsetN += mnConfig.baseN) {
        if (unlikely(offsetN + mnConfig.baseN >= curCubeSingleN)) {
            curVecBaseN = curCubeSingleN - offsetN;
        }
        uint32_t alignBaseN = Ceil(curVecBaseN, uint32_t(8)) * 8;  //  8: num int32_t in 32B ub block
        uint32_t curVecBaseM = vecBaseM;
        DataCopyScale(curVecBaseN, alignBaseN, scaleOffset + offsetN);
        if (hasBias) {
            DataCopyBias(curVecBaseN, alignBaseN, scaleOffset + offsetN);
        }
        uint64_t mmOutOffset = mnConfig.workSpaceOffset + offsetN * mnConfig.baseM;
        CrossCoreWaitFlag(SYNC_AIC_TO_AIV);
        for (uint32_t offsetM = 0; offsetM < curCubeSingleM; offsetM += vecBaseM, vecCount++) {
            if (taskRation != 0 && vecCount % taskRation != subBlockIdx) {
                continue;
            }
            if (unlikely(offsetM + vecBaseM >= curCubeSingleM)) {
                curVecBaseM = curCubeSingleM - offsetM;
            }
            // 使用AscendDequant接口做perchannel反量化
            LocalTensor<cT::T> mmOutLocal = vecInQueue.AllocTensor<cT::T>();
            DataCopy2DDimParams dimParams{curVecBaseM, curVecBaseN, curVecBaseN};
            DataCopyPad2D(mmOutLocal, mmOutGm[mmOutOffset + offsetM * curVecBaseN], dimParams);
            vecInQueue.EnQue(mmOutLocal);
            VectorAtomicParams vecAParams{curVecBaseM, curVecBaseN, alignBaseN, offsetM, mGlobalOffset,
                mnConfig.nIdx * mnConfig.singleN + offsetN, outOffset + offsetM * tiling->n + offsetN};
            ComputeDequantAndActivate(mnConfig, vecAParams);
            VectorAtomicProcess(vecAParams);
        }
        if constexpr (std::is_same_v<typename P::SCALE_TYPE, float>) {
            scaleInQueue.FreeTensor(scaleInUb);
        }
    }
    CrossCoreSetFlag<2, PIPE_MTE2>(SYNC_AIV_TO_AIC);  // 2: mode为2, group内同步
}

template <class P>
__aicore__ inline void QuantGroupMatmul<P>::ComputeDequantProcess(uint32_t computeSize)
{
    if (hasBias) {
        Add(mulsResultLocal, dequantMiddleResult, biasCalcLocal, computeSize);
    }

    // 2: 分配两份大小给高层api作为临时空间
    LocalTensor<DTYPE_OUT> yLocalInUb = vecOutQueue.AllocTensor<DTYPE_OUT>();
    // Cast后获得最终输出
    Cast(yLocalInUb, mulsResultLocal, RoundMode::CAST_RINT, computeSize);
    PipeBarrier<PIPE_V>();
    vecOutQueue.EnQue(yLocalInUb);
}

template <class P>
__aicore__ inline void QuantGroupMatmul<P>::ComputeDequantAndActivate(MNConfig& mnConfig, const VectorAtomicParams& vecAParams)
{
    DataCopyPerTokenScaleAndBrcb(mnConfig, vecAParams.curVecBaseM, vecAParams.alignBaseN, vecAParams.offsetM);
    LocalTensor<int32_t> mmOutInUb = vecInQueue.DeQue<cT::T>();

    LocalTensor<float> scaleBuf;
    if constexpr (std::is_same_v<typename P::SCALE_TYPE, bfloat16_t>) {
        scaleBuf = mulsResultLocal[vecAParams.alignBaseN];
    } else {
        scaleBuf = scaleInUb;
    }

    AscendDequant(dequantMiddleResult, mmOutInUb, scaleBuf, sharedTmpLocal,
                  {vecAParams.curVecBaseM, vecAParams.alignBaseN, vecAParams.curVecBaseN});
    PipeBarrier<PIPE_V>();
    vecInQueue.FreeTensor(mmOutInUb);
    
    uint32_t computeSize = vecAParams.curVecBaseM * vecAParams.alignBaseN;
    
    // pertoken反量化
    if constexpr (P::combine) {
        LocalTensor<DTYPE_OUT> yLocalInUb = vecOutQueue.AllocTensor<DTYPE_OUT>();
        if (hasBias) {
            Mul(dequantMiddleResult, dequantMiddleResult, pertokenBrcbLocal, computeSize);
            PipeBarrier<PIPE_V>();
            Add(yLocalInUb, dequantMiddleResult, biasCalcLocal, computeSize);
            PipeBarrier<PIPE_V>();
        } else {
            Mul(yLocalInUb, dequantMiddleResult, pertokenBrcbLocal, computeSize);
        }

        vecOutQueue.EnQue(yLocalInUb);
        return;
    }
    Mul(mulsResultLocal, dequantMiddleResult, pertokenBrcbLocal, computeSize);
    PipeBarrier<PIPE_V>();
    ComputeDequantProcess(computeSize);
}

template <class P>
__aicore__ inline void QuantGroupMatmul<P>::DataCopyScale(uint32_t curBaseN, uint32_t alignBaseN, uint64_t scaleOffset)
{
    // GM拷贝scale
    DataCopyPadExtParams<typename P::SCALE_TYPE> padParams;
    DataCopyExtParams scaleParams{1, static_cast<uint32_t>(curBaseN * sizeof(typename P::SCALE_TYPE)), 1, 1, 0};
    LocalTensor<typename P::SCALE_TYPE> scaleLocal = scaleInQueue.AllocTensor<typename P::SCALE_TYPE>();
    DataCopyPad(scaleLocal, scaleGm[scaleOffset], scaleParams, padParams);
    scaleInQueue.EnQue(scaleLocal);

    scaleInUb = scaleInQueue.DeQue<typename P::SCALE_TYPE>();
    scaleInUb.SetSize(alignBaseN);

    if constexpr (std::is_same_v<typename P::SCALE_TYPE, bfloat16_t>) {
        Cast(mulsResultLocal[alignBaseN], scaleInUb, AscendC::RoundMode::CAST_NONE, alignBaseN);
        scaleInQueue.FreeTensor(scaleInUb);
    } 
}

template <class P>
__aicore__ inline void QuantGroupMatmul<P>::DataCopyBias(uint32_t curBaseN, uint32_t alignBaseN, uint64_t offset)
{
    // GM拷贝scale
    DataCopyPadExtParams<bfloat16_t> padParams;
    DataCopyExtParams biasParams{1, static_cast<uint32_t>(curBaseN * sizeof(bfloat16_t)), 1, 1, 0};
    LocalTensor<bfloat16_t> biasLocal = biasInQueue.AllocTensor<bfloat16_t>();
    DataCopyPad(biasLocal, biasGm[offset], biasParams, padParams);
    biasInQueue.EnQue(biasLocal);

    biasInUb = biasInQueue.DeQue<bfloat16_t>();
    biasInUb.SetSize(alignBaseN);

    Cast(mulsResultLocal, biasInUb, AscendC::RoundMode::CAST_NONE, alignBaseN);
    PipeBarrier<PIPE_V>();
    biasInQueue.FreeTensor(biasInUb);
}

template <class P>
__aicore__ inline void QuantGroupMatmul<P>::DataCopyPerTokenScaleAndBrcb(MNConfig& mnConfig,
        uint32_t curBaseM, uint32_t alignBaseN, uint32_t offsetM)
{
    uint64_t vecBaseMOffset = mnConfig.offsetM + mnConfig.mIdx * mnConfig.singleM + offsetM;
    uint32_t alignBaseM = (hasPertokenScale != 0) ? (Ceil(curBaseM, uint32_t(8)) * 8) : 0;  //  8: num int32_t in 32B ub block
    DataCopyPadExtParams<float> padParams;
    DataCopyExtParams perTokenScaleParams{1, static_cast<uint32_t>(curBaseM * sizeof(float)), 0, 0, 0};
    LocalTensor<float> perTokenScaleLocal = perTokenScaleInQueue.AllocTensor<float>();
    if (hasPertokenScale != 0) {
        // GM拷贝per token scale
        DataCopyPad(perTokenScaleLocal, perTokenScaleGm[vecBaseMOffset], perTokenScaleParams, padParams);
    }

    if (P::combine && tiling->scatterAdd) {
        DataCopyPad(perTokenScaleLocal[alignBaseM], logitsGm[vecBaseMOffset], perTokenScaleParams, padParams);
    }
    perTokenScaleInQueue.EnQue(perTokenScaleLocal);

    perTokenScaleInUb = perTokenScaleInQueue.DeQue<float>();
    auto scaleTmp = perTokenScaleInUb;

    if (P::combine && tiling->scatterAdd) {
        if (hasPertokenScale) {
            Mul(dequantMiddleResult, perTokenScaleInUb, perTokenScaleInUb[alignBaseM], curBaseM);
            scaleTmp = dequantMiddleResult;
        } else {
            scaleTmp = perTokenScaleInUb[alignBaseM];
        }
        PipeBarrier<PIPE_V>();
    }

    const uint32_t broadCastDst[BROADCAST_DIM] = {curBaseM, alignBaseN};
    const uint32_t broadCastSrc[BROADCAST_DIM] = {curBaseM, 1};
    BroadCast<float, BROADCAST_DIM, 1>(pertokenBrcbLocal, scaleTmp, broadCastDst, broadCastSrc, sharedTmpLocal);

    if (hasBias) {
        BroadCast<float, BROADCAST_DIM, 1>(biasCalcLocal, perTokenScaleLocal[alignBaseM], broadCastDst, broadCastSrc, sharedTmpLocal);
        for (int i = 0; i < curBaseM; i++) {
            Mul(biasCalcLocal[alignBaseN * i], biasCalcLocal[alignBaseN * i], mulsResultLocal, alignBaseN);
        }
    }

    perTokenScaleInQueue.FreeTensor(perTokenScaleInUb);
}
}  // namespace GroupedMatmulFinalizeRouting
#endif