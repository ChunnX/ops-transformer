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
 * \file matmul_all_reduce_weight_quant_310.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_WEIGHT_QUANT_310_H
#define MATMUL_ALL_REDUCE_WEIGHT_QUANT_310_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#ifdef __CCE_KT_TEST__
#include "rac_server_stub.h"
#else
#include "rac_server.h"
#endif
#include "../common.h"
#include "mm_allreduce.h"
#include "../../3rd/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2_weight_nz_performance.h"
#include "../arch22/weight_quant_matmul_all_reduce_tiling_data.h"

namespace MatmulAllReduceImpl {
using namespace AscendC;
using namespace Mc2WeightQuantBatchMatmulV2;
template <
    typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
    Mc2QuantType antiQuantType, bool hasAntiQuantOffset>
class MatmulAllReduceWeightQuant310
{
public:
    __aicore__ inline MatmulAllReduceWeightQuant310()
    {}
    __aicore__ inline void Init(
        GM_ADDR aGM, GM_ADDR bGM, GM_ADDR antiquantScaleGM, GM_ADDR antiquantOffsetGM, GM_ADDR biasGM, GM_ADDR cGM,
        GM_ADDR workspaceGM, Mc2Tiling::WeightQuantMatmulAllReduceNzTilingData* tilDat, TPipe* tPipe,
        HcclServer* hcclSrv);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InnerProcess(
        uint32_t tileLoopCnt, Mc2WeightQuantBatchMatmulV2NzTilingData* mmTiling, uint32_t turnShift, int32_t coreNum);
    Mc2Tiling::WeightQuantMatmulAllReduceNzTilingData* td_;
    HcclServer* hccl_;
    TPipe* tp_;
    GM_ADDR aPtr_;
    GM_ADDR bPtr_;
    GM_ADDR biasPtr_;
    GM_ADDR aqScalePtr_;
    GM_ADDR aqOffPtr_;
    GM_ADDR wsPtr_;
    GM_ADDR cPtr_;
    bool notifyFl_{false};
    int32_t coreCnt_{0};
};

template <
    typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
    Mc2QuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void
MatmulAllReduceWeightQuant310<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset>::
    InnerProcess(uint32_t tileLoopCnt, Mc2WeightQuantBatchMatmulV2NzTilingData* mmTiling, uint32_t turnShift, int32_t coreNum)
{
    const uint64_t aOff = CalcShapeOffset(sizeof(xType), mmTiling->mSize, mmTiling->kSize);
    const uint64_t cOff = CalcShapeOffset(sizeof(yType), mmTiling->mSize, mmTiling->nSize);
    if (GetBlockIdx() < coreNum) {
        Mc2WeightQuantBatchMatmulV2WeightNzPerformanceKernel<
            xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset>
            op;
        for (uint32_t i = 1U; i <= tileLoopCnt; ++i) {
            if (i == 1U) {
                tp_->Reset();
                op.Init(
                    aPtr_, bPtr_, aqScalePtr_, aqOffPtr_, nullptr, nullptr, biasPtr_, cPtr_, wsPtr_,
                    mmTiling, tp_);
            } else {
                op.UpdateGlobalAddr(
                    aPtr_, bPtr_, aqScalePtr_, aqOffPtr_, nullptr, nullptr, biasPtr_, cPtr_, wsPtr_);
            }
            op.Process();
            hccl_->TurnNotifyRun(block_idx, coreNum, i + turnShift);
            aPtr_ += aOff;
            cPtr_ += cOff;
        }
    } else {
        for (uint32_t i = 1U; i <= tileLoopCnt; ++i) {
            aPtr_ += aOff;
            cPtr_ += cOff;
        }
    }
}

template <
    typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
    Mc2QuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void
MatmulAllReduceWeightQuant310<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR antiquantScaleGM, GM_ADDR antiquantOffsetGM, GM_ADDR biasGM, GM_ADDR cGM,
    GM_ADDR workspaceGM, Mc2Tiling::WeightQuantMatmulAllReduceNzTilingData* tilDat, TPipe* tPipe, HcclServer* hcclSrv)
{
    __gm__ HcclCombinOpParam* context = (__gm__ HcclCombinOpParam*)(GetHcclContext<0>());
    __gm__ uint8_t* wsMsg = (__gm__ uint8_t*)(context->WorkSpace + tilDat->msg.notifyOff);
    TBuf<TPosition::VECCALC> tmpBuf;
    tPipe->InitBuffer(tmpBuf, 256);
    auto&& cfg = tilDat->param;
    auto&& tiling = tilDat->tilematmulTiling.matmulTiling;
    hcclSrv->Init(wsMsg, (tilDat->msg).debugMode, tmpBuf);
    workspaceGM += cfg.nd2NzWorkLen;
    workspaceGM += cfg.biasLen;
    GM_ADDR syncPtr = workspaceGM;
    int32_t tileNum = tilDat->tilematmulTiling.cubeBlockDimN * tilDat->tilematmulTiling.cubeBlockDimM;
    int32_t tailNum = tilDat->tailmatmulTiling.cubeBlockDimN * tilDat->tailmatmulTiling.cubeBlockDimM;
    coreCnt_ = tileNum > tailNum ? tileNum : tailNum;
    hcclSrv->InitSoftSync(syncPtr, coreCnt_, tmpBuf);
    workspaceGM += AC_MAX_AIV * DEFAULT_BLK_NUM;

    if (tilDat->msg.useBufferType == MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_WINDOW_IN &&
        context->config.determinism != 1) {
        cPtr_ = (__gm__ uint8_t*)(context->windowsIn[context->rankId]);
    } else {
        cPtr_ = cGM;
    }
    td_ = tilDat;
    hccl_ = hcclSrv;
    tp_ = tPipe;
    aPtr_ = aGM;
    bPtr_ = bGM;
    biasPtr_ = biasGM;
    aqScalePtr_ = antiquantScaleGM;
    aqOffPtr_ = antiquantOffsetGM;
    wsPtr_ = workspaceGM;
    if (GetBlockIdx() == 0 && (g_coreType == AIC || g_coreType == MIX)) {
        notifyFl_ = true;
    }
}

template <
    typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
    Mc2QuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void MatmulAllReduceWeightQuant310<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset>::Process()
{
    auto&& mc2Tiling = td_->param;
    int32_t tileNum = td_->tilematmulTiling.cubeBlockDimN * td_->tilematmulTiling.cubeBlockDimM;
    int32_t tailNum = td_->tailmatmulTiling.cubeBlockDimN * td_->tailmatmulTiling.cubeBlockDimM;
    InnerProcess(mc2Tiling.tileLoopCnt, &td_->tilematmulTiling, 0U, tileNum);
    if (mc2Tiling.tailM != 0U) {
        InnerProcess(mc2Tiling.tailCnt, &td_->tailmatmulTiling, mc2Tiling.tileLoopCnt, tailNum);
    }
    if (notifyFl_) {
        hccl_->TurnWait(mc2Tiling.tileLoopCnt + mc2Tiling.tailCnt);
    }
}

__aicore__ inline void MatMulEmptyTensorBrcBias(GM_ADDR biasGM, GM_ADDR cGM,
    Mc2Tiling::WeightQuantMatmulAllReduceNzTilingData* tilDat, TBuf<TPosition::VECCALC>& tmpBuf)
{
    // 搬运biase对齐部分
    int32_t cSzHalf = (tilDat->param.rankN * tilDat->param.rankM) * sizeof(DTYPE_X1) / sizeof(DTYPE_Y);
    GlobalTensor<DTYPE_Y> cGlHalf;
    cGlHalf.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_Y*>(cGM), cSzHalf);
    LocalTensor<DTYPE_Y> bias = tmpBuf.Get<DTYPE_Y>();
    GlobalTensor<DTYPE_Y> biasGl;
    biasGl.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_Y*>(biasGM));
    TBuffAddr bfAddr;
    bfAddr.logicPos = (uint8_t)QuePosition::VECCALC;
    uint32_t elemCnt = 32 / sizeof(DTYPE_Y);
    uint32_t alignSz = (tilDat->param.rankN / elemCnt) * elemCnt;
    if (alignSz > 0) {
        DataCopy(bias, biasGl, alignSz);
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(eventID);
        WaitFlag<HardEvent::MTE2_MTE3>(eventID);
        for (uint32_t idx = 0; idx < tilDat->param.rankM; ++i) {
            uint32_t offDst = i * tilDat->param.rankN;
            DataCopy(cGlHalf[offDst], bias, alignSz);
        }
    }
    if (tilDat->param.rankN % elemCnt) {
        // 搬运biase非对齐部分
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventID);
        WaitFlag<HardEvent::MTE3_MTE2>(eventID);
        DataCopy(bias, biasGl[tilDat->param.rankN - elemCnt], elemCnt);
        eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(eventID);
        WaitFlag<HardEvent::MTE2_MTE3>(eventID);
        for (uint32_t idx = 0; idx < tilDat->param.rankM; ++i) {
            uint32_t offDst = (i + 1) * tilDat->param.rankN - elemCnt;
            DataCopy(cGlHalf[offDst], bias, elemCnt);
        }
    }
}

__aicore__ inline void WeightQuantEmptyTensorKernel(GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR workspaceGM,
    Mc2Tiling::WeightQuantMatmulAllReduceNzTilingData* tilDat, HcclServer* hcclSrv)
{
    TBuf<TPosition::VECCALC> tmpBuf;
    GetTPipePtr()->InitBuffer(tmpBuf, TOTAL_UB_SIZE);
    // 初始化hccl
    __gm__ HcclCombinOpParam* context = (__gm__ HcclCombinOpParam*)(GetHcclContext<0>());
    __gm__ uint8_t* wsMsg = (__gm__ uint8_t*)(context->WorkSpace + (tilDat->msg).notifyOff);
    hcclSrv->Init(wsMsg, (tilDat->msg).debugMode, tmpBuf);
    if (tilDat->tilematmulTiling.matmulTiling.usedCoreNum > 1) {
        hcclSrv->InitSoftSync(workspaceGM, tilDat->tilematmulTiling.matmulTiling.usedCoreNum, tmpBuf);
    }
    // 初始化输出tensor
    int32_t cSz = (tilDat->param.rankN * tilDat->param.rankM) * sizeof(DTYPE_X1) / sizeof(int32_t);
    GlobalTensor<int32_t> cGl;
    cGl.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(cGM), cSz);
    if (block_idx == 0) {
        InitOutput<int32_t>(cGl, cSz, (int32_t)0);
        if (tilDat->tilematmulTiling.matmulTiling.isBias) {
            MatMulEmptyTensorBrcBias(biasGM, cGM, tilDat, tmpBuf);
        }
    }
    // 通知AICPU计算结束
    if (block_idx == 0) {
        hcclSrv->TurnNotifyRun(block_idx, tilDat->tilematmulTiling.matmulTiling.usedCoreNum, 1);
        // 通过一个核轮询并清除数据，防止多核之间写后读依赖
        hcclSrv->TurnWait(tilDat->param.tileLoopCnt + tilDat->param.tailCnt);
    }
}

#define INVOKE_WEIGHT_QUANT_BMM_OP_IMPL_310(templateClass, ...)                                                        \
    do {                                                                                                               \
        GET_TILING_DATA_MEMBER(Mc2Tiling::WeightQuantMatmulAllReduceNzTilingData, msg, msg, tilingGM);                            \
        if (msg.debugMode != static_cast<uint8_t>(DebugMode::MC2_DEBUG_ONLY_AICPU)) {                                  \
            GET_TILING_DATA_WITH_STRUCT(Mc2Tiling::WeightQuantMatmulAllReduceNzTilingData, tilDat, tilingGM);                 \
            templateClass<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_Y, __VA_ARGS__> op;                                       \
            op.Init(                                                                                                   \
                aGM, bGM, antiquantScaleGM, antiquantOffsetGM, biasGM, cGM, userWS, &tilDat, &tPipe, &hcclSrv); \
            op.Process();                                                                                              \
        }                                                                                                              \
    } while (0)
} // namespace MatmulAllReduceImpl
#endif // MATMUL_ALL_REDUCE_WEIGHT_QUANT_310_H