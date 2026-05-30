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
 * \file matmul_all_reduce_quant_bmm.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_QUANT_BMM_H
#define MATMUL_ALL_REDUCE_QUANT_BMM_H

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
#include "../../3rd/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3.h"
#include "../arch22/quant_matmul_all_reduce_tiling_data.h"

namespace MatmulAllReduceImpl {
using namespace AscendC;
using namespace DequantBmm;
template <typename aType, typename bType, typename biasType, typename cType, bool aTrans, bool bTrans>
class MatmulAllReduceQuantBmm
{
public:
    __aicore__ inline MatmulAllReduceQuantBmm()
    {}
    __aicore__ inline void Init(
        GM_ADDR aGM, GM_ADDR bGM, GM_ADDR dequantGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR workspaceGM,
        Mc2Tiling::QuantMatmulAllReduceTilingData* tilDat, TPipe* tPipe, HcclServer* hcclSrv);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InnerProcess(uint32_t tileLoopCnt, Mc2QuantBatchMatmulV3TilingData& quant_tiling, uint32_t turnShift);

private:
    Mc2Tiling::QuantMatmulAllReduceTilingData* tilDat_;
    HcclServer* hcclSrv_;
    TPipe* tPipe_;
    GM_ADDR cGM_;
    GM_ADDR aGM_;
    GM_ADDR bGM_;
    GM_ADDR biasGM_;
    GM_ADDR dequantGM_;
    GM_ADDR workspaceGM_;
    TBuf<TPosition::VECCALC> tempBuf2_;
};

template <typename aType, typename bType, typename biasType, typename cType, bool aTrans, bool bTrans>
__aicore__ inline void MatmulAllReduceQuantBmm<aType, bType, biasType, cType, aTrans, bTrans>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR dequantGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR workspaceGM,
    Mc2Tiling::QuantMatmulAllReduceTilingData* tilDat, TPipe* tPipe, HcclServer* hcclSrv)
{
    GetTPipePtr()->InitBuffer(tempBuf2_, TOTAL_UB_SIZE);
    auto&& cfg = tilDat->param;

    __gm__ HcclCombinOpParam* context = (__gm__ HcclCombinOpParam*)(GetHcclContext<0>());
    __gm__ uint8_t* wsMsg2 = (__gm__ uint8_t*)(context->WorkSpace + (tilDat->msg).notifyOff);
    hcclSrv->Init(wsMsg2, (tilDat->msg).debugMode, tempBuf2_);

    workspaceGM += cfg.nd2NzWorkLen;
    workspaceGM += cfg.biasLen;
    GM_ADDR syncPtr2 = workspaceGM;
    hcclSrv->InitSoftSync(syncPtr2, GetBlockNum(), tempBuf2_);
    workspaceGM += AC_MAX_AIV * DEFAULT_BLK_NUM;

    tilDat_ = tilDat;
    hcclSrv_ = hcclSrv;
    tPipe_ = tPipe;
    aGM_ = aGM;
    bGM_ = bGM;
    cGM_ = cGM;
    biasGM_ = biasGM;
    dequantGM_ = dequantGM;
    workspaceGM_ = workspaceGM;
}

template <typename aType, typename bType, typename biasType, typename cType, bool aTrans, bool bTrans>
__aicore__ inline void MatmulAllReduceQuantBmm<aType, bType, biasType, cType, aTrans, bTrans>::Process()
{
    auto&& cfg = tilDat_->param;
    if (g_coreType == AIV) {
        return;
    }

    InnerProcess(cfg.tileLoopCnt, tilDat_->tilematmulTiling, 0);
    if (cfg.tailM != 0U) {
        InnerProcess(cfg.tailCnt, tilDat_->tailmatmulTiling, cfg.tileLoopCnt);
    }

    if (GetBlockIdx() == 0 && (g_coreType == AIC || g_coreType == MIX)) {
        hcclSrv_->TurnWait(cfg.tileLoopCnt + cfg.tailCnt);
    }
}

template <typename aType, typename bType, typename biasType, typename cType, bool aTrans, bool bTrans>
__aicore__ inline void MatmulAllReduceQuantBmm<aType, bType, biasType, cType, aTrans, bTrans>::InnerProcess(
    uint32_t tileLoopCnt, Mc2QuantBatchMatmulV3TilingData& quant_tiling, uint32_t turnShift)
{
    const uint64_t aOff = CalcShapeOffset(sizeof(aType), quant_tiling.matmulTiling.M, quant_tiling.matmulTiling.Ka);
    const uint64_t cOff = CalcShapeOffset(sizeof(cType), quant_tiling.matmulTiling.M, quant_tiling.matmulTiling.N);
    BmmDequant<aType, bType, FORMAT_X1, FORMAT_X2, biasType, uint64_t, cType, aTrans, bTrans> op;
    for (uint32_t idx = 0; idx < tileLoopCnt; i++) {
        if (i == 0) {
            tPipe_->Reset();
            op.InitTbuf(tempBuf2_);
            op.Init(aGM_, bGM_, biasGM_, dequantGM_, cGM_, workspaceGM_, &quant_tiling, tPipe_);
        } else {
            op.UpdateGlobalAddr(aGM_, bGM_, biasGM_, dequantGM_, cGM_, workspaceGM_);
        }

        op.Process();
        hcclSrv_->TurnNotifyRun(block_idx, GetBlockNum(), i + 1 + turnShift);
        aGM_ += aOff;
        cGM_ += cOff;
    }
}

#define INVOKE_QUANT_BMM_OP_IMPL(templateClass, ...)                                              \
    do {                                                                                          \
        GET_TILING_DATA_WITH_STRUCT(Mc2Tiling::QuantMatmulAllReduceTilingData, tilDat, tilingGM);        \
        templateClass<DTYPE_X1, DTYPE_X2, int32_t, DTYPE_Y, __VA_ARGS__> op;                      \
        op.Init(aGM, bGM, dequantGM, biasGM, cGM, workspaceGM, &tilDat, &tPipe, &hcclSrv); \
        op.Process();                                                                             \
    } while (0)
} // namespace MatmulAllReduceImpl
#endif // MATMUL_ALL_REDUCE_QUANT_BMM_H