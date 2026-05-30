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
 * \file matmul_all_reduce_unquant_310.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_UNQUANT_310_H
#define MATMUL_ALL_REDUCE_UNQUANT_310_H

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
#include "../matmul_all_reduce_tiling_key.h"
#include "../../3rd/mat_mul_v3/op_kernel/mat_mul_base_kernel.h"
#include "../../3rd/mat_mul_v3/op_kernel/mat_mul_unaligned_base_kernel.h"
#include "../arch22/unquant_matmul_all_reduce_tiling_data.h"

namespace MatmulAllReduceImpl {
using namespace AscendC;
template <typename aType, typename bType, typename biasType, typename cType, bool mixNdNz>
class MatmulAllReduceUnquant310
{
public:
    __aicore__ inline MatmulAllReduceUnquant310()
    {}
    __aicore__ inline void Init(
        GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR workspaceGM,
        Mc2Tiling::UnQuantMatmulAllReduceTilingData* tiling, TPipe* tPipe, HcclServer* hcclSrv);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InnerProcess(uint32_t tileLoopCnt, Mc2MatmulV3TilingData& mm_tiling, uint32_t turnShift);

private:
    Mc2Tiling::UnQuantMatmulAllReduceTilingData* tiling_;
    HcclServer* hcclSrv_;
    TPipe* tPipe_;
    GM_ADDR cGM_;
    GM_ADDR aGM_;
    GM_ADDR bGM_;
    GM_ADDR biasGM_;
    GM_ADDR workspaceGM_;
    TBuf<TPosition::VECCALC> tempBuf3_;
};

template <typename aType, typename bType, typename biasType, typename cType, bool mixNdNz>
__aicore__ inline void MatmulAllReduceUnquant310<aType, bType, biasType, cType, mixNdNz>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR workspaceGM,
    Mc2Tiling::UnQuantMatmulAllReduceTilingData* tiling, TPipe* tPipe, HcclServer* hcclSrv)
{
    GetTPipePtr()->InitBuffer(tempBuf3_, TOTAL_UB_SIZE);
    auto&& cfg = tiling->param;

    __gm__ HcclCombinOpParam* context = (__gm__ HcclCombinOpParam*)(GetHcclContext<0>());
    __gm__ uint8_t* wsMsg3 = (__gm__ uint8_t*)(context->WorkSpace + (tiling->msg).notifyOff);
    hcclSrv->Init(wsMsg3, (tiling->msg).debugMode, tempBuf3_);

    workspaceGM += cfg.nd2NzWorkLen;
    workspaceGM += cfg.biasLen;
    GM_ADDR syncPtr3 = workspaceGM;
    hcclSrv->InitSoftSync(syncPtr3, GetBlockNum(), tempBuf3_);
    workspaceGM += AC_MAX_AIV * DEFAULT_BLK_NUM;

    tiling_ = tiling;
    hcclSrv_ = hcclSrv;
    tPipe_ = tPipe;
    aGM_ = aGM;
    bGM_ = bGM;
    cGM_ = cGM;
    biasGM_ = biasGM;
    workspaceGM_ = workspaceGM;
}

template <typename aType, typename bType, typename biasType, typename cType, bool mixNdNz>
__aicore__ inline void MatmulAllReduceUnquant310<aType, bType, biasType, cType, mixNdNz>::Process()
{
    auto&& cfg = tiling_->param;
    if (g_coreType == AIV) {
        return;
    }

    InnerProcess(cfg.tileLoopCnt, tiling_->tilematmulTiling, 0);
    if (cfg.tailM != 0U) {
        InnerProcess(cfg.tailCnt, tiling_->tailmatmulTiling, cfg.tileLoopCnt);
    }

    if (GetBlockIdx() == 0) {
        hcclSrv_->TurnWait(cfg.tileLoopCnt + cfg.tailCnt);
    }
}

#define MATMUL_ALL_REDUCE_TEMPLATE(                                                                           \
    mmop, tileLoopCnt, tPipe, aGM, bGM, cGM, biasGM, workspaceGM, mm_tiling, hcclSrv, turnShift, aOff, cOff) \
    for (uint32_t idx = 0; idx < tileLoopCnt; i++) {                                                                  \
        if (i == 0) {                                                                                         \
            tPipe_->Reset();                                                                                  \
            mmop.Init(aGM_, bGM_, cGM_, biasGM_, nullptr, workspaceGM_, &mm_tiling, tPipe_);                  \
        } else {                                                                                              \
            mmop.UpdateGlobalTensor(aGM_, bGM_, cGM_, biasGM_, nullptr, workspaceGM_);                        \
        }                                                                                                     \
        mmop.Process();                                                                                       \
        hcclSrv_->TurnNotifyRun(block_idx, GetBlockNum(), i + 1 + turnShift);                                  \
        aGM_ += aOff;                                                                                      \
        cGM_ += cOff;                                                                                      \
    }

template <typename aType, typename bType, typename biasType, typename cType, bool mixNdNz>
__aicore__ inline void MatmulAllReduceUnquant310<aType, bType, biasType, cType, mixNdNz>::InnerProcess(
    uint32_t tileLoopCnt, Mc2MatmulV3TilingData& mm_tiling, uint32_t turnShift)
{
    const uint64_t aOff = CalcShapeOffset(sizeof(aType), mm_tiling.matmulTiling.M, mm_tiling.matmulTiling.Ka);
    const uint64_t cOff = CalcShapeOffset(sizeof(cType), mm_tiling.matmulTiling.M, mm_tiling.matmulTiling.N);
    using aMatmulType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aType, false>;
    using cMatmulType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, cType>;
    using biasMatmulType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, biasType>;
    if (tiling_->param.isTransposeB == 0) {
        using bMatmulType = MatmulType<AscendC::TPosition::GM, CubeFormat::NZ, bType, false>;
        if constexpr (mixNdNz == MAT_MUL_V3_MIXND2NZ_TRUE) {
            Mc2MatmulBaseUnAlignedKernel<
                aMatmulType, bMatmulType, cMatmulType, biasMatmulType, Mc2MatmulBaseBlock, MM_CFG_VEC_ND2NZ>
                mmop;
            MATMUL_ALL_REDUCE_TEMPLATE(
                mmop, tileLoopCnt, tPipe_, aGM_, bGM_, cGM_, biasGM_, workspaceGM_, mm_tiling, hcclSrv_, turnShift, aOff,
                cOff);
        } else if constexpr (mixNdNz == MAT_MUL_V3_MIXND2NZ_FALSE) {
            Mc2MatmulBaseKernel<aMatmulType, bMatmulType, cMatmulType, biasMatmulType, Mc2MatmulBaseBlock, MM_CFG_VEC_ND2NZ>
                mmop;
            MATMUL_ALL_REDUCE_TEMPLATE(
                mmop, tileLoopCnt, tPipe_, aGM_, bGM_, cGM_, biasGM_, workspaceGM_, mm_tiling, hcclSrv_, turnShift, aOff,
                cOff);
        }
    } else {
        using bMatmulType = MatmulType<AscendC::TPosition::GM, CubeFormat::NZ, bType, true>;
        if constexpr (mixNdNz == MAT_MUL_V3_MIXND2NZ_TRUE) {
            Mc2MatmulBaseUnAlignedKernel<
                aMatmulType, bMatmulType, cMatmulType, biasMatmulType, Mc2MatmulBaseBlock, MM_CFG_VEC_ND2NZ>
                mmop;
            MATMUL_ALL_REDUCE_TEMPLATE(
                mmop, tileLoopCnt, tPipe_, aGM_, bGM_, cGM_, biasGM_, workspaceGM_, mm_tiling, hcclSrv_, turnShift, aOff,
                cOff);
        } else if constexpr (mixNdNz == MAT_MUL_V3_MIXND2NZ_FALSE) {
            Mc2MatmulBaseKernel<aMatmulType, bMatmulType, cMatmulType, biasMatmulType, Mc2MatmulBaseBlock, MM_CFG_VEC_ND2NZ>
                mmop;
            MATMUL_ALL_REDUCE_TEMPLATE(
                mmop, tileLoopCnt, tPipe_, aGM_, bGM_, cGM_, biasGM_, workspaceGM_, mm_tiling, hcclSrv_, turnShift, aOff,
                cOff);
        }
    }
}

__aicore__ inline void MatMulEmptyTensorKernelUnquantNzBias(GM_ADDR biasGM, GM_ADDR cGM,
    Mc2Tiling::UnQuantMatmulAllReduceTilingData* tiling, TBuf<TPosition::VECCALC>& tempBuf3)
{
    // 搬运biase对齐部分
    int32_t cSz3Half = (tiling->param.rankN * tiling->param.rankM) * sizeof(DTYPE_X1) / sizeof(DTYPE_Y);
    GlobalTensor<DTYPE_Y> cGl3Half;
    cGl3Half.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_Y*>(cGM), cSz3Half);
    LocalTensor<DTYPE_Y> bias = tempBuf3.Get<DTYPE_Y>();
    GlobalTensor<DTYPE_Y> biasGlobal;
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_Y*>(biasGM));
    TBuffAddr buffAddr;
    buffAddr.logicPos = (uint8_t)QuePosition::VECCALC;
    uint32_t eleCnt = 32 / sizeof(DTYPE_Y);
    uint32_t alSize = (tiling->param.rankN / eleCnt) * eleCnt;
    if (alSize > 0) {
        DataCopy(bias, biasGlobal, alSize);
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(eventID);
        WaitFlag<HardEvent::MTE2_MTE3>(eventID);
        for (uint32_t idx = 0; idx < tiling->param.rankM; ++i) {
            uint32_t offsetDst = i * tiling->param.rankN;
            DataCopy(cGl3Half[offsetDst], bias, alSize);
        }
    }
    if (tiling->param.rankN % eleCnt) {
        // 搬运biase非对齐部分
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventID);
        WaitFlag<HardEvent::MTE3_MTE2>(eventID);
        DataCopy(bias, biasGlobal[tiling->param.rankN - eleCnt], eleCnt);
        eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(eventID);
        WaitFlag<HardEvent::MTE2_MTE3>(eventID);
        for (uint32_t idx = 0; idx < tiling->param.rankM; ++i) {
            uint32_t offsetDst = (i + 1) * tiling->param.rankN - eleCnt;
            DataCopy(cGl3Half[offsetDst], bias, eleCnt);
        }
    }
}

__aicore__ inline void MatMulEmptyTensorKernelUnquantNz(
    GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR workspaceGM, Mc2Tiling::UnQuantMatmulAllReduceTilingData* tiling,
    HcclServer* hcclSrv)
{
    TBuf<TPosition::VECCALC> tempBuf3;
    GetTPipePtr()->InitBuffer(tempBuf3, TOTAL_UB_SIZE);
    // 初始化hccl
    __gm__ HcclCombinOpParam* context = (__gm__ HcclCombinOpParam*)(GetHcclContext<0>());
    __gm__ uint8_t* wsMsg3 = (__gm__ uint8_t*)(context->WorkSpace + (tiling->msg).notifyOff);
    hcclSrv->Init(wsMsg3, (tiling->msg).debugMode, tempBuf3);
    if (tiling->tilematmulTiling.matmulTiling.usedCoreNum > 1) {
        hcclSrv->InitSoftSync(workspaceGM, tiling->tilematmulTiling.matmulTiling.usedCoreNum, tempBuf3);
    }
    // 初始化输出tensor
    int32_t cSz3 = (tiling->param.rankN * tiling->param.rankM) * sizeof(DTYPE_X1) / sizeof(int32_t);
    GlobalTensor<int32_t> cGl3;
    cGl3.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(cGM), cSz3);
    if (block_idx == 0) {
        InitOutput<int32_t>(cGl3, cSz3, (int32_t)0);
        if (tiling->tilematmulTiling.matmulTiling.isBias) {
            MatMulEmptyTensorKernelUnquantNzBias(biasGM, cGM, tiling, tempBuf3);
        }
    }
    // 通知AICPU计算结束
    if (block_idx == 0) {
        hcclSrv->TurnNotifyRun(block_idx, tiling->tilematmulTiling.matmulTiling.usedCoreNum, 1);
        // 通过一个核轮询并清除数据，防止多核之间写后读依赖
        hcclSrv->TurnWait(tiling->param.tileLoopCnt + tiling->param.tailCnt);
    }
}

#define INVOKE_UNQUANT_BMM_OP_IMPL_310(templateClass)                                        \
    do {                                                                                     \
        GET_TILING_DATA_MEMBER(Mc2Tiling::UnQuantMatmulAllReduceTilingData, msg, msg, tilingGM);        \
        if (msg.debugMode == static_cast<uint8_t>(DebugMode::MC2_DEBUG_ONLY_AICPU)) {        \
            continue;                                                                        \
        }                                                                                    \
        GET_TILING_DATA_WITH_STRUCT(Mc2Tiling::UnQuantMatmulAllReduceTilingData, tiling, tilingGM); \
        templateClass<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_Y, MIXND2NZ> op;                              \
        op.Init(aGM, bGM, biasGM, cGM, userWS, &tiling, &tPipe, &hcclSrv);            \
        op.Process();                                                                        \
    } while (0)
} // namespace MatmulAllReduceImpl
#endif // MATMUL_ALL_REDUCE_UNQUANT_310_H
