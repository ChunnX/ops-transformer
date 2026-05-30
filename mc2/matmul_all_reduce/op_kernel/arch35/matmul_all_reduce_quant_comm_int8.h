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
 * \file matmul_all_reduce_quant_comm_int8.h
 * \brief
 */

#ifndef MATMUL_ALL_REDUCE_QUANT_COMM_INT8_H
#define MATMUL_ALL_REDUCE_QUANT_COMM_INT8_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "../common.h"

#include "../../3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_cube_on_the_fly.h"
#include "matmul_all_reduce_add_x3.h"
#include "matmul_all_reduce_quant_perchannel.h"
#include "matmul_all_reduce_quant_mul_cast.h"
#include "matmul_all_reduce_dequant_perchannel.h"

namespace MatmulAllReduceImpl {
constexpr uint32_t MAX_HANDLE_NUM = 16U;
constexpr uint32_t NUM_TWO = 2U;

using namespace AscendC;
using namespace MatmulAllReduceQuantMulCastImpl;
using namespace MatmulAllReduceQuantPerchannelImpl;
using namespace MatmulAllReduceDequantPerchannelImpl;
template <typename XType, typename WType, typename YType, class MmType, Mc2CoreType CoreType, int commMode>
class MatmulAllReduceQuantCommInt8
{
public:
    __aicore__ inline MatmulAllReduceQuantCommInt8()
    {}
    __aicore__ inline void Init(
        GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR addGM, GM_ADDR dequantScaleGM, GM_ADDR pertokenGM,
        GM_ADDR commQuantScale1GM, GM_ADDR commQuantScale2GM, GM_ADDR cGM, GM_ADDR workspaceGM,
        Mc2Tiling::QuantMatmulAllReduceTilingDataA5* tilDat, TPipe* tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InnerProcess(
        MmType& mmOp, uint32_t tileCnt, DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams* mmTiling, uint32_t isAdd, uint32_t needUbBuffer,
        uint32_t curPadM, bool isTailFlag);
    __aicore__ inline void PrepareInit();
    __aicore__ inline uint32_t SendCountCheck(uint32_t prepareIndex);

    Mc2Tiling::QuantMatmulAllReduceTilingDataA5* tilDat2_;
    TPipe* tp3_;
    GM_ADDR aGM_;
    GM_ADDR bGM_;
    GM_ADDR biasGM_;
    GM_ADDR addGM_;
    GM_ADDR dequantScaleGM_;
    GM_ADDR pertokenGM_;
    GM_ADDR commQuantScale1GM_;
    GM_ADDR commQuantScale2GM_;
    GM_ADDR cGM_;
    GM_ADDR workspaceGM_;
    GM_ADDR rdcScatterInGM_;
    GM_ADDR rdcScatterOutGM_;
    GM_ADDR aGatherInGM_;
    GM_ADDR aGatherOutGM_;
    typename HcclTypeSelector<commMode>::type hccl_;
    GM_ADDR outGM_;
    uint32_t tilePadMatM_ = 0U;
    uint32_t tailPadMatM_ = 0U;
    uint32_t tilePadCnt_ = 0U;
    uint32_t tailPadCnt_ = 0U;

    // 仅在0核上使用
    AscendC::HcclHandle reduceScatterHandleId_[MAX_HANDLE_NUM] = {0};
    AscendC::HcclHandle allGatherHandleId_[MAX_HANDLE_NUM] = {0};
    GM_ADDR rdcScatterSendGM_[MAX_HANDLE_NUM] = {0};
    GM_ADDR aGatherSendGM_[MAX_HANDLE_NUM] = {0};
    GM_ADDR RdcScatterRecvGM_[MAX_HANDLE_NUM] = {0};
    GM_ADDR aGatherRecvGM_[MAX_HANDLE_NUM] = {0};
    int reduceScatterCommitIdx_ = 0;
    int reduceScatterWaitIdx_ = 0;
    int allGatherCommitIdx_ = 0;
    int allGatherWaitIdx_ = 0;
    // 所有核
    bool isSendTileFlag_ = false;
};

template <typename XType, typename WType, typename YType, class MmType, Mc2CoreType CoreType, int commMode>
__aicore__ inline void MatmulAllReduceQuantCommInt8<XType, WType, YType, MmType, CoreType, commMode>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR addGM, GM_ADDR dequantScaleGM, GM_ADDR pertokenGM,
    GM_ADDR commQuantScale1GM, GM_ADDR commQuantScale2GM, GM_ADDR cGM, GM_ADDR workspaceGM,
    Mc2Tiling::QuantMatmulAllReduceTilingDataA5* tilDat, TPipe* tPipe)
{
    __gm__ HcclCombinOpParam* context = (__gm__ HcclCombinOpParam*)(GetHcclContext<0>());
    OOMInit(context);
    hccl_.InitV2(GetHcclContext<0>(), tilDat); 
    hccl_.SetCcTilingV2(offsetof(Mc2Tiling::QuantMatmulAllReduceTilingDataA5, mc2CcTiling));  
    hccl_.SetCcTilingV2(offsetof(Mc2Tiling::QuantMatmulAllReduceTilingDataA5, mc2CcTilingComm)); 
    tilDat2_ = tilDat;
    tp3_ = tPipe;
    aGM_ = aGM;
    bGM_ = bGM;
    biasGM_ = biasGM;
    addGM_ = addGM;
    dequantScaleGM_ = dequantScaleGM;
    pertokenGM_ = pertokenGM;
    commQuantScale1GM_ = commQuantScale1GM;
    commQuantScale2GM_ = commQuantScale2GM;
    workspaceGM_ = workspaceGM;
    cGM_ = cGM;
    outGM_ = cGM;
    rdcScatterInGM_ = workspaceGM_ + tilDat2_->param.commWorkSpaceSize;     // reduceScatter输入
    aGatherInGM_ = rdcScatterInGM_ + tilDat2_->param.commInt8WorkSpace;   // allGather输入
    aGatherOutGM_ = aGatherInGM_ + tilDat2_->param.commInt8WorkSpace;      // allGather输出
    rdcScatterOutGM_ = aGatherOutGM_ + tilDat2_->param.commInt8WorkSpace; // reduceScatter输出
}

template <typename XType, typename WType, typename YType, class MmType, Mc2CoreType CoreType, int commMode>
__aicore__ inline uint32_t MatmulAllReduceQuantCommInt8<
    XType, WType, YType, MmType, CoreType, commMode>::SendCountCheck(uint32_t prepareIndex)
{
    uint32_t txCnt = tilePadCnt_ / tilDat2_->param.rankDim;
    if (prepareIndex >= tilDat2_->param.tileCnt) {
        txCnt = tailPadCnt_ / tilDat2_->param.rankDim;
    }
    return txCnt;
}

template <typename XType, typename WType, typename YType, class MmType, Mc2CoreType CoreType, int commMode>
__aicore__ inline void MatmulAllReduceQuantCommInt8<XType, WType, YType, MmType, CoreType, commMode>::PrepareInit()
{
    auto&& mc2Cfg = tilDat2_->param;
    uint32_t rankCnt = mc2Cfg.rankDim;
    uint32_t tileMatM = tilDat2_->tilematmulTiling.matmulTiling.M; // 头块 pad 大小更新
    tilePadMatM_ = tileMatM;
    if ((tileMatM % rankCnt) != 0) { // 按照 M 来 pad
        tilePadMatM_ += rankCnt - (tileMatM % rankCnt);
    }
    uint32_t tailMatM = tilDat2_->tailmatmulTiling.matmulTiling.M;
    tailPadMatM_ = tailMatM;
    if ((tailMatM % rankCnt) != 0) {
        tailPadMatM_ += rankCnt - (tailMatM % rankCnt);
    }

    tilePadCnt_ = tilePadMatM_ * tilDat2_->tilematmulTiling.matmulTiling.N; // 一个MM头块的数据个数
    tailPadCnt_ = tailPadMatM_ * tilDat2_->tailmatmulTiling.matmulTiling.N; // 一个MM尾块的数据个数
    const int64_t tilePadSingleOff = tilePadCnt_ / rankCnt;
    const int64_t tailPadSingleOff = tailPadCnt_ / rankCnt;

    for (uint32_t i = 0U; i < mc2Cfg.tileCnt; ++i) { // 头块偏移
        const int64_t indexOffsetTile = i * tilePadCnt_;
        rdcScatterSendGM_[i] = rdcScatterInGM_ + indexOffsetTile * sizeof(int8_t);
        RdcScatterRecvGM_[i] = rdcScatterOutGM_ + indexOffsetTile * sizeof(float);
        aGatherSendGM_[i] = aGatherInGM_ + indexOffsetTile * sizeof(int8_t);
        aGatherRecvGM_[i] = aGatherOutGM_ + indexOffsetTile * sizeof(int8_t);
    }
    for (uint32_t i = 0U; i < mc2Cfg.tailCnt; ++i) { // 尾块偏移
        const int64_t tailIdxOff = mc2Cfg.tileCnt * tilePadCnt_ + i * tailPadCnt_;
        rdcScatterSendGM_[mc2Cfg.tileCnt + i] = rdcScatterInGM_ + tailIdxOff * sizeof(int8_t);
        RdcScatterRecvGM_[mc2Cfg.tileCnt + i] = rdcScatterOutGM_ + tailIdxOff * sizeof(float);
        aGatherSendGM_[mc2Cfg.tileCnt + i] = aGatherInGM_ + tailIdxOff * sizeof(int8_t);
        aGatherRecvGM_[mc2Cfg.tileCnt + i] = aGatherOutGM_ + tailIdxOff * sizeof(int8_t);
    }

    if (g_coreType == AscendC::AIV) {
        uint32_t nowReduceScatterIdx = 0U;
        uint32_t nowAllGatherIdx = 0U;
        uint32_t numN = (mc2Cfg.tileCnt + mc2Cfg.tailCnt) / NUM_TWO;
        uint32_t numReN = (mc2Cfg.tileCnt + mc2Cfg.tailCnt) % NUM_TWO;
        for (uint32_t i = 0U; i < numN; ++i) { // 按总核数下发
            reduceScatterHandleId_[nowReduceScatterIdx] = hccl_.template ReduceScatter<false>(
                rdcScatterSendGM_[nowReduceScatterIdx], RdcScatterRecvGM_[nowReduceScatterIdx],
                SendCountCheck(nowReduceScatterIdx), AscendC::HCCL_DATA_TYPE_INT8, HcclReduceOp::HCCL_REDUCE_SUM, 0, 1);
            nowReduceScatterIdx++;
            reduceScatterHandleId_[nowReduceScatterIdx] = hccl_.template ReduceScatter<false>(
                rdcScatterSendGM_[nowReduceScatterIdx], RdcScatterRecvGM_[nowReduceScatterIdx],
                SendCountCheck(nowReduceScatterIdx), AscendC::HCCL_DATA_TYPE_INT8, HcclReduceOp::HCCL_REDUCE_SUM, 0, 1);
            nowReduceScatterIdx++;

            allGatherHandleId_[nowAllGatherIdx] = hccl_.template AllGather<false>(
                aGatherSendGM_[nowAllGatherIdx], aGatherRecvGM_[nowAllGatherIdx], SendCountCheck(nowAllGatherIdx),
                AscendC::HCCL_DATA_TYPE_INT8, 0);
            nowAllGatherIdx++;
            allGatherHandleId_[nowAllGatherIdx] = hccl_.template AllGather<false>(
                aGatherSendGM_[nowAllGatherIdx], aGatherRecvGM_[nowAllGatherIdx], SendCountCheck(nowAllGatherIdx),
                AscendC::HCCL_DATA_TYPE_INT8, 0);
            nowAllGatherIdx++;
        }

        if (numReN != 0U) { // 余数下发
            reduceScatterHandleId_[nowReduceScatterIdx] = hccl_.template ReduceScatter<false>(
                rdcScatterSendGM_[nowReduceScatterIdx], RdcScatterRecvGM_[nowReduceScatterIdx],
                SendCountCheck(nowReduceScatterIdx), AscendC::HCCL_DATA_TYPE_INT8, HcclReduceOp::HCCL_REDUCE_SUM, 0, 1);
            nowReduceScatterIdx++;
            
            allGatherHandleId_[nowAllGatherIdx] = hccl_.template AllGather<false>(
                aGatherSendGM_[nowAllGatherIdx], aGatherRecvGM_[nowAllGatherIdx], SendCountCheck(nowAllGatherIdx),
                AscendC::HCCL_DATA_TYPE_INT8, 0);
            nowAllGatherIdx++;
        }
    }
}

template <typename XType, typename WType, typename YType, class MmType, Mc2CoreType CoreType, int commMode>
__aicore__ inline void MatmulAllReduceQuantCommInt8<
    XType, WType, YType, MmType, CoreType, commMode>::InnerProcess(
    MmType& mmOp, uint32_t tileCnt, DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams* mmTiling, uint32_t isAdd,
    uint32_t needUbBuffer, uint32_t curPadM, bool isTailFlag)
{
    const uint64_t aOffset = CalcShapeOffset(sizeof(XType), mmTiling->matmulTiling.M, mmTiling->matmulTiling.Ka);
    const uint64_t cOffset = CalcShapeOffset(sizeof(YType), mmTiling->matmulTiling.M, mmTiling->matmulTiling.N);
    const uint64_t tempOffset = CalcShapeOffset(sizeof(int8_t), curPadM, mmTiling->matmulTiling.N);

    for (uint32_t i = 0U; i < tileCnt; ++i) {
        tp3_->Reset();
        mmOp.Init(aGM_, bGM_, biasGM_, dequantScaleGM_, pertokenGM_, cGM_, workspaceGM_, mmTiling, tp3_);
        mmOp.Process();
        SyncAll<false>();
        if (isAdd) {
            MatmulAllReduceAddX3Kernel<YType>(
                cGM_, addGM_, cOffset / sizeof(YType), tilDat2_->param.addX3UbCnt, tp3_);
            addGM_ += cOffset;
            SyncAll<false>();
        }
        // bf16->int8 存在 rdcScatterInGM_
        MatmulAllReduceQuantPerchannelCommInt8<YType>(
            cGM_, commQuantScale1GM_, rdcScatterInGM_, tp3_, mmTiling->matmulTiling.N, mmTiling->matmulTiling.M);
        SyncAll<false>();

        if (g_coreType == AscendC::AIV) {
            hccl_.Commit(reduceScatterHandleId_[reduceScatterCommitIdx_]);
            reduceScatterCommitIdx_++;
        }
        if (g_coreType == AscendC::AIV && isSendTileFlag_) { // 需要同步前一块
            hccl_.Wait(reduceScatterHandleId_[reduceScatterWaitIdx_]);
            SyncAll();
            if (isTailFlag && (i == 0U)) {
                MatmulAllReduceQuantMulCastCommInt8<YType, commMode>(
                    rdcScatterOutGM_, commQuantScale1GM_, commQuantScale2GM_, aGatherInGM_, tilePadMatM_,
                    mmTiling->matmulTiling.N, tp3_, hccl_);
                rdcScatterOutGM_ += tilePadMatM_ * mmTiling->matmulTiling.N * sizeof(float);
                aGatherInGM_ += tilePadMatM_ * mmTiling->matmulTiling.N * sizeof(int8_t);
            } else {
                MatmulAllReduceQuantMulCastCommInt8<YType, commMode>(
                    rdcScatterOutGM_, commQuantScale1GM_, commQuantScale2GM_, aGatherInGM_, curPadM,
                    mmTiling->matmulTiling.N, tp3_, hccl_);
                rdcScatterOutGM_ += curPadM * mmTiling->matmulTiling.N * sizeof(float);
                aGatherInGM_ += curPadM * mmTiling->matmulTiling.N * sizeof(int8_t);
            }
            SyncAll();
            hccl_.Commit(allGatherHandleId_[reduceScatterWaitIdx_]);
            reduceScatterWaitIdx_++;
        }
        isSendTileFlag_ = true;
        aGM_ += aOffset;
        cGM_ += cOffset;                  // 偏原始大小
        rdcScatterInGM_ += tempOffset; // 偏移 padM*N 大小
    }
}

template <typename XType, typename WType, typename YType, class MmType, Mc2CoreType CoreType, int commMode>
__aicore__ inline void MatmulAllReduceQuantCommInt8<XType, WType, YType, MmType, CoreType, commMode>::Process()
{
    auto&& mc2Cfg = tilDat2_->param;

    // reducescatter allgather Prepare + InnerProcess
    PrepareInit();
    MmType opTile;
    InnerProcess(
        opTile, mc2Cfg.tileCnt, &tilDat2_->tilematmulTiling, mc2Cfg.isAdd, mc2Cfg.needUbBuffer, tilePadMatM_,
        false);
    if (mc2Cfg.tailMatM != 0U) {
        MmType opTail;
        InnerProcess(
            opTail, mc2Cfg.tailCnt, &tilDat2_->tailmatmulTiling, mc2Cfg.isAdd, mc2Cfg.needUbBuffer,
            tailPadMatM_, true);
    }

    // 最后一次的 reduceScatter 任务没有 wait+mul/cast，以及最后一次的 allgather 任务没有下发
    if (g_coreType == AscendC::AIV) {
        hccl_.Wait(reduceScatterHandleId_[reduceScatterWaitIdx_]);
        SyncAll();
        uint32_t lastN = tilDat2_->tilematmulTiling.matmulTiling.N;
        uint32_t padM = tilePadMatM_;
        if (mc2Cfg.tailMatM != 0U) {
            lastN = tilDat2_->tailmatmulTiling.matmulTiling.N;
            padM = tailPadMatM_;
        }
        MatmulAllReduceQuantMulCastCommInt8<YType, commMode>(
            rdcScatterOutGM_, commQuantScale1GM_, commQuantScale2GM_, aGatherInGM_, padM, lastN, tp3_, hccl_);
        SyncAll();
        hccl_.Commit(allGatherHandleId_[reduceScatterWaitIdx_]);
        reduceScatterWaitIdx_++;
    }

    if (g_coreType == AscendC::AIV) {
        // wait所有的allGather任务 + dequant
        const uint64_t outGmTileOffset =
            tilDat2_->tilematmulTiling.matmulTiling.M * tilDat2_->tilematmulTiling.matmulTiling.N * sizeof(YType);
        const uint64_t outGmTailOffset =
            tilDat2_->tailmatmulTiling.matmulTiling.M * tilDat2_->tailmatmulTiling.matmulTiling.N * sizeof(YType);
        for (uint32_t i = 0U; i < (mc2Cfg.tileCnt + mc2Cfg.tailCnt); ++i) { // 尾块偏移
            hccl_.Wait(allGatherHandleId_[i]);
            SyncAll();
            if (i < mc2Cfg.tileCnt) {
                MatmulAllReduceDequantPerchannelCommInt8<YType>(
                    aGatherOutGM_, commQuantScale2GM_, outGM_, tp3_, tilDat2_->tilematmulTiling.matmulTiling.N,
                    tilDat2_->tilematmulTiling.matmulTiling.M);
                outGM_ += outGmTileOffset;
                aGatherOutGM_ += tilePadCnt_ * sizeof(int8_t);
                SyncAll();
            } else {
                MatmulAllReduceDequantPerchannelCommInt8<YType>(
                    aGatherOutGM_, commQuantScale2GM_, outGM_, tp3_, tilDat2_->tailmatmulTiling.matmulTiling.N,
                    tilDat2_->tailmatmulTiling.matmulTiling.M);
                outGM_ += outGmTailOffset;
                aGatherOutGM_ += tailPadCnt_ * sizeof(int8_t);
                SyncAll();
            }
        }
        hccl_.Finalize();
    }
}

#define INVOKE_MC2_QUANT_COMM_INT8_910_OP_IMPL(templateClass, coreType, commMode, scaleType, ...)              \
    do {                                                                                                       \
        GET_TILING_DATA_WITH_STRUCT(Mc2Tiling::QuantMatmulAllReduceTilingDataA5, tilDat, tilingGM);        \
        MC2GmAddrs addrs = {aGM, bGM, biasGM, addGM, cGM, workspaceGM, cGM};                                   \
        QuantGmAddrs quantAddrs = {nullptr, nullptr, nullptr, dequantGM, pertokenGM};                          \
        using OpType = templateClass<                                                                          \
            DTYPE_X1, DTYPE_X2, scaleType, DTYPE_BIAS, DTYPE_Y, X1_FORMAT, X2_FORMAT, Y_FORMAT, __VA_ARGS__>;  \
        MatmulAllReduceQuantCommInt8<DTYPE_X1, DTYPE_X2, DTYPE_Y, OpType, coreType, commMode> op;              \
        op.Init(                                                                                               \
            aGM, bGM, biasGM, addGM, dequantGM, pertokenGM, commQuantScale1GM, commQuantScale2GM, cGM, userWS, \
            &tilDat, &tPipe);                                                                              \
        op.Process();                                                                                          \
        tPipe.Destroy();                                                                                       \
    } while (0)

} // namespace MatmulAllReduceImpl
#endif // MAtMUL_ALL_REDUCE_QUANT_COMM_INT8_H