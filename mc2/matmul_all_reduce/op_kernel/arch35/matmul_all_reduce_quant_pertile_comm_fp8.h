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
 * \file matmul_all_reduce_quant_pertile_comm_fp8.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_QUANT_PERTILE_COMM_FP8_H
#define MATMUL_ALL_REDUCE_QUANT_PERTILE_COMM_FP8_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "../common.h"

#include "matmul_all_reduce_base.h"
#include "matmul_all_reduce_reduce_sum.h"
#include "matmul_all_reduce_dynamic_quant_pertile.h"
#include "matmul_all_reduce_dynamic_quant_pertile_utils.h"
#if __has_include("../../3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_mix_online_dynamic.h")
#include "../../3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_mix_online_dynamic.h"
#else
#include "../3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_mix_online_dynamic.h"
#endif
#include "../common/matmul_all_reduce_element_wise_add.h"

namespace MatmulAllReduceImpl {
constexpr uint32_t PERTILE_MAX_HANDLE_ID_NUM = 16;
constexpr uint32_t NUM_TWO_PERTILE = 2;

using namespace AscendC;
using namespace MatmulAllReduceReduceSumImpl;
using namespace MatmulAllReduceDynamicQuantPertileImpl;
template <typename XType, typename WType, typename YType, class MmType, Mc2CoreType CoreType>
class MatmulAllReduceQuantPertileCommFp8 {
public:
    __aicore__ inline MatmulAllReduceQuantPertileCommFp8() 
    {
    }

    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR addGM, GM_ADDR dequantScaleGM,
                                GM_ADDR pertokenGM, GM_ADDR commQuantScale1GM, GM_ADDR commQuantScale2GM, GM_ADDR cGM, 
                                GM_ADDR workspaceGM, Mc2Tiling::QuantMatmulAllReduceTilingDataA5* tilingData, TPipe* tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InnerProcess(MmType &mmOp, MatmulAllReduceDynamicQuantPertile<XType, float> &quantOp,
                                        uint32_t tileCnt, DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams *mmTiling, uint32_t curPadM,
                                        uint32_t isAdd, bool isTail);
    __aicore__ inline void StepOneTurn(MmType &mmOp, MatmulAllReduceDynamicQuantPertile<XType, float> &quantOp,
                                       DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams *mmTiling, uint32_t curPadM, bool isTail,
                                       bool isFirst);
    __aicore__ inline void ProcessLast(MatmulAllReduceDynamicQuantPertile<XType, YType> &quantOp);
    __aicore__ inline void PrepareInit();
    __aicore__ inline void InitCommTasks();
    __aicore__ inline uint32_t SendCountCheck(uint32_t prepareIndex);
    Mc2Tiling::QuantMatmulAllReduceTilingDataA5* tilingData_;

    TPipe* tPipe_;
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
    GM_ADDR outGM_;
    GM_ADDR all2allInGM_;
    GM_ADDR all2allOutGM_;
    GM_ADDR reduceSumInGM_;
    GM_ADDR reduceSumOutGM_;
    GM_ADDR allGatherInGM_;
    GM_ADDR allGatherOutGM_;
    bool notifyFlag_{false};
    Hccl<HCCL_SERVER_TYPE_CCU> hccl_;
    AscendC::HcclDataType hcclType_ = AscendC::HCCL_DATA_TYPE_RESERVED;

    // 仅在0核上使用
    AscendC::HcclHandle all2allHandleId_[PERTILE_MAX_HANDLE_ID_NUM] = {0};
    AscendC::HcclHandle allGatherHandleId_[PERTILE_MAX_HANDLE_ID_NUM] = {0};
    GM_ADDR all2allSendGM_[PERTILE_MAX_HANDLE_ID_NUM] = {0};
    GM_ADDR all2allRecvGM_[PERTILE_MAX_HANDLE_ID_NUM] = {0};
    GM_ADDR allGatherSendGM_[PERTILE_MAX_HANDLE_ID_NUM] = {0};
    GM_ADDR allGatherRecvGM_[PERTILE_MAX_HANDLE_ID_NUM] = {0};
    int all2allCommitIdx_ = 0;
    int all2allWaitIdx_ = 0;
    // 所有核
    bool isSendTileFlag_ = false;
    uint32_t tilePadM_ = 0U;
    uint32_t tileM_ = 0U;
    uint32_t tileN_ = 0U;
    uint32_t tailPadM_ = 0U;
    uint32_t tilePadDataCnt_ = 0U;
    uint32_t tailPadDataCnt_ = 0U;
    uint32_t tileScaleCnt_ = 0U;
    uint32_t tailScaleCnt_ = 0U;
    uint32_t tileOneLineSCnt_ = 0U;
    uint32_t tailOneLineSCnt_ = 0U;
    uint32_t rankNum_ = 0U;
    uint32_t coreNum_ = 0U;
    uint32_t maxProcRowsQuant_ = 0U;
    uint32_t maxProcRowsDequant_ = 0U;
    uint32_t maxProcRowsDequantLast_ = 0U;
};

template <typename XType, typename WType, typename YType, class MmType, Mc2CoreType CoreType>
__aicore__ inline void MatmulAllReduceQuantPertileCommFp8<XType, WType, YType, MmType, CoreType>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR addGM, GM_ADDR dequantScaleGM, GM_ADDR pertokenGM,
    GM_ADDR commQuantScale1GM, GM_ADDR commQuantScale2GM, GM_ADDR cGM, GM_ADDR workspaceGM,
    Mc2Tiling::QuantMatmulAllReduceTilingDataA5* tilingData, TPipe* tPipe)
{
    __gm__ HcclCombinOpParam* context = (__gm__ HcclCombinOpParam*)(GetHcclContext<0>());
    OOMInit(context);
    hccl_.InitV2(GetHcclContext<0>(), tilingData);
    hccl_.SetCcTilingV2(offsetof(Mc2Tiling::QuantMatmulAllReduceTilingDataA5, mc2CcTiling));
    hccl_.SetCcTilingV2(offsetof(Mc2Tiling::QuantMatmulAllReduceTilingDataA5, mc2CcTilingCommQuant));
    tilingData_ = tilingData;
    rankNum_ = tilingData_->param.rankDim;
    tPipe_ = tPipe;
    aGM_ = aGM;
    bGM_ = bGM;
    biasGM_ = biasGM;
    addGM_ = addGM;
    dequantScaleGM_ = dequantScaleGM;
    pertokenGM_ = pertokenGM;
    commQuantScale1GM_ = commQuantScale1GM;
    commQuantScale2GM_ = commQuantScale2GM;
    workspaceGM_ = workspaceGM;
    outGM_ = cGM;
    cGM_ = workspaceGM_ + tilingData_->param.commWorkSpaceSize;
    coreNum_ = GetBlockNum() * GetTaskRation();
    if constexpr(std::is_same<XType, fp8_e5m2_t>::value) {
        hcclType_ = AscendC::HCCL_DATA_TYPE_FP8E5M2;
    } else if constexpr(std::is_same<XType, fp8_e4m3fn_t>::value) {
        hcclType_ = AscendC::HCCL_DATA_TYPE_FP8E4M3;
    }
    if ASCEND_IS_AIV { // V核的0核下发通信任务
        if (GetBlockIdx() == 0) {
            notifyFlag_ = true;
        }
    }
}

template <typename XType, typename WType, typename YType, class MmType, Mc2CoreType CoreType>
__aicore__ inline uint32_t MatmulAllReduceQuantPertileCommFp8<XType, WType, YType, MmType, CoreType>::SendCountCheck(
    uint32_t prepareIndex)
{
    uint32_t sendCount = (tilePadDataCnt_ * sizeof(XType) + tileScaleCnt_ * sizeof(float)) / sizeof(XType);
    if (prepareIndex >= tilingData_->param.tileCnt) {
        sendCount = (tailPadDataCnt_ * sizeof(XType) + tailScaleCnt_ * sizeof(float)) / sizeof(XType);
    }
    return sendCount / rankNum_;
}

template <typename XType, typename WType, typename YType, class MmType, Mc2CoreType CoreType>
__aicore__ inline void MatmulAllReduceQuantPertileCommFp8<XType, WType, YType, MmType, CoreType>::PrepareInit()
{
    auto&& mc2Tiling = tilingData_->param;
    tileM_ = tilingData_->tilematmulTiling.matmulTiling.M;
    tileN_ = tilingData_->tilematmulTiling.matmulTiling.N;
    uint32_t tailN = tilingData_->tailmatmulTiling.matmulTiling.N;
    tilePadM_ = tileM_;
    if ((tileM_ % rankNum_) != 0) {
        tilePadM_ += rankNum_ - (tileM_ % rankNum_);
    }
    uint32_t tailM = tilingData_->tailmatmulTiling.matmulTiling.M;
    tailPadM_ = tailM;
    if ((tailM % rankNum_) != 0) {
        tailPadM_ += rankNum_ - (tailM % rankNum_);
    }
    uint64_t commFp32WorkSpace =
        (tilePadM_ * tileN_ * mc2Tiling.tileCnt + tailPadM_ * tailN * mc2Tiling.tailCnt) * sizeof(float);
    all2allInGM_ = cGM_ + commFp32WorkSpace;
    all2allOutGM_ = all2allInGM_ + tilingData_->param.commInt8WorkSpace;
    reduceSumInGM_ = all2allOutGM_ + tilingData_->param.commInt8WorkSpace;
    reduceSumOutGM_ = reduceSumInGM_ + commFp32WorkSpace;
    allGatherInGM_ = reduceSumOutGM_ + commFp32WorkSpace / rankNum_;
    allGatherOutGM_ = allGatherInGM_ + tilingData_->param.commInt8WorkSpace / rankNum_;
    tileOneLineSCnt_ = Ceil(tileN_, TILELEN);
    tailOneLineSCnt_ = Ceil(tailN, TILELEN);
    tilePadDataCnt_ = tilePadM_ * tileN_;
    tailPadDataCnt_ = tailPadM_ * tailN;
    tileScaleCnt_ = tilePadM_ * tileOneLineSCnt_;
    tailScaleCnt_ = tailPadM_ * tailOneLineSCnt_;
    maxProcRowsQuant_ = GetMaxProcRows<XType, float>(true, tilingData_->param.dynamicQuantTempBuffSize);
    maxProcRowsDequant_ = GetMaxProcRows<XType, float>(false, tilingData_->param.dynamicQuantTempBuffSize);
    maxProcRowsDequantLast_ = GetMaxProcRows<XType, YType>(false, tilingData_->param.dynamicQuantTempBuffSize);
    for (uint32_t i = 0U; i < mc2Tiling.tileCnt; i++) { // 头块偏移
        const int64_t indexOffsetTile = (tilePadDataCnt_ * sizeof(XType) + tileScaleCnt_ * sizeof(float)) * i;
        all2allSendGM_[i] = all2allInGM_ + indexOffsetTile;
        all2allRecvGM_[i] = all2allOutGM_ + indexOffsetTile;
        allGatherSendGM_[i] = allGatherInGM_ + indexOffsetTile / rankNum_;
        allGatherRecvGM_[i] = allGatherOutGM_ + indexOffsetTile;
    }
    for (uint32_t i = 0U; i < mc2Tiling.tailCnt; i++) { // 尾块偏移
        const int64_t indexOffsetTail =
            mc2Tiling.tileCnt * (tilePadDataCnt_ * sizeof(XType) + tileScaleCnt_ * sizeof(float)) +
            (tailPadDataCnt_ * sizeof(XType) + tailScaleCnt_ * sizeof(float)) * i;
        all2allSendGM_[mc2Tiling.tileCnt + i] = all2allInGM_ + indexOffsetTail;
        all2allRecvGM_[mc2Tiling.tileCnt + i] = all2allOutGM_ + indexOffsetTail;
        allGatherSendGM_[mc2Tiling.tileCnt + i] = allGatherInGM_ + indexOffsetTail / rankNum_;
        allGatherRecvGM_[mc2Tiling.tileCnt + i] = allGatherOutGM_ + indexOffsetTail;
    }
    InitCommTasks();
}

template <typename XType, typename WType, typename YType, class MmType, Mc2CoreType CoreType>
__aicore__ inline void MatmulAllReduceQuantPertileCommFp8<XType, WType, YType, MmType, CoreType>::InitCommTasks()
{
    if ASCEND_IS_AIC {
        return;
    }
    if (GetBlockIdx() == 0) {
        auto&& mc2Tiling = tilingData_->param;
        uint32_t nowAll2allIdx = 0U;
        uint32_t nowAllGatherIdx = 0U;
        uint32_t numN = (mc2Tiling.tileCnt + mc2Tiling.tailCnt) / NUM_TWO_PERTILE;
        uint32_t numReN = (mc2Tiling.tileCnt + mc2Tiling.tailCnt) % NUM_TWO_PERTILE;
        for (uint32_t i = 0U; i < numN; i++) {
            all2allHandleId_[nowAll2allIdx] = hccl_.AlltoAll<false>(
                all2allSendGM_[nowAll2allIdx], all2allRecvGM_[nowAll2allIdx], SendCountCheck(nowAll2allIdx), hcclType_);
            nowAll2allIdx++;
            all2allHandleId_[nowAll2allIdx] = hccl_.AlltoAll<false>(
                all2allSendGM_[nowAll2allIdx], all2allRecvGM_[nowAll2allIdx], SendCountCheck(nowAll2allIdx), hcclType_);
            nowAll2allIdx++;
            allGatherHandleId_[nowAllGatherIdx] = hccl_.AllGather<false>(
                allGatherSendGM_[nowAllGatherIdx], allGatherRecvGM_[nowAllGatherIdx], SendCountCheck(nowAllGatherIdx),
                hcclType_, 0);
            nowAllGatherIdx++;
            allGatherHandleId_[nowAllGatherIdx] = hccl_.AllGather<false>(
                allGatherSendGM_[nowAllGatherIdx], allGatherRecvGM_[nowAllGatherIdx], SendCountCheck(nowAllGatherIdx),
                hcclType_, 0);
            nowAllGatherIdx++;
        }
        if (numReN != 0U) {
            all2allHandleId_[nowAll2allIdx] = hccl_.AlltoAll<false>(
                all2allSendGM_[nowAll2allIdx], all2allRecvGM_[nowAll2allIdx], SendCountCheck(nowAll2allIdx), hcclType_);
            nowAll2allIdx++;
            allGatherHandleId_[nowAllGatherIdx] = hccl_.AllGather<false>(
                allGatherSendGM_[nowAllGatherIdx], allGatherRecvGM_[nowAllGatherIdx], SendCountCheck(nowAllGatherIdx),
                hcclType_, 0);
            nowAllGatherIdx++;
        }
    }
}

template <typename XType, typename WType, typename YType, class MmType, Mc2CoreType CoreType>
__aicore__ inline void MatmulAllReduceQuantPertileCommFp8<XType, WType, YType, MmType, CoreType>::InnerProcess(
    MmType &mmOp, MatmulAllReduceDynamicQuantPertile<XType, float> &quantOp, uint32_t tileCnt,
    DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams *mmTiling, uint32_t curPadM, uint32_t isAdd, bool isTail)
{
    uint32_t oneLineSCnt = isTail ? tailOneLineSCnt_ : tileOneLineSCnt_;
    uint32_t quantNandSLen = mmTiling->matmulTiling.N * sizeof(XType) + oneLineSCnt * sizeof(float);
    const uint64_t aOffset = CalcShapeOffset(sizeof(XType), mmTiling->matmulTiling.M, mmTiling->matmulTiling.Ka);
    const uint64_t cOffset = CalcShapeOffset(sizeof(float), mmTiling->matmulTiling.M, mmTiling->matmulTiling.N);
    const uint64_t addOffset = CalcShapeOffset(sizeof(YType), mmTiling->matmulTiling.M, mmTiling->matmulTiling.N);
    const uint64_t all2allInOffset = quantNandSLen * mmTiling->matmulTiling.M;
    const uint64_t pertokenOffset = mmTiling->matmulTiling.M * sizeof(float);
    for (uint32_t i = 0U; i < tileCnt; i++) {
        tPipe_->Reset();
        mmOp.Init(aGM_, bGM_, dequantScaleGM_, nullptr, biasGM_, pertokenGM_, cGM_, workspaceGM_, mmTiling, tPipe_);
        mmOp.Process();
        SyncAll<false>();
        if (isAdd) {
            MatmulAllReduceElementWiseAddKernel<float, YType>(cGM_, addGM_, cOffset / sizeof(float),
                                                              tilingData_->param.addX3UbCnt, tPipe_);
            addGM_ += addOffset;
            SyncAll<false>();
        }
        // matmul, add和quant独立进行，分别分核，在大shape场景性能可能更好。为适配其他场景可考虑将三者做一个mix版本
        quantOp.Init(cGM_, all2allInGM_, mmTiling->matmulTiling.M, mmTiling->matmulTiling.N, oneLineSCnt, coreNum_,
                     maxProcRowsQuant_, true, tPipe_);
        quantOp.Process(mmTiling->matmulTiling.N, coreNum_, quantNandSLen, true);
        SyncAll<false>();
        if (notifyFlag_) {
            hccl_.Commit(all2allHandleId_[all2allCommitIdx_]);
            all2allCommitIdx_++;
        }
        if (isSendTileFlag_) {
            StepOneTurn(mmOp, quantOp, mmTiling, curPadM, isTail, i == 0);
            SyncAll<false>();
        }
        isSendTileFlag_ = true;
        aGM_ += aOffset;
        cGM_ += cOffset;
        all2allInGM_ += all2allInOffset;
        pertokenGM_ += pertokenOffset;
    }
}

template <typename XType, typename WType, typename YType, class MmType, Mc2CoreType CoreType>
__aicore__ inline void MatmulAllReduceQuantPertileCommFp8<XType, WType, YType, MmType, CoreType>::StepOneTurn(
    MmType &mmOp, MatmulAllReduceDynamicQuantPertile<XType, float> &quantOp, DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams *mmTiling,
    uint32_t curPadM, bool isTail, bool isFirst)
{
    if ASCEND_IS_AIC {
        return;
    }
    uint32_t padM = (isTail && isFirst) ? tilePadM_ : curPadM;
    uint32_t tileM = (isTail && isFirst) ? tileM_ : mmTiling->matmulTiling.M;
    uint32_t tileN = (isTail && isFirst) ? tileN_ : mmTiling->matmulTiling.N;
    uint32_t oneLineSCnt = Ceil(tileN, TILELEN);
    uint32_t quantNandSLen = tileN * sizeof(XType) + oneLineSCnt * sizeof(float);
    uint64_t all2allOutOffset = padM * quantNandSLen;
    uint64_t reduceSumInOffset = padM * tileN * sizeof(float);
    uint64_t reduceSumOutOffset = (padM / rankNum_) * tileN * sizeof(float);
    uint64_t allGatherInOffset = (padM / rankNum_) * quantNandSLen;
    if (notifyFlag_) {
        hccl_.Wait(all2allHandleId_[all2allWaitIdx_]);
    }
    SyncAll();
    // dequant, reduce和dequant独立进行，分别分核，在大shape场景性能可能更好。为适配其他场景可考虑将三者做一个mix版本
    quantOp.Init(all2allOutGM_, reduceSumInGM_, tileM, tileN, oneLineSCnt, coreNum_, maxProcRowsDequant_, false,
                 tPipe_);
    quantOp.Process(tileN, coreNum_, quantNandSLen, false);
    SyncAll();
    MatmulAllReduceReduceSumKernel<float>(reduceSumInGM_, reduceSumOutGM_, padM * tileN, rankNum_, tPipe_);
    SyncAll();
    quantOp.Init(reduceSumOutGM_, allGatherInGM_, padM / rankNum_, tileN, oneLineSCnt, coreNum_, maxProcRowsQuant_,
                 true, tPipe_);
    quantOp.Process(tileN, coreNum_, quantNandSLen, true);
    SyncAll();
    all2allOutGM_ += all2allOutOffset;
    reduceSumInGM_ += reduceSumInOffset;
    reduceSumOutGM_ += reduceSumOutOffset;
    allGatherInGM_ += allGatherInOffset;
    if (notifyFlag_) {
        hccl_.Commit(allGatherHandleId_[all2allWaitIdx_]);
        all2allWaitIdx_++;
    }
}

template <typename XType, typename WType, typename YType, class MmType, Mc2CoreType CoreType>
__aicore__ inline void MatmulAllReduceQuantPertileCommFp8<XType, WType, YType, MmType, CoreType>::Process()
{
    auto&& mc2Tiling = tilingData_->param;
    PrepareInit();
    MmType opTile;
    MatmulAllReduceDynamicQuantPertile<XType, float> quantOpTile;
    InnerProcess(opTile, quantOpTile, mc2Tiling.tileCnt, &tilingData_->tilematmulTiling, tilePadM_, mc2Tiling.isAdd,
                 false);
    if (mc2Tiling.tailM != 0U) {
        MmType opTail;
        MatmulAllReduceDynamicQuantPertile<XType, float> quantOpTail;
        InnerProcess(opTail, quantOpTail, mc2Tiling.tailCnt, &tilingData_->tailmatmulTiling, tailPadM_, mc2Tiling.isAdd,
                     true);
        StepOneTurn(opTail, quantOpTail, &tilingData_->tailmatmulTiling, tailPadM_, true, false);
    } else {
        StepOneTurn(opTile, quantOpTile, &tilingData_->tilematmulTiling, tilePadM_, false, false);
    }
    MatmulAllReduceDynamicQuantPertile<XType, YType> lastDequant;
    ProcessLast(lastDequant);
    if (notifyFlag_) {
        hccl_.Finalize();
    }
}

template <typename XType, typename WType, typename YType, class MmType, Mc2CoreType CoreType>
__aicore__ inline void MatmulAllReduceQuantPertileCommFp8<XType, WType, YType, MmType, CoreType>::ProcessLast(
    MatmulAllReduceDynamicQuantPertile<XType, YType> &quantOp)
{
    auto&& mc2Tiling = tilingData_->param;
    if ASCEND_IS_AIV {
        for (uint32_t i = 0U; i < (mc2Tiling.tileCnt + mc2Tiling.tailCnt); i++) {
            if (notifyFlag_) {
                hccl_.Wait(allGatherHandleId_[i]);
            }
            SyncAll();
            if (i < mc2Tiling.tileCnt) {
                uint32_t quantNandSLen =
                    tilingData_->tilematmulTiling.matmulTiling.N * sizeof(XType) + tileOneLineSCnt_ * sizeof(float);
                uint64_t allGatherOutOffset = tilingData_->tilematmulTiling.matmulTiling.M * quantNandSLen;
                uint64_t outOffset = tilingData_->tilematmulTiling.matmulTiling.M *
                                     tilingData_->tilematmulTiling.matmulTiling.N * sizeof(YType);
                quantOp.Init(allGatherOutGM_, outGM_, tilingData_->tilematmulTiling.matmulTiling.M,
                             tilingData_->tilematmulTiling.matmulTiling.N, tileOneLineSCnt_, coreNum_,
                             maxProcRowsDequantLast_, false, tPipe_);
                quantOp.Process(tilingData_->tilematmulTiling.matmulTiling.N, coreNum_, quantNandSLen, false);
                allGatherOutGM_ += allGatherOutOffset;
                outGM_ += outOffset;
            } else {
                uint32_t quantNandSLen =
                    tilingData_->tailmatmulTiling.matmulTiling.N * sizeof(XType) + tailOneLineSCnt_ * sizeof(float);
                uint64_t allGatherOutOffset = tilingData_->tailmatmulTiling.matmulTiling.M * quantNandSLen;
                uint64_t outOffset = tilingData_->tailmatmulTiling.matmulTiling.M *
                                     tilingData_->tailmatmulTiling.matmulTiling.N * sizeof(YType);
                quantOp.Init(allGatherOutGM_, outGM_, tilingData_->tailmatmulTiling.matmulTiling.M,
                             tilingData_->tailmatmulTiling.matmulTiling.N, tailOneLineSCnt_, coreNum_,
                             maxProcRowsDequantLast_, false, tPipe_);
                quantOp.Process(tilingData_->tailmatmulTiling.matmulTiling.N, coreNum_, quantNandSLen, false);                   
                allGatherOutGM_ += allGatherOutOffset;
                outGM_ += outOffset;
            }
            SyncAll();
        }
    }
}

#define INVOKE_MC2_QUANT_PERTILE_COMM_FP8_910_OP_IMPL(templateClass, coreType, isATrans, isBTrans...)                  \
    do {                                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(Mc2Tiling::QuantMatmulAllReduceTilingDataA5, tilingData, tilingGM);                \
        MC2GmAddrs addrs = {aGM, bGM, biasGM, addGM, cGM, workspaceGM, cGM};                                           \
        QuantGmAddrs quantAddrs = {nullptr, nullptr, nullptr, dequantGM, pertokenGM};                                  \
        using OpType = templateClass<DTYPE_X1, DTYPE_X2, float, DTYPE_BIAS, float, float, X1_FORMAT, X2_FORMAT,        \
                                     Y_FORMAT, isATrans, isBTrans, DTYPE_LOC_LOCAL,                                    \
                                     Mc2QuantBatchMatmulV3::Mc2QuantBmmAswBlock, MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG>;    \
        MatmulAllReduceQuantPertileCommFp8<DTYPE_X1, DTYPE_X2, DTYPE_Y, OpType, coreType> op;                          \
        op.Init(aGM, bGM, biasGM, addGM, dequantGM, pertokenGM, commQuantScale1GM, commQuantScale2GM, cGM, userWS,     \
                &tilingData, &tPipe);                                                                                  \
        op.Process();                                                                                                  \
        tPipe.Destroy();                                                                                               \
    } while (0)

} // naemspace MatmulAllReduceImpl
#endif // MATMUL_ALL_REDUCE_QUANT_PERTILE_COMM_FP8_H