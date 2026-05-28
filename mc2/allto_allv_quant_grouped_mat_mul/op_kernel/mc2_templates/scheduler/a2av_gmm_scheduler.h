/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file pipeline_template_comm_compute.h
 * \brief
 */

#ifndef MC2_PIPELINE_TEMPLATE_COMM_COMPUTE_H
#define MC2_PIPELINE_TEMPLATE_COMM_COMPUTE_H

#include "../a2av_gmm_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

using namespace AscendC;

namespace MC2KernelTemplate {
template <typename CommOpType, typename ComputeOpType, typename LocalComputeOpType, typename TilingDataType,
          typename GmmTilingDataType, typename GmmArrayAddrType>
class A2avGmmScheduler {
public:
    __aicore__ inline A2avGmmScheduler() {}
    __aicore__ inline void Init(GM_ADDR gmmxGM, GM_ADDR gmmweightGM, GM_ADDR mmxOptionalGM, GM_ADDR mmweightOptionalGM,
                                GM_ADDR gmmxScaleGM, GM_ADDR gmmWeightScaleGM, GM_ADDR mmxScaleGM,
                                GM_ADDR mmWeightScaleGM, GM_ADDR gmmyGM, GM_ADDR mmyOptionalGM,
                                GM_ADDR permuteOutOptionalGM, GM_ADDR workspaceGM, GM_ADDR tilingGM,
                                GmmArrayAddrType *gmmArrayAddrIn, GmmArrayAddrType *mmArrayAddrIn, TPipe *tPipe,
                                bool isA2avGmmFlag);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void End();

private:
    __aicore__ inline void ProcessScaleCommBatchMode();
    __aicore__ inline void ProcessCommRoundMode();

private:
    CommOpType commOp;
    ComputeOpType computeOp;
    LocalComputeOpType localComputeOp;
    GM_ADDR commOutGm_ = nullptr;
    GM_ADDR gmmxPermuteGM_ = nullptr;
    GM_ADDR gmmxScaleGm_ = nullptr;
    GM_ADDR gmmxScalePermuteOutGm_ = nullptr;
    GM_ADDR gmmxScaleCommOutGm_ = nullptr;
    GM_ADDR computeGmmxGM_ = nullptr;
    GM_ADDR computeScaleGm_ = nullptr;
    GM_ADDR computeWorkspaceGm_ = nullptr;
    const TilingDataType *tilingData_ = nullptr;
    uint32_t e_ = 0U;
    uint32_t expertNum_ = 1;
    bool isNeedPermute_ = false;
};

template <typename CommOpType, typename ComputeOpType, typename LocalComputeOpType, typename TilingDataType,
          typename GmmTilingDataType, typename GmmArrayAddrType>
__aicore__ inline void A2avGmmScheduler<CommOpType, ComputeOpType, LocalComputeOpType, TilingDataType,
    GmmTilingDataType, GmmArrayAddrType>::Init(GM_ADDR gmmxGM, GM_ADDR gmmweightGM, GM_ADDR mmxOptionalGM,
    GM_ADDR mmweightOptionalGM, GM_ADDR gmmxScaleGM, GM_ADDR gmmWeightScaleGM, GM_ADDR mmxScaleGM,
    GM_ADDR mmWeightScaleGM, GM_ADDR gmmyGM, GM_ADDR mmyOptionalGM, GM_ADDR permuteOutOptionalGM, GM_ADDR workspaceGM,
    GM_ADDR tilingGM, GmmArrayAddrType *gmmArrayAddrIn, GmmArrayAddrType *mmArrayAddrIn, TPipe *tPipe, bool isA2avGmmFlag)
{
    GET_TILING_DATA(tilingData, tilingGM);
    tilingData_ = &tilingData;
    e_ = tilingData_->taskTilingInfo.e;
    expertNum_ = tilingData_->taskTilingInfo.expertNum;
    gmmxScaleGm_ = gmmxScaleGM;
    uint64_t a2avLaunchNum = CeilDiv(e_, expertNum_);
    uint64_t a2avLimit = MX_QUANT_MODE ? CeilDiv(MAX_HCCL_PREPARE_NUM, 2) : MAX_HCCL_PREPARE_NUM;
    expertNum_ = a2avLaunchNum > a2avLimit ? CeilDiv(e_, a2avLimit) : expertNum_;
    isNeedPermute_ = expertNum_ > 1;
    uint64_t workspaceOffset = 0;
    uint64_t gmmxSize = Align(CeilDiv((tilingData_->taskTilingInfo.A) * (tilingData_->taskTilingInfo.H1),
                                PACK_FACTOR) * X_TYPE_SIZE, TENSOR_LIST_SIZE);
    if (!tilingData_->isPermuteOut) {
        commOutGm_ = workspaceGM + workspaceOffset;
        workspaceOffset += gmmxSize;
        gmmxPermuteGM_ = workspaceGM + workspaceOffset;
        workspaceOffset += gmmxSize;
        computeGmmxGM_ = isNeedPermute_ ? gmmxPermuteGM_ : commOutGm_;
    } else {
        GM_ADDR tempGM = workspaceGM + workspaceOffset;
        workspaceOffset += gmmxSize;
        commOutGm_ = isNeedPermute_ ? tempGM : permuteOutOptionalGM;
        gmmxPermuteGM_ = isNeedPermute_ ? permuteOutOptionalGM : tempGM;
        computeGmmxGM_ = permuteOutOptionalGM;
    }
    computeScaleGm_ = gmmxScaleGm_;
    const void *hcclInitTiling = &(tilingData_->hcclA2avTilingInfo.hcclInitTiling);
    uint64_t hcclCcTilingOffset = offsetof(TilingDataType, hcclA2avTilingInfo) +
                                  offsetof(MC2KernelTemplate::HcclA2avTilingInfo, a2avCcTiling);
    commOp.Init(hcclInitTiling, hcclCcTilingOffset, &tilingData_->taskTilingInfo, gmmxGM, commOutGm_, gmmxPermuteGM_);
    if constexpr (MX_QUANT_MODE) {
        uint64_t scaleAxis = CeilDiv(tilingData_->taskTilingInfo.H1, SCALE_ALIGNMENT_BLOCK_SIZE) * 2;
        uint64_t permuteScaleOutSize = Align(tilingData_->taskTilingInfo.A * scaleAxis, TENSOR_LIST_SIZE);
        gmmxScaleCommOutGm_ = workspaceGM + workspaceOffset;
        workspaceOffset += permuteScaleOutSize;
        gmmxScalePermuteOutGm_ = workspaceGM + workspaceOffset;
        workspaceOffset += permuteScaleOutSize;
        commOp.InitScaleBuffer(gmmxScaleGm_, gmmxScaleCommOutGm_, gmmxScalePermuteOutGm_);
        computeScaleGm_ = isNeedPermute_ ? gmmxScalePermuteOutGm_ : gmmxScaleCommOutGm_;
    }
    computeWorkspaceGm_ = workspaceGM + workspaceOffset;
    if (tilingData_->isNeedMM != 0) {
        localComputeOp.Init(mmxOptionalGM, mmweightOptionalGM, mmxScaleGM, mmWeightScaleGM, mmyOptionalGM,
                            computeWorkspaceGm_, tilingData_, &tilingData_->mmQuantTilingData, mmArrayAddrIn, tPipe,
                            isA2avGmmFlag);
    }
    computeGmmxGM_ = isNeedPermute_ ? gmmxPermuteGM_ : commOutGm_;
    computeOp.Init(computeGmmxGM_, gmmweightGM, computeScaleGm_, gmmWeightScaleGM, gmmyGM, computeWorkspaceGm_,
                   tilingData_, &tilingData_->gmmQuantTilingData, gmmArrayAddrIn, tPipe, isA2avGmmFlag);
}

template <typename CommOpType, typename ComputeOpType, typename LocalComputeOpType, typename TilingDataType,
          typename GmmTilingDataType, typename GmmArrayAddrType>
__aicore__ inline void A2avGmmScheduler<CommOpType, ComputeOpType, LocalComputeOpType, TilingDataType,
    GmmTilingDataType, GmmArrayAddrType>::Process()
{
    if (tilingData_->isNeedMM != 0) {
        localComputeOp.Process(0, 1);
    }
    ProcessCommRoundMode();
    this->End();
}

template <typename CommOpType, typename ComputeOpType, typename LocalComputeOpType, typename TilingDataType,
          typename GmmTilingDataType, typename GmmArrayAddrType>
__aicore__ inline void A2avGmmScheduler<CommOpType, ComputeOpType, LocalComputeOpType, TilingDataType,
    GmmTilingDataType, GmmArrayAddrType>::End()
{
    commOp.End();
    computeOp.End();
    localComputeOp.End();
}

template <typename CommOpType, typename ComputeOpType, typename LocalComputeOpType, typename TilingDataType,
          typename GmmTilingDataType, typename GmmArrayAddrType>
__aicore__ inline void A2avGmmScheduler<CommOpType, ComputeOpType, LocalComputeOpType, TilingDataType,
    GmmTilingDataType, GmmArrayAddrType>::ProcessCommRoundMode()
{
    for (uint32_t startExpertIdx = 0; startExpertIdx < e_; startExpertIdx += expertNum_) {
        uint32_t actualExpertNum = (startExpertIdx + expertNum_ > e_) ? (e_ - startExpertIdx) : expertNum_;
        if constexpr (MX_QUANT_MODE) {
            commOp.LaunchCommScale(startExpertIdx, actualExpertNum);
        }
        commOp.LaunchCommData(startExpertIdx, actualExpertNum);
    }
    for (uint32_t startExpertIdx = 0; startExpertIdx < e_; startExpertIdx += expertNum_) {
        uint32_t actualExpertNum = (startExpertIdx + expertNum_ > e_) ? (e_ - startExpertIdx) : expertNum_;
        if constexpr (MX_QUANT_MODE) {
            commOp.WaitScale(startExpertIdx);
            if (isNeedPermute_) {
                commOp.PermuteScale(startExpertIdx, actualExpertNum);
            }
        }
        commOp.Wait(startExpertIdx);
        if (isNeedPermute_) {
            commOp.PermuteData(startExpertIdx, actualExpertNum);
        }
        SyncAll<false>();
        computeOp.Process(startExpertIdx, actualExpertNum);
    }
}

}; // namespace MC2KernelTemplate
#endif