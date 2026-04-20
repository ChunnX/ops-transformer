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

constexpr int32_t MXFP_DIVISOR_SIZE = 64;
constexpr int32_t MXFP_MULTI_BASE_SIZE = 2;

template <typename CommOpType, typename ComputeOpType, typename LocalComputeOpType, typename TilingDataType,
          typename GmmTilingDataType, typename GmmArrayAddrType, bool IsNeedMM>
class A2avGmmScheduler {
public:
    __aicore__ inline void Init(GM_ADDR gmmxGM, GM_ADDR gmmweightGM, GM_ADDR mmxOptionalGM, GM_ADDR mmweightOptionalGM,
                                GM_ADDR gmmxScaleGM, GM_ADDR gmmWeightScaleGM, GM_ADDR mmxScaleGM,
                                GM_ADDR mmWeightScaleGM, GM_ADDR gmmyGM, GM_ADDR mmyOptionalGM,
                                GM_ADDR permuteOutOptionalGM, GM_ADDR workspaceGM, GM_ADDR tilingGM,
                                GmmArrayAddrType *gmmArrayAddrIn, GmmArrayAddrType *mmArrayAddrIn, TPipe *tPipe,
                                bool isA2avGmmFlag)
    {
        GET_TILING_DATA(tilingData, tilingGM);
        tilingData_ = &tilingData;
        e_ = tilingData_->taskTilingInfo.e;
        h1_ = tilingData_->taskTilingInfo.H1;
        a_ = tilingData_->taskTilingInfo.A;
        epWorldSize_ = tilingData_->taskTilingInfo.epWorldSize;
        gmmxScaleGm_ = gmmxScaleGM;
        workspaceGM_ = workspaceGM;
        const void *hcclInitTiling = &(tilingData_->hcclA2avTilingInfo.hcclInitTiling);
        uint64_t hcclCcTilingOffset = offsetof(TilingDataType, hcclA2avTilingInfo) +
                                      offsetof(MC2KernelTemplate::HcclA2avTilingInfo, a2avCcTiling);
        uint64_t workspaceOffset = 0;
        if (tilingData_->isPermuteOut) {
            commOutGm_ = permuteOutOptionalGM;
        } else {
            commOutGm_ = workspaceGM_;
            workspaceOffset =
                workspaceOffset +
                Align((tilingData_->taskTilingInfo.A) * (tilingData_->taskTilingInfo.H1), TENSOR_LIST_SIZE);
        }
        if (MX_QUANT_MODE) {
            uint64_t scaleAxis = MXFP_MULTI_BASE_SIZE * CeilDiv(h1_, static_cast<uint64_t>(MXFP_DIVISOR_SIZE));
            uint64_t scalePermuteOutSize = Align(a_ * scaleAxis, TENSOR_LIST_SIZE);
            gmmxScalePermuteOutGm_ = workspaceGM_ + workspaceOffset;
            workspaceOffset += scalePermuteOutSize;
            computeScaleGm_ = gmmxScalePermuteOutGm_;

            constexpr uint32_t SCALE_BATCH_THRESHOLD_ROUND = 32U;
            if (e_ >= SCALE_BATCH_THRESHOLD_ROUND) {
                uint64_t scaleCommOutSize = Align(e_ * epWorldSize_ * scaleAxis, TENSOR_LIST_SIZE);
                gmmxScaleCommOutGm_ = workspaceGM_ + workspaceOffset;
                workspaceOffset += scaleCommOutSize;
            } else {
                gmmxScaleCommOutGm_ = gmmxScalePermuteOutGm_;
            }

            commOp.InitScaleBuffer(gmmxScaleGm_, gmmxScaleCommOutGm_, gmmxScalePermuteOutGm_);
        } else {
            computeScaleGm_ = gmmxScaleGm_;
        }
        GM_ADDR preCommBuffer = workspaceGM_ + workspaceOffset;
        workspaceOffset += Align(2U * epWorldSize_ * epWorldSize_ * e_ * sizeof(uint64_t), TENSOR_LIST_SIZE);
        commOp.Init(hcclInitTiling, hcclCcTilingOffset, &tilingData_->taskTilingInfo, gmmxGM, commOutGm_, preCommBuffer,
                    workspaceGM_);
        groupListGM_ = workspaceGM_ + workspaceOffset;
        computeOp.Init(commOutGm_, gmmweightGM, computeScaleGm_, gmmWeightScaleGM, gmmyGM, groupListGM_, tilingData_,
                       &tilingData_->gmmQuantTilingData, gmmArrayAddrIn, tPipe, isA2avGmmFlag);
        if (IsNeedMM) {
            localComputeOp.Init(mmxOptionalGM, mmweightOptionalGM, mmxScaleGM, mmWeightScaleGM, mmyOptionalGM,
                                groupListGM_, tilingData_, &tilingData_->mmQuantTilingData, mmArrayAddrIn, tPipe,
                                isA2avGmmFlag);
        }
    }

    __aicore__ inline void Process()
    {
        commOp.InitPreComm();

        if (IsNeedMM) {
            localComputeOp.Process(0);
        }

        commOp.WaitPreComm();
        SyncAll<false>();

        localBaseA_ = commOp.GetLocalBaseA();
        totalRound_ = commOp.GetTotalRound();

        if ASCEND_IS_AIV {
            if (GetBlockIdx() == 0) {
                AscendC::printf("[Scheduler] rank%u: totalRound=%u localBaseA=%u\n", commOp.GetRankId(), totalRound_,
                                localBaseA_);
            }
        }

        if constexpr (MX_QUANT_MODE) {
            constexpr uint32_t MX_MAX_HANDLE_NUM = 31U;
            if (totalRound_ > MX_MAX_HANDLE_NUM) {
                ProcessScaleBatchMode();
            } else {
                ProcessScaleRoundMode();
            }
        } else {
            ProcessCommRoundMode();
        }
        this->End();
    }

private:
    static constexpr uint32_t MX_MAX_HANDLE_NUM = 31U;

    __aicore__ inline void ProcessScaleBatchMode()
    {
        commOp.LaunchScaleAllOnce();
        for (uint32_t roundIdx = 0U; roundIdx < totalRound_; roundIdx++) {
            commOp.LaunchMultiRound(roundIdx);
        }
        commOp.WaitScaleBatch();
        // SyncAll<false>();
        commOp.PermuteScale();
        // SyncAll<false>();
        for (uint32_t roundIdx = 0U; roundIdx < totalRound_; roundIdx++) {
            commOp.WaitMultiRound(roundIdx);
            // SyncAll<false>();
            computeOp.ProcessAll(roundIdx, localBaseA_);
        }
    }

    __aicore__ inline void ProcessScaleRoundMode()
    {
        for (uint32_t roundIdx = 0U; roundIdx < totalRound_; roundIdx++) {
            commOp.LaunchScaleMultiRound(roundIdx);
            commOp.LaunchMultiRound(roundIdx);
            commOp.WaitScaleMultiRound(roundIdx);
            commOp.WaitMultiRound(roundIdx);
            SyncAll<false>();
            computeOp.ProcessAll(roundIdx, localBaseA_);
        }
    }

    __aicore__ inline void ProcessCommRoundMode()
    {
        for (uint32_t roundIdx = 0U; roundIdx < totalRound_; roundIdx++) {
            commOp.LaunchMultiRound(roundIdx);
            commOp.WaitMultiRound(roundIdx);
            SyncAll<false>();
            computeOp.ProcessAll(roundIdx, localBaseA_);
        }
    }

protected:
    __aicore__ inline void End()
    {
        commOp.End();
        computeOp.End();
        localComputeOp.End();
    }

private:
    CommOpType commOp;
    ComputeOpType computeOp;
    LocalComputeOpType localComputeOp;
    GM_ADDR commOutGm_ = nullptr;
    GM_ADDR gmmxScaleGm_ = nullptr;
    GM_ADDR gmmxScalePermuteOutGm_ = nullptr;
    GM_ADDR gmmxScaleCommOutGm_ = nullptr;
    GM_ADDR computeScaleGm_ = nullptr;
    GM_ADDR workspaceGM_ = nullptr;
    GM_ADDR groupListGM_ = nullptr;
    const TilingDataType *tilingData_ = nullptr;
    uint64_t e_ = 0U;
    uint64_t h1_ = 0U;
    uint64_t a_ = 0U;
    uint64_t epWorldSize_ = 0U;
    uint32_t localBaseA_ = 0U;
    uint32_t totalRound_ = 0U;
};
}; // namespace MC2KernelTemplate
#endif