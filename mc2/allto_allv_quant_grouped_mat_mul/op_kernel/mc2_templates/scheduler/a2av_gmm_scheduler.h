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
        // init member variables
        GET_TILING_DATA(tilingData, tilingGM);
        tilingData_ = &tilingData;
        e_ = tilingData_->taskTilingInfo.e;
        gmmxScaleGm_ = gmmxScaleGM;
        isScaleBatchMode_ = e_ >= SCALE_COMM_BATCH_THRESHOLD;

        uint64_t workspaceOffset = 0;

        // 1. x通信输出区域：isPermuteOut=true时不占用workspace
        if (!tilingData_->isPermuteOut) {
            uint64_t commOutLen =
                Align(CeilDiv((tilingData_->taskTilingInfo.A) * (tilingData_->taskTilingInfo.H1), PACK_FACTOR),
                      TENSOR_LIST_SIZE);
            commOutGm_ = workspaceGM + workspaceOffset;
            workspaceOffset += commOutLen;
        } else {
            commOutGm_ = permuteOutOptionalGM;
        }
        computeScaleGm_ = gmmxScaleGm_;

        // gmmX comm init
        const void *hcclInitTiling = &(tilingData_->hcclA2avTilingInfo.hcclInitTiling);
        uint64_t hcclCcTilingOffset = offsetof(TilingDataType, hcclA2avTilingInfo) +
                                      offsetof(MC2KernelTemplate::HcclA2avTilingInfo, a2avCcTiling);
        commOp.Init(hcclInitTiling, hcclCcTilingOffset, &tilingData_->taskTilingInfo, gmmxGM, commOutGm_);

        // 2. scale permuteOut区域：MX场景scale重排后的输出（按专家排列）
        if constexpr (MX_QUANT_MODE) {
            uint64_t scaleAxis = CeilDiv(tilingData_->taskTilingInfo.H1, SCALE_ALIGNMENT_BLOCK_SIZE) * SCALE_MULTIPLIER;
            uint64_t permuteScaleOutSize = Align(tilingData_->taskTilingInfo.A * scaleAxis, TENSOR_LIST_SIZE);
            gmmxScalePermuteOutGm_ = workspaceGM + workspaceOffset;
            workspaceOffset += permuteScaleOutSize;

            // 3. scale commOut区域：仅e>=32时需要，scale AlltoAllV通信输出（按rank排列）
            if (isScaleBatchMode_) {
                uint64_t scaleCommOutLen =
                    Align(static_cast<uint64_t>(tilingData_->taskTilingInfo.e) *
                              static_cast<uint64_t>(tilingData_->taskTilingInfo.epWorldSize) * scaleAxis,
                          TENSOR_LIST_SIZE);
                gmmxScaleCommOutGm_ = workspaceGM + workspaceOffset;
                workspaceOffset += scaleCommOutLen;
            } else {
                gmmxScaleCommOutGm_ = gmmxScalePermuteOutGm_;
            }

            commOp.InitScaleBuffer(gmmxScaleGm_, gmmxScalePermuteOutGm_, gmmxScaleCommOutGm_);
            computeScaleGm_ = gmmxScalePermuteOutGm_;
        }

        // 4. 计算类使用的workspace起始地址（偏移后的workspace）
        computeWorkspaceGm_ = workspaceGM + workspaceOffset;

        // mm compute init
        if (IsNeedMM) {
            localComputeOp.Init(mmxOptionalGM, mmweightOptionalGM, mmxScaleGM, mmWeightScaleGM, mmyOptionalGM,
                                computeWorkspaceGm_, tilingData_, &tilingData_->mmQuantTilingData, mmArrayAddrIn, tPipe,
                                isA2avGmmFlag);
        }
        // gmm compute init
        computeOp.Init(commOutGm_, gmmweightGM, computeScaleGm_, gmmWeightScaleGM, gmmyGM, computeWorkspaceGm_,
                       tilingData_, &tilingData_->gmmQuantTilingData, gmmArrayAddrIn, tPipe, isA2avGmmFlag);
    }

    __aicore__ inline void Process()
    {
        if (IsNeedMM) {
            localComputeOp.Process(0);
            SyncAll<false>();
        }
        if constexpr (MX_QUANT_MODE) {
            if (isScaleBatchMode_) {
                ProcessScaleCommBatchMode();
            } else {
                ProcessScaleCommRoundMode();
            }
        } else {
            ProcessCommRoundMode();
        }
        this->End();
    }

private:
    static constexpr uint32_t SCALE_COMM_BATCH_THRESHOLD = 32U;

    __aicore__ inline void ProcessScaleCommBatchMode()
    {
        AscendC::printf("ProcessScaleCommBatchMode: e=%u\n", e_);
        commOp.LaunchScaleAllOnce(0, e_);
        for (uint32_t expertIdx = 0U; expertIdx < e_; expertIdx++) {
            commOp.Launch(expertIdx, 1);
        }
        commOp.WaitScale(0);
        SyncAll<false>();
        commOp.PermuteScale();
        SyncAll<false>();
        for (uint32_t expertIdx = 0U; expertIdx < e_; expertIdx++) {
            commOp.Wait(expertIdx);
            SyncAll<false>();
            computeOp.Process(expertIdx);
        }
    }

    __aicore__ inline void ProcessScaleCommRoundMode()
    {
        uint32_t max_hccl_handle_num = 31U;
        uint32_t totalRound = (e_ + max_hccl_handle_num - 1) / max_hccl_handle_num;
        for (uint32_t roundIdx = 0; roundIdx < totalRound; roundIdx++) {
            uint32_t startExpertIdx = roundIdx * max_hccl_handle_num;
            uint32_t endExpertIdx = (roundIdx + 1) * max_hccl_handle_num;
            endExpertIdx = endExpertIdx > e_ ? e_ : endExpertIdx;
            for (uint32_t expertIdx = startExpertIdx; expertIdx < endExpertIdx; expertIdx++) {
                commOp.LaunchScaleBeforeCompute(expertIdx, 1);
                commOp.Launch(expertIdx, 1);
            }
            for (uint32_t expertIdx = startExpertIdx; expertIdx < endExpertIdx; expertIdx++) {
                commOp.WaitScale(expertIdx);
                commOp.Wait(expertIdx);
                SyncAll<false>();
                computeOp.Process(expertIdx);
            }
        }
    }

    __aicore__ inline void ProcessCommRoundMode()
    {
        uint32_t max_hccl_handle_num = 63U;
        uint32_t totalRound = (e_ + max_hccl_handle_num - 1) / max_hccl_handle_num;
        for (uint32_t roundIdx = 0; roundIdx < totalRound; roundIdx++) {
            uint32_t startExpertIdx = roundIdx * max_hccl_handle_num;
            uint32_t endExpertIdx = (roundIdx + 1) * max_hccl_handle_num;
            endExpertIdx = endExpertIdx > e_ ? e_ : endExpertIdx;
            for (uint32_t expertIdx = startExpertIdx; expertIdx < endExpertIdx; expertIdx++) {
                commOp.Launch(expertIdx, 1);
            }
            for (uint32_t expertIdx = startExpertIdx; expertIdx < endExpertIdx; expertIdx++) {
                commOp.Wait(expertIdx);
                SyncAll<false>();
                computeOp.Process(expertIdx);
            }
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
    static constexpr uint64_t SCALE_ALIGNMENT_BLOCK_SIZE = 64U;
    static constexpr uint64_t SCALE_MULTIPLIER = 2U;

    CommOpType commOp;
    ComputeOpType computeOp;
    LocalComputeOpType localComputeOp;
    GM_ADDR commOutGm_ = nullptr;
    GM_ADDR gmmxScaleGm_ = nullptr;
    GM_ADDR gmmxScalePermuteOutGm_ = nullptr;
    GM_ADDR gmmxScaleCommOutGm_ = nullptr;
    GM_ADDR computeScaleGm_ = nullptr;
    GM_ADDR computeWorkspaceGm_ = nullptr;
    const TilingDataType *tilingData_ = nullptr;
    uint32_t e_ = 0U;
    bool isScaleBatchMode_ = false;
};
}; // namespace MC2KernelTemplate
#endif