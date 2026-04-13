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
            gmmxScaleCommOutGm_ = workspaceGM_ + workspaceOffset;
            computeScaleGm_ = gmmxScaleCommOutGm_;
            commOp.InitScaleBuffer(gmmxScaleGm_, gmmxScaleCommOutGm_);
            workspaceOffset = workspaceOffset +
                              Align(a_ * MXFP_MULTI_BASE_SIZE * CeilDiv(h1_, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)),
                                    TENSOR_LIST_SIZE);
        } else {
            computeScaleGm_ = gmmxScaleGm_;
        }
        GM_ADDR preCommBuffer = workspaceGM_ + workspaceOffset;
        workspaceOffset += Align((epWorldSize_ + 1) * epWorldSize_ * e_ * sizeof(uint64_t), TENSOR_LIST_SIZE);
        AscendC::printf("\n=== Scheduler Workspace Layout ===\n");
        AscendC::printf("workspace base=%p\n", workspaceGM_);
        AscendC::printf("commOutGm=%p (offset=%lu)\n", commOutGm_, 0UL);
        if (tilingData_->isPermuteOut) {
            AscendC::printf("  (permuteOut provided externally, not in workspace)\n");
        }
        if (MX_QUANT_MODE) {
            AscendC::printf("gmmxScaleCommOutGm=%p\n", gmmxScaleCommOutGm_);
        }
        AscendC::printf("preCommBuffer=%p\n", preCommBuffer);
        commOp.Init(hcclInitTiling, hcclCcTilingOffset, &tilingData_->taskTilingInfo, gmmxGM, commOutGm_,
                    preCommBuffer, workspaceGM_);
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
        commOp.WaitPreComm();
        
        // 从 commOp 获取动态计算的 baseA 和 totalRound
        baseA_ = commOp.GetBaseA();
        totalRound_ = commOp.GetTotalRound();
        
        AscendC::printf("=== Scheduler Process ===\n");
        AscendC::printf("baseA=%u, totalRound=%u\n", baseA_, totalRound_);

        if (IsNeedMM) {
            localComputeOp.Process(0);
            SyncAll<false>();
        }

        uint32_t maxRoundsPerCommBatch = MX_QUANT_MODE ? 31U : 63U;
        uint32_t commBatchCount = (totalRound_ + maxRoundsPerCommBatch - 1) / maxRoundsPerCommBatch;

        for (uint32_t commBatchIdx = 0U; commBatchIdx < commBatchCount; commBatchIdx++) {
            uint32_t startRound = commBatchIdx * maxRoundsPerCommBatch;
            uint32_t endRound = (commBatchIdx + 1) * maxRoundsPerCommBatch;
            if (endRound > totalRound_) {
                endRound = totalRound_;
            }

            for (uint32_t baseARoundIdx = startRound; baseARoundIdx < endRound; baseARoundIdx++) {
                uint32_t slotIdx = baseARoundIdx - startRound;

                if constexpr (MX_QUANT_MODE) {
                    commOp.LaunchScaleMultiRound(baseARoundIdx, slotIdx);
                }
                commOp.LaunchMultiRound(baseARoundIdx, slotIdx);

                if constexpr (MX_QUANT_MODE) {
                    commOp.WaitScaleMultiRound(slotIdx);
                }
                commOp.WaitMultiRound(slotIdx);
                SyncAll<false>();
                computeOp.ProcessAll(baseARoundIdx, baseA_);
            }
        }
        this->End();
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
    GM_ADDR gmmxScaleCommOutGm_ = nullptr;
    GM_ADDR computeScaleGm_ = nullptr;
    GM_ADDR workspaceGM_ = nullptr;
    GM_ADDR groupListGM_ = nullptr;
    const TilingDataType *tilingData_ = nullptr;
    uint64_t e_ = 0U;
    uint64_t h1_ = 0U;
    uint64_t a_ = 0U;
    uint64_t epWorldSize_ = 0U;
    uint32_t baseA_ = 0U;
    uint32_t totalRound_ = 0U;
};
}; // namespace MC2KernelTemplate
#endif