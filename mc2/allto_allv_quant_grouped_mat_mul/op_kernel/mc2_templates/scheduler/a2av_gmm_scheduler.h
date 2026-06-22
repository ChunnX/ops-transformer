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
        isCombinedMode_ = tilingData_->taskTilingInfo.combineExpertSize > 1U;
        // workspace偏移
        uint64_t workspaceOffset = 0;
        /**
         * workspace排布：
         * 0、通信重排输出，大小为commOutLen
         * 1、
        **/
        if (isCombinedMode_ || !tilingData_->isPermuteOut) {
            // 0、合并通信模式用来存储未重排的通信结果、无permuteOut输出用来存储通信结果
            commOutGm_ = workspaceGM + workspaceOffset;
            uint64_t commOutLen = Align(
                CeilDiv((tilingData_->taskTilingInfo.A) * (tilingData_->taskTilingInfo.H1), PACK_FACTOR) * X_TYPE_SIZE,
                TENSOR_LIST_SIZE);
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
        // scale commOut区域：MX场景scale通信输出
        if constexpr (MX_QUANT_MODE) {
            uint64_t scaleAxis = CeilDiv(tilingData_->taskTilingInfo.H1, SCALE_ALIGNMENT_BLOCK_SIZE) * 2;
            uint64_t permuteScaleOutSize = Align(tilingData_->taskTilingInfo.A * scaleAxis, TENSOR_LIST_SIZE);
            gmmxScaleCommOutGm_ = workspaceGM + workspaceOffset;
            workspaceOffset += permuteScaleOutSize;
            if (isScaleBatchMode_ && !isCombinedMode_) {
                gmmxScalePermuteOutGm_ = workspaceGM + workspaceOffset;
                workspaceOffset += permuteScaleOutSize;
            } else {
                gmmxScalePermuteOutGm_ = gmmxScaleCommOutGm_;
            }
            commOp.InitScaleBuffer(gmmxScaleGm_, gmmxScaleCommOutGm_, gmmxScalePermuteOutGm_);
            computeScaleGm_ = isCombinedMode_ ? gmmxScaleCommOutGm_ : gmmxScalePermuteOutGm_;
        }
        if (isCombinedMode_) {
            uint64_t combinedGroupListSize = sizeof(int64_t) *
                tilingData_->taskTilingInfo.combineExpertSize * tilingData_->taskTilingInfo.epWorldSize;
            combinedGroupListGm_ = workspaceGM + workspaceOffset;
            workspaceOffset += combinedGroupListSize;
            uint64_t yOffsetListSize = sizeof(int64_t) *
                tilingData_->taskTilingInfo.combineExpertSize * tilingData_->taskTilingInfo.epWorldSize;
            yOffsetListGm_ = workspaceGM + workspaceOffset;
            workspaceOffset += yOffsetListSize;
            if (tilingData_->isPermuteOut) {
                // AscendC::printf("InitOutputPermuteBuffer commOutGm_ = %p, permuteOutOptionalGM = %p\n", commOutGm_, permuteOutOptionalGM);
                commOp.InitOutputPermuteBuffer(commOutGm_, permuteOutOptionalGM);
            }
        }
        computeWorkspaceGm_ = workspaceGM + workspaceOffset;
        if (tilingData_->isNeedMM != 0) {
            localComputeOp.Init(mmxOptionalGM, mmweightOptionalGM, mmxScaleGM, mmWeightScaleGM, mmyOptionalGM,
                                computeWorkspaceGm_, tilingData_,
                                &tilingData_->mmQuantTilingData, mmArrayAddrIn, tPipe, isA2avGmmFlag);
        }
        computeOp.Init(commOutGm_, gmmweightGM, computeScaleGm_, gmmWeightScaleGM, gmmyGM,
                       computeWorkspaceGm_, tilingData_,
                       &tilingData_->gmmQuantTilingData, gmmArrayAddrIn, tPipe, isA2avGmmFlag);
    }

    __aicore__ inline void Process()
    {
        if (tilingData_->isNeedMM != 0) {
            localComputeOp.Process(0);
        }
        if (isCombinedMode_) {
            ProcessCombinedExpertMode();
        } else if constexpr (MX_QUANT_MODE) {
            if (isScaleBatchMode_) {
                ProcessScaleCommBatchMode();
            } else {
                ProcessCommRoundMode();
            }
        } else {
            ProcessCommRoundMode();
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
    __aicore__ inline void ProcessScaleCommBatchMode()
    {
        commOp.LaunchScaleBeforeCompute(0, e_);
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

    __aicore__ inline void ProcessCommRoundMode()
    {
        for (uint32_t expertIdx = 0; expertIdx < e_; expertIdx++) {
            if constexpr (MX_QUANT_MODE) {
                commOp.LaunchScaleBeforeCompute(expertIdx, 1);
            }
            commOp.Launch(expertIdx, 1);
        }
        for (uint32_t expertIdx = 0; expertIdx < e_; expertIdx++) {
            if constexpr (MX_QUANT_MODE) {
                commOp.WaitScale(expertIdx);
            }
            commOp.Wait(expertIdx);
            SyncAll<false>();
            computeOp.Process(expertIdx);
        }
    }

    __aicore__ inline void ProcessCombinedExpertMode()
    {
        uint32_t commTurn = tilingData_->taskTilingInfo.commTurn;
        uint32_t combineExpertSize = tilingData_->taskTilingInfo.combineExpertSize;
        bool isPermuteOut = tilingData_->isPermuteOut;

        for (uint32_t turnIdx = 0U; turnIdx < commTurn; turnIdx++) {
            uint32_t realCombineSize = (turnIdx == commTurn - 1U) ?
                (e_ - turnIdx * combineExpertSize) : combineExpertSize;
            commOp.LaunchCombinedAic(turnIdx, realCombineSize);
            if constexpr (MX_QUANT_MODE) {
                // commOp.LaunchScaleCombinedAic(turnIdx, realCombineSize);
            }
        }

        for (uint32_t turnIdx = 0U; turnIdx < commTurn; turnIdx++) {
            uint32_t realCombineSize = (turnIdx == commTurn - 1U) ?
                (e_ - turnIdx * combineExpertSize) : combineExpertSize;
            commOp.WaitCombinedAic(turnIdx);

            if (isPermuteOut) {
                uint32_t notifyFlagId = NOTIFY_FLAG_ID_BASE + (turnIdx % FLAG_ID_ROTATE_COUNT);
                if ASCEND_IS_AIC {
                    AscendC::printf("CrossCoreSetFlag notifyFlagId = %d\n", notifyFlagId);
                    CrossCoreSetFlag<0x2, PIPE_FIX>(notifyFlagId);
                }
                if ASCEND_IS_AIV {
                    AscendC::printf("CrossCoreWaitFlag notifyFlagId = %d\n", notifyFlagId);
                    CrossCoreWaitFlag(notifyFlagId);
                    AscendC::printf("PermuteOutput turnIdx = %d, realCombineSize = %d\n", turnIdx, realCombineSize);
                    commOp.PermuteOutput(turnIdx, realCombineSize);
                }
            }

            if constexpr (MX_QUANT_MODE) {
                // AscendC::printf("WaitScaleCombinedAic turnIdx = %d\n", turnIdx);
                // commOp.WaitScaleCombinedAic(turnIdx);
            }

            // computeOp.ProcessCombined(turnIdx, realCombineSize, combinedGroupListGm_, yOffsetListGm_);
        }
    }

    CommOpType commOp;
    ComputeOpType computeOp;
    LocalComputeOpType localComputeOp;
    GM_ADDR commOutGm_ = nullptr;
    GM_ADDR gmmxScaleGm_ = nullptr;
    GM_ADDR gmmxScalePermuteOutGm_ = nullptr;
    GM_ADDR gmmxScaleCommOutGm_ = nullptr;
    GM_ADDR computeScaleGm_ = nullptr;
    GM_ADDR computeWorkspaceGm_ = nullptr;
    GM_ADDR combinedGroupListGm_ = nullptr;
    GM_ADDR yOffsetListGm_ = nullptr;
    const TilingDataType *tilingData_ = nullptr;
    uint32_t e_ = 0U;
    bool isScaleBatchMode_ = false;
    bool isCombinedMode_ = false;

    static constexpr uint32_t NOTIFY_FLAG_ID_BASE = 8U;
    static constexpr uint32_t FLAG_ID_ROTATE_COUNT = 3U;
};
}; // namespace MC2KernelTemplate
#endif