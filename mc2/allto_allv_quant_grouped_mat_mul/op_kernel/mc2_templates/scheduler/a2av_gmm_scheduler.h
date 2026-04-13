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
        commOutGm_ = tilingData_->isPermuteOut ? permuteOutOptionalGM : workspaceGM;
        computeScaleGm_ = gmmxScaleGm_; // 没有scale通信时为gmmxScaleGm_
        // gmmX comm init
        const void *hcclInitTiling = &(tilingData_->hcclA2avTilingInfo.hcclInitTiling);
        uint64_t hcclCcTilingOffset = offsetof(TilingDataType, hcclA2avTilingInfo) +
                                      offsetof(MC2KernelTemplate::HcclA2avTilingInfo, a2avCcTiling);
        commOp.Init(hcclInitTiling, hcclCcTilingOffset, &tilingData_->taskTilingInfo, gmmxGM, commOutGm_);
        // gmmXScale comm init(MX quant)
        if constexpr (MX_QUANT_MODE) {
            uint64_t commOutLen =
                Align(CeilDiv((tilingData_->taskTilingInfo.A) * (tilingData_->taskTilingInfo.H1), PACK_FACTOR),
                TENSOR_LIST_SIZE);
            gmmxScaleCommOutGm_ = tilingData_->isPermuteOut ? workspaceGM : workspaceGM + commOutLen;
            commOp.InitScaleBuffer(gmmxScaleGm_, gmmxScaleCommOutGm_);
            computeScaleGm_ = gmmxScaleCommOutGm_; // 存在scale通信时为通信后permuteScaleout的地址
        }
        // mm compute init
        if (IsNeedMM) {
            localComputeOp.Init(mmxOptionalGM, mmweightOptionalGM, mmxScaleGM, mmWeightScaleGM, mmyOptionalGM,
                                workspaceGM, tilingData_, &tilingData_->mmQuantTilingData, mmArrayAddrIn, tPipe,
                                isA2avGmmFlag);
        }
        // gmm compute init
        computeOp.Init(commOutGm_, gmmweightGM, computeScaleGm_, gmmWeightScaleGM, gmmyGM, workspaceGM, tilingData_,
                       &tilingData_->gmmQuantTilingData, gmmArrayAddrIn, tPipe, isA2avGmmFlag);
    }

    __aicore__ inline void Process()
    {
        // mm compute
        if (IsNeedMM) {
            localComputeOp.Process(0);
            SyncAll<false>();
        }
        uint32_t max_hccl_handle_num = MX_QUANT_MODE ? 31U : 63U;
        uint32_t totalRound = (e_ + max_hccl_handle_num - 1) / max_hccl_handle_num;
        for (uint32_t roundIdx = 0; roundIdx < totalRound; roundIdx++) {
            uint32_t startExpertIdx = roundIdx * max_hccl_handle_num;
            uint32_t endExpertIdx = (roundIdx + 1) * max_hccl_handle_num;
            endExpertIdx = endExpertIdx > e_ ? e_ : endExpertIdx;
            for (uint32_t expertIdx = startExpertIdx; expertIdx < endExpertIdx; expertIdx++) {
                if constexpr (MX_QUANT_MODE) {
                    commOp.LaunchScaleBeforeCompute(expertIdx, 1);
                }
                commOp.Launch(expertIdx, 1);
            }
            for (uint32_t expertIdx = startExpertIdx; expertIdx < endExpertIdx; expertIdx++) {
                if constexpr (MX_QUANT_MODE) {
                    commOp.WaitScale(expertIdx);
                }
                commOp.Wait(expertIdx);
                SyncAll<false>();
                computeOp.Process(expertIdx);
            }
        }
        // end
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
    const TilingDataType *tilingData_ = nullptr;
    uint32_t e_ = 0U;
};
}; // namespace MC2KernelTemplate
#endif