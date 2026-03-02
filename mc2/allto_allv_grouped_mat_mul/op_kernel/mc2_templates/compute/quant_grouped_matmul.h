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
 * \file quant_grouped_matmul.h
 * \brief
 */
 
#ifndef MC2_QUANT_GROUPED_MATMUL_H
#define MC2_QUANT_GROUPED_MATMUL_H

#include "kernel_operator.h"

#if __has_include("../../../3rd/grouped_matmul/op_kernel/gqmm_cube_on_the_fly.h")
#include "../../../3rd/grouped_matmul/op_kernel/gqmm_cube_on_the_fly.h"
#else
#include "../../../../3rd/grouped_matmul/op_kernel/gqmm_cube_on_the_fly.h"
#endif

using namespace AscendC;

namespace MC2KernelTemplate {
constexpr uint64_t GROUP_LIST_INDEX = 0;

template <typename TilingDataType, typename GmmTilingDataType, class xType, class wType, class scaleType, class yType,
    CubeFormat wFormat, bool aTrans, bool bTrans, bool isLocal>
class QuantGroupedMatmul {
public:
    __aicore__ inline void Init(GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR xScaleGM, GM_ADDR weightScaleGM, GM_ADDR yGM,
        GM_ADDR workspaceGM, const TilingDataType *tilingData, const GmmTilingDataType *gmmTilingData,
        TILING_TYPE *gmmArrayAddrIn, TPipe *tPipe)
    {
        if ASCEND_IS_AIV {
            return ;
        }
        xGM_ = xGM;
        wGM_ = weightGM;
        xScaleGM_ = xScaleGM;
        weightScaleGM_ = weightScaleGM;
        yGM_ = yGM;
        tilingData_ = tilingData;
        tPipe_ = tPipe;
        workspaceGM_ = workspaceGM;
        gmmTilingData_ = gmmTilingData;
        gmmArrayAddrIn_ = gmmArrayAddrIn;

        expertNumInOneRank_ = tilingData_->taskTilingInfo.e;
        epWorldSize_ = tilingData_->taskTilingInfo.epWorldSize;
        h1_ = tilingData_->taskTilingInfo.H1;
        n1_ = tilingData_->taskTilingInfo.N1;
        bs_ = tilingData_->taskTilingInfo.BS;
        a_ = tilingData_->taskTilingInfo.A;
        groupListGm_ = tilingData_->isPermuteOut ? workspaceGM_ : workspaceGM_ + a_ * h1_;

        xGlobalBuffer_.SetGlobalBuffer((__gm__ xType *)this->xGM_);
        wGlobalBuffer_.SetGlobalBuffer((__gm__ wType *)this->wGM_);
        yGlobalBuffer_.SetGlobalBuffer((__gm__ yType *)this->yGM_);
        groupListGlobalBuffer_.SetGlobalBuffer((__gm__ int64_t *)groupListGm_);
        xScaleGlobalBuffer_.SetGlobalBuffer((__gm__ scaleType *)xScaleGM);
        wScaleGlobalBuffer_.SetGlobalBuffer((__gm__ scaleType *)weightScaleGM);

        const auto *recvCnt = &tilingData_->taskTilingInfo.recvCnt[0];
        for (uint32_t e = 0U; e < expertNumInOneRank_; e++) {
            for (uint32_t i = 0U; i < epWorldSize_; i++) {
                expertTokenNum_[e] += static_cast<uint64_t>(recvCnt[e + i * expertNumInOneRank_]);
            }
        }
    }

    __aicore__ inline void Process(uint32_t expertIdx)
    {
        if ASCEND_IS_AIV {
            return ;
        }
        if (expertTokenNum_[expertIdx] == 0) {
            return ;
        }
        uint64_t groupListToken = isLocal ? bs_ : expertTokenNum_[expertIdx];
        groupListGlobalBuffer_.SetValue(GROUP_LIST_INDEX, groupListToken);
        AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
            AscendC::DcciDst::CACHELINE_OUT>(groupListGlobalBuffer_);
        this->UpdateAddr(expertIdx);
        Mc2GroupedMatmul::Mc2GmmASWKernel<xType, wType, biasType, scaleType, yType, wFormat, aTrans, bTrans> gmmASWKernel;
        tPipe_->Reset();
        gmmASWKernel.Init(xGM_, wGM_, nullptr, xScaleGM_, groupListGm_, weightScaleGM_, yGM_, workspaceGM_,
            &gmmTilingData_->gmmQuantParams, &gmmTilingData_->mmTilingData, gmmArrayAddrIn_, tPipe_);
        gmmASWKernel.Process();
    }

    __aicore__ inline void End() {
        if ASCEND_IS_AIV {
            return ;
        }
    }

protected:
    __aicore__ inline void UpdateAddr(uint32_t expertIdx)
    {
        xGM_ = (GM_ADDR)xGlobalBuffer_.GetPhyAddr(expertTokenOffset_ * h1_);
        wGM_ = (GM_ADDR)wGlobalBuffer_.GetPhyAddr(expertIdx * h1_ * n1_);
        yGM_ = (GM_ADDR)yGlobalBuffer_.GetPhyAddr(expertTokenOffset_ * n1_);
        expertTokenOffset_ += expertTokenNum_[expertIdx];
    }

private:
    using biasType = float;

    GM_ADDR xGM_;
    GM_ADDR wGM_;
    GM_ADDR xScaleGM_;
    GM_ADDR weightScaleGM_;
    GM_ADDR yGM_;
    GM_ADDR groupListGm_;
    GM_ADDR workspaceGM_;
    GlobalTensor<xType> xGlobalBuffer_;
    GlobalTensor<wType> wGlobalBuffer_;
    GlobalTensor<scaleType> xScaleGlobalBuffer_;
    GlobalTensor<scaleType> wScaleGlobalBuffer_;
    GlobalTensor<yType> yGlobalBuffer_;
    GlobalTensor<int64_t> groupListGlobalBuffer_;
    const TilingDataType *tilingData_;
    TPipe *tPipe_;
    uint64_t expertTokenNum_[32] = {0};
    uint64_t expertTokenOffset_ = 0;
    uint64_t expertNumInOneRank_ = 0;
    uint64_t epWorldSize_ = 0;
    uint64_t h1_;
    uint64_t n1_;
    uint64_t bs_;
    uint64_t a_;
    const GmmTilingDataType *gmmTilingData_;
    TILING_TYPE *gmmArrayAddrIn_;
};
} // namespace MC2KernelTemplate
#endif
// MC2_QUANT_GROUPED_MATMUL_H