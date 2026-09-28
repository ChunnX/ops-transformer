/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file grouped_matmul_swiglu_quant_v2_weight_quant_asw_kernel.h
 * \brief
 */

#pragma once

#include "weight_quant_tool_asw.h"
#include "weight_quant_grouped_matmul_asw_block.h"
#include "grouped_matmul_swiglu_quant_v2_weight_quant_asw_cube_compute.h"
#include "grouped_matmul_swiglu_quant_v2_weight_quant_asw_vec_compute.h"
#include "grouped_matmul_swiglu_quant_v2_weight_quant_asw_tiling_data.h"
#include "../../grouped_matmul_swiglu_quant_v2_utils.h"
using namespace WeightQuantTools;
using AswParams = GMMSQArch35Tiling::GMMSQWeightQuantAswParams;
using namespace GroupedMatmulDequantSwigluQuant;

namespace GMMSQWeightQuantAsw {

constexpr uint64_t UB_Y_F32_SIZE = 256 * 1024;
constexpr uint64_t UB_Y_F32_OFFSET = 0;
constexpr uint64_t SYNC_MODE4 = 4UL;
constexpr uint16_t AIC_SYNC_AIV_FLAG = 4;
constexpr uint16_t AIV_SYNC_AIC_FLAG = 6;

#define GMMSQ_WQ_ASW_TEMPLATE_CLASS_PARAMS                                                                             \
    template <typename xType, typename wType, typename xScaleType, typename weightScaleType, typename yType,           \
              typename yScaleType>
#define GMMSQ_WQ_ASW_TEMPLATE_FUNC_PARAMS xType, wType, xScaleType, weightScaleType, yType, yScaleType

GMMSQ_WQ_ASW_TEMPLATE_CLASS_PARAMS
class GMMSQWeightQuantASWKernel {
public:
    __aicore__ inline GMMSQWeightQuantASWKernel()
    {
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR xScale, GM_ADDR weightScale, GM_ADDR groupList,
                                GM_ADDR y, GM_ADDR yScale, const AswParams *__restrict params,
                                const TCubeTiling *__restrict tilingData);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void ProcessSingleBlock(uint64_t roundIdx, bool enablePreAlloc);
    __aicore__ inline void UpdateGlobalAddr();
    __aicore__ inline void SetMNK(uint32_t groupIdx, uint64_t &mSize, uint64_t &nSize, uint64_t &kSize);
    __aicore__ inline bool EnablePreAlloc();
    __aicore__ inline void NotifyVector();
    __aicore__ inline void WaitForVector();
    __aicore__ inline void NotifyCube();
    __aicore__ inline void WaitForCube();
    __aicore__ inline bool IsLastGroupAndNeedSplit(uint32_t groupIdx);
    __aicore__ inline bool IsLastGroupAndRound(uint32_t groupIdx, uint64_t roundIdx);
    __aicore__ inline void End();

    WeightQuantASWBlock block_;
    GMMSQWeightQuantASWCubeCompute<xType, wType, xScaleType, weightScaleType> cubeCompute_;
    GMMSQWeightQuantASWVecCompute<yType, yScaleType> vecCompute_;

    LocalTensor<float> yF32Buffer_;

    __gm__ xType *xGmAddr_;
    __gm__ wType *weightGmAddr_;
    __gm__ xScaleType *xScaleGmAddr_;
    __gm__ weightScaleType *weightScaleGmAddr_;
    __gm__ yType *yGmAddr_;
    __gm__ yScaleType *yScaleGmAddr_;

    const TCubeTiling *__restrict mmTilingData_ = nullptr;

    uint32_t blockIdx_ = 0;
    uint32_t groupNum_ = 0;
    uint8_t groupListType_ = 0;
    uint64_t preOffset_ = 0;
    AscendC::GlobalTensor<int64_t> groupListGlobal_;
};

GMMSQ_WQ_ASW_TEMPLATE_CLASS_PARAMS
__aicore__ inline void GMMSQWeightQuantASWKernel<GMMSQ_WQ_ASW_TEMPLATE_FUNC_PARAMS>::Init(
    GM_ADDR x, GM_ADDR weight, GM_ADDR xScale, GM_ADDR weightScale, GM_ADDR groupList, GM_ADDR y, GM_ADDR yScale,
    const AswParams *__restrict params, const TCubeTiling *__restrict tilingData)
{
    blockIdx_ = AscendC::GetBlockIdx();
    mmTilingData_ = tilingData;
    groupNum_ = params->groupNum;
    groupListType_ = params->groupListType;

    xGmAddr_ = reinterpret_cast<__gm__ xType *>(x);
    weightGmAddr_ = GetTensorAddr<wType>(0, weight);
    xScaleGmAddr_ = reinterpret_cast<__gm__ xScaleType *>(xScale);
    weightScaleGmAddr_ = GetTensorAddr<weightScaleType>(0, weightScale);
    yGmAddr_ = reinterpret_cast<__gm__ yType *>(y);
    yScaleGmAddr_ = reinterpret_cast<__gm__ yScaleType *>(yScale);

    groupListGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(groupList));

    block_.Init(tilingData, blockIdx_);

    yF32Buffer_ = LocalTensor<float>(AscendC::TPosition::LCM, UB_Y_F32_OFFSET, UB_Y_F32_SIZE);

    if ASCEND_IS_AIC {
        cubeCompute_.Init(yF32Buffer_, params->l0cSize, mmTilingData_);
    }

    if ASCEND_IS_AIV {
        vecCompute_.Init(yF32Buffer_);
        NotifyCube();
    }
}

GMMSQ_WQ_ASW_TEMPLATE_CLASS_PARAMS
__aicore__ inline void GMMSQWeightQuantASWKernel<GMMSQ_WQ_ASW_TEMPLATE_FUNC_PARAMS>::SetMNK(uint32_t groupIdx,
                                                                                            uint64_t &mSize,
                                                                                            uint64_t &nSize,
                                                                                            uint64_t &kSize)
{
    mSize = GetSplitValueFromGroupList(groupIdx, preOffset_, 0, groupListType_, groupListGlobal_);
    nSize = mmTilingData_->N;
    kSize = mmTilingData_->Ka;
}

GMMSQ_WQ_ASW_TEMPLATE_CLASS_PARAMS
__aicore__ inline void GMMSQWeightQuantASWKernel<GMMSQ_WQ_ASW_TEMPLATE_FUNC_PARAMS>::UpdateGlobalAddr()
{
    if ASCEND_IS_AIC {
        cubeCompute_.UpdateGlobalAddr(
            reinterpret_cast<__gm__ xType *>(xGmAddr_ + block_.params_.xGroupAddrOffset),
            reinterpret_cast<__gm__ wType *>(weightGmAddr_ + block_.params_.wGroupAddrOffset),
            reinterpret_cast<__gm__ xScaleType *>(xScaleGmAddr_ + block_.params_.pertokenScaleGroupAddrOffset),
            reinterpret_cast<__gm__ weightScaleType *>(weightScaleGmAddr_ +
                                                       block_.params_.antiquantScaleGroupAddrOffset));
    }

    if ASCEND_IS_AIV {
        vecCompute_.UpdateGlobalAddr(
            reinterpret_cast<__gm__ yType *>(yGmAddr_ + block_.params_.yGroupAddrOffset),
            reinterpret_cast<__gm__ yScaleType *>(yScaleGmAddr_ + block_.params_.yScaleGroupAddrOffset));
    }
}

GMMSQ_WQ_ASW_TEMPLATE_CLASS_PARAMS
__aicore__ inline bool GMMSQWeightQuantASWKernel<GMMSQ_WQ_ASW_TEMPLATE_FUNC_PARAMS>::EnablePreAlloc()
{
    uint64_t kL1Size = WeightQuantTools::Min<uint64_t>(block_.params_.k, mmTilingData_->stepKa * mmTilingData_->baseK);
    uint64_t aL1Size = block_.params_.baseM * kL1Size * sizeof(xType);
    uint64_t bL1Size = 0UL;
    // 本函数调用时singleCoreN未完成计算，使用splitBaseN * 2进行粗略计算
    if constexpr (AscendC::IsSupportB4<wType>()) {
        bL1Size = (block_.params_.splitBaseN * SWIGLU_N_HALF_FACTOR * kL1Size) >> 1;
    } else {
        bL1Size = block_.params_.splitBaseN * SWIGLU_N_HALF_FACTOR * kL1Size * sizeof(wType);
    }
    // AL1+BL1载入数据量较大(128*1024B，经验值)时，开启prealloc会导致后2轮的mte2抢占前面的ostd，导致前面的带宽跟不上cube计算。
    // 当K较小时，单轮计算量较小，如果不开启prealloc，下一轮MN的计算的mte2无法提前发出，会导致mte2断流。
    return !(aL1Size + bL1Size >= 128 * 1024 && block_.params_.k > 1024);
}

GMMSQ_WQ_ASW_TEMPLATE_CLASS_PARAMS
__aicore__ inline void GMMSQWeightQuantASWKernel<GMMSQ_WQ_ASW_TEMPLATE_FUNC_PARAMS>::Process()
{
    for (uint32_t groupIdx = 0; groupIdx < groupNum_; ++groupIdx) {
        uint64_t mSize = 0;
        uint64_t nSize = 0;
        uint64_t kSize = 0;
        // 更新每个group的M, N, K和GM offset
        SetMNK(groupIdx, mSize, nSize, kSize);
        block_.template UpdateGroupOffset<true, xType, wType, weightScaleType, yType, CubeFormat::NZ,
                                          SwiGLUGroupAddrUpdater>(mSize, nSize, kSize, groupIdx);

        if (mSize <= 0 || kSize <= 0 || nSize <= 0) {
            continue;
        }

        block_.template UpdateGroupParams<true>(groupIdx == groupNum_ - 1);
        bool enablePreAlloc = EnablePreAlloc();
        // 最后一组最后一轮尾块重切分
        if (IsLastGroupAndNeedSplit(groupIdx)) {
            block_.template CalcTailTile<false, true, true>();
            block_.UpdateTailTile();
        }

        // 更新每个group的GM地址
        UpdateGlobalAddr();

        for (uint64_t roundIdx = 0; roundIdx < block_.params_.round; ++roundIdx) {
            bool isLastGroupRound = IsLastGroupAndRound(groupIdx, roundIdx);
            // 计算基本块索引
            block_.UpdateBasicIndex(roundIdx, isLastGroupRound);
            // 计算基本块大小
            block_.template UpdateBlockParams<false, true, CubeFormat::NZ, true>(roundIdx, isLastGroupRound);
            // 计算基本块地址偏移
            block_
                .template CalcGMOffset<true, xType, wType, weightScaleType, CubeFormat::NZ, SwiGLUBlockOffsetUpdater>();
            ProcessSingleBlock(roundIdx, enablePreAlloc);
        }
    }
    End();
}

GMMSQ_WQ_ASW_TEMPLATE_CLASS_PARAMS
__aicore__ inline void
GMMSQWeightQuantASWKernel<GMMSQ_WQ_ASW_TEMPLATE_FUNC_PARAMS>::ProcessSingleBlock(uint64_t roundIdx, bool enablePreAlloc)
{
    if ASCEND_IS_AIC {
        cubeCompute_.MatmulCompute(block_.params_, block_.offset_, enablePreAlloc);
        WaitForVector();
        cubeCompute_.GetTensorC(block_.params_);
        NotifyVector();
    }

    if ASCEND_IS_AIV {
        WaitForCube();
        vecCompute_.IterateNzNkWithAiv(block_.params_, block_.offset_);
        NotifyCube();
    }
}

GMMSQ_WQ_ASW_TEMPLATE_CLASS_PARAMS
__aicore__ inline void GMMSQWeightQuantASWKernel<GMMSQ_WQ_ASW_TEMPLATE_FUNC_PARAMS>::NotifyVector()
{
    AscendC::CrossCoreSetFlag<SYNC_MODE4, PIPE_FIX>(AIC_SYNC_AIV_FLAG);
}

GMMSQ_WQ_ASW_TEMPLATE_CLASS_PARAMS
__aicore__ inline void GMMSQWeightQuantASWKernel<GMMSQ_WQ_ASW_TEMPLATE_FUNC_PARAMS>::WaitForVector()
{
    AscendC::CrossCoreWaitFlag<SYNC_MODE4, PIPE_FIX>(AIV_SYNC_AIC_FLAG);
}

GMMSQ_WQ_ASW_TEMPLATE_CLASS_PARAMS
__aicore__ inline void GMMSQWeightQuantASWKernel<GMMSQ_WQ_ASW_TEMPLATE_FUNC_PARAMS>::NotifyCube()
{
    AscendC::CrossCoreSetFlag<SYNC_MODE4, PIPE_V>(AIV_SYNC_AIC_FLAG);
}

GMMSQ_WQ_ASW_TEMPLATE_CLASS_PARAMS
__aicore__ inline void GMMSQWeightQuantASWKernel<GMMSQ_WQ_ASW_TEMPLATE_FUNC_PARAMS>::WaitForCube()
{
    AscendC::CrossCoreWaitFlag<SYNC_MODE4, PIPE_V>(AIC_SYNC_AIV_FLAG);
}

GMMSQ_WQ_ASW_TEMPLATE_CLASS_PARAMS
__aicore__ inline bool
GMMSQWeightQuantASWKernel<GMMSQ_WQ_ASW_TEMPLATE_FUNC_PARAMS>::IsLastGroupAndNeedSplit(uint32_t groupIdx)
{
    return groupIdx == groupNum_ - 1 && (block_.GetEndBlockIdx() + 1) <= mmTilingData_->usedCoreNum / 2;
}

GMMSQ_WQ_ASW_TEMPLATE_CLASS_PARAMS
__aicore__ inline bool
GMMSQWeightQuantASWKernel<GMMSQ_WQ_ASW_TEMPLATE_FUNC_PARAMS>::IsLastGroupAndRound(uint32_t groupIdx, uint64_t roundIdx)
{
    return groupIdx == groupNum_ - 1 && roundIdx == block_.params_.round - 1 && blockIdx_ <= block_.GetEndBlockIdx();
}

GMMSQ_WQ_ASW_TEMPLATE_CLASS_PARAMS
__aicore__ inline void GMMSQWeightQuantASWKernel<GMMSQ_WQ_ASW_TEMPLATE_FUNC_PARAMS>::End()
{
    if ASCEND_IS_AIC {
        WaitForVector();
        cubeCompute_.EndSync();
    }
}

} // namespace GMMSQWeightQuantAsw