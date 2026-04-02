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
 * \file grouped_matmul_mxfp8fp4_kernel_resplit.h
 * \brief
 */
#ifndef GROUPED_MATMUL_MXFP8FP4_KERNEL_RESPLIT_H
#define GROUPED_MATMUL_MXFP8FP4_KERNEL_RESPLIT_H

#include "../../../grouped_matmul_utils.h"
#include "../../grouped_matmul_tiling_data_apt.h"
#include "../block/block_mmad.h"
#include "../block/grouped_matmul_scheduler_n_resplit.h"
#include "../prologue/block_prologue.h"
#include "include/experimental/tensor_api/tensor.h"
#include "kernel_operator_list_tensor_intf.h"

template <typename T, typename PtrT>
__aicore__ inline __gm__ T* GetTensorAddr(uint64_t index, PtrT tensorPtr)
{
    // 将输入的任意指针强制转换为 ListTensorDesc 要求的 __gm__ void*
    AscendC::ListTensorDesc listTensorDesc(reinterpret_cast<__gm__ void*>(tensorPtr));
    return listTensorDesc.GetDataPtr<T>(index);
}

using WeightQuantBatchMatmulV2::Arch35::CeilDivide;

namespace Kernel {

#define GROUPED_MATMUL_RESPLIT_KERNEL_TEMPLATE_PARAM                                                                    \
    template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler, class BlockPrologue>

#define GROUPED_MATMUL_RESPLIT_KERNEL_CLASS                                                                             \
    GroupedMatmul<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler, BlockPrologue>

GROUPED_MATMUL_RESPLIT_KERNEL_TEMPLATE_PARAM
class GROUPED_MATMUL_RESPLIT_KERNEL_CLASS {
public:
    struct Params {
        ProblemShape problemShape;
        typename BlockMmad::Params mmad;
        typename BlockScheduler::Params scheduler;
        typename BlockPrologue::Params prologue;
        uint64_t groupNum;
        int8_t groupType;
        uint64_t groupListType;
        GM_ADDR groupList;
    };

    __aicore__ inline GroupedMatmul() = default;
    __aicore__ inline void operator()(const Params &params);
private:
    template <typename TensorA, typename TensorScaleA, typename TensorY, typename TensorScaleB>
    struct AicScheduleHandler {
        BlockMmad &blockMmad;
        const TensorA &tensorAGm;
        const TensorScaleA &tensorScaleAGm;
        const TensorY &tensorYGm;
        const TensorScaleB &tensorScaleBGm;
        TensorA &tensorBlockAGm;
        TensorScaleA &tensorBlockScaleAGm;
        uint64_t kSize;
        uint64_t scaleKSize;

        __aicore__ inline void OnM(uint64_t mOffset, uint64_t mL1Size) const
        {
            tensorBlockAGm = tensorAGm(AscendC::Te::MakeCoord(mOffset, 0), AscendC::Te::MakeShape(mL1Size, kSize));
            tensorBlockScaleAGm =
                tensorScaleAGm(AscendC::Te::MakeCoord(mOffset, 0), AscendC::Te::MakeShape(mL1Size, scaleKSize));
                // TODO
            // uint64_t aSize = mL1Size * kSize * sizeof(typename BlockMmad::AType);
            // if (mL1Size <= 512 &&
            //     aSize <= static_cast<uint64_t>(gmmBaseTiling.cubeNumBlocksN) * A_L1_MAX_SIZE_WITH_BIAS_QUANT &&
            //     (gmmBaseTiling.coreNum % gmmBaseTiling.cubeNumBlocksN == 0)) {
            //     uint64_t aPrefetchSize = AscendC::CeilAlign(
            //         CeilDivide(mL1Size * kSize, static_cast<uint64_t>(gmmBaseTiling.cubeNumBlocksN)), 64UL);
            //     blockMmad.PrefetchA(aPrefetchSize, mL1Size * kSize);
            // }
        }

        __aicore__ inline void OnMN(uint64_t mOffset, uint64_t mL1Size, uint64_t nOffset, uint64_t nL1Size) const
        {
            auto tensorBlockYGm =
                tensorYGm(AscendC::Te::MakeCoord(mOffset, nOffset), AscendC::Te::MakeShape(mL1Size, nL1Size));
            auto tensorBlockScaleBGm =
                tensorScaleBGm(AscendC::Te::MakeCoord(0, nOffset), AscendC::Te::MakeShape(scaleKSize, nL1Size));
            blockMmad(tensorBlockAGm, tensorBlockYGm, tensorBlockScaleAGm, tensorBlockScaleBGm);
        }
    };

    struct AivScheduleHandler {
        BlockPrologue &blockPrologue;
        __gm__ typename BlockPrologue::BType *weightGm;
        __gm__ typename BlockPrologue::BiasType *biasGm;
        bool &weightL2Cacheable;
        uint64_t mSize;
        uint64_t kSize;
        uint64_t nAlign;
        bool isCacheLineUnaligned;

        __aicore__ inline void OnM(uint64_t, uint64_t mL1Size) const
        {
            weightL2Cacheable = mL1Size < mSize || isCacheLineUnaligned;
        }

        __aicore__ inline void OnMN(uint64_t, uint64_t mL1Size, uint64_t nOffset, uint64_t nL1Size) const
        {
            blockPrologue(weightGm, biasGm, weightL2Cacheable, mL1Size, kSize, nL1Size, nOffset, nAlign);
        }
    };

    __aicore__ inline uint64_t GetSplitValueFromGroupList(const Params& params, uint64_t groupIdx);

    __gm__ typename BlockMmad::AType *xGm_;
    __gm__ typename BlockMmad::ScaleBType *antiquantScaleGm_;
    __gm__ typename BlockMmad::CType *yGm_;
    __gm__ typename BlockMmad::ScaleAType *perTokenScaleGm_;

    __gm__ typename BlockPrologue::BType *weightGm_;
    __gm__ typename BlockPrologue::BiasType *biasGm_ = nullptr;

    // TODO 替换tensor-api
    GlobalTensor<int64_t> groupListGm_;

    uint64_t preOffset_ = 0;
};

GROUPED_MATMUL_RESPLIT_KERNEL_TEMPLATE_PARAM
__aicore__ inline void GROUPED_MATMUL_RESPLIT_KERNEL_CLASS::operator()(const Params &params)
{
    preOffset_ = 0;

    if ASCEND_IS_AIC {
        xGm_ = GetTensorAddr<typename BlockMmad::AType>(0, params.mmad.ptrA);
        antiquantScaleGm_ =
            GetTensorAddr<typename BlockMmad::ScaleBType>(0, params.mmad.ptrScaleB);
        perTokenScaleGm_ = reinterpret_cast<__gm__ typename BlockMmad::ScaleAType *>(params.mmad.ptrScaleA);
        yGm_ = GetTensorAddr<typename BlockMmad::CType>(0, params.mmad.ptrC);
    }
    if ASCEND_IS_AIV {
        weightGm_ = GetTensorAddr<typename BlockPrologue::BType>(0, params.prologue.ptrB);
        if (params.mmad.hasBias) {
            biasGm_ = GetTensorAddr<typename BlockPrologue::BiasType>(0, params.prologue.ptrBias);
        }
    }
    if (params.groupList != nullptr) {
        groupListGm_.SetGlobalBuffer((__gm__ int64_t *)params.groupList);
    }

    const uint64_t kSize = AscendC::Std::get<1>(params.problemShape);
    const uint64_t nSize = AscendC::Std::get<2>(params.problemShape);
    const uint64_t nAlign = AscendC::CeilAlign(nSize, static_cast<uint64_t>(BLOCK_CUBE));
    // 128B是256个B4
    const bool isCacheLineUnaligned = kSize % 256 != 0;
    const uint64_t scaleKSize = CeilDivide(kSize, static_cast<uint64_t>(64)) * 2;
    BlockScheduler scheduler(params.scheduler);
    if ASCEND_IS_AIC {
        using TensorLayoutA = typename BlockMmad::LayoutA;
        using TensorLayoutC = typename BlockMmad::LayoutC;
        using TensorLayoutScaleA = typename BlockMmad::LayoutScaleA;
        using TensorLayoutScaleB = typename BlockMmad::LayoutScaleB;
        BlockMmad blockMmad(params.mmad);
        uint64_t startBasicBlockId = 0;
        for (uint32_t groupIdx = 0; groupIdx < AscendC::Std::get<3>(params.problemShape); ++groupIdx) {
            uint64_t mSize = GetSplitValueFromGroupList(params, groupIdx);
            if (mSize > 0 && nSize > 0) {
                auto tensorAGm =
                    AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(xGm_), TensorLayoutA{}(mSize, kSize));
                auto tensorYGm =
                    AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(yGm_), TensorLayoutC{}(mSize, nSize));
                auto tensorScaleAGm = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(perTokenScaleGm_),
                                                              TensorLayoutScaleA{}(mSize, scaleKSize));
                auto tensorScaleBGm = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(antiquantScaleGm_),
                                                              TensorLayoutScaleB{}(scaleKSize, nSize));
                decltype(tensorAGm) tensorBlockAGm;
                decltype(tensorScaleAGm) tensorBlockScaleAGm;
                AicScheduleHandler<decltype(tensorAGm), decltype(tensorScaleAGm), decltype(tensorYGm),
                                   decltype(tensorScaleBGm)>
                    handler = {
                        blockMmad,
                        tensorAGm,
                        tensorScaleAGm,
                        tensorYGm,
                        tensorScaleBGm,
                        tensorBlockAGm,
                        tensorBlockScaleAGm,
                        kSize,
                        scaleKSize};
                scheduler(startBasicBlockId, mSize, handler);
                xGm_ += mSize * kSize;
                antiquantScaleGm_ += nSize * scaleKSize;
                perTokenScaleGm_ += mSize * scaleKSize;
                yGm_ += mSize * nSize;
            }
        }
    } else {
        BlockPrologue blockPrologue(params.prologue);
        uint64_t startBasicBlockId = 0;
        for (uint32_t groupIdx = 0; groupIdx < AscendC::Std::get<3>(params.problemShape); ++groupIdx) {
            uint64_t mSize = GetSplitValueFromGroupList(params, groupIdx);
            if (mSize > 0 && nSize > 0) {
                bool weightL2Cacheable = isCacheLineUnaligned;
                AivScheduleHandler handler = {
                    blockPrologue,
                    weightGm_,
                    biasGm_,
                    weightL2Cacheable,
                    mSize,
                    kSize,
                    nAlign,
                    isCacheLineUnaligned};
                scheduler(startBasicBlockId, mSize, handler);
                // 4bit，地址偏移单位为8bit
                weightGm_ += (nSize * kSize) >> 1;
                if (params.prologue.hasBias) {
                    biasGm_ += nSize;
                }
            }
        }
    }
}

GROUPED_MATMUL_RESPLIT_KERNEL_TEMPLATE_PARAM
__aicore__ inline uint64_t GROUPED_MATMUL_RESPLIT_KERNEL_CLASS::GetSplitValueFromGroupList(const Params& params,
    uint64_t groupIdx)
{
    uint64_t splitValue = 0;
    if (likely(params.groupType != -1)) {
        if (params.groupListType == 0) {
            uint64_t offset = static_cast<uint64_t>(groupListGm_.GetValue(groupIdx));
            splitValue = offset - preOffset_;
            preOffset_ = offset;
        } else {
            splitValue = static_cast<uint64_t>(groupListGm_.GetValue(groupIdx));
        }
    }
    return splitValue;
}

#undef GROUPED_MATMUL_RESPLIT_KERNEL_CLASS
#undef GROUPED_MATMUL_RESPLIT_KERNEL_TEMPLATE_PARAM
} // namespace Kernel

#endif  // GROUPED_MATMUL_MXFP8FP4_KERNEL_RESPLIT_H
