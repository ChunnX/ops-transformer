/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file masked_causal_conv1d_backward_arch35.cpp
 * \brief MaskedCausalConv1dBackward tiling implementation
 */

#include "masked_causal_conv1d_backward_tiling_arch35.h"
#include <algorithm>
#include "securec.h"

namespace optiling {


using MaskedCausalConv1dBackwardArch35Tiling::MaskedCausalConv1dBackwardTilingDataV35;

bool MaskedCausalConv1dBackwardTiling::IsCapable()
{
    // 详细校验在 CheckInputParams 中完成，这里返回 true
    return true;
}

ge::graphStatus MaskedCausalConv1dBackwardTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr =
            reinterpret_cast<const MaskedCausalConv1dBackwardArch35CompileInfo *>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        totalCoreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        totalCoreNum_ = static_cast<uint64_t>(ascendcPlatform.GetCoreNumAiv());
        OP_CHECK_IF(totalCoreNum_ == 0UL, OP_LOGE(context_->GetNodeName(), "coreNum is 0"), return ge::GRAPH_FAILED);
        uint64_t ubSize = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        OP_CHECK_IF(ubSize == static_cast<uint64_t>(0), OP_LOGE(context_->GetNodeName(), "ubSize is 0"),
                    return ge::GRAPH_FAILED);
        ubSize_ = static_cast<uint64_t>(ubSize);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedCausalConv1dBackwardTiling::ValidateGradOutputShape()
{
    auto shape = context_->GetInputShape(GRAD_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shape);
    auto origin = shape->GetOriginShape();

    OP_CHECK_IF(origin.GetDimNum() != 3,
                OP_LOGE(context_->GetNodeName(), "grad_output dim num must be 3, but got %lu", origin.GetDimNum()),
                return ge::GRAPH_FAILED);

    S_ = static_cast<int64_t>(origin.GetDim(DIM_0));
    B_ = static_cast<int64_t>(origin.GetDim(DIM_1));
    H_ = static_cast<int64_t>(origin.GetDim(DIM_2));

    // 约束范围（来自需求文档）：S∈[1, 32K], B∈[1, 8], H∈[384, 24576]
    OP_CHECK_IF(S_ <= 0 || S_ > (1 << 15), OP_LOGE(context_->GetNodeName(), "invalid S=%ld", S_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(B_ <= 0 || B_ > 8, OP_LOGE(context_->GetNodeName(), "invalid B=%ld", B_), return ge::GRAPH_FAILED);
    // OP_CHECK_IF(H_ < 384 || H_ > (192 * 128), OP_LOGE(context_->GetNodeName(), "invalid H=%ld", H_),
    //             return ge::GRAPH_FAILED);

    // H 必须按 64 对齐以便 H 方向切分
    OP_CHECK_IF((H_ % DIM_ALIGN_ELEMENT) != 0,
                OP_LOGE(context_->GetNodeName(), "H must be multiple of %ld, got %ld", DIM_ALIGN_ELEMENT, H_),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedCausalConv1dBackwardTiling::ValidateInputShape()
{
    auto goShape = context_->GetInputShape(GRAD_OUTPUT_INDEX)->GetOriginShape();

    auto inShape = context_->GetInputShape(INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inShape);
    auto inOrigin = inShape->GetOriginShape();

    OP_CHECK_IF(inOrigin.GetDimNum() != 3,
                OP_LOGE(context_->GetNodeName(), "input dim num must be 3, but got %lu", inOrigin.GetDimNum()),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(goShape.GetDim(DIM_0) != inOrigin.GetDim(DIM_0) || goShape.GetDim(DIM_1) != inOrigin.GetDim(DIM_1) ||
                    goShape.GetDim(DIM_2) != inOrigin.GetDim(DIM_2),
                OP_LOGE(context_->GetNodeName(), "input shape mismatch with grad_output"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedCausalConv1dBackwardTiling::ValidateWeightShape()
{
    auto wShape = context_->GetInputShape(WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, wShape);
    auto wOrigin = wShape->GetOriginShape();

    OP_CHECK_IF(wOrigin.GetDimNum() != 2,
                OP_LOGE(context_->GetNodeName(), "weight dim num must be 2, but got %lu", wOrigin.GetDimNum()),
                return ge::GRAPH_FAILED);

    W_ = static_cast<int64_t>(wOrigin.GetDim(DIM_0));
    auto wH = static_cast<int64_t>(wOrigin.GetDim(DIM_1));

    OP_CHECK_IF(W_ != 3, OP_LOGE(context_->GetNodeName(), "weight[0]=W must be 3, but got %ld", W_),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(wH != H_, OP_LOGE(context_->GetNodeName(), "weight dim1 %ld must match H %ld", wH, H_),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedCausalConv1dBackwardTiling::ValidateMaskShape()
{
    auto maskShape = context_->GetOptionalInputShape(MASK_INDEX);
    if (maskShape == nullptr) {
        hasMask_ = 0;
        return ge::GRAPH_SUCCESS;
    }
    hasMask_ = 1;
    auto mOrigin = maskShape->GetOriginShape();

    OP_CHECK_IF(mOrigin.GetDimNum() != 2,
                OP_LOGE(context_->GetNodeName(), "mask dim num must be 2, but got %lu", mOrigin.GetDimNum()),
                return ge::GRAPH_FAILED);

    auto mB = static_cast<int64_t>(mOrigin.GetDim(DIM_0));
    auto mS = static_cast<int64_t>(mOrigin.GetDim(DIM_1));

    OP_CHECK_IF(mB != B_ || mS != S_,
                OP_LOGE(context_->GetNodeName(), "mask shape must be [B,S]=[%ld,%ld], got [%ld,%ld]", B_, S_, mB, mS),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedCausalConv1dBackwardTiling::ValidateGradOutputType()
{
    dataType_ = context_->GetInputDesc(GRAD_OUTPUT_INDEX)->GetDataType();
    OP_CHECK_IF(dataType_ != ge::DataType::DT_FLOAT16 && dataType_ != ge::DataType::DT_BF16,
                OP_LOGE(context_->GetNodeName(), "grad_output dtype must be FLOAT16 or BF16, but got %s",
                        Ops::Base::ToString(dataType_).c_str()),
                return ge::GRAPH_FAILED);
    dtypeSize_ = DTYPE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedCausalConv1dBackwardTiling::ValidateInputType()
{
    auto t = context_->GetInputDesc(INPUT_INDEX)->GetDataType();
    OP_CHECK_IF(t != dataType_,
                OP_LOGE(context_->GetNodeName(), "input dtype %s must match grad_output dtype %s",
                        Ops::Base::ToString(t).c_str(), Ops::Base::ToString(dataType_).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedCausalConv1dBackwardTiling::ValidateWeightType()
{
    auto t = context_->GetInputDesc(WEIGHT_INDEX)->GetDataType();
    OP_CHECK_IF(t != dataType_,
                OP_LOGE(context_->GetNodeName(), "weight dtype %s must match grad_output dtype %s",
                        Ops::Base::ToString(t).c_str(), Ops::Base::ToString(dataType_).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedCausalConv1dBackwardTiling::ValidateMaskType()
{
    auto desc = context_->GetOptionalInputDesc(MASK_INDEX);
    if (desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto t = desc->GetDataType();
    OP_CHECK_IF(
        t != ge::DataType::DT_UINT8,
        OP_LOGE(context_->GetNodeName(), "mask dtype must be UINT8, but got %s", Ops::Base::ToString(t).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedCausalConv1dBackwardTiling::CheckInputParams()
{
    // Shapes
    OP_CHECK_IF(ValidateGradOutputShape() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "grad_output shape validation failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateInputShape() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "input shape validation failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateWeightShape() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "weight shape validation failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateMaskShape() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "mask shape validation failed"), return ge::GRAPH_FAILED);

    // Types
    OP_CHECK_IF(ValidateGradOutputType() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "grad_output type validation failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateInputType() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "input type validation failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateWeightType() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "weight type validation failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateMaskType() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "mask type validation failed"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedCausalConv1dBackwardTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE("MaskedCausalConv1dBackward", "context is null"), return ge::GRAPH_FAILED);

    // 收集形状和类型，并做完整校验
    OP_CHECK_IF(CheckInputParams() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "MaskedCausalConv1dBackward CheckInputParams FAILED."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedCausalConv1dBackwardTiling::ComputeInterCoreSplit()
{
    // 仅沿 H 方向按 64 切分
    // 根据文档：主核处理的H会比尾核多64(hMainSize = hTailSize + 64)
    int64_t tiles = H_ / DIM_ALIGN_ELEMENT;
    OP_CHECK_IF(tiles <= 0, OP_LOGE(context_->GetNodeName(), "H(%ld) < %ld", H_, DIM_ALIGN_ELEMENT),
                return ge::GRAPH_FAILED);

    int64_t maxCores = static_cast<int64_t>(totalCoreNum_);
    usedCoreNum_ = std::min<int64_t>(tiles, std::max<int64_t>(1, maxCores));

    if (H_ % usedCoreNum_ == 0) {
        // H能均分，则认为没有尾核
        hMainCoreCnt_ = usedCoreNum_;
        hTailCoreCnt_ = 0;
        hMainSize_ = H_ / usedCoreNum_;
        hTailSize_ = hMainSize_;  // 文档要求：hTailSize仍等于hMainSize
    } else {
        // 计算主核和尾核的分配
        int64_t baseTiles = H_ / (usedCoreNum_ * DIM_ALIGN_ELEMENT);
        hTailSize_ = baseTiles * DIM_ALIGN_ELEMENT;
        hMainSize_ = hTailSize_ + DIM_ALIGN_ELEMENT;  // 主核比尾核多64

        // 计算需要多少主核
        int64_t totalWithMain = hMainSize_ * usedCoreNum_;
        int64_t excess = totalWithMain - H_;
        hTailCoreCnt_ = excess / DIM_ALIGN_ELEMENT;
        hMainCoreCnt_ = usedCoreNum_ - hTailCoreCnt_;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedCausalConv1dBackwardTiling::ComputeIntraCoreUbTiling()
{
    // UB 预留系统空间
    int64_t availableUbSize = static_cast<int64_t>(ubSize_) - SYSTEM_RESERVED_UB_SIZE;
    OP_CHECK_IF(availableUbSize <= 0, OP_LOGE(context_->GetNodeName(), "available UB size <= 0, ubSize=%lu", ubSize_),
                return ge::GRAPH_FAILED);

    // 根据文档：保证H不小于64元素情况下，多载BS
    // UB先按H=64(128B)，全载核内BS
    int64_t minH = DIM_ALIGN_ELEMENT;  // 64
    int64_t perCoreH = (hMainCoreCnt_ > 0) ? hMainSize_ : hTailSize_;

    // Buffer占用估算函数
    // 参数 s_effective 是有效输出行数（不包含重叠部分）
    auto calculateBufferSize = [&](int64_t h, int64_t b, int64_t s_effective) -> int64_t {
        // 根据文档第4节Buffer设计
        // gradOutputQueue 和 maskQueue 需要加上重叠的2行（W-1）
        int64_t s_with_overlap = s_effective + (W_ - 1);

        // gradOutputQueue: hUb * bUb * (sUb + W-1) * sizeof(DTYPE) - double buffer
        int64_t gradOutputSize = h * b * s_with_overlap * static_cast<int64_t>(dtypeSize_);

        // inputQueue: hUb * bUb * sUb * sizeof(DTYPE) - double buffer
        int64_t inputSize = h * b * s_effective * static_cast<int64_t>(dtypeSize_);

        // weightQueue: hUb * W * sizeof(DTYPE) - double buffer
        int64_t weightSize = h * W_ * static_cast<int64_t>(dtypeSize_);

        // maskQueue: bUb * (sUb + W-1) - double buffer (BOOL类型，每个元素1字节)
        // 注意：maskQueue 始终分配，即使 mask=None 时 kernel 也会使用全1的mask
        int64_t maskSize = b * s_with_overlap;

        // gradInputQueue: hUb * bUb * sUb * sizeof(DTYPE) - double buffer
        int64_t gradInputSize = h * b * s_effective * static_cast<int64_t>(dtypeSize_);

        // gradWeightQueue: hUb * W * sizeof(DTYPE) - double buffer
        int64_t gradWeightSize = h * W_ * static_cast<int64_t>(dtypeSize_);

        // tempBuff: hUb * W * sizeof(float) - TBuf (单buffer，用于临时存储权重梯度)
        int64_t tempBuffSize = h * W_ * sizeof(float);

        // 所有Queue都是double buffer，tempBuff是单buffer
        int64_t totalSize = BUFFER_NUM * (gradOutputSize + inputSize + weightSize + maskSize +
                                          gradInputSize + gradWeightSize) + tempBuffSize;
        return totalSize;
    };

    // 首先尝试H=64，全载BS
    // 注意：sUB_ 存储的是有效输出行数（不包含重叠部分）
    hUB_ = minH;
    bUB_ = B_;
    sUB_ = S_;  // 有效输出大小
    if (calculateBufferSize(hUB_, bUB_, sUB_) <= availableUbSize) {  //H=64能全载BS，看H能否增加
        // 计算理论最大H（简化公式，基于Buffer占用）
        int64_t s_effective = sUB_;
        int64_t s_with_overlap = s_effective + (W_ - 1);
        int64_t coeff = B_ * s_with_overlap * static_cast<int64_t>(dtypeSize_) * BUFFER_NUM +
                        2 * B_ * s_effective * static_cast<int64_t>(dtypeSize_) * BUFFER_NUM +
                        2 * W_ * static_cast<int64_t>(dtypeSize_) * BUFFER_NUM +
                        W_ * sizeof(float);
        // maskQueue 始终存在，大小为 bUb * (sUb + W-1)
        int64_t const_cost = BUFFER_NUM * B_ * s_with_overlap;
        int64_t maxH = std::min(perCoreH, (availableUbSize - const_cost) / coeff);
        hUB_ = (maxH / minH) * minH;  // 对齐到64

        // 确保至少是minH
        if (hUB_ < minH) {
            hUB_ = minH;
        }
    } else {
        // 不能全载，先压缩B
        bUB_ = 1;
        if (calculateBufferSize(hUB_, bUB_, sUB_) <= availableUbSize) { //B=1时能全载S，尝试增加B
            // 计算最大B
            int64_t s_effective = sUB_;
            int64_t s_with_overlap = s_effective + (W_ - 1);
            int64_t coeff_b = hUB_ * s_with_overlap * static_cast<int64_t>(dtypeSize_) * BUFFER_NUM+
                              2 * hUB_ * s_effective * static_cast<int64_t>(dtypeSize_) * BUFFER_NUM;
            int64_t fixed_cost = 2 * hUB_ * W_ * static_cast<int64_t>(dtypeSize_) * BUFFER_NUM +
                                 hUB_ * W_ * sizeof(float);
            int64_t maxB = std::min(B_, (availableUbSize - fixed_cost) / coeff_b);
            bUB_ = std::max<int64_t>(1, maxB);
        } else {  //即便B=1也不能全载S，切S
            // 计算最大的有效输出大小
            // Buffer总大小 = s * coeff_s + const_cost_s
            // coeff_s = (3 input/output queues) * BUFFER_NUM * h * b * dtype + maskQueue * BUFFER_NUM * b
            // const_cost_s = BUFFER_NUM * (gradOutput/mask的常数项 + weight相关) + tempBuff
            int64_t coeff_s = 3 *  hUB_ * bUB_ * static_cast<int64_t>(dtypeSize_) * BUFFER_NUM+
                              bUB_ * BUFFER_NUM;
            int64_t const_cost_s = 2 * hUB_ * bUB_ * static_cast<int64_t>(dtypeSize_) * BUFFER_NUM +
                                   2 * hUB_ * W_ * static_cast<int64_t>(dtypeSize_) * BUFFER_NUM +
                                   2 * bUB_ * BUFFER_NUM +
                                   hUB_ * W_ * sizeof(float);

            int64_t min_s_effective = 1;
            int64_t maxS = (availableUbSize - const_cost_s) / coeff_s;
            sUB_ = std::max(min_s_effective, maxS);
        }
    }

    // 计算主核的循环次数和尾块大小
    // sUB_ 是有效输出行数（不包含重叠部分）
    hLoopCnt_ = (perCoreH + hUB_ - 1) / hUB_;
    bLoopCnt_ = (B_ + bUB_ - 1) / bUB_;
    sLoopCnt_ = (S_ + sUB_ - 1) / sUB_;

    // 计算主核的主块和尾块大小
    ubMainFactorH_ = hUB_;
    ubTailFactorH_ = (hLoopCnt_ == 1) ? perCoreH : (perCoreH % hUB_ == 0 ? hUB_ : perCoreH % hUB_);

    ubMainFactorB_ = bUB_;
    ubTailFactorB_ = (bLoopCnt_ == 1) ? B_ : (B_ % bUB_ == 0 ? bUB_ : B_ % bUB_);

    // ubMainFactorS 和 ubTailFactorS 存储有效输出大小（不包含重叠部分）
    ubMainFactorS_ = sUB_;

    // 计算尾块：有效输出的余数
    int64_t s_effective_remain = S_ % sUB_;
    if (s_effective_remain == 0 && sLoopCnt_ > 0) {
        // 能整除，尾块和主块一样
        ubTailFactorS_ = sUB_;
    } else if (s_effective_remain > 0) {
        // 有余数，尾块 = 余数
        ubTailFactorS_ = s_effective_remain;
    } else {
        // sLoopCnt_ == 0 的异常情况
        ubTailFactorS_ = sUB_;
    }

    // 计算尾核的参数（如果有尾核）
    if (hTailCoreCnt_ > 0 && hTailSize_ > 0) {
        // 尾核可能需要不同的H循环次数
        int64_t tailCoreH = hTailSize_;
        tailHLoopCnt_ = (tailCoreH + hUB_ - 1) / hUB_;
        tailBLoopCnt_ = bLoopCnt_;  // B和S的循环次数保持一致
        tailSLoopCnt_ = sLoopCnt_;

        tailCoreUbMainFactorH_ = hUB_;
        tailCoreUbTailFactorH_ = (tailHLoopCnt_ == 1) ? hUB_ :
                                 (tailCoreH % hUB_ == 0 ? hUB_ : tailCoreH % hUB_);

        tailCoreUbMainFactorB_ = ubMainFactorB_;
        tailCoreUbTailFactorB_ = ubTailFactorB_;

        tailCoreUbMainFactorS_ = ubMainFactorS_;  // 有效输出大小，不包含重叠
        tailCoreUbTailFactorS_ = ubTailFactorS_;  // 有效输出大小，不包含重叠
    } else {
        // 没有尾核，或尾核参数与主核相同
        tailHLoopCnt_ = hLoopCnt_;
        tailBLoopCnt_ = bLoopCnt_;
        tailSLoopCnt_ = sLoopCnt_;

        tailCoreUbMainFactorH_ = ubMainFactorH_;
        tailCoreUbTailFactorH_ = ubTailFactorH_;
        tailCoreUbMainFactorB_ = ubMainFactorB_;
        tailCoreUbTailFactorB_ = ubTailFactorB_;
        tailCoreUbMainFactorS_ = ubMainFactorS_;
        tailCoreUbTailFactorS_ = ubTailFactorS_;
    }

    // 保留原有的兼容性变量（用于日志输出）
    hUBTail_ = ubTailFactorH_;
    bUBTail_ = ubTailFactorB_;
    sUBTail_ = ubTailFactorS_;
    allUb_ = calculateBufferSize(ubMainFactorH_, ubMainFactorB_, ubMainFactorS_);

    return ge::GRAPH_SUCCESS;
}


ge::graphStatus MaskedCausalConv1dBackwardTiling::DoOpTiling()
{
    OP_CHECK_IF(ComputeInterCoreSplit() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "ComputeInterCoreSplit failed"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(ComputeIntraCoreUbTiling() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "ComputeIntraCoreUbTiling failed"), return ge::GRAPH_FAILED);

    // 填充 tilingData_ - 按照文档3.3节的结构体定义
    // 核间切分参数
    tilingData_.hMainCoreCnt = hMainCoreCnt_;
    tilingData_.hTailCoreCnt = hTailCoreCnt_;
    tilingData_.hMainSize = hMainSize_;
    tilingData_.hTailSize = hTailSize_;

    // 主核循环参数
    tilingData_.hLoopCnt = hLoopCnt_;
    tilingData_.bLoopCnt = bLoopCnt_;
    tilingData_.sLoopCnt = sLoopCnt_;

    // 主核UB切块参数
    tilingData_.ubMainFactorH = ubMainFactorH_;
    tilingData_.ubTailFactorH = ubTailFactorH_;
    tilingData_.ubMainFactorB = ubMainFactorB_;
    tilingData_.ubTailFactorB = ubTailFactorB_;
    tilingData_.ubMainFactorS = ubMainFactorS_;
    tilingData_.ubTailFactorS = ubTailFactorS_;

    // 尾核循环参数
    tilingData_.tailHLoopCnt = tailHLoopCnt_;
    tilingData_.tailBLoopCnt = tailBLoopCnt_;
    tilingData_.tailSLoopCnt = tailSLoopCnt_;

    // 尾核UB切块参数
    tilingData_.tailCoreUbMainFactorH = tailCoreUbMainFactorH_;
    tilingData_.tailCoreUbTailFactorH = tailCoreUbTailFactorH_;
    tilingData_.tailCoreUbMainFactorB = tailCoreUbMainFactorB_;
    tilingData_.tailCoreUbTailFactorB = tailCoreUbTailFactorB_;
    tilingData_.tailCoreUbMainFactorS = tailCoreUbMainFactorS_;
    tilingData_.tailCoreUbTailFactorS = tailCoreUbTailFactorS_;

    // 全局参数
    tilingData_.hasMask = hasMask_;
    tilingData_.S = S_;
    tilingData_.B = B_;
    tilingData_.H = H_;
    tilingData_.W = W_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedCausalConv1dBackwardTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MaskedCausalConv1dBackwardTiling::GetTilingKey() const
{
    // 可按 dtype 区分，当前返回 0
    if (dataType_ == ge::DataType::DT_BF16) {
        return TILING_KEY_MASK_EDCAUSAL_CONV1D_BACKWARD_BF16;
    } else if (dataType_ == ge::DataType::DT_FLOAT16) {
        return TILING_KEY_MASK_EDCAUSAL_CONV1D_BACKWARD_FP16;
    }
    
}

ge::graphStatus MaskedCausalConv1dBackwardTiling::GetWorkspaceSize()
{
    auto platformInfo = context_->GetPlatformInfo();
    uint32_t sysWorkspaceSize = 0;
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    }
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = static_cast<size_t>(0UL + sysWorkspaceSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedCausalConv1dBackwardTiling::PostTiling()
{
    // 设置核数
    context_->SetBlockDim(static_cast<uint32_t>(usedCoreNum_));

    // 保存 tiling 数据
    auto *raw = context_->GetRawTilingData();
    auto tilingDataSize = sizeof(MaskedCausalConv1dBackwardTilingDataV35);
    errno_t ret = memcpy_s(raw->GetData(), raw->GetCapacity(), reinterpret_cast<void *>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    raw->SetDataSize(tilingDataSize);

    context_->SetTilingKey(GetTilingKey());
    return ge::GRAPH_SUCCESS;
}

void MaskedCausalConv1dBackwardTiling::DumpTilingInfo()
{
    OP_LOGI(context_->GetNodeName(), "=== MaskedCausalConv1dBackward DumpTilingInfo ===");
    OP_LOGI(context_->GetNodeName(), "S=%ld B=%ld H=%ld W=%ld", S_, B_, H_, W_);
    OP_LOGI(context_->GetNodeName(), "hasMask=%ld dtypeSize=%zu", hasMask_, dtypeSize_);
    OP_LOGI(context_->GetNodeName(), "totalCoreNum=%lu usedCoreNum=%ld", totalCoreNum_, usedCoreNum_);
    OP_LOGI(context_->GetNodeName(), "hMainCoreCnt=%ld hTailCoreCnt=%ld", hMainCoreCnt_, hTailCoreCnt_);
    OP_LOGI(context_->GetNodeName(), "hMainSize=%ld hTailSize=%ld", hMainSize_, hTailSize_);
    OP_LOGI(context_->GetNodeName(), "hUB=%ld bUB=%ld sUB=%ld", hUB_, bUB_, sUB_);
    OP_LOGI(context_->GetNodeName(), "hLoopCnt=%ld bLoopCnt=%ld sLoopCnt=%ld", hLoopCnt_, bLoopCnt_, sLoopCnt_);
    OP_LOGI(context_->GetNodeName(), "hUBTail=%ld bUBTail=%ld sUBTail=%ld", hUBTail_, bUBTail_, sUBTail_);
    OP_LOGI(context_->GetNodeName(), "allUb=%ld", allUb_);
}


} // namespace optiling
