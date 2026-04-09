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

    OP_CHECK_IF(origin.GetDimNum() != NUM_3,
                OP_LOGE(context_->GetNodeName(), "grad_output dim num must be 3, but got %lu", origin.GetDimNum()),
                return ge::GRAPH_FAILED);

    S_ = static_cast<int64_t>(origin.GetDim(DIM_0));
    B_ = static_cast<int64_t>(origin.GetDim(DIM_1));
    H_ = static_cast<int64_t>(origin.GetDim(DIM_2));

    OP_CHECK_IF(S_ <= 0 || S_ > S_MAX, OP_LOGE(context_->GetNodeName(), "invalid S=%ld", S_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(B_ <= 0 || B_ > B_MAX, OP_LOGE(context_->GetNodeName(), "invalid B=%ld", B_), return ge::GRAPH_FAILED);
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

    OP_CHECK_IF(inOrigin.GetDimNum() != NUM_3,
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

    OP_CHECK_IF(wOrigin.GetDimNum() != DIM_2,
                OP_LOGE(context_->GetNodeName(), "weight dim num must be 2, but got %lu", wOrigin.GetDimNum()),
                return ge::GRAPH_FAILED);

    W_ = static_cast<int64_t>(wOrigin.GetDim(DIM_0));
    auto wH = static_cast<int64_t>(wOrigin.GetDim(DIM_1));

    OP_CHECK_IF(W_ != NUM_3, OP_LOGE(context_->GetNodeName(), "weight[0]=W must be 3, but got %ld", W_),
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

    OP_CHECK_IF(mOrigin.GetDimNum() != NUM_2,
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
        t != ge::DataType::DT_BOOL,
        OP_LOGE(context_->GetNodeName(), "mask dtype must be DT_BOOL, but got %s", Ops::Base::ToString(t).c_str()),
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
    // 核处理的H会比尾核多64(hMainSize = hTailSize + 64)
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
        hTailSize_ = hMainSize_;  // hTailSize仍等于hMainSize
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
    OP_LOGI(context_->GetNodeName(), "ubSize is %ld", ubSize_);
    OP_LOGI(context_->GetNodeName(), "availableUbSize is %ld", availableUbSize);
    OP_CHECK_IF(availableUbSize <= 0, OP_LOGE(context_->GetNodeName(), "available UB size <= 0, ubSize=%lu", ubSize_),
                return ge::GRAPH_FAILED);

    // UB先按H=64(128B)，全载核内BS，核内仅切 S, sUB_ 存储的是有效输出行数（不包含重叠部分）
    int64_t minH = DIM_ALIGN_ELEMENT;  // 64
    int64_t perCoreH = (hMainCoreCnt_ > 0) ? hMainSize_ : hTailSize_;

    hUB_ = minH;

    // Buffer占用估算函数
    // 参数 s_effective 是有效输出行数（不包含重叠部分）
    auto calculateBufferSize = [&](int64_t h, int64_t s_effective) -> int64_t {
        int64_t currentSLoopCnt = (S_ + s_effective - 1) / s_effective;
        int64_t sCacheLevels_ = 1;
        {
            int64_t t = currentSLoopCnt;
            while (t > 1) { ++sCacheLevels_; t >>= 1; }
        }
        
        int64_t s_with_overlap = s_effective + (W_ - 1);
        int64_t gradOutputSize = h * s_with_overlap * static_cast<int64_t>(dtypeSize_);
        int64_t inputSize = h * s_effective * static_cast<int64_t>(dtypeSize_);
        int64_t weightSize = h * W_ * static_cast<int64_t>(dtypeSize_);
        int64_t maskSize = 0;
        if (hasMask_) {
            maskSize = ((s_with_overlap + ALIGN_BYTES - 1) / ALIGN_BYTES) * ALIGN_BYTES;
            int64_t vecBytes   = ((s_with_overlap * static_cast<int64_t>(dtypeSize_) + (ALIGN_BYTES - 1)) / ALIGN_BYTES) * ALIGN_BYTES;
            int64_t planeBytes = s_with_overlap * h * static_cast<int64_t>(dtypeSize_);
            maskSize += vecBytes;
            maskSize += planeBytes;
        }
        int64_t gradInputSize = h * s_effective * static_cast<int64_t>(dtypeSize_);
        int64_t gradWeightSize = h * W_ * static_cast<int64_t>(dtypeSize_);
        int64_t tempBuffSize = h * W_ * B_ * sizeof(float);
        int64_t prodBufSize = W_ * s_effective * h * sizeof(float);
        int64_t sumBufSize = h * sizeof(float);

        int64_t wsSize = W_ * sCacheLevels_ * h * sizeof(float);
        int64_t totalSize = BUFFER_NUM * (gradOutputSize + inputSize + weightSize + maskSize +
                                          gradInputSize + gradWeightSize) + tempBuffSize
                                          + prodBufSize + sumBufSize + wsSize;

        return totalSize;
    };

    // 用 calculateBufferSize 作为真实成本函数，执行二分搜索以找到最大 sUB_，避免线性近似误差
    int64_t lowS = 1;
    int64_t highS = std::max<int64_t>(1, S_);
    while (lowS <= highS) {
        int64_t midS = (lowS + highS) / NUM_2;
        int64_t need = calculateBufferSize(hUB_, midS);
        if (need <= availableUbSize) {
            sUB_ = midS;        // 可行，尝试更大
            lowS = midS + 1;
        } else {
            highS = midS - 1;  // 不可行，缩小范围
        }
    }
    // 保护：不超过 S_
    if (sUB_ > S_) sUB_ = S_;

    // 循环次数
    hLoopCnt_ = (perCoreH + hUB_ - 1) / hUB_;
    sLoopCnt_ = (S_ + sUB_ - 1) / sUB_;

    // 主块与尾块
    ubMainFactorH_ = hUB_;
    ubTailFactorH_ = (hLoopCnt_ == 1) ? perCoreH : (perCoreH % hUB_ == 0 ? hUB_ : perCoreH % hUB_);

    // 均分 S 维
    if (sLoopCnt_ <= 0) {
        sLoopCnt_ = 1;
    }
    ubMainFactorS_ = (S_ + sLoopCnt_ - 1) / sLoopCnt_; // ceil
    ubTailFactorS_ = (S_ / sLoopCnt_);                  // floor
    ubMainFactorSCount_ = S_ % sLoopCnt_;                  // 需要分配主块大小的次数
    ubTailFactorSCount_ = sLoopCnt_ - mainCnt;             // 其余分配尾块大小

    // 尾核
    if (hTailCoreCnt_ > 0 && hTailSize_ > 0) {
        // 尾核可能需要不同的H循环次数
        int64_t tailCoreH = hTailSize_;
        tailHLoopCnt_ = (tailCoreH + hUB_ - 1) / hUB_;
        tailSLoopCnt_ = sLoopCnt_;

        tailCoreUbMainFactorH_ = hUB_;
        tailCoreUbTailFactorH_ = (tailHLoopCnt_ == 1) ? hUB_ :
                                 (tailCoreH % hUB_ == 0 ? hUB_ : tailCoreH % hUB_);

        tailCoreUbMainFactorS_ = ubMainFactorS_;  // 有效输出大小，不包含重叠
        tailCoreUbTailFactorS_ = ubTailFactorS_;  // 有效输出大小，不包含重叠
    } else {
        // 没有尾核，或尾核参数与主核相同
        tailHLoopCnt_ = hLoopCnt_;
        tailSLoopCnt_ = sLoopCnt_;

        tailCoreUbMainFactorH_ = ubMainFactorH_;
        tailCoreUbTailFactorH_ = ubTailFactorH_;
        tailCoreUbMainFactorS_ = ubMainFactorS_;
        tailCoreUbTailFactorS_ = ubTailFactorS_;
    }

    allUb_ = calculateBufferSize(ubMainFactorH_, ubMainFactorS_);
    OP_LOGI(context_->GetNodeName(), "allUb_ = %ld", allUb_);
    return ge::GRAPH_SUCCESS;
}


ge::graphStatus MaskedCausalConv1dBackwardTiling::DoOpTiling()
{
    OP_CHECK_IF(ComputeInterCoreSplit() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "ComputeInterCoreSplit failed"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(ComputeIntraCoreUbTiling() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "ComputeIntraCoreUbTiling failed"), return ge::GRAPH_FAILED);

    tilingData_.hMainCoreCnt = hMainCoreCnt_;
    tilingData_.hTailCoreCnt = hTailCoreCnt_;
    tilingData_.hMainSize = hMainSize_;
    tilingData_.hTailSize = hTailSize_;
    tilingData_.hLoopCnt = hLoopCnt_;
    tilingData_.sLoopCnt = sLoopCnt_;
    tilingData_.ubMainFactorH = ubMainFactorH_;
    tilingData_.ubTailFactorH = ubTailFactorH_;
    tilingData_.ubMainFactorS = ubMainFactorS_;
    tilingData_.ubTailFactorS = ubTailFactorS_;
    tilingData_.tailHLoopCnt = tailHLoopCnt_;
    tilingData_.tailSLoopCnt = tailSLoopCnt_;
    tilingData_.tailCoreUbMainFactorH = tailCoreUbMainFactorH_;
    tilingData_.tailCoreUbTailFactorH = tailCoreUbTailFactorH_;
    tilingData_.ubMainFactorSCount = ubMainFactorSCount_;
    tilingData_.ubTailFactorSCount = ubTailFactorSCount_;    
    tilingData_.tailCoreUbMainFactorS = tailCoreUbMainFactorS_;
    tilingData_.tailCoreUbTailFactorS = tailCoreUbTailFactorS_;
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
    context_->SetBlockDim(static_cast<uint32_t>(usedCoreNum_));

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

    OP_LOGI(context_->GetNodeName(), "hMainCoreCnt = %ld", hMainCoreCnt_);
    OP_LOGI(context_->GetNodeName(), "hTailCoreCnt = %ld", hTailCoreCnt_);
    OP_LOGI(context_->GetNodeName(), "hMainSize = %ld", hMainSize_);
    OP_LOGI(context_->GetNodeName(), "hTailSize = %ld", hTailSize_);
    OP_LOGI(context_->GetNodeName(), "hLoopCnt = %ld", hLoopCnt_);
    OP_LOGI(context_->GetNodeName(), "sLoopCnt = %ld", sLoopCnt_);
    OP_LOGI(context_->GetNodeName(), "ubMainFactorH = %ld", ubMainFactorH_);
    OP_LOGI(context_->GetNodeName(), "ubTailFactorH = %ld", ubTailFactorH_);
    OP_LOGI(context_->GetNodeName(), "ubMainFactorS = %ld", ubMainFactorS_);
    OP_LOGI(context_->GetNodeName(), "ubTailFactorS = %ld", ubTailFactorS_);
    OP_LOGI(context_->GetNodeName(), "ubMainFactorSCount = %ld", ubMainFactorSCount_);
    OP_LOGI(context_->GetNodeName(), "ubTailFactorSCount = %ld", ubTailFactorSCount_);
    OP_LOGI(context_->GetNodeName(), "tailHLoopCnt = %ld", tailHLoopCnt_);
    OP_LOGI(context_->GetNodeName(), "tailSLoopCnt = %ld", tailSLoopCnt_);
    OP_LOGI(context_->GetNodeName(), "tailCoreUbMainFactorH = %ld", tailCoreUbMainFactorH_);
    OP_LOGI(context_->GetNodeName(), "tailCoreUbTailFactorH = %ld", tailCoreUbTailFactorH_);
    OP_LOGI(context_->GetNodeName(), "tailCoreUbMainFactorS = %ld", tailCoreUbMainFactorS_);
    OP_LOGI(context_->GetNodeName(), "tailCoreUbTailFactorS = %ld", tailCoreUbTailFactorS_);
    OP_LOGI(context_->GetNodeName(), "hasMask = %ld", hasMask_);
    OP_LOGI(context_->GetNodeName(), "S = %ld", S_);
    OP_LOGI(context_->GetNodeName(), "B = %ld", B_);
    OP_LOGI(context_->GetNodeName(), "H = %ld", H_);
    OP_LOGI(context_->GetNodeName(), "W = %ld", W_);

}


} // namespace optiling
