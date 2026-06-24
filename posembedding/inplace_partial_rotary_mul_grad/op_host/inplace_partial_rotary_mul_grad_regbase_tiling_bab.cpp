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
 * \file inplace_partial_rotary_mul_grad_regbase_tiling_bab.cpp
 * \brief BAB template tiling for InplacePartialRotaryMulGrad (BSND layout, cosb_==1)
 */
#include "inplace_partial_rotary_mul_grad_regbase_tiling.h"
namespace optiling {
constexpr uint64_t TILING_KEY_BAB = 203;
constexpr uint64_t INPLACE_PARTIAL_ROTARY_MUL_GRAD_BAB_TILING_PRIORITY = 20000;
constexpr uint32_t MIN_UB_LOAD_D_NUM = 4;
constexpr uint32_t DOUBLE_BUFFER = 2;

class InplacePartialRotaryMulGradRegbaseTilingBab : public InplacePartialRotaryMulGradRegbaseTiling {
public:
    explicit InplacePartialRotaryMulGradRegbaseTilingBab(gert::TilingContext *context)
        : InplacePartialRotaryMulGradRegbaseTiling(context)
    {
    }

protected:
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

    bool IsCapable() override
    {
        if (Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_) &&
            (layout_ == InplacePartialRotaryMulGradLayout::BSND) && (cosb_ == 1)) {
            return true;
        }
        return false;
    }

private:
    int64_t coreNum_ = 0;
    int64_t blockNumB_ = 0;
    int64_t blockFactorB_ = 0;
    int64_t blockNumS_ = 0;
    int64_t blockFactorS_ = 0;
    int64_t ubFactorS_ = 1;
    int64_t ubFactorB_ = 1;
    int64_t ubLoopNumN_ = 0;
    int64_t ubFactorN_ = 1;
    int64_t ubTailFactorN_ = 0;
    int64_t ubSize_ = 0;
    InplacePartialRotaryMulGradRegbaseTilingData tilingDataBab_;

    void SplitCore();
    ge::graphStatus SplitUb();
};

ge::graphStatus InplacePartialRotaryMulGradRegbaseTilingBab::DoOpTiling()
{
    ubSize_ = aicoreParams_.ubSize;
    coreNum_ = static_cast<int64_t>(aicoreParams_.numBlocks);

    ge::graphStatus status = SplitUb();
    if (status != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "SplitUb Failed.");
        return ge::GRAPH_FAILED;
    }
    SplitCore();
    if (blockNumB_ * blockNumS_ > coreNum_) {
        OP_LOGE(context_->GetNodeName(), "split coreNum [%ld] large than coreNum[%ld]", blockNumB_ * blockNumS_,
                coreNum_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void InplacePartialRotaryMulGradRegbaseTilingBab::SplitCore()
{
    auto blockFactorB1 = Ops::Base::CeilDiv(b_, coreNum_);
    auto blockNumB1 = Ops::Base::CeilDiv(b_, blockFactorB1);
    auto blockNumS1 = std::min(coreNum_ / blockNumB1, s_);
    auto blockFactorS1 = Ops::Base::CeilDiv(s_, blockNumS1);
    blockNumS1 = Ops::Base::CeilDiv(s_, blockFactorS1);
    auto usedCoreNum1 = blockNumB1 * blockNumS1;

    auto blockFactorS2 = Ops::Base::CeilDiv(s_, coreNum_);
    auto blockNumS2 = Ops::Base::CeilDiv(s_, blockFactorS2);
    auto blockNumB2 = std::min(coreNum_ / blockNumS2, b_);
    auto blockFactorB2 = Ops::Base::CeilDiv(b_, blockNumB2);
    blockNumB2 = Ops::Base::CeilDiv(b_, blockFactorB2);
    auto usedCoreNum2 = blockNumB2 * blockNumS2;

    auto ubFactorS1 = std::min(ubFactorS_, blockFactorS1);
    auto ubFactorS2 = std::min(ubFactorS_, blockFactorS2);
    if (usedCoreNum1 * ubFactorS1 >= usedCoreNum2 * ubFactorS2) {
        blockNumB_ = blockNumB1;
        blockFactorB_ = blockFactorB1;
        blockNumS_ = blockNumS1;
        blockFactorS_ = blockFactorS1;
        usedCoreNum_ = usedCoreNum1;
        ubFactorS_ = ubFactorS1;
    } else {
        blockNumB_ = blockNumB2;
        blockFactorB_ = blockFactorB2;
        blockNumS_ = blockNumS2;
        blockFactorS_ = blockFactorS2;
        usedCoreNum_ = usedCoreNum2;
        ubFactorS_ = ubFactorS2;
    }
}

ge::graphStatus InplacePartialRotaryMulGradRegbaseTilingBab::SplitUb()
{
    int64_t typeSize = ge::GetSizeByDataType(dtype_);
    int64_t cosTypeSize = ge::GetSizeByDataType(cosDtype_);
    int64_t maxTypeSize = std::max(typeSize, cosTypeSize);
    int64_t dAlign = Ops::Base::CeilAlign(sliceLength_ * maxTypeSize / dSplitCoef_, blockSize_) * dSplitCoef_;

    int64_t canLoadDNum = Ops::Base::FloorDiv(ubSize_, dAlign);
    if (canLoadDNum < MIN_UB_LOAD_D_NUM + MIN_UB_LOAD_D_NUM) {
        OP_LOGE(context_->GetNodeName(), "ubSize_ can't load 8 d_, d_ = %ld.", sliceLength_);
        return ge::GRAPH_FAILED;
    }
    canLoadDNum = canLoadDNum / MIN_UB_LOAD_D_NUM;
    int64_t ubLoopNum = Ops::Base::CeilDiv(n_, (canLoadDNum - 1));
    ubFactorN_ = std::min(Ops::Base::CeilDiv(n_, ubLoopNum), MAX_COPY_BLOCK_COUNT / dSplitCoef_);
    ubLoopNumN_ = Ops::Base::CeilDiv(n_, ubFactorN_);
    ubTailFactorN_ = (n_ % ubFactorN_ == 0) ? ubFactorN_ : n_ % ubFactorN_;
    int64_t ubFactorS = Ops::Base::FloorDiv(canLoadDNum, n_ + 1);
    ubFactorS_ = (ubFactorS == 0) ? 1 : ubFactorS;

    int64_t dyBytes = 4 * ubFactorS_ * ubFactorN_ * dAlign;
    int64_t cosBytes = 4 * ubFactorS_ * dAlign;
    OP_CHECK_IF(dyBytes + cosBytes > static_cast<int64_t>(ubSize_),
                OP_LOGE(context_->GetNodeName(), "UB overflow: need %ld > %ld.", dyBytes + cosBytes, ubSize_),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradRegbaseTilingBab::PostTiling()
{
    tilingDataBab_.set_b(b_);
    tilingDataBab_.set_s(s_);
    tilingDataBab_.set_d(d_);
    tilingDataBab_.set_n(n_);
    tilingDataBab_.set_ubFactorS(ubFactorS_);
    tilingDataBab_.set_ubFactorB(ubFactorB_);
    tilingDataBab_.set_ubLoopNumN(ubLoopNumN_);
    tilingDataBab_.set_ubFactorN(ubFactorN_);
    tilingDataBab_.set_ubTailFactorN(ubTailFactorN_);
    tilingDataBab_.set_blockNumB(blockNumB_);
    tilingDataBab_.set_blockFactorB(blockFactorB_);
    tilingDataBab_.set_blockNumS(blockNumS_);
    tilingDataBab_.set_blockFactorS(blockFactorS_);
    tilingDataBab_.set_usedCoreNum(usedCoreNum_);
    tilingDataBab_.set_rotaryMode(static_cast<int64_t>(rotaryMode_));
    tilingDataBab_.set_sliceStart(sliceStart_);
    tilingDataBab_.set_sliceEnd(sliceEnd_);
    tilingDataBab_.set_sliceLength(sliceLength_);
    tilingDataBab_.set_dSplitCoef(dSplitCoef_);

    // 设置 userWorkspace
    size_t *userWorkspaceSize = context_->GetWorkspaceSizes(WORKSPACE_COUNT);
    OP_CHECK_NULL_WITH_CONTEXT(context_, userWorkspaceSize);
    userWorkspaceSize[0] = RESERVED_WORKSPACE;

    auto rawTilingDataPtr = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTilingDataPtr);
    if (tilingDataBab_.GetDataSize() > rawTilingDataPtr->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }

    tilingDataBab_.SaveToBuffer(rawTilingDataPtr->GetData(), rawTilingDataPtr->GetCapacity());
    rawTilingDataPtr->SetDataSize(tilingDataBab_.GetDataSize());

    tilingKey_ = static_cast<uint64_t>(TILING_KEY_BAB);
    context_->SetTilingKey(tilingKey_);
    context_->SetBlockDim(usedCoreNum_);

    OP_LOGI(context_->GetNodeName(),
            "InplacePartialRotaryMulGradBAB tiling: B=%ld S=%ld D=%ld N=%ld "
            "blockNumB=%ld blockFactorB=%ld blockNumS=%ld blockFactorS=%ld "
            "ubFactorS=%ld ubFactorB=%ld ubLoopNumN=%ld ubFactorN=%ld ubTailFactorN=%ld "
            "usedCoreNum=%ld rotaryMode=%ld dSplitCoef=%ld "
            "sliceStart=%ld sliceEnd=%ld sliceLength=%ld "
            " tilingKey=%lu",
            b_, s_, d_, n_, blockNumB_, blockFactorB_, blockNumS_, blockFactorS_, ubFactorS_, ubFactorB_, ubLoopNumN_,
            ubFactorN_, ubTailFactorN_, usedCoreNum_, static_cast<int64_t>(rotaryMode_), dSplitCoef_, sliceStart_,
            sliceEnd_, sliceLength_, tilingKey_);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(InplacePartialRotaryMulGrad, InplacePartialRotaryMulGradRegbaseTilingBab,
                             INPLACE_PARTIAL_ROTARY_MUL_GRAD_BAB_TILING_PRIORITY);
} // namespace optiling
