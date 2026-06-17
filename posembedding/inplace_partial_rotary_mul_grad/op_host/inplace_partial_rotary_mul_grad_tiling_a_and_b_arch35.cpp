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
 * \file inplace_partial_rotary_mul_grad_tiling_a_and_b_arch35.cpp
 * \brief
 */
#include "rotary_position_embedding_grad_tiling.h"

using namespace AscendC;

namespace optiling {

constexpr uint64_t ROPE_GRAD_A_AND_B_TILING_PRIORITY = 40000;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t UB_FACTOR = 4;
constexpr int64_t HALF_INTERLEAVE_COEF = 2;
constexpr int64_t QUARTER_MODE_COEF = 4;

class InplacePartialRotaryMulGradRegBaseTilingClassAAndB : public InplacePartialRotaryMulGradRegBaseTilingClass {
public:
    explicit InplacePartialRotaryMulGradRegBaseTilingClassAAndB(gert::TilingContext *context)
        : InplacePartialRotaryMulGradRegBaseTilingClass(context)
    {
    }

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus PostTiling() override;
    void SetTilingData();

private:
    ge::graphStatus MergeDim();
    ge::graphStatus SplitCore();
    ge::graphStatus ComputeUbFactor();

    int64_t blockNumB_{0};
    int64_t blockFactorB_{0};
    int64_t blockNumS_{0};
    int64_t blockFactorS_{0};
    int64_t ubFactorB_{0};
    int64_t ubFactorS_{0};
    int64_t ubFactorN_{0};
    int64_t ubLoopNumN_{0};
    int64_t ubTailFactorN_{0};
};

bool InplacePartialRotaryMulGradRegBaseTilingClassAAndB::IsCapable()
{
    // 处理全boardcast和不boardcast的情况
    return (Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) &&
           (layout_ == RopeLayout::NO_BROADCAST || layout_ == RopeLayout::BROADCAST_BSN);
}

ge::graphStatus InplacePartialRotaryMulGradRegBaseTilingClassAAndB::MergeDim()
{
    b_ = b_ * n_ * s_;
    n_ = 1;
    s_ = 1;
    if (layout_ == RopeLayout::NO_BROADCAST) {
        cosb_ = b_;
    } else {
        cosb_ = 1;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradRegBaseTilingClassAAndB::SplitCore()
{
    blockFactorB_ = Ops::Base::CeilDiv(static_cast<uint64_t>(b_), aicoreParams_.numBlocks);
    blockNumB_ = Ops::Base::CeilDiv(static_cast<int64_t>(b_), blockFactorB_);
    usedCoreNum_ = blockNumB_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradRegBaseTilingClassAAndB::ComputeUbFactor()
{
    ubFactorB_ = 1;
    // UB占用大小：
    // 111d：2 * (2b + 1) * dAlign
    // b1sd：2 * 4 * b * dAlign
    auto maxDtypeSize = std::max(ge::GetSizeByDataType(dtype_), ge::GetSizeByDataType(cosDtype_));
    int64_t dSize = Ops::Base::CeilAlign(d_ * maxDtypeSize / dSplitCoef_, this->blockSize_) * dSplitCoef_;
    int64_t numOfDAvailable = Ops::Base::FloorDiv(static_cast<int64_t>(aicoreParams_.ubSize), DOUBLE_BUFFER * dSize);
    OP_CHECK_IF(numOfDAvailable < UB_FACTOR,
                OP_LOGE(context_, "D is too big to load in ub, ubSize is %ld bytes, loading requires %ld bytes.",
                        static_cast<int64_t>(aicoreParams_.ubSize), UB_FACTOR * dSize * (ubFactorB_ + 1)),
                return ge::GRAPH_FAILED);
    if (layout_ == RopeLayout::NO_BROADCAST) {
        numOfDAvailable /= UB_FACTOR;
        ubFactorB_ = std::min(blockFactorB_, numOfDAvailable);
    } else {
        numOfDAvailable -= 1;
        numOfDAvailable /= DOUBLE_BUFFER;
        ubFactorB_ = std::min(blockFactorB_, numOfDAvailable);
    }

    ubFactorB_ = std::min(ubFactorB_, MAX_COPY_BLOCK_COUNT / dSplitCoef_);

    return ge::GRAPH_SUCCESS;
}

void InplacePartialRotaryMulGradRegBaseTilingClassAAndB::SetTilingData()
{
    tilingData_->inplacePartialRotaryMulGradABParams.b = b_;
    tilingData_->inplacePartialRotaryMulGradABParams.s = s_;
    tilingData_->inplacePartialRotaryMulGradABParams.d = d_;
    tilingData_->inplacePartialRotaryMulGradABParams.n = n_;
    tilingData_->inplacePartialRotaryMulGradABParams.blockNumB = blockNumB_;
    tilingData_->inplacePartialRotaryMulGradABParams.blockFactorB = blockFactorB_;
    tilingData_->inplacePartialRotaryMulGradABParams.blockNumS = blockNumS_;
    tilingData_->inplacePartialRotaryMulGradABParams.blockFactorS = blockFactorS_;
    tilingData_->inplacePartialRotaryMulGradABParams.ubFactorS = ubFactorS_;
    tilingData_->inplacePartialRotaryMulGradABParams.ubFactorB = ubFactorB_;
    tilingData_->inplacePartialRotaryMulGradABParams.ubLoopNumN = ubLoopNumN_;
    tilingData_->inplacePartialRotaryMulGradABParams.ubFactorN = ubFactorN_;
    tilingData_->inplacePartialRotaryMulGradABParams.ubTailFactorN = ubTailFactorN_;
    tilingData_->inplacePartialRotaryMulGradABParams.usedCoreNum = usedCoreNum_;
    tilingData_->inplacePartialRotaryMulGradABParams.rotaryMode = static_cast<int64_t>(rotaryMode_);
    tilingData_->inplacePartialRotaryMulGradABParams.sliceStart = sliceStart_;
    tilingData_->inplacePartialRotaryMulGradABParams.sliceEnd = sliceEnd_;
    tilingData_->inplacePartialRotaryMulGradABParams.sliceLength = sliceLength_;
    tilingData_->inplacePartialRotaryMulGradABParams.dCosFlag = dCosFlag_;

    OP_LOGI(context_->GetNodeName(),
            "RopeGradRegBaseTilingClassAAndB tilingData: "
            "B is %ld, S is %ld, D is %ld, N is %ld, blockNumB %ld, blockFactorB_ is %ld, "
            "blockNumS %ld, blockFactorS is %ld, ubFactorS is %ld, ubFactorB is %ld,"
            "ubLoopNumN is %ld, ubFactorN is %ld, ubTailFactorN is %ld, rotaryMode is %ld, tilingKey is %ld,"
            "usedCoreNum = %ld.",
            tilingData_->inplacePartialRotaryMulGradABParams.b, tilingData_->inplacePartialRotaryMulGradABParams.s,
            tilingData_->inplacePartialRotaryMulGradABParams.d, tilingData_->inplacePartialRotaryMulGradABParams.n,
            tilingData_->inplacePartialRotaryMulGradABParams.blockNumB,
            tilingData_->inplacePartialRotaryMulGradABParams.blockFactorB,
            tilingData_->inplacePartialRotaryMulGradABParams.blockNumS,
            tilingData_->inplacePartialRotaryMulGradABParams.blockFactorS,
            tilingData_->inplacePartialRotaryMulGradABParams.ubFactorS,
            tilingData_->inplacePartialRotaryMulGradABParams.ubFactorB,
            tilingData_->inplacePartialRotaryMulGradABParams.ubLoopNumN,
            tilingData_->inplacePartialRotaryMulGradABParams.ubFactorN,
            tilingData_->inplacePartialRotaryMulGradABParams.ubTailFactorN,
            tilingData_->inplacePartialRotaryMulGradABParams.rotaryMode, tilingKey_,
            tilingData_->inplacePartialRotaryMulGradABParams.usedCoreNum);
}

ge::graphStatus InplacePartialRotaryMulGradRegBaseTilingClassAAndB::DoOpTiling()
{
    OP_CHECK_IF(MergeDim() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "failed to merge dim."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(SplitCore() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "failed to split core."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ComputeUbFactor() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "failed to compute ub factor."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(InitTilingData() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "InitTilingData failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(TilingReduce() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "TilingReduce failed."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradRegBaseTilingClassAAndB::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradRegBaseTilingClassAAndB::PostTiling()
{
    SetTilingData();
    SetRotaryXTilingData();
    uint32_t tilingKey = (layout_ == RopeLayout::NO_BROADCAST ? static_cast<uint32_t>(ropeGradTilingKey::TILING_KEY_A) :
                                                                static_cast<uint32_t>(ropeGradTilingKey::TILING_KEY_B));
    SetTilingKeyBlockDim(tilingKey);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(RotaryPositionEmbeddingGrad, InplacePartialRotaryMulGradRegBaseTilingClassAAndB,
                             ROPE_GRAD_A_AND_B_TILING_PRIORITY);
} // namespace optiling