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
 * \file inplace_partial_rotary_mul_grad_regbase_tiling_a_and_b.cpp
 * \brief A/B template tiling for InplacePartialRotaryMulGrad (NO_BROADCAST / BROADCAST_BSN)
 */
#include "inplace_partial_rotary_mul_grad_regbase_tiling.h"

namespace optiling {
constexpr uint64_t TILING_KEY_A = 205;
constexpr uint64_t TILING_KEY_B = 206;
constexpr uint64_t INPLACE_PARTIAL_ROTARY_MUL_GRAD_A_AND_B_TILING_PRIORITY = 40000;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t UB_FACTOR = 4;
constexpr int64_t HALF_INTERLEAVE_COEF = 2;
constexpr int64_t QUARTER_MODE_COEF = 4;

class InplacePartialRotaryMulGradRegbaseTilingAAndB : public InplacePartialRotaryMulGradRegbaseTiling {
public:
    explicit InplacePartialRotaryMulGradRegbaseTilingAAndB(gert::TilingContext *context)
        : InplacePartialRotaryMulGradRegbaseTiling(context)
    {
    }

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    void SetTilingData();

private:
    ge::graphStatus MergeDim();
    ge::graphStatus SplitCore();
    ge::graphStatus ComputeUbFactor();

    int64_t blockNumB_{0};
    int64_t blockFactorB_{0};
    int64_t ubFactorB_{0};
};

bool InplacePartialRotaryMulGradRegbaseTilingAAndB::IsCapable()
{
    return (Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) &&
           (layout_ == InplacePartialRotaryMulGradLayout::NO_BROADCAST ||
            layout_ == InplacePartialRotaryMulGradLayout::BROADCAST_BSN);
}

ge::graphStatus InplacePartialRotaryMulGradRegbaseTilingAAndB::MergeDim()
{
    b_ = b_ * n_ * s_;
    n_ = 1;
    s_ = 1;
    if (layout_ == InplacePartialRotaryMulGradLayout::NO_BROADCAST) {
        cosb_ = b_;
    } else {
        cosb_ = 1;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradRegbaseTilingAAndB::SplitCore()
{
    blockFactorB_ = Ops::Base::CeilDiv(b_, static_cast<int64_t>(aicoreParams_.numBlocks));
    blockNumB_ = Ops::Base::CeilDiv(b_, blockFactorB_);
    usedCoreNum_ = blockNumB_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradRegbaseTilingAAndB::ComputeUbFactor()
{
    ubFactorB_ = 1;
    int64_t typeSize = ge::GetSizeByDataType(dtype_);
    int64_t cosTypeSize = ge::GetSizeByDataType(cosDtype_);
    int64_t maxTypeSize = std::max(typeSize, cosTypeSize);
    int64_t dSize = Ops::Base::CeilAlign(sliceLength_ * maxTypeSize / dSplitCoef_, this->blockSize_) * dSplitCoef_;

    int64_t numOfDAvailable = Ops::Base::FloorDiv(static_cast<int64_t>(aicoreParams_.ubSize), DOUBLE_BUFFER * dSize);
    OP_CHECK_IF(numOfDAvailable < UB_FACTOR,
                OP_LOGE(context_, "D is too big to load in ub, ubSize is %ld bytes, loading requires %ld bytes.",
                        static_cast<int64_t>(aicoreParams_.ubSize), UB_FACTOR * dSize * (ubFactorB_ + 1)),
                return ge::GRAPH_FAILED);

    if (layout_ == InplacePartialRotaryMulGradLayout::NO_BROADCAST) {
        numOfDAvailable /= UB_FACTOR;
        ubFactorB_ = std::min(blockFactorB_, numOfDAvailable);
    } else {
        // BROADCAST_BSN: cos+sin single buffer，对齐 reference rope_grad_regbase_tiling_a_and_b
        numOfDAvailable -= 1;
        numOfDAvailable /= DOUBLE_BUFFER;
        ubFactorB_ = std::min(blockFactorB_, numOfDAvailable);
    }
    ubFactorB_ = std::min(ubFactorB_, MAX_COPY_BLOCK_COUNT / dSplitCoef_);
    return ge::GRAPH_SUCCESS;
}

void InplacePartialRotaryMulGradRegbaseTilingAAndB::SetTilingData()
{
    tilingData_->inplacePartialRotaryMulGradParams.b = b_;
    tilingData_->inplacePartialRotaryMulGradParams.s = s_;
    tilingData_->inplacePartialRotaryMulGradParams.d = d_;
    tilingData_->inplacePartialRotaryMulGradParams.n = n_;
    tilingData_->inplacePartialRotaryMulGradParams.blockNumB = blockNumB_;
    tilingData_->inplacePartialRotaryMulGradParams.blockFactorB = blockFactorB_;
    tilingData_->inplacePartialRotaryMulGradParams.ubFactorB = ubFactorB_;
    tilingData_->inplacePartialRotaryMulGradParams.usedCoreNum = usedCoreNum_;
    tilingData_->inplacePartialRotaryMulGradParams.rotaryMode = static_cast<int64_t>(rotaryMode_);
    tilingData_->inplacePartialRotaryMulGradParams.sliceStart = sliceStart_;
    tilingData_->inplacePartialRotaryMulGradParams.sliceEnd = sliceEnd_;
    tilingData_->inplacePartialRotaryMulGradParams.sliceLength = sliceLength_;
    tilingData_->inplacePartialRotaryMulGradParams.dSplitCoef = dSplitCoef_;
    tilingData_->sliceLength = sliceLength_;
}

ge::graphStatus InplacePartialRotaryMulGradRegbaseTilingAAndB::DoOpTiling()
{
    OP_CHECK_IF(MergeDim() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "failed to merge dim."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(SplitCore() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "failed to split core."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ComputeUbFactor() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "failed to compute ub factor."), return ge::GRAPH_FAILED);

    if (tilingData_ == nullptr) {
        tilingData_ = context_->GetTilingData<InplacePartialRotaryMulGradTilingData>();
        OP_CHECK_IF(tilingData_ == nullptr, OP_LOGE(context_->GetNodeName(), "get tilingdata ptr failed"),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF((memset_s(tilingData_, sizeof(InplacePartialRotaryMulGradTilingData), 0,
                          sizeof(InplacePartialRotaryMulGradTilingData)) != EOK),
                OP_LOGE(context_->GetNodeName(), "memset tilingdata failed"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradRegbaseTilingAAndB::PostTiling()
{
    SetTilingData();

    uint32_t tilingKey = (layout_ == InplacePartialRotaryMulGradLayout::NO_BROADCAST)
                             ? static_cast<uint32_t>(TILING_KEY_A)
                             : static_cast<uint32_t>(TILING_KEY_B);

    tilingKey_ = static_cast<uint64_t>(tilingKey);
    context_->SetBlockDim(usedCoreNum_);

    OP_LOGI(context_->GetNodeName(),
            "InplacePartialRotaryMulGradAAndB tiling: B=%ld S=%ld D=%ld N=%ld "
            "blockNumB=%ld blockFactorB=%ld ubFactorB=%ld usedCoreNum=%ld rotaryMode=%ld "
            "sliceStart=%ld sliceEnd=%ld sliceLength=%ld tilingKey=%u",
            b_, s_, d_, n_, blockNumB_, blockFactorB_, ubFactorB_, usedCoreNum_,
            static_cast<int64_t>(rotaryMode_), sliceStart_, sliceEnd_, sliceLength_, tilingKey);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(InplacePartialRotaryMulGrad, InplacePartialRotaryMulGradRegbaseTilingAAndB,
                             INPLACE_PARTIAL_ROTARY_MUL_GRAD_A_AND_B_TILING_PRIORITY);
} // namespace optiling
