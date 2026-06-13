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
 * \file inplace_partial_rotary_mul_grad_regbase_tiling_aba_and_ba.cpp
 * \brief ABA/BA template tiling for InplacePartialRotaryMulGrad (BNSD layout)
 */
#include "inplace_partial_rotary_mul_grad_regbase_tiling.h"

namespace optiling {
constexpr uint64_t TILING_KEY_ABA = 201;
constexpr uint64_t TILING_KEY_BA = 202;
constexpr uint64_t INPLACE_PARTIAL_ROTARY_MUL_GRAD_ABA_AND_BA_TILING_PRIORITY = 10000;
constexpr int64_t UB_FACTOR = 4;
constexpr int64_t HALF_INTERLEAVE_COEF = 2;
constexpr int64_t QUARTER_MODE_COEF = 4;

class InplacePartialRotaryMulGradRegbaseTilingAbaAndBa : public InplacePartialRotaryMulGradRegbaseTiling {
public:
    explicit InplacePartialRotaryMulGradRegbaseTilingAbaAndBa(gert::TilingContext *context)
        : InplacePartialRotaryMulGradRegbaseTiling(context)
    {
    }

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    void SetTilingData();

private:
    ge::graphStatus SplitCore();
    ge::graphStatus ComputeUbFactor();

    int64_t ubFactorB_{1};
    int64_t ubFactorS_{0};
    int64_t ubFactorN_{0};
    int64_t ubLoopNumN_{0};
    int64_t ubTailFactorN_{0};
    int64_t blockNumB_{0};
    int64_t blockFactorB_{0};
    int64_t blockNumS_{0};
    int64_t blockFactorS_{0};
};

bool InplacePartialRotaryMulGradRegbaseTilingAbaAndBa::IsCapable()
{
    return Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_) &&
           (layout_ == InplacePartialRotaryMulGradLayout::BNSD);
}

ge::graphStatus InplacePartialRotaryMulGradRegbaseTilingAbaAndBa::SplitCore()
{
    auto numBlocks = static_cast<int64_t>(aicoreParams_.numBlocks);
    auto blockFactorB1 = Ops::Base::CeilDiv(b_, numBlocks);
    auto blockNumB1 = Ops::Base::CeilDiv(b_, blockFactorB1);
    auto blockNumS1 = std::min(numBlocks / blockNumB1, s_);
    auto blockFactorS1 = Ops::Base::CeilDiv(s_, blockNumS1);
    blockNumS1 = Ops::Base::CeilDiv(s_, blockFactorS1);
    auto usedCoreNum1 = blockNumB1 * blockNumS1;

    auto blockFactorS2 = Ops::Base::CeilDiv(s_, numBlocks);
    auto blockNumS2 = Ops::Base::CeilDiv(s_, blockFactorS2);
    auto blockNumB2 = std::min(numBlocks / blockNumS2, b_);
    auto blockFactorB2 = Ops::Base::CeilDiv(b_, blockNumB2);
    blockNumB2 = Ops::Base::CeilDiv(b_, blockFactorB2);
    auto usedCoreNum2 = blockNumB2 * blockNumS2;

    if (usedCoreNum1 >= usedCoreNum2) {
        blockNumB_ = blockNumB1;
        blockFactorB_ = blockFactorB1;
        blockNumS_ = blockNumS1;
        blockFactorS_ = blockFactorS1;
        usedCoreNum_ = usedCoreNum1;
    } else {
        blockNumB_ = blockNumB2;
        blockFactorB_ = blockFactorB2;
        blockNumS_ = blockNumS2;
        blockFactorS_ = blockFactorS2;
        usedCoreNum_ = usedCoreNum2;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradRegbaseTilingAbaAndBa::ComputeUbFactor()
{
    int64_t typeSize = ge::GetSizeByDataType(dtype_);
    int64_t cosTypeSize = ge::GetSizeByDataType(cosDtype_);
    int64_t maxTypeSize = std::max(typeSize, cosTypeSize);
    int64_t dSize = Ops::Base::CeilAlign(sliceLength_ * maxTypeSize / dSplitCoef_, this->blockSize_) * dSplitCoef_;

    int64_t numOfDAvailable =
        Ops::Base::FloorDiv(static_cast<int64_t>(aicoreParams_.ubSize), UB_FACTOR * dSize);
    OP_CHECK_IF(numOfDAvailable < 2,
                OP_LOGE(context_, "D is too big, ubSize is %ld, loading requires %ld.",
                        static_cast<int64_t>(aicoreParams_.ubSize), UB_FACTOR * 2 * dSize),
                return ge::GRAPH_FAILED);

    ubFactorS_ = std::min(blockFactorS_, Ops::Base::FloorDiv(numOfDAvailable, ubFactorB_ + 1));
    ubFactorS_ = std::min(ubFactorS_, MAX_COPY_BLOCK_COUNT / dSplitCoef_);
    ubFactorS_ = std::max(ubFactorS_, int64_t(1));
    numOfDAvailable = numOfDAvailable / ubFactorS_;

    if (numOfDAvailable > ubFactorB_ + 1) {
        if (cosb_ == 1) {
            numOfDAvailable -= 1;
            ubFactorN_ = std::min(n_, numOfDAvailable);
            numOfDAvailable = numOfDAvailable / ubFactorN_;
        } else {
            ubFactorN_ = std::min(n_, numOfDAvailable - 1);
            numOfDAvailable = numOfDAvailable / (ubFactorN_ + 1);
        }
        if (numOfDAvailable > 1) {
            ubFactorB_ = std::min(blockFactorB_, numOfDAvailable);
        }
    }
    // 最终验证：总 UB 不超限
    int64_t dAlign = Ops::Base::CeilAlign(sliceLength_ * maxTypeSize / dSplitCoef_, this->blockSize_) *
                     dSplitCoef_;
    // dAlign 已包含 maxTypeSize 乘法，不再乘 typeSize/cosTypeSize
    int64_t dyBytes = 4 * ubFactorB_ * ubFactorS_ * ubFactorN_ * dAlign;
    int64_t cosBytes = (cosb_ == 1) ? (4 * ubFactorS_ * dAlign)
                                     : (4 * ubFactorB_ * ubFactorS_ * dAlign);
    OP_CHECK_IF(dyBytes + cosBytes > static_cast<int64_t>(aicoreParams_.ubSize),
                OP_LOGE(context_, "UB overflow: need %ld > %ld.", dyBytes + cosBytes,
                        static_cast<int64_t>(aicoreParams_.ubSize)),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradRegbaseTilingAbaAndBa::DoOpTiling()
{
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

void InplacePartialRotaryMulGradRegbaseTilingAbaAndBa::SetTilingData()
{
    tilingData_->inplacePartialRotaryMulGradParams.b = b_;
    tilingData_->inplacePartialRotaryMulGradParams.s = s_;
    tilingData_->inplacePartialRotaryMulGradParams.d = d_;
    tilingData_->inplacePartialRotaryMulGradParams.n = n_;
    tilingData_->inplacePartialRotaryMulGradParams.blockNumB = blockNumB_;
    tilingData_->inplacePartialRotaryMulGradParams.blockFactorB = blockFactorB_;
    tilingData_->inplacePartialRotaryMulGradParams.blockNumS = blockNumS_;
    tilingData_->inplacePartialRotaryMulGradParams.blockFactorS = blockFactorS_;
    tilingData_->inplacePartialRotaryMulGradParams.ubFactorS = ubFactorS_;
    tilingData_->inplacePartialRotaryMulGradParams.ubFactorB = ubFactorB_;
    tilingData_->inplacePartialRotaryMulGradParams.ubLoopNumN = ubLoopNumN_;
    tilingData_->inplacePartialRotaryMulGradParams.ubFactorN = ubFactorN_;
    tilingData_->inplacePartialRotaryMulGradParams.ubTailFactorN = ubTailFactorN_;
    tilingData_->inplacePartialRotaryMulGradParams.usedCoreNum = usedCoreNum_;
    tilingData_->inplacePartialRotaryMulGradParams.rotaryMode = static_cast<int64_t>(rotaryMode_);
    tilingData_->inplacePartialRotaryMulGradParams.sliceStart = sliceStart_;
    tilingData_->inplacePartialRotaryMulGradParams.sliceEnd = sliceEnd_;
    tilingData_->inplacePartialRotaryMulGradParams.sliceLength = sliceLength_;
    tilingData_->inplacePartialRotaryMulGradParams.dSplitCoef = dSplitCoef_;
    tilingData_->sliceLength = sliceLength_;
}

ge::graphStatus InplacePartialRotaryMulGradRegbaseTilingAbaAndBa::PostTiling()
{
    SetTilingData();

    uint32_t tilingKey = (cosb_ == 1) ? static_cast<uint32_t>(TILING_KEY_BA)
                                       : static_cast<uint32_t>(TILING_KEY_ABA);

    tilingKey_ = static_cast<uint64_t>(tilingKey);
    context_->SetBlockDim(usedCoreNum_);

    OP_LOGI(context_->GetNodeName(),
            "InplacePartialRotaryMulGradABAAndBA tiling: B=%ld S=%ld D=%ld N=%ld "
            "blockNumB=%ld blockFactorB=%ld blockNumS=%ld blockFactorS=%ld "
            "ubFactorS=%ld ubFactorB=%ld ubLoopNumN=%ld ubFactorN=%ld ubTailFactorN=%ld "
            "usedCoreNum=%ld rotaryMode=%ld sliceStart=%ld sliceEnd=%ld sliceLength=%ld tilingKey=%u",
            b_, s_, d_, n_, blockNumB_, blockFactorB_, blockNumS_, blockFactorS_, ubFactorS_, ubFactorB_, ubLoopNumN_,
            ubFactorN_, ubTailFactorN_, usedCoreNum_, static_cast<int64_t>(rotaryMode_), sliceStart_, sliceEnd_,
            sliceLength_, tilingKey);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(InplacePartialRotaryMulGrad, InplacePartialRotaryMulGradRegbaseTilingAbaAndBa,
                             INPLACE_PARTIAL_ROTARY_MUL_GRAD_ABA_AND_BA_TILING_PRIORITY);
} // namespace optiling
