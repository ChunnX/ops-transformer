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
 * \file inplace_partial_rotary_mul_grad_tiling_ab_arch35.cpp
 * \brief
 */
#include "op_host/tiling_templates_registry.h"
#include "inplace_partial_rotary_mul_grad_tiling.h"

namespace optiling {
constexpr int64_t CONST_TWO = 2;
constexpr int64_t CONST_FOUR = 4;
constexpr int64_t DB_FLAG = 2;

class InplacePartialRotaryMulGradRegBaseTilingClassAB : public InplacePartialRotaryMulGradRegBaseTilingClass {
public:
    explicit InplacePartialRotaryMulGradRegBaseTilingClassAB(gert::TilingContext *context)
        : InplacePartialRotaryMulGradRegBaseTilingClass(context)
    {
    }

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

private:
    int64_t blockNumBS_ = 0;
    int64_t blockFactorBS_ = 0;
    int64_t blockTailBS_ = 0;
    int64_t blockNumN_ = 0;
    int64_t blockFactorN_ = 0;
    int64_t blockTailN_ = 0;
    int64_t ubFactorBS_ = 0;
    int64_t ubFactorN_ = 0;
    int64_t dAlign_ = 0;
    int64_t bn_ = 0;
};

ge::graphStatus InplacePartialRotaryMulGradRegBaseTilingClassAB::DoOpTiling()
{
    int64_t bs = b_ * s_;
    bn_ = n_;
    if (cosb_ == 1) {
        bs = s_;
        bn_ = b_ * n_;
    }
    int64_t typeSize = std::max(ge::GetSizeByDataType(dtype_), ge::GetSizeByDataType(cosDtype_));
    dAlign_ = Ops::Base::CeilAlign(d_ / dSplitCoef_, blockSize_ / typeSize) * dSplitCoef_;

    blockFactorBS_ = Ops::Base::CeilDiv(bs, int64_t(aicoreParams_.numBlocks));
    blockNumBS_ = Ops::Base::CeilDiv(bs, blockFactorBS_);
    blockTailBS_ = bs - (blockNumBS_ - 1) * blockFactorBS_;

    if (bs <= int64_t(aicoreParams_.numBlocks) / CONST_TWO) {
        blockNumN_ = aicoreParams_.numBlocks / blockNumBS_;
        blockFactorN_ = Ops::Base::CeilDiv(bn_, blockNumN_);
        blockNumN_ = Ops::Base::CeilDiv(bn_, blockFactorN_);
        blockTailN_ = bn_ - (blockNumN_ - 1) * blockFactorN_;
    } else {
        blockNumN_ = 1;
        blockFactorN_ = bn_;
        blockTailN_ = bn_;
    }
    usedCoreNum_ = blockNumBS_ * blockNumN_;
    int64_t baseBufferSize = dAlign_ * typeSize;
    int64_t baseBlockInUb =
        Ops::Base::FloorAlign(static_cast<int64_t>(aicoreParams_.ubSize / CONST_TWO / DB_FLAG), blockSize_) /
        baseBufferSize;
    OP_CHECK_IF(baseBlockInUb < 1, OP_LOGE(context_->GetNodeName(), "ubSize can't load 8 d size, d = %ld.", d_),
                return ge::GRAPH_FAILED);

    ubFactorN_ = std::min(blockFactorN_, baseBlockInUb - 1);
    ubFactorN_ = std::min(ubFactorN_, MAX_COPY_BLOCK_COUNT / dSplitCoef_);

    ubFactorBS_ = std::min(Ops::Base::FloorDiv(baseBlockInUb, ubFactorN_ + 1), blockFactorBS_);
    ubFactorBS_ = (ubFactorBS_ == 0) ? 1 : ubFactorBS_;
    OP_CHECK_IF(InitTilingData() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "InitTilingData failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(TilingReduce() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "TilingReduce failed."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradRegBaseTilingClassAB::PostTiling()
{
    tilingData_->inplacePartialRotaryMulGradABParams.b = b_;
    tilingData_->inplacePartialRotaryMulGradABParams.s = s_;
    tilingData_->inplacePartialRotaryMulGradABParams.d = d_;
    tilingData_->inplacePartialRotaryMulGradABParams.n = bn_;
    tilingData_->inplacePartialRotaryMulGradABParams.dAlign = dAlign_;
    tilingData_->inplacePartialRotaryMulGradABParams.dSplitCoef = dSplitCoef_;
    tilingData_->inplacePartialRotaryMulGradABParams.blockNumBS = blockNumBS_;
    tilingData_->inplacePartialRotaryMulGradABParams.blockFactorBS = blockFactorBS_;
    tilingData_->inplacePartialRotaryMulGradABParams.blockTailBS = blockTailBS_;
    tilingData_->inplacePartialRotaryMulGradABParams.blockNumN = blockNumN_;
    tilingData_->inplacePartialRotaryMulGradABParams.blockFactorN = blockFactorN_;
    tilingData_->inplacePartialRotaryMulGradABParams.blockTailN = blockTailN_;
    tilingData_->inplacePartialRotaryMulGradABParams.ubFactorBS = ubFactorBS_;
    tilingData_->inplacePartialRotaryMulGradABParams.ubFactorN = ubFactorN_;
    tilingData_->inplacePartialRotaryMulGradABParams.usedCoreNum = usedCoreNum_;
    tilingData_->inplacePartialRotaryMulGradABParams.rotaryMode = static_cast<int64_t>(rotaryMode_);
    tilingData_->inplacePartialRotaryMulGradABParams.sliceStart = sliceStart_;
    tilingData_->inplacePartialRotaryMulGradABParams.sliceEnd = sliceEnd_;
    tilingData_->inplacePartialRotaryMulGradABParams.sliceLength = sliceLength_;
    tilingData_->inplacePartialRotaryMulGradABParams.dCosFlag = dCosFlag_;
    SetRotaryXTilingData();
    SetTilingKeyBlockDim(static_cast<uint32_t>(ropeGradTilingKey::TILING_KEY_AB));

    OP_LOGI(
        context_->GetNodeName(),
        "InplacePartialRotaryMulGradRegBaseTilingClassAB tilingData is B: %ld, S: %ld, N: %ld, D: %ld, kAlign: %ld, "
        "dSplitCoef: %ld, BlockNumBS: %ld, BlockFactorBS: %ld, BlockTailBS: %ld, BlockNumN: %ld, "
        "BlockFactorN: %ld, blockTailN: %ld, ubFactorBS: %ld, ubFactorN: %ld, "
        "usedCoreNum_: %ld, RotaryMode: %ld, TilingKey: %ld.",
        tilingData_->inplacePartialRotaryMulGradABParams.b, tilingData_->inplacePartialRotaryMulGradABParams.s,
        tilingData_->inplacePartialRotaryMulGradABParams.n, tilingData_->inplacePartialRotaryMulGradABParams.d,
        tilingData_->inplacePartialRotaryMulGradABParams.dAlign,
        tilingData_->inplacePartialRotaryMulGradABParams.dSplitCoef,
        tilingData_->inplacePartialRotaryMulGradABParams.blockNumBS,
        tilingData_->inplacePartialRotaryMulGradABParams.blockFactorBS,
        tilingData_->inplacePartialRotaryMulGradABParams.blockTailBS,
        tilingData_->inplacePartialRotaryMulGradABParams.blockNumN,
        tilingData_->inplacePartialRotaryMulGradABParams.blockFactorN,
        tilingData_->inplacePartialRotaryMulGradABParams.blockTailN,
        tilingData_->inplacePartialRotaryMulGradABParams.ubFactorBS,
        tilingData_->inplacePartialRotaryMulGradABParams.ubFactorN,
        tilingData_->inplacePartialRotaryMulGradABParams.usedCoreNum,
        tilingData_->inplacePartialRotaryMulGradABParams.rotaryMode, tilingKey_);
    return ge::GRAPH_SUCCESS;
}

bool InplacePartialRotaryMulGradRegBaseTilingClassAB::IsCapable()
{
    if (!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) {
        return false;
    }

    OP_LOGI(context_->GetNodeName(), "layout: %ld", static_cast<int64_t>(layout_));
    // 1. qk:bsnd, cos:bs1d 2. qk:sbnd, cos:sb1d 3. qk:sbnd, cos:s11d 4. qk:bsnd, cos:b11d
    return layout_ == InplacePartialRotaryMulGradLayout::SBND;
}

REGISTER_OPS_TILING_TEMPLATE(InplacePartialRotaryMulGrad, InplacePartialRotaryMulGradRegBaseTilingClassAB, 25000);
} // namespace optiling