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
 * \file apply_rotary_pos_emb_grad_tiling_a_and_b.cpp
 * \brief A template tiling (NO_BROADCAST → Layout=3, DxTilingKey=205)
 *
 * grad_query_embed/grad_key_embed/cos/sin shape 完全一致, 无广播轴
 * MergeDim: B×N×S 合并为 b_, cosb_=b_ (全同, cos 与 dy 一一对应)
 * 参照 rope_grad_regbase_tiling_a_and_b.cpp 的 A 分支
 */
#include "apply_rotary_pos_emb_grad_tiling.h"

using namespace AscendC;

namespace optiling {
constexpr uint64_t PRIORITY = 40000;
constexpr int64_t DBUF = 2, UB_F = 4, HCOEF = 2;

class ApplyRopeGradRegBaseTilingClassA : public ApplyRopeGradRegbaseTilingClass
{
public:
    explicit ApplyRopeGradRegBaseTilingClassA(gert::TilingContext* ctx)
        : ApplyRopeGradRegbaseTilingClass(ctx) {}

protected:
    bool IsCapable() override {
        return Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_) &&
               (layout_ == ApplyRopeLayout::NO_BROADCAST);
    }
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override { return ge::GRAPH_SUCCESS; }
    ge::graphStatus PostTiling() override;
    void SetTilingData();

private:
    ge::graphStatus MergeDim();
    ge::graphStatus SplitCore();
    ge::graphStatus ComputeUbFactor();

    int64_t blockNumB_{0}, blockFactorB_{0};
    int64_t blockNumS_{0}, blockFactorS_{0};
    int64_t ubFactorB_{0}, ubFactorS_{0}, ubFactorN_{0};
    int64_t ubLoopNumN_{0}, ubTailFactorN_{0};
};

ge::graphStatus ApplyRopeGradRegBaseTilingClassA::MergeDim()
{
    b_ = b_ * nQ_ * s_;
    nQ_ = 1;
    nK_ = 1;
    s_ = 1;
    // NO_BROADCAST: cos == dy → cosb_ = merged b_
    cosb_ = b_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRopeGradRegBaseTilingClassA::SplitCore()
{
    blockFactorB_ = Ops::Base::CeilDiv(static_cast<uint64_t>(b_), aicoreParams_.numBlocks);
    blockNumB_ = Ops::Base::CeilDiv(static_cast<int64_t>(b_), blockFactorB_);
    usedCoreNum_ = blockNumB_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRopeGradRegBaseTilingClassA::ComputeUbFactor()
{
    ubFactorB_ = 1;
    int64_t typeSize = ge::GetSizeByDataType(dtype_);
    int64_t dSize = Ops::Base::CeilAlign(d_ * typeSize / dSplitCoef_, blockSize_) * dSplitCoef_;
    int64_t numDAvail = Ops::Base::FloorDiv(
        static_cast<int64_t>(aicoreParams_.ubSize), DBUF * dSize);
    OP_CHECK_IF(numDAvail < UB_F, OP_LOGE(context_, "D too big for UB"), return ge::GRAPH_FAILED);
    // NO_BROADCAST: cosb_ == b_ → 需要双缓冲 cos/sin + grad/x
    numDAvail /= UB_F;
    ubFactorB_ = std::min(blockFactorB_, numDAvail);
    ubFactorB_ = std::min(ubFactorB_, MAX_COPY_BLOCK_COUNT / dSplitCoef_);
    ubFactorS_ = 1;
    ubFactorN_ = 1;
    ubLoopNumN_ = 1;
    ubTailFactorN_ = 1;
    blockNumS_ = 1;
    blockFactorS_ = 1;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRopeGradRegBaseTilingClassA::DoOpTiling()
{
    CHK(MergeDim());
    CHK(SplitCore());
    CHK(ComputeUbFactor());
    CHK(InitTilingData());
    return ge::GRAPH_SUCCESS;
}

void ApplyRopeGradRegBaseTilingClassA::SetTilingData()
{
    tilingData_->ropeGradParams.layout = static_cast<uint64_t>(layout_);
    tilingData_->ropeGradParams.b = b_;
    tilingData_->ropeGradParams.s = s_;
    tilingData_->ropeGradParams.d = d_;
    tilingData_->ropeGradParams.nQ = nQ_;
    tilingData_->ropeGradParams.nK = nK_;
    tilingData_->ropeGradParams.blockNumB = blockNumB_;
    tilingData_->ropeGradParams.blockFactorB = blockFactorB_;
    tilingData_->ropeGradParams.blockNumS = blockNumS_;
    tilingData_->ropeGradParams.blockFactorS = blockFactorS_;
    tilingData_->ropeGradParams.ubFactorS = ubFactorS_;
    tilingData_->ropeGradParams.ubFactorB = ubFactorB_;
    tilingData_->ropeGradParams.ubLoopNumN = ubLoopNumN_;
    tilingData_->ropeGradParams.ubFactorN = ubFactorN_;
    tilingData_->ropeGradParams.ubTailFactorN = ubTailFactorN_;
    tilingData_->ropeGradParams.usedCoreNum = usedCoreNum_;
    tilingData_->ropeGradParams.coreNum = aicoreParams_.numBlocks;
    tilingData_->ropeGradParams.rotaryMode = rotaryMode_;
    tilingData_->ropeGradParams.cosb = cosb_;
}

ge::graphStatus ApplyRopeGradRegBaseTilingClassA::PostTiling()
{
    SetTilingData();
    tilingData_->rotaryXParams.b = b_;
    tilingData_->rotaryXParams.s = 1;
    tilingData_->rotaryXParams.d = d_;
    tilingData_->rotaryXParams.nQ = 1;
    tilingData_->rotaryXParams.nK = 1;
    tilingData_->rotaryXParams.usedCoreNum = usedCoreNum_;
    tilingData_->rotaryXParams.blockFactorN = blockFactorB_;
    tilingData_->rotaryXParams.blockTailFactorN =
        (b_ % blockFactorB_ == 0) ? blockFactorB_ : (b_ % blockFactorB_);
    tilingData_->rotaryXParams.ubFactorN = ubFactorB_;
    tilingData_->rotaryXParams.rotaryMode = rotaryMode_;
    SetTilingKeyBlockDim(static_cast<uint32_t>(ApplyDxTilingKey::KEY_A),
                         static_cast<uint32_t>(layout_));
    tilingData_->dCosFlag = dCosFlag_;
    tilingData_->layout = static_cast<uint32_t>(layout_);
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_CLASS(ApplyRopeGradRegBaseTilingClassA, PRIORITY);
} // namespace optiling
