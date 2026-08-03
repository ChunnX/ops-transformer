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
 * \file apply_rotary_pos_emb_grad_tiling_bab.cpp
 * \brief BAB tiling (BSND layout, cos 1S1D)
 * 双路反向 Q+K
 * 参照 rope_grad_regbase_tiling_bab.cpp
 */
#include "apply_rotary_pos_emb_grad_tiling.h"

namespace optiling {
constexpr uint64_t APPLY_ROPE_GRAD_BAB_TILING_PRIORITY = 20000;
constexpr uint32_t MIN_UB_LOAD_D_NUM = 4;
constexpr uint32_t DOUBLE_BUFFER = 2;

class ApplyRopeGradRegBaseTilingClassBAB : public ApplyRopeGradRegbaseTilingClass
{
public:
    explicit ApplyRopeGradRegBaseTilingClassBAB(gert::TilingContext* context_)
        : ApplyRopeGradRegbaseTilingClass(context_) {}
    ~ApplyRopeGradRegBaseTilingClassBAB() override = default;
    void Reset(gert::TilingContext* context_) override
    {
        ApplyRopeGradRegbaseTilingClass::Reset(context_);
    }

protected:
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    bool IsCapable() override
    {
        // BSND layout (含 TND 映射后的 1S1D): cosb_==1, dy B==1
        if (Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_) &&
            ((layout_ == ApplyRopeLayout::BSND) || (isTndLayout_)) &&
            (cosb_ == 1)) {
            return true;
        }
        return false;
    }

private:
    int64_t coreNum_ = 0;
    int64_t blockNumB_ = 0, blockFactorB_ = 0;
    int64_t blockNumS_ = 0, blockFactorS_ = 0;
    int64_t ubFactorS_ = 1, ubFactorB_ = 1;
    int64_t ubLoopNumN_ = 0, ubFactorN_ = 1, ubTailFactorN_ = 0;
    int64_t ubSize_ = 0;

    void SplitCore();
    ge::graphStatus SplitUb();
    void PrintTilingData();
};

ge::graphStatus ApplyRopeGradRegBaseTilingClassBAB::DoOpTiling()
{
    ubSize_ = aicoreParams_.ubSize;
    coreNum_ = aicoreParams_.numBlocks;
    ge::graphStatus status = SplitUb();
    if (status != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "SplitUb Failed.");
        return ge::GRAPH_FAILED;
    }
    SplitCore();
    if (blockNumB_ * blockNumS_ > coreNum_) {
        OP_LOGE(context_->GetNodeName(), "split coreNum [%ld] large than coreNum[%ld]",
            blockNumB_ * blockNumS_, coreNum_);
        return ge::GRAPH_FAILED;
    }
    usedCoreNum_ = blockNumB_ * blockNumS_;
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

void ApplyRopeGradRegBaseTilingClassBAB::SplitCore()
{
    blockFactorB_ = Ops::Base::CeilDiv(b_, coreNum_);
    blockNumB_ = Ops::Base::CeilDiv(b_, blockFactorB_);
    blockFactorS_ = Ops::Base::CeilDiv(s_, coreNum_ / blockNumB_);
    blockNumS_ = Ops::Base::CeilDiv(s_, blockFactorS_);
}

ge::graphStatus ApplyRopeGradRegBaseTilingClassBAB::SplitUb()
{
    int64_t typeSize = ge::GetSizeByDataType(dtype_);
    int64_t blockLen = Ops::Base::CeilAlign(d_ / dSplitCoef_, blockSize_ / typeSize) * dSplitCoef_;
    int64_t dSize = blockLen * typeSize;
    int64_t maxN = (nQ_ > nK_) ? nQ_ : nK_;
    // UB 中可容纳的 D 数量
    int64_t dNumInUb = Ops::Base::FloorAlign(
        static_cast<int64_t>(ubSize_ / (MIN_UB_LOAD_D_NUM * DOUBLE_BUFFER)), blockSize_) / dSize;
    OP_CHECK_IF(dNumInUb < 1,
        OP_LOGE(context_->GetNodeName(), "ubSize can't load one d size, d = %ld.", d_),
        return ge::GRAPH_FAILED);

    ubFactorN_ = dNumInUb;
    ubFactorS_ = Ops::Base::FloorAlign(
        static_cast<int64_t>(ubSize_ / ((MIN_UB_LOAD_D_NUM + 1) * DOUBLE_BUFFER)), blockSize_) / dSize;
    ubFactorS_ = std::max(ubFactorS_, static_cast<int64_t>(1));

    if (ubFactorS_ >= blockFactorS_) {
        ubFactorS_ = blockFactorS_;
        int64_t remainDNumInUb = Ops::Base::FloorAlign(
            static_cast<int64_t>(ubSize_ / ((2 * ubFactorS_ + maxN * ubFactorS_) * DOUBLE_BUFFER)), blockSize_) / dSize;
        ubFactorN_ = std::max(remainDNumInUb, static_cast<int64_t>(1));
    }

    ubLoopNumN_ = Ops::Base::CeilDiv(maxN, ubFactorN_);
    ubTailFactorN_ = (maxN % ubFactorN_ == 0) ? ubFactorN_ : (maxN % ubFactorN_);
    OP_CHECK_IF(
        (ubFactorS_ == 0) || (ubFactorN_ == 0),
        OP_LOGE(context_->GetNodeName(), "ub factor S %ld or ub factor N %ld == 0", ubFactorS_, ubFactorN_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRopeGradRegBaseTilingClassBAB::PostTiling()
{
    OP_CHECK_IF(InitTilingData() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "InitTilingData Failed."), return ge::GRAPH_FAILED);

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
    tilingData_->ropeGradParams.coreNum = coreNum_;
    tilingData_->ropeGradParams.rotaryMode = rotaryMode_;

    uint32_t layoutVal = static_cast<uint32_t>(layout_);
    SetTilingKeyBlockDim(static_cast<uint32_t>(ApplyDxTilingKey::KEY_BAB), layoutVal);

    if (dCosFlag_ == 1) {
        SetRotaryXTilingData();
    }
    tilingData_->dCosFlag = dCosFlag_;
    tilingData_->layout = static_cast<uint32_t>(layout_);
    return ge::GRAPH_SUCCESS;
}

void ApplyRopeGradRegBaseTilingClassBAB::PrintTilingData()
{
    OP_LOGI(context_->GetNodeName(),
        "%s b=%ld s=%ld d=%ld nQ=%ld nK=%ld blockNumB=%ld blockFactorB=%ld blockNumS=%ld blockFactorS=%ld "
        "ubLoopNumN=%ld ubFactorN=%ld ubTailFactorN=%ld ubFactorS=%ld ubFactorB=%ld usedCoreNum=%ld coreNum=%ld",
        context_->GetNodeName(), b_, s_, d_, nQ_, nK_, blockNumB_, blockFactorB_, blockNumS_, blockFactorS_,
        ubLoopNumN_, ubFactorN_, ubTailFactorN_, ubFactorS_, ubFactorB_, usedCoreNum_, coreNum_);
}

REGISTER_TILING_CLASS(ApplyRopeGradRegBaseTilingClassBAB, APPLY_ROPE_GRAD_BAB_TILING_PRIORITY);

} // namespace optiling
