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
 * \file apply_rotary_pos_emb_grad_tiling_ab.cpp
 * \brief AB tiling (SBND layout)
 * 双路反向 Q+K
 * 参照 rope_grad_regbase_tiling_ab.cpp
 */
#include "op_host/tiling_templates_registry.h"
#include "apply_rotary_pos_emb_grad_tiling.h"

namespace optiling {
constexpr int64_t CONST_TWO = 2;
constexpr int64_t CONST_FOUR = 4;
constexpr int64_t DB_FLAG = 2;
constexpr uint64_t APPLY_ROPE_GRAD_AB_TILING_PRIORITY = 25000;

class ApplyRopeGradRegBaseTilingClassAB : public ApplyRopeGradRegbaseTilingClass
{
public:
    explicit ApplyRopeGradRegBaseTilingClassAB(gert::TilingContext* context)
        : ApplyRopeGradRegbaseTilingClass(context) {}

protected:
    bool IsCapable() override
    {
        // SBND layout
        return Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_) && (layout_ == ApplyRopeLayout::SBND);
    }
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

private:
    int64_t blockNumBS_ = 0, blockFactorBS_ = 0, blockTailBS_ = 0;
    int64_t blockNumN_ = 0, blockFactorN_ = 0, blockTailN_ = 0;
    int64_t ubFactorBS_ = 0, ubFactorN_ = 0;
    int64_t dAlign_ = 0;
    int64_t bs_ = 0, bn_ = 0;
};

ge::graphStatus ApplyRopeGradRegBaseTilingClassAB::DoOpTiling()
{
    bs_ = b_ * s_;
    bn_ = (nQ_ > nK_) ? nQ_ : nK_;  // 用 max(nQ, nK)
    if (cosb_ == 1) {
        bs_ = s_;
        bn_ = b_ * ((nQ_ > nK_) ? nQ_ : nK_);
    }
    int64_t typeSize = ge::GetSizeByDataType(dtype_);
    dAlign_ = Ops::Base::CeilAlign(d_ / dSplitCoef_, blockSize_ / typeSize) * dSplitCoef_;

    blockFactorBS_ = Ops::Base::CeilDiv(bs_, int64_t(aicoreParams_.numBlocks));
    blockNumBS_ = Ops::Base::CeilDiv(bs_, blockFactorBS_);
    blockTailBS_ = bs_ - (blockNumBS_ - 1) * blockFactorBS_;

    if (bs_ <= int64_t(aicoreParams_.numBlocks) / CONST_TWO) {
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
    int64_t baseBlockInUb = Ops::Base::FloorAlign(
        static_cast<int64_t>(aicoreParams_.ubSize / CONST_TWO / DB_FLAG), blockSize_) / baseBufferSize;
    OP_CHECK_IF(baseBlockInUb < 1,
        OP_LOGE(context_->GetNodeName(), "ubSize can't load 8 d size, d = %ld.", d_),
        return ge::GRAPH_FAILED);

    ubFactorN_ = baseBlockInUb;
    ubFactorBS_ = Ops::Base::FloorAlign(
        static_cast<int64_t>(aicoreParams_.ubSize / CONST_FOUR / DB_FLAG), blockSize_) / baseBufferSize;
    ubFactorBS_ = std::max(ubFactorBS_, static_cast<int64_t>(1));

    if (ubFactorBS_ >= blockFactorBS_) {
        ubFactorBS_ = blockFactorBS_;
        int64_t remainBlockInUb = Ops::Base::FloorAlign(
            static_cast<int64_t>(aicoreParams_.ubSize / ((CONST_TWO * ubFactorBS_) * DB_FLAG)), blockSize_) / baseBufferSize;
        ubFactorN_ = std::max(remainBlockInUb, static_cast<int64_t>(1));
    }

    if (bn_ <= int64_t(aicoreParams_.numBlocks) * CONST_TWO && blockNumBS_ == 1) {
        ubFactorN_ = blockFactorN_;
        ubFactorBS_ = std::min(blockFactorBS_, ubFactorBS_);
    }

    OP_CHECK_IF(
        (ubFactorBS_ == 0) || (ubFactorN_ == 0),
        OP_LOGE(context_->GetNodeName(), "ub factor BS %ld or ub factor N %ld == 0", ubFactorBS_, ubFactorN_),
        return ge::GRAPH_FAILED);

    OP_LOGI(context_->GetNodeName(),
        "b=%ld s=%ld d=%ld nQ=%ld nK=%ld bs=%ld bn=%ld blockNumBS=%ld blockFactorBS=%ld blockTailBS=%ld "
        "blockNumN=%ld blockFactorN=%ld blockTailN=%ld ubFactorBS=%ld ubFactorN=%ld usedCoreNum=%ld",
        b_, s_, d_, nQ_, nK_, bs_, bn_,
        blockNumBS_, blockFactorBS_, blockTailBS_,
        blockNumN_, blockFactorN_, blockTailN_,
        ubFactorBS_, ubFactorN_, usedCoreNum_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRopeGradRegBaseTilingClassAB::PostTiling()
{
    OP_CHECK_IF(InitTilingData() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "InitTilingData Failed."), return ge::GRAPH_FAILED);

    tilingData_->ropeGradABParams.b = b_;
    tilingData_->ropeGradABParams.s = s_;
    tilingData_->ropeGradABParams.d = d_;
    tilingData_->ropeGradABParams.nQ = nQ_;
    tilingData_->ropeGradABParams.nK = nK_;
    tilingData_->ropeGradABParams.dAlign = dAlign_;
    tilingData_->ropeGradABParams.dSplitCoef = dSplitCoef_;
    tilingData_->ropeGradABParams.blockNumBS = blockNumBS_;
    tilingData_->ropeGradABParams.blockFactorBS = blockFactorBS_;
    tilingData_->ropeGradABParams.blockTailBS = blockTailBS_;
    tilingData_->ropeGradABParams.blockNumN = blockNumN_;
    tilingData_->ropeGradABParams.blockFactorN = blockFactorN_;
    tilingData_->ropeGradABParams.blockTailN = blockTailN_;
    tilingData_->ropeGradABParams.ubFactorBS = ubFactorBS_;
    tilingData_->ropeGradABParams.ubFactorN = ubFactorN_;
    tilingData_->ropeGradABParams.usedCoreNum = usedCoreNum_;
    tilingData_->ropeGradABParams.rotaryMode = rotaryMode_;

    uint32_t layoutVal = static_cast<uint32_t>(layout_);
    SetTilingKeyBlockDim(static_cast<uint32_t>(ApplyDxTilingKey::KEY_AB), layoutVal);

    if (dCosFlag_ == 1) {
        SetRotaryXTilingData();
    }
    tilingData_->dCosFlag = dCosFlag_;
    tilingData_->layout = static_cast<uint32_t>(layout_);
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_CLASS(ApplyRopeGradRegBaseTilingClassAB, APPLY_ROPE_GRAD_AB_TILING_PRIORITY);

} // namespace optiling
