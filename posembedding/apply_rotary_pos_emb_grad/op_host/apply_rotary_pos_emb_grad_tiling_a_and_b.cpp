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
 * \brief A/B tiling（NO_BROADCAST + BROADCAST_BSN → Layout=3）
 * A (205): dy shape == cos shape，cos 与 dy 完全同形，需要逐行加载 cos/sin
 * B (206): cos shape == 111D（全广播），cos/sin 只加载一次全程复用
 *
 * 4D 场景：BNSD layout 的 dy + 全同 cos → A；BNSD + 111D cos → B
 * TND 场景：TND + 1ND cos → A/B（B=1 时 A 退化为 B），TND + TND cos → 可能也是 A
 *
 * MergeDim 将所有维度（B×N×S）合并为一维 b_，然后逐 b_ 分核处理
 * 双路 Q+K：合并时 nQ_ 和 nK_ 中取 max 参与 b_ 计算
 * 参照 rope_grad_regbase_tiling_a_and_b.cpp
 */
#include "apply_rotary_pos_emb_grad_tiling.h"

using namespace AscendC;

namespace optiling {
constexpr uint64_t APPLY_ROPE_GRAD_A_AND_B_TILING_PRIORITY = 40000;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t UB_FACTOR = 4;
constexpr int64_t HALF_INTERLEAVE_COEF = 2;

class ApplyRopeGradRegBaseTilingClassAAndB : public ApplyRopeGradRegbaseTilingClass
{
public:
    explicit ApplyRopeGradRegBaseTilingClassAAndB(gert::TilingContext* context)
        : ApplyRopeGradRegbaseTilingClass(context) {}

protected:
    bool IsCapable() override
    {
        // NO_BROADCAST (全同) 和 BROADCAST_BSN (111D 全广播) 都用 Layout=3
        // 通过 cosb_ 区分: cosb_==1→B(206), cosb_!=1→A(205)
        return (Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) &&
               (layout_ == ApplyRopeLayout::NO_BROADCAST_AND_BROADCAST_BSN);
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
    int64_t ubFactorB_{0}, ubFactorS_{0};
    int64_t ubFactorN_{0}, ubLoopNumN_{0}, ubTailFactorN_{0};
};

ge::graphStatus ApplyRopeGradRegBaseTilingClassAAndB::MergeDim()
{
    // 将所有维度（B×N×S）合并为一维 b_，然后逐块分核处理
    // 双路：取 max(nQ, nK) 参与合并（Q 和 K 交替放入同一个 b_ 循环中处理）
    int64_t maxN = (nQ_ > nK_) ? nQ_ : nK_;
    b_ = b_ * maxN * s_;
    nQ_ = 1;
    nK_ = 1;
    s_ = 1;
    // cosb_ 由 JudgeLayoutByShape 设置：
    //   NO_BROADCAST → cosb_ = b_（cos 和 dy 同形，需要逐行加载）
    //   BROADCAST_BSN → cosb_ = 1（cos 是 111D，只加载一次）
    // MergeDim 后 B/N/S 都合并了，需要同步更新 cosb_：全同则 cosb_ = b_
    if (cosb_ != b_ / (maxN * s_)) {
        // 检查: 原始 cosb_ 是否等于原始 b_
        // 如果是，说明是 NO_BROADCAST，现在 b_ 变成了 b_*N*S，cosb_ 也该同步扩大
        if (cosb_ > 1 && cosb_ == b_ / (maxN * s_)) {
            cosb_ = b_;
        } else {
            // BROADCAST_BSN: cosb_ = 1，不变
            cosb_ = 1;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRopeGradRegBaseTilingClassAAndB::SplitCore()
{
    blockFactorB_ = Ops::Base::CeilDiv(static_cast<uint64_t>(b_), aicoreParams_.numBlocks);
    blockNumB_ = Ops::Base::CeilDiv(static_cast<int64_t>(b_), blockFactorB_);
    usedCoreNum_ = blockNumB_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRopeGradRegBaseTilingClassAAndB::ComputeUbFactor()
{
    ubFactorB_ = 1;
    // UB 大小：
    //   NO_BROADCAST (cosb_ != 1): 2 × (2b + 1) × dAlign — cos+sin 也要双缓冲
    //   BROADCAST_BSN (cosb_ == 1): 2 × 4 × b × dAlign — cos/sin 只占 1 份
    int64_t typeSize = ge::GetSizeByDataType(dtype_);
    int64_t dSize = Ops::Base::CeilAlign(d_ * typeSize / dSplitCoef_, blockSize_) * dSplitCoef_;
    int64_t numOfDAvailable = Ops::Base::FloorDiv(
        static_cast<int64_t>(aicoreParams_.ubSize), DOUBLE_BUFFER * dSize);
    OP_CHECK_IF(
        numOfDAvailable < UB_FACTOR,
        OP_LOGE(context_, "D is too big to load in ub, ubSize is %ld bytes, loading requires %ld bytes.",
            static_cast<int64_t>(aicoreParams_.ubSize), UB_FACTOR * dSize * (ubFactorB_ + 1)),
        return ge::GRAPH_FAILED);
    if (cosb_ != 1) {
        // NO_BROADCAST: 所有输入都需要双缓冲
        numOfDAvailable /= UB_FACTOR;
        ubFactorB_ = std::min(blockFactorB_, numOfDAvailable);
    } else {
        // BROADCAST_BSN: cos/sin 只占一份
        numOfDAvailable -= 1;               // cos/sin 占 1 个 D
        numOfDAvailable /= DOUBLE_BUFFER;    // grad/x 双缓冲
        ubFactorB_ = std::min(blockFactorB_, numOfDAvailable);
    }

    ubFactorB_ = std::min(ubFactorB_, MAX_COPY_BLOCK_COUNT / dSplitCoef_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRopeGradRegBaseTilingClassAAndB::DoOpTiling()
{
    OP_CHECK_IF(
        MergeDim() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "failed to merge dim."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        SplitCore() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "failed to split core."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ComputeUbFactor() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "failed to compute ub factor."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        InitTilingData() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "InitTilingData failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        TilingReduce() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "TilingReduce failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void ApplyRopeGradRegBaseTilingClassAAndB::SetTilingData()
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

    OP_LOGI(
        context_->GetNodeName(),
        "AAndB tilingData: B=%ld S=%ld D=%ld nQ=%ld nK=%ld blockNumB=%ld blockFactorB=%ld "
        "ubFactorB=%ld ubFactorS=%ld ubLoopNumN=%ld ubFactorN=%ld ubTailFactorN=%ld "
        "cosb=%ld usedCoreNum=%ld rotaryMode=%ld layout=%lu",
        tilingData_->ropeGradParams.b, tilingData_->ropeGradParams.s, tilingData_->ropeGradParams.d,
        tilingData_->ropeGradParams.nQ, tilingData_->ropeGradParams.nK,
        tilingData_->ropeGradParams.blockNumB, tilingData_->ropeGradParams.blockFactorB,
        tilingData_->ropeGradParams.ubFactorB, tilingData_->ropeGradParams.ubFactorS,
        tilingData_->ropeGradParams.ubLoopNumN, tilingData_->ropeGradParams.ubFactorN,
        tilingData_->ropeGradParams.ubTailFactorN, cosb_,
        tilingData_->ropeGradParams.usedCoreNum, rotaryMode_,
        static_cast<uint64_t>(layout_));
}

ge::graphStatus ApplyRopeGradRegBaseTilingClassAAndB::PostTiling()
{
    SetTilingData();
    if (dCosFlag_ == 1) {
        SetRotaryXTilingData();
    }
    // cosb_ == 1 → 全广播 (BROADCAST_BSN) → KEY_B (206)
    // cosb_ != 1 → 全同 (NO_BROADCAST) → KEY_A (205)
    uint32_t dxKey = (cosb_ == 1) ? static_cast<uint32_t>(ApplyDxTilingKey::KEY_B)
                                  : static_cast<uint32_t>(ApplyDxTilingKey::KEY_A);
    uint32_t layoutVal = static_cast<uint32_t>(layout_);
    SetTilingKeyBlockDim(dxKey, layoutVal);
    tilingData_->dCosFlag = dCosFlag_;
    tilingData_->layout = static_cast<uint32_t>(layout_);
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_CLASS(ApplyRopeGradRegBaseTilingClassAAndB, APPLY_ROPE_GRAD_A_AND_B_TILING_PRIORITY);

} // namespace optiling
