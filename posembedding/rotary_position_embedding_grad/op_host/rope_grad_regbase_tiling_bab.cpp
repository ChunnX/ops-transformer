/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file rope_grad_regbase_tiling_bab.cc
 * \brief
 */

#include "rotary_position_embedding_grad_tiling.h"
#include <functional>
#include <iostream>

namespace optiling {
constexpr uint64_t ROPE_GRAD_BAB_TILING_PRIORITY = 20000;
constexpr uint32_t MIN_UB_LOAD_D_NUM = 4; // x, y或in, cos输入开doubleBuffer
constexpr uint32_t DOUBLE_BUFFER = 2;

class RopeGradRegBaseTilingClassBAB : public RopeGradRegBaseTilingClass {
public:
    explicit RopeGradRegBaseTilingClassBAB(gert::TilingContext *context_) : RopeGradRegBaseTilingClass(context_)
    {
    }
    ~RopeGradRegBaseTilingClassBAB() override = default;
    void Reset(gert::TilingContext *context_) override
    {
        RopeGradRegBaseTilingClass::Reset(context_);
    }

protected:
    // 计算数据切分
    ge::graphStatus DoOpTiling() override;
    // 设置Tiling数据
    ge::graphStatus PostTiling() override;

    bool IsCapable() override
    {
        // BSND format, 1s1d模版，后续可扩展支持所有bab类型的boardcast
        if (Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_) && (layout_ == RopeLayout::BSND) &&
            (cosb_ == 1)) {
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
    int64_t ubLoopNumN_ = 0;    // 核内处理N循环了多少次
    int64_t ubFactorN_ = 1;     // 每次循环处理多少个N
    int64_t ubTailFactorN_ = 0; // 最后一次循环处理多少N
    int64_t ubSize_ = 0;

    void SplitCore();
    ge::graphStatus SplitUb();
    void PrintTilingData();
};

ge::graphStatus RopeGradRegBaseTilingClassBAB::DoOpTiling()
{
    ubSize_ = aicoreParams_.ubSize;
    std::cout << "ubSize_ = " << ubSize_ << std::endl;
    coreNum_ = aicoreParams_.numBlocks;
    std::cout << "coreNum_ = " << coreNum_ << std::endl;
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
    std::cout << "InitTilingData" << std::endl;
    OP_CHECK_IF(InitTilingData() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "InitTilingData failed."),
                return ge::GRAPH_FAILED);
    std::cout << "TilingReduce" << std::endl;
    OP_CHECK_IF(TilingReduce() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "TilingReduce failed."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void RopeGradRegBaseTilingClassBAB::SplitCore()
{
    // 尝试先对B分核，再尝试优先对S分核，比较二者切分之后的总核数
    auto blockFactorB1 = Ops::Base::CeilDiv(b_, coreNum_);
    std::cout << "blockFactorB1 = " << blockFactorB1 << std::endl;
    auto blockNumB1 = Ops::Base::CeilDiv(b_, blockFactorB1);
    std::cout << "blockNumB1 = " << blockNumB1 << std::endl;
    auto blockNumS1 = std::min(coreNum_ / blockNumB1, s_);
    std::cout << "blockNumS1 = " << blockNumS1 << std::endl;
    auto blockFactorS1 = Ops::Base::CeilDiv(s_, blockNumS1);
    std::cout << "blockFactorS1 = " << blockFactorS1 << std::endl;
    blockNumS1 = Ops::Base::CeilDiv(s_, blockFactorS1);
    std::cout << "blockNumS1 = " << blockNumS1 << std::endl;
    auto usedCoreNum1 = blockNumB1 * blockNumS1;
    std::cout << "usedCoreNum1 = " << usedCoreNum1 << std::endl;

    auto blockFactorS2 = Ops::Base::CeilDiv(s_, coreNum_);
    std::cout << "blockFactorS2 = " << blockFactorS2 << std::endl;
    auto blockNumS2 = Ops::Base::CeilDiv(s_, blockFactorS2);
    std::cout << "blockNumS2 = " << blockNumS2 << std::endl;
    auto blockNumB2 = std::min(coreNum_ / blockNumS2, b_);
    std::cout << "blockNumB2 = " << blockNumB2 << std::endl;
    auto blockFactorB2 = Ops::Base::CeilDiv(b_, blockNumB2);
    std::cout << "blockFactorB2 = " << blockFactorB2 << std::endl;
    blockNumB2 = Ops::Base::CeilDiv(b_, blockFactorB2);
    std::cout << "blockNumB2 = " << blockNumB2 << std::endl;
    auto usedCoreNum2 = blockNumB2 * blockNumS2;
    std::cout << "usedCoreNum2 = " << usedCoreNum2 << std::endl;

    // ubFactorS 很小的时候，选择核数多的，ubFactorS大于blockFactorS时， 综合考虑分核和UB切分
    auto ubFactorS1 = std::min(ubFactorS_, blockFactorS1);
    std::cout << "ubFactorS1 = " << ubFactorS1 << std::endl;
    auto ubFactorS2 = std::min(ubFactorS_, blockFactorS2);
    std::cout << "ubFactorS2 = " << ubFactorS2 << std::endl;
    if (usedCoreNum1 * ubFactorS1 >= usedCoreNum2 * ubFactorS2) {
        blockNumB_ = blockNumB1;
        std::cout << "blockNumB_ = " << blockNumB_ << std::endl;
        blockFactorB_ = blockFactorB1;
        std::cout << "blockFactorB_ = " << blockFactorB_ << std::endl;
        blockNumS_ = blockNumS1;
        std::cout << "blockNumS_ = " << blockNumS_ << std::endl;
        blockFactorS_ = blockFactorS1;
        std::cout << "blockFactorS_ = " << blockFactorS_ << std::endl;
        usedCoreNum_ = usedCoreNum1;
        std::cout << "usedCoreNum_ = " << usedCoreNum_ << std::endl;
        ubFactorS_ = ubFactorS1;
        std::cout << "ubFactorS_ = " << ubFactorS_ << std::endl;
    } else {
        blockNumB_ = blockNumB2;
        std::cout << "blockNumB_ = " << blockNumB_ << std::endl;
        blockFactorB_ = blockFactorB2;
        std::cout << "blockFactorB_ = " << blockFactorB_ << std::endl;
        blockNumS_ = blockNumS2;
        std::cout << "blockNumS_ = " << blockNumS_ << std::endl;
        blockFactorS_ = blockFactorS2;
        std::cout << "blockFactorS_ = " << blockFactorS_ << std::endl;
        usedCoreNum_ = usedCoreNum2;
        std::cout << "usedCoreNum_ = " << usedCoreNum_ << std::endl;
        ubFactorS_ = ubFactorS2;
        std::cout << "ubFactorS_ = " << ubFactorS_ << std::endl;
    }
    return;
}

ge::graphStatus RopeGradRegBaseTilingClassBAB::SplitUb()
{
    uint32_t typeSize = ge::GetSizeByDataType(dtype_);
    std::cout << "typeSize = " << typeSize << std::endl;
    int64_t dAlign = Ops::Base::CeilAlign(d_ * typeSize / dSplitCoef_, blockSize_) * dSplitCoef_;
    std::cout << "dAlign = " << dAlign << std::endl;
    // interleave模式需要补pad, D对齐
    int64_t canLoadDNum = Ops::Base::FloorDiv(ubSize_, dAlign);
    std::cout << "canLoadDNum = " << canLoadDNum << std::endl;
    if (canLoadDNum < MIN_UB_LOAD_D_NUM + MIN_UB_LOAD_D_NUM) {
        OP_LOGE(context_->GetNodeName(), "ubSize_ can't load 10 d_, d_ = %ld.", d_);
        return ge::GRAPH_FAILED;
    }
    canLoadDNum = canLoadDNum / MIN_UB_LOAD_D_NUM;
    std::cout << "canLoadDNum = " << canLoadDNum << std::endl;
    int64_t ubLoopNum = Ops::Base::CeilDiv(n_, (canLoadDNum - 1)); // 这里减去sin, cos 开doubleBuffer
    std::cout << "ubLoopNum = " << ubLoopNum << std::endl;
    ubFactorN_ = std::min(Ops::Base::CeilDiv(n_, ubLoopNum), MAX_COPY_BLOCK_COUNT / dSplitCoef_);
    std::cout << "ubFactorN_ = " << ubFactorN_ << std::endl;
    ubLoopNumN_ = Ops::Base::CeilDiv(n_, ubFactorN_);
    std::cout << "ubLoopNumN_ = " << ubLoopNumN_ << std::endl;
    ubTailFactorN_ = (n_ % ubFactorN_ == 0) ? ubFactorN_ : n_ % ubFactorN_;
    std::cout << "ubTailFactorN_ = " << ubTailFactorN_ << std::endl;
    int64_t ubFactorS = Ops::Base::FloorDiv(canLoadDNum, n_ + 1);
    ubFactorS_ = (ubFactorS == 0) ? 1 : ubFactorS;
    std::cout << "ubFactorS_ = " << ubFactorS_ << std::endl;
    return ge::GRAPH_SUCCESS;
}

void RopeGradRegBaseTilingClassBAB::PrintTilingData()
{
    std::cout << "=== PostTiling RopeGrad BAB ===" << std::endl
              << "  B=" << b_ << " S=" << s_ << " D=" << d_ << " N=" << n_ << std::endl
              << "  blockNumB=" << blockNumB_ << " blockFactorB=" << blockFactorB_ << " blockNumS=" << blockNumS_
              << " blockFactorS=" << blockFactorS_ << std::endl
              << "  ubFactorS=" << ubFactorS_ << " ubFactorB=" << ubFactorB_ << " ubLoopNumN=" << ubLoopNumN_
              << " ubFactorN=" << ubFactorN_ << " ubTailFactorN=" << ubTailFactorN_ << std::endl
              << "  usedCoreNum=" << usedCoreNum_ << " rotaryMode=" << static_cast<int64_t>(rotaryMode_)
              << " dSplitCoef=" << dSplitCoef_ << std::endl
              << "  dCosFlag=" << dCosFlag_ << " tilingKey=" << tilingKey_ << std::endl;

    OP_LOGI(context_->GetNodeName(),
            "RopeGradRegBaseTilingBAB tilingData: B=%ld S=%ld D=%ld N=%ld "
            "blockNumB=%ld blockFactorB=%ld blockNumS=%ld blockFactorS=%ld "
            "ubFactorS=%ld ubFactorB=%ld ubLoopNumN=%ld ubFactorN=%ld ubTailFactorN=%ld "
            "usedCoreNum=%ld rotaryMode=%ld dSplitCoef=%ld dCosFlag=%u tilingKey=%lu",
            b_, s_, d_, n_, blockNumB_, blockFactorB_, blockNumS_, blockFactorS_, ubFactorS_, ubFactorB_, ubLoopNumN_,
            ubFactorN_, ubTailFactorN_, usedCoreNum_, static_cast<int64_t>(rotaryMode_), dSplitCoef_, dCosFlag_,
            tilingKey_);
    return;
}

ge::graphStatus RopeGradRegBaseTilingClassBAB::PostTiling()
{
    tilingData_->ropeGradParams.b = b_;
    tilingData_->ropeGradParams.s = s_;
    tilingData_->ropeGradParams.d = d_;
    tilingData_->ropeGradParams.n = n_;
    tilingData_->ropeGradParams.ubFactorS = ubFactorS_;
    tilingData_->ropeGradParams.ubFactorB = ubFactorB_;
    tilingData_->ropeGradParams.ubLoopNumN = ubLoopNumN_;
    tilingData_->ropeGradParams.ubFactorN = ubFactorN_;
    tilingData_->ropeGradParams.ubTailFactorN = ubTailFactorN_;
    tilingData_->ropeGradParams.blockNumB = blockNumB_;
    tilingData_->ropeGradParams.blockFactorB = blockFactorB_;
    tilingData_->ropeGradParams.blockNumS = blockNumS_;
    tilingData_->ropeGradParams.blockFactorS = blockFactorS_;
    tilingData_->ropeGradParams.usedCoreNum = usedCoreNum_;
    tilingData_->ropeGradParams.rotaryMode = static_cast<int64_t>(rotaryMode_);
    tilingData_->dCosFlag = dCosFlag_;
    SetRotaryXTilingData();
    SetTilingKeyBlockDim(static_cast<uint32_t>(ropeGradTilingKey::TILING_KEY_BAB));
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(RotaryPositionEmbeddingGrad, RopeGradRegBaseTilingClassBAB, ROPE_GRAD_BAB_TILING_PRIORITY);
} // namespace optiling
