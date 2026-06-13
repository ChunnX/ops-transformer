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
#include <iostream>

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

    void SplitCore();
    ge::graphStatus SplitUb();
};

ge::graphStatus InplacePartialRotaryMulGradRegbaseTilingBab::DoOpTiling()
{
    ubSize_ = aicoreParams_.ubSize;
    std::cout << "ubSize_ = " << ubSize_ << std::endl;
    coreNum_ = static_cast<int64_t>(aicoreParams_.numBlocks);
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

void InplacePartialRotaryMulGradRegbaseTilingBab::SplitCore()
{
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
}

ge::graphStatus InplacePartialRotaryMulGradRegbaseTilingBab::SplitUb()
{
    int64_t typeSize = ge::GetSizeByDataType(dtype_);
    std::cout << "typeSize = " << typeSize << std::endl;
    int64_t cosTypeSize = ge::GetSizeByDataType(cosDtype_);
    std::cout << "cosTypeSize = " << cosTypeSize << std::endl;
    int64_t maxTypeSize = std::max(typeSize, cosTypeSize);
    std::cout << "maxTypeSize = " << maxTypeSize << std::endl;
    int64_t dAlign = Ops::Base::CeilAlign(sliceLength_ * maxTypeSize / dSplitCoef_, blockSize_) * dSplitCoef_;
    std::cout << "dAlign = " << dAlign << std::endl;

    int64_t canLoadDNum = Ops::Base::FloorDiv(ubSize_, dAlign);
    std::cout << "canLoadDNum = " << canLoadDNum << std::endl;
    if (canLoadDNum < MIN_UB_LOAD_D_NUM + MIN_UB_LOAD_D_NUM) {
        OP_LOGE(context_->GetNodeName(), "ubSize_ can't load 8 d_, d_ = %ld.", sliceLength_);
        return ge::GRAPH_FAILED;
    }
    canLoadDNum = canLoadDNum / MIN_UB_LOAD_D_NUM;
    std::cout << "canLoadDNum = " << canLoadDNum << std::endl;
    int64_t ubLoopNum = Ops::Base::CeilDiv(n_, (canLoadDNum - 1));
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

    // 最终验证：总 UB 不超限
    int64_t dyBytes = 4 * ubFactorS_ * ubFactorN_ * dAlign;
    int64_t cosBytes = 4 * ubFactorS_ * dAlign;
    std::cout << "dyBytes = " << dyBytes << std::endl;
    std::cout << "cosBytes = " << cosBytes << std::endl;
    std::cout << "totalBytes = " << dyBytes + cosBytes << std::endl;
    OP_CHECK_IF(dyBytes + cosBytes > static_cast<int64_t>(ubSize_),
                OP_LOGE(context_->GetNodeName(), "UB overflow: need %ld > %ld.", dyBytes + cosBytes, ubSize_),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradRegbaseTilingBab::PostTiling()
{
    tilingData_->inplacePartialRotaryMulGradParams.b = b_;
    tilingData_->inplacePartialRotaryMulGradParams.s = s_;
    tilingData_->inplacePartialRotaryMulGradParams.d = d_;
    tilingData_->inplacePartialRotaryMulGradParams.n = n_;
    tilingData_->inplacePartialRotaryMulGradParams.ubFactorS = ubFactorS_;
    tilingData_->inplacePartialRotaryMulGradParams.ubFactorB = ubFactorB_;
    tilingData_->inplacePartialRotaryMulGradParams.ubLoopNumN = ubLoopNumN_;
    tilingData_->inplacePartialRotaryMulGradParams.ubFactorN = ubFactorN_;
    tilingData_->inplacePartialRotaryMulGradParams.ubTailFactorN = ubTailFactorN_;
    tilingData_->inplacePartialRotaryMulGradParams.blockNumB = blockNumB_;
    tilingData_->inplacePartialRotaryMulGradParams.blockFactorB = blockFactorB_;
    tilingData_->inplacePartialRotaryMulGradParams.blockNumS = blockNumS_;
    tilingData_->inplacePartialRotaryMulGradParams.blockFactorS = blockFactorS_;
    tilingData_->inplacePartialRotaryMulGradParams.usedCoreNum = usedCoreNum_;
    tilingData_->inplacePartialRotaryMulGradParams.rotaryMode = static_cast<int64_t>(rotaryMode_);
    tilingData_->inplacePartialRotaryMulGradParams.sliceStart = sliceStart_;
    tilingData_->inplacePartialRotaryMulGradParams.sliceEnd = sliceEnd_;
    tilingData_->inplacePartialRotaryMulGradParams.sliceLength = sliceLength_;
    tilingData_->inplacePartialRotaryMulGradParams.dSplitCoef = dSplitCoef_;
    tilingData_->sliceLength = sliceLength_;

    tilingKey_ = static_cast<uint64_t>(TILING_KEY_BAB);
    context_->SetBlockDim(usedCoreNum_);

    std::cout << "=== PostTiling InplacePartialRotaryMulGrad BAB ===" << std::endl
              << "  B=" << b_ << " S=" << s_ << " D=" << d_ << " N=" << n_ << std::endl
              << "  blockNumB=" << blockNumB_ << " blockFactorB=" << blockFactorB_ << " blockNumS=" << blockNumS_
              << " blockFactorS=" << blockFactorS_ << std::endl
              << "  ubFactorS=" << ubFactorS_ << " ubFactorB=" << ubFactorB_ << " ubLoopNumN=" << ubLoopNumN_
              << " ubFactorN=" << ubFactorN_ << " ubTailFactorN=" << ubTailFactorN_ << std::endl
              << "  usedCoreNum=" << usedCoreNum_ << " rotaryMode=" << static_cast<int64_t>(rotaryMode_)
              << " dSplitCoef=" << dSplitCoef_ << std::endl
              << "  sliceStart=" << sliceStart_ << " sliceEnd=" << sliceEnd_ << " sliceLength=" << sliceLength_
              << std::endl
              << "  tilingDataSliceLength=" << tilingData_->sliceLength << " tilingKey=" << tilingKey_ << std::endl;

    OP_LOGI(context_->GetNodeName(),
            "InplacePartialRotaryMulGradBAB tiling: B=%ld S=%ld D=%ld N=%ld "
            "blockNumB=%ld blockFactorB=%ld blockNumS=%ld blockFactorS=%ld "
            "ubFactorS=%ld ubFactorB=%ld ubLoopNumN=%ld ubFactorN=%ld ubTailFactorN=%ld "
            "usedCoreNum=%ld rotaryMode=%ld dSplitCoef=%ld "
            "sliceStart=%ld sliceEnd=%ld sliceLength=%ld "
            "tilingDataSliceLength=%ld tilingKey=%lu",
            b_, s_, d_, n_, blockNumB_, blockFactorB_, blockNumS_, blockFactorS_, ubFactorS_, ubFactorB_, ubLoopNumN_,
            ubFactorN_, ubTailFactorN_, usedCoreNum_, static_cast<int64_t>(rotaryMode_), dSplitCoef_, sliceStart_,
            sliceEnd_, sliceLength_, tilingData_->sliceLength, tilingKey_);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(InplacePartialRotaryMulGrad, InplacePartialRotaryMulGradRegbaseTilingBab,
                             INPLACE_PARTIAL_ROTARY_MUL_GRAD_BAB_TILING_PRIORITY);
} // namespace optiling
