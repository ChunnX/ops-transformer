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
 * \file inplace_partial_rotary_mul_grad_regbase_tiling_ab.cpp
 * \brief AB template tiling for InplacePartialRotaryMulGrad (SBND layout)
 */
#include "op_host/tiling_templates_registry.h"
#include "inplace_partial_rotary_mul_grad_regbase_tiling.h"

namespace optiling {
constexpr uint64_t TILING_KEY_AB = 204;
constexpr uint64_t INPLACE_PARTIAL_ROTARY_MUL_GRAD_AB_TILING_PRIORITY = 25000;
constexpr int64_t CONST_TWO = 2;
constexpr int64_t DB_FLAG = 2;

class InplacePartialRotaryMulGradRegbaseTilingAb : public InplacePartialRotaryMulGradRegbaseTiling {
public:
    explicit InplacePartialRotaryMulGradRegbaseTilingAb(gert::TilingContext *context)
        : InplacePartialRotaryMulGradRegbaseTiling(context)
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

ge::graphStatus InplacePartialRotaryMulGradRegbaseTilingAb::DoOpTiling()
{
    int64_t bs = b_ * s_;
    bn_ = n_;
    if (cosb_ == 1) {
        bs = s_;
        bn_ = b_ * n_;
    }

    // dAlign 基于 sliceLength 和 max(typeSize, cosTypeSize)，所有 buffer 统一对齐
    int64_t typeSize = ge::GetSizeByDataType(dtype_);
    int64_t cosTypeSize = ge::GetSizeByDataType(cosDtype_);
    int64_t maxTypeSize = std::max(typeSize, cosTypeSize);
    dAlign_ = Ops::Base::CeilAlign(sliceLength_ / dSplitCoef_, blockSize_ / maxTypeSize) * dSplitCoef_;

    // 分核 (BS 维度)
    blockFactorBS_ = Ops::Base::CeilDiv(bs, int64_t(aicoreParams_.numBlocks));
    blockNumBS_ = Ops::Base::CeilDiv(bs, blockFactorBS_);
    blockTailBS_ = bs - (blockNumBS_ - 1) * blockFactorBS_;

    // 分核 (N 维度)
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

    // UB 分块
    // UB 布局: gradInQueue(DB=2) + dxQueue(DB=2) + cosInQueue(DB=2) + sinInQueue(DB=2)
    // 所有 buffer 统一按 dAlign_ * maxTypeSize 对齐，适配所有 dtype 组合
    int64_t dSizeX = dAlign_ * maxTypeSize;

    int64_t baseBlockInUb =
        Ops::Base::FloorAlign(static_cast<int64_t>(aicoreParams_.ubSize / CONST_TWO / DB_FLAG), blockSize_);
    int64_t baseBlockX = baseBlockInUb / dSizeX;

    // 同精度时 cos+sin 占 1 份 dSizeX；混合精度时也占 1 份（buffer 统一大小）
    OP_CHECK_IF(baseBlockX < 1, OP_LOGE(context_->GetNodeName(), "ubSize can't load AB grad, d = %ld.", sliceLength_),
                return ge::GRAPH_FAILED);

    ubFactorN_ = std::min(blockFactorN_, baseBlockX - 1);
    ubFactorN_ = std::min(ubFactorN_, MAX_COPY_BLOCK_COUNT / dSplitCoef_);
    ubFactorN_ = std::max(ubFactorN_, int64_t(1));

    ubFactorBS_ = std::min(Ops::Base::FloorDiv(baseBlockX, ubFactorN_ + 1), blockFactorBS_);
    ubFactorBS_ = std::max(ubFactorBS_, int64_t(1));

    // 最终验证：总 UB 不超限
    int64_t totalUbBytes = 4 * ubFactorBS_ * dAlign_ * (typeSize * ubFactorN_ + cosTypeSize);
    OP_CHECK_IF(totalUbBytes > static_cast<int64_t>(aicoreParams_.ubSize) / CONST_TWO,
                OP_LOGE(context_->GetNodeName(), "UB overflow: need %ld > %ld.", totalUbBytes,
                        static_cast<int64_t>(aicoreParams_.ubSize) / CONST_TWO),
                return ge::GRAPH_FAILED);

    // 初始化 tiling data
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

ge::graphStatus InplacePartialRotaryMulGradRegbaseTilingAb::PostTiling()
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
    tilingData_->sliceLength = sliceLength_;

    // 单 tiling key，不需要 reduce
    tilingKey_ = static_cast<uint64_t>(TILING_KEY_AB);
    context_->SetBlockDim(usedCoreNum_);

    OP_LOGI(context_->GetNodeName(),
            "InplacePartialRotaryMulGradAB tiling: B=%ld S=%ld N=%ld D=%ld dAlign=%ld dSplitCoef=%ld "
            "blockNumBS=%ld blockFactorBS=%ld blockTailBS=%ld "
            "blockNumN=%ld blockFactorN=%ld blockTailN=%ld "
            "ubFactorBS=%ld ubFactorN=%ld usedCoreNum=%ld rotaryMode=%ld "
            "sliceStart=%ld sliceEnd=%ld sliceLength=%ld",
            b_, s_, bn_, d_, dAlign_, dSplitCoef_, blockNumBS_, blockFactorBS_, blockTailBS_, blockNumN_, blockFactorN_,
            blockTailN_, ubFactorBS_, ubFactorN_, usedCoreNum_, static_cast<int64_t>(rotaryMode_), sliceStart_,
            sliceEnd_, sliceLength_);
    return ge::GRAPH_SUCCESS;
}

bool InplacePartialRotaryMulGradRegbaseTilingAb::IsCapable()
{
    if (!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) {
        return false;
    }
    // SBND layout: dy(S,B,N,D), cos(S,B,1,D) 或 cos(S,1,1,D)
    return layout_ == InplacePartialRotaryMulGradLayout::SBND;
}

REGISTER_OPS_TILING_TEMPLATE(InplacePartialRotaryMulGrad, InplacePartialRotaryMulGradRegbaseTilingAb,
                             INPLACE_PARTIAL_ROTARY_MUL_GRAD_AB_TILING_PRIORITY);
} // namespace optiling
