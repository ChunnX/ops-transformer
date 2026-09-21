/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstring>
#include <limits>
#include <string>

#include "log/log.h"
#include "qbsa_fp8_tiling.h"
#include "../../../op_kernel/arch35/common/quant_block_sparse_attn_const.h"
#include "../../../op_kernel/arch35/common/quant_block_sparse_attn_tiling_key_registry.h"

namespace optiling {
namespace {
constexpr const char *kOpName = "QuantBlockSparseAttn";
constexpr uint8_t QBSA_PA_LAYOUT_TYPE_BNBD = 0U;
constexpr uint8_t QBSA_COMPAT_MASK_NONE = 0U;
constexpr uint8_t QBSA_COMPAT_MASK_RIGHT_DOWN_CAUSAL = 2U;
constexpr uint32_t QBSA_ATTEN_MASK_DEFAULT_S2_SIZE = 2048U;

void FillFp8CommonParams(const QuantBlockSparseAttnTilingInfo &info, uint32_t usedCoreNum,
                         QuantBlockSparseAttnFp8TilingData &tilingData)
{
    auto &common = tilingData.commonParams;
    auto &base = common.baseParams;
    base.bSize = info.bSize;
    base.t1Size = info.qTokenNum;
    base.n2Size = info.n2Size;
    base.gSize = info.gSize;
    base.dSize = info.dSize;
    base.dSizeV = info.dSizeV;
    base.coreNum = usedCoreNum;
    base.scaleValue = info.softmaxScaleVal;
    if (info.opParamInfo.pScale.shape != nullptr) {
        const int64_t size = info.opParamInfo.pScale.shape->GetStorageShape().GetShapeSize();
        base.pScaleShapeSize = size > 0 ? static_cast<uint32_t>(size) : 0U;
    }

    auto &pa = common.paParams;
    pa.blockTableDim2 = info.maxBlockNumPerBatch;
    pa.paBlockNumSum = info.paBlockNumSum;
    pa.paLayoutType = QBSA_PA_LAYOUT_TYPE_BNBD;
    pa.blockSize = info.paBlockSizeVal;
    pa.kvBlockSize = info.kvBlockSizeVal;
    pa.qBlockSize = info.qBlockSizeVal;
    pa.paBlockStride = info.paBlockStrideVal;
    pa.isRowInvalid = 1U;

    common.sparseParams.sparseSeqLenStride = info.qbMax;
    common.sparseParams.sparseIndicesStride = info.sparseCount;
    common.maskParams.preTokens = std::numeric_limits<int32_t>::max();
    common.maskParams.nextTokens = 0;
    common.maskParams.attenMaskS2Size = QBSA_ATTEN_MASK_DEFAULT_S2_SIZE;
    common.maskParams.attenMaskCompressMode =
        info.maskModeVal == QBSA_MASK_MODE_CAUSAL ? QBSA_COMPAT_MASK_RIGHT_DOWN_CAUSAL : QBSA_COMPAT_MASK_NONE;
    common.outputParams.totalOutputSize = static_cast<int64_t>(info.qTokenNum) * info.n1Size * info.dSizeV;
    common.outputParams.totalSoftMaxLseOutputSize = static_cast<int64_t>(info.qTokenNum) * info.n1Size;
}
} // namespace

ge::graphStatus QbsaFp8Arch35Tiling::BuildTilingData()
{
    const uint64_t bnCount = static_cast<uint64_t>(tilingInfo_.bSize) * tilingInfo_.n1Size;
    totalTaskNum_ = bnCount * tilingInfo_.gS1OuterSize;
    if (totalTaskNum_ == 0U || totalTaskNum_ > std::numeric_limits<uint32_t>::max()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "bSize * n1Size * gS1OuterSize", std::to_string(totalTaskNum_),
                                              "FP8 task num must be in range [1, UINT32_MAX]");
        return ge::GRAPH_FAILED;
    }
    usedCoreNum_ = static_cast<uint32_t>(std::min<uint64_t>(GetAicCoreNum(), totalTaskNum_));
    std::memset(&tilingData_, 0, sizeof(tilingData_));
    FillFp8CommonParams(tilingInfo_, usedCoreNum_, tilingData_);

    auto &fp8 = tilingData_.fp8Params;
    fp8.layoutType = tilingInfo_.layoutQValue == QBSA_LAYOUT_Q_NTD_VALUE ? QBSA_REGBASE_LAYOUT_TYPE_NTD :
                                                                           QBSA_REGBASE_LAYOUT_TYPE_TND;
    fp8.seqUsedQlenSize = tilingInfo_.bSize;
    fp8.seqUsedKvlenSize = tilingInfo_.bSize;
    fp8.isKvContinuous = 1U;
    fp8.isGqa = tilingInfo_.isGqa ? 1U : 0U;
    fp8.s1OuterSize = tilingInfo_.qbMax;
    for (uint32_t boundaryIdx = 0U; boundaryIdx <= usedCoreNum_; ++boundaryIdx) {
        const uint64_t taskOffset = totalTaskNum_ * boundaryIdx / usedCoreNum_;
        fp8.bnStartIdx[boundaryIdx] = static_cast<uint32_t>(taskOffset / tilingInfo_.gS1OuterSize);
    }
    const int64_t aivCoreNum = static_cast<int64_t>(usedCoreNum_) * 2;
    fp8.singleCoreSize =
        static_cast<uint32_t>((tilingData_.commonParams.outputParams.totalOutputSize + aivCoreNum - 1) / aivCoreNum);
    fp8.needInit = 1U;

    tilingKey_ =
        BuildQbsaFp8TilingKey(tilingInfo_.layoutQValue, tilingInfo_.maskModeVal, tilingInfo_.returnSoftmaxLseVal);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QbsaFp8Arch35Tiling::SaveTilingData()
{
    return PublishTilingData(tilingData_);
}
} // namespace optiling
