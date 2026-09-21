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
#include "qbsa_mxfp8_tiling.h"
#include "../../../op_kernel/arch35/common/quant_block_sparse_attn_const.h"
#include "../../../op_kernel/arch35/common/quant_block_sparse_attn_tiling_key_registry.h"

namespace optiling {
namespace {
constexpr const char *kOpName = "QuantBlockSparseAttn";
constexpr uint8_t QBSA_PA_LAYOUT_TYPE_BNBD = 0U;
constexpr uint8_t QBSA_COMPAT_MASK_NONE = 0U;
constexpr uint8_t QBSA_COMPAT_MASK_RIGHT_DOWN_CAUSAL = 2U;
constexpr uint32_t QBSA_ATTEN_MASK_DEFAULT_S2_SIZE = 2048U;
constexpr uint32_t QBSA_MXFP8_VALUE_SCALE_LAST_DIM = 2U;

QuantBlockSparseAttnMxStrideParams MakePaBnbdStride(uint32_t n2Size, uint32_t blockSize, uint32_t lastDim)
{
    QuantBlockSparseAttnMxStrideParams stride;
    stride.n2Stride = static_cast<uint64_t>(blockSize) * lastDim;
    stride.bnStride = static_cast<uint64_t>(n2Size) * stride.n2Stride;
    return stride;
}

void FillMxfp8CommonParams(const QuantBlockSparseAttnTilingInfo &info, uint32_t usedCoreNum,
                           QuantBlockSparseAttnMxfp8TilingData &tilingData)
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

ge::graphStatus QbsaMxfp8Arch35Tiling::BuildTilingData()
{
    const uint64_t bnCount = static_cast<uint64_t>(tilingInfo_.bSize) * tilingInfo_.n1Size;
    totalTaskNum_ = bnCount * tilingInfo_.gS1OuterSize;
    if (totalTaskNum_ == 0U || totalTaskNum_ > std::numeric_limits<uint32_t>::max()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "bSize * n1Size * gS1OuterSize", std::to_string(totalTaskNum_),
                                              "MXFP8 task num must be in range [1, UINT32_MAX]");
        return ge::GRAPH_FAILED;
    }
    usedCoreNum_ = static_cast<uint32_t>(std::min<uint64_t>(GetAicCoreNum(), totalTaskNum_));
    std::memset(&tilingData_, 0, sizeof(tilingData_));
    FillMxfp8CommonParams(tilingInfo_, usedCoreNum_, tilingData_);

    auto &mx = tilingData_.mxParams;
    mx.scaleGroupSize = QBSA_MXFP8_SCALE_GROUP_SIZE;
    mx.scaleLastDim = QBSA_MXFP8_SCALE_LAST_DIM;
    mx.queryScaleDSize = QBSACeilDiv(tilingInfo_.dSize, QBSA_MXFP8_SCALE_GROUP_SIZE);
    mx.keyScaleDSize = QBSACeilDiv(tilingInfo_.dSize, QBSA_MXFP8_SCALE_GROUP_SIZE);
    mx.valueScaleBlockSize = QBSACeilDiv(tilingInfo_.paBlockSizeVal, QBSA_MXFP8_SCALE_GROUP_SIZE);
    mx.valueScaleDSize = tilingInfo_.dSizeV * QBSA_MXFP8_VALUE_SCALE_LAST_DIM;
    mx.keyStrides = MakePaBnbdStride(tilingInfo_.n2Size, tilingInfo_.paBlockSizeVal, tilingInfo_.dSize);
    mx.valueStrides = MakePaBnbdStride(tilingInfo_.n2Size, tilingInfo_.paBlockSizeVal, tilingInfo_.dSizeV);
    mx.kScaleStrides =
        MakePaBnbdStride(tilingInfo_.n2Size, tilingInfo_.paBlockSizeVal, mx.keyScaleDSize * mx.scaleLastDim);
    mx.vScaleStrides = MakePaBnbdStride(tilingInfo_.n2Size, mx.valueScaleBlockSize, mx.valueScaleDSize);
    mx.pScaleDtype = MX_PSCALE_DTYPE_E8M0;
    if (tilingData_.commonParams.baseParams.pScaleShapeSize > 0U && tilingInfo_.opParamInfo.pScale.desc != nullptr &&
        tilingInfo_.opParamInfo.pScale.desc->GetDataType() == ge::DT_FLOAT) {
        mx.pScaleDtype = MX_PSCALE_DTYPE_FP32;
    }

    tilingKey_ =
        BuildQbsaMxfp8TilingKey(tilingInfo_.layoutQValue, tilingInfo_.maskModeVal, tilingInfo_.returnSoftmaxLseVal);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QbsaMxfp8Arch35Tiling::SaveTilingData()
{
    return PublishTilingData(tilingData_);
}
} // namespace optiling
