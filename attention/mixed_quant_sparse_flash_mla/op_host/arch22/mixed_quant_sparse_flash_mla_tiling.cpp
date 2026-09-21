/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../mixed_quant_sparse_flash_mla_tiling.h"
#include "../../op_kernel/mixed_quant_sparse_flash_mla_template_tiling_key.h"
#include <algorithm>

using namespace ge;
using namespace AscendC;
namespace optiling {

ge::graphStatus MixedQuantSparseFlashMlaTiling::DoTurboQuantTiling(MQSMLATilingInfo *tilingInfo)
{
    constexpr uint32_t S2_BASE_SIZE = 512;
    constexpr uint32_t BYTE_BLOCK = 32;
    constexpr uint32_t MM1_RES_ELEM_SIZE = 4;
    constexpr uint32_t VEC1_RES_ELEM_SIZE = 2;
    constexpr uint32_t MM2_RES_ELEM_SIZE = 4;
    constexpr uint32_t VEC2_RES_ELEM_SIZE = 4;
    constexpr uint32_t PRELOAD_NUM = 2;
    constexpr uint32_t MERGE_CACHE_GM_BUF_NUM = 3;
    constexpr uint32_t MERGE_CACHE_ELEM_SIZE = 2;
    auto align = [](uint32_t value, uint32_t alignment) { return (value + alignment - 1) / alignment * alignment; };

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo->platformInfo);
    const uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    const uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    context_->SetBlockDim(ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum));

    const uint32_t mBaseSize = tilingInfo->gSize;
    const uint32_t s2BaseSizeAlign = align(S2_BASE_SIZE, BYTE_BLOCK);
    // The kernel splits the full TND M dimension into tiles no larger than mBaseSize.
    const uint32_t cubeMSize = mBaseSize;
    const uint32_t mmResUbSize = s2BaseSizeAlign * align(cubeMSize, 16U);
    const uint32_t bmm2ResUbSize = align(tilingInfo->qkHeadDim, BYTE_BLOCK) * align(cubeMSize, 16U);

    size_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    workspaceSize += static_cast<size_t>(PRELOAD_NUM) * mmResUbSize * (MM1_RES_ELEM_SIZE + VEC1_RES_ELEM_SIZE) * aicNum;
    workspaceSize +=
        static_cast<size_t>(PRELOAD_NUM) * bmm2ResUbSize * (MM2_RES_ELEM_SIZE + VEC2_RES_ELEM_SIZE) * aicNum;
    workspaceSize += static_cast<size_t>(MERGE_CACHE_GM_BUF_NUM) * S2_BASE_SIZE * tilingInfo->qkHeadDim *
                     MERGE_CACHE_ELEM_SIZE * aicNum;
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE(tilingInfo->opName, "workspace sizes is nullptr"),
                return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize;

    tilingData_.tqBaseParams.set_batchSize(tilingInfo->bSize);
    tilingData_.tqBaseParams.set_qSeqSize(tilingInfo->s1Size);
    tilingData_.tqBaseParams.set_kvSeqSize(tilingInfo->s2Size);
    tilingData_.tqBaseParams.set_paBlockSize(tilingInfo->oriBlockSize);
    tilingData_.tqBaseParams.set_oriBlockSize(tilingInfo->oriBlockSize);
    tilingData_.tqBaseParams.set_cmpBlockSize(tilingInfo->cmpBlockSize);
    tilingData_.tqBaseParams.set_oriMaxBlockNumPerBatch(tilingInfo->oriMaxBlockNumPerBatch);
    tilingData_.tqBaseParams.set_nNumOfQInOneGroup(tilingInfo->gSize);
    tilingData_.tqBaseParams.set_actualLenDimsQ(tilingInfo->actualLenDimsQ);
    tilingData_.tqBaseParams.set_actualLenDimsKV(tilingInfo->actualLenDimsKV);
    tilingData_.tqBaseParams.set_softmaxScale(tilingInfo->softmaxScale);
    tilingData_.tqBaseParams.set_outputLayout(static_cast<uint32_t>(tilingInfo->outLayout));
    tilingData_.tqBaseParams.set_oriMaskMode(tilingInfo->oriMaskMode);
    tilingData_.tqBaseParams.set_oriKvStride0(tilingInfo->oriKvStride);
    tilingData_.tqBaseParams.set_oriWinLeft(tilingInfo->oriWinLeft);
    tilingData_.tqBaseParams.set_oriWinRight(tilingInfo->oriWinRight);
    tilingData_.tqBaseParams.set_sparseBlockSize(tilingInfo->sparseBlockSize);
    tilingData_.tqBaseParams.set_usedCoreNum(aicNum);
    tilingData_.tqBaseParams.set_mmResUbSize(mmResUbSize);
    tilingData_.tqBaseParams.set_bmm2ResUbSize(bmm2ResUbSize);
    tilingData_.tqBaseParams.set_mBaseSize(mBaseSize);
    tilingData_.tqBaseParams.set_s2BaseSize(S2_BASE_SIZE);
    tilingData_.tqBaseParams.set_returnSoftmaxLse(tilingInfo->returnSoftmaxLse);
    tilingData_.tqBaseParams.set_kvQuantMode(static_cast<uint32_t>(tilingInfo->quantMode));
    tilingData_.tqCmpParams.set_cmpMaxBlockNumPerBatch(tilingInfo->cmpMaxBlockNumPerBatch);
    tilingData_.tqCmpParams.set_sparseBlockCount(tilingInfo->cmpSparseBlockCount);
    tilingData_.tqCmpParams.set_cmpRatio(tilingInfo->cmpRatio);
    tilingData_.tqCmpParams.set_cmpMaskMode(tilingInfo->cmpMaskMode);
    tilingData_.tqCmpParams.set_cmpKvStride0(tilingInfo->cmpKvStride);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    const uint64_t tilingKey = GET_TPL_TILING_KEY(
        0U, static_cast<uint32_t>(tilingInfo->qLayout), static_cast<uint32_t>(tilingInfo->kvLayout), CSA_TEMPLATE, 0U,
        static_cast<uint32_t>(tilingInfo->quantMode), DTYPE_FP8_E4M3FN, 0U, 0U, 0U);
    context_->SetTilingKey(tilingKey);
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
