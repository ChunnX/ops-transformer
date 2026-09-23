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
 * \file kv_cache_turbo_quant_tiling.cpp
 * \brief
 */
#include "kv_cache_turbo_quant_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"

using namespace ge;

namespace optiling {

constexpr uint32_t KV_INPUT_INDEX = 0;
constexpr uint32_t QJL_INPUT_INDEX = 2;
constexpr size_t ATTR_MSE_BITS_INDEX = 0;
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t KV_DIMS = 3;
constexpr int64_t MATRIX_DIMS = 2;
constexpr int64_t MIN_KV_HEADS = 4;
constexpr int64_t MAX_KV_HEADS = 32;
constexpr int64_t QJL_DIM_ALIGN = 8;


static ge::graphStatus TilingKvCacheTurboQuant(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape *kvShape = context->GetInputShape(KV_INPUT_INDEX);
    if (kvShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &kv = kvShape->GetStorageShape();
    if (kv.GetDimNum() != KV_DIMS || kv.GetDim(2) != HEAD_DIM) {
        return ge::GRAPH_FAILED;
    }
    const int64_t numTokens = kv.GetDim(0);
    const int64_t numKvHeads = kv.GetDim(1);
    if (numTokens <= 0 || numKvHeads < MIN_KV_HEADS || numKvHeads > MAX_KV_HEADS) {
        return ge::GRAPH_FAILED;
    }
    const int64_t totalVec = numTokens * numKvHeads;

    const gert::StorageShape *qjlShape = context->GetInputShape(QJL_INPUT_INDEX);
    if (qjlShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &qjl = qjlShape->GetStorageShape();
    if (qjl.GetDimNum() != MATRIX_DIMS || qjl.GetDim(1) != HEAD_DIM) {
        return ge::GRAPH_FAILED;
    }
    const int64_t qjlDim = qjl.GetDim(0);
    if (qjlDim <= 0 || qjlDim > HEAD_DIM || (qjlDim % QJL_DIM_ALIGN) != 0) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t *pBits = attrs->GetAttrPointer<int64_t>(ATTR_MSE_BITS_INDEX);
    const int64_t mseBits = (pBits != nullptr) ? *pBits : 3;
    if (mseBits != 2 && mseBits != 3 && mseBits != 4) {
        return ge::GRAPH_FAILED;
    }

    // each (token, head) vector is independent; assign whole vectors to vector cores
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t availableCoreNum = ascendcPlatform.GetCoreNumAiv();
    if (availableCoreNum == 0) {
        availableCoreNum = 1;
    }
    const uint32_t blockDim =
        (totalVec < static_cast<int64_t>(availableCoreNum)) ? static_cast<uint32_t>(totalVec) : availableCoreNum;

    KvCacheTurboQuantTilingData tiling;
    tiling.set_totalVec(totalVec);
    tiling.set_qjlDim(qjlDim);
    tiling.set_mseBits(mseBits);
    tiling.set_coreNum(blockDim);
    context->SetBlockDim(blockDim);
    context->SetTilingKey(0);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // no GM workspace: both matrices and all intermediates are resident in UB
    size_t *workspaces = context->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = 0;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(KvCacheTurboQuant).Tiling(TilingKvCacheTurboQuant);

} // namespace optiling
