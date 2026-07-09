/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file stem_oam_prep_paged_kv_tiling_simd.cpp
 * \brief StemOamPrepPagedKV simd Tiling implementation
 */
#include "stem_oam_prep_paged_kv_tiling.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

bool StemPrepKVTiling::IsCapable()
{
    return true;
}

void StemPrepKVTiling::ContinuousStridesCompute(gert::Shape &shape, gert::Stride &inputStride)
{
    inputStride.SetDimNum(shape.GetDimNum());
    int32_t maxDim = static_cast<int32_t>(shape.GetDimNum()) - 1;
    int64_t tmpStride = 1; 
    for (int32_t j = maxDim; j >= 0; --j) {
        inputStride.SetStride(j, tmpStride);
        tmpStride *= shape.GetDim(j);
    }
}

ge::graphStatus StemPrepKVTiling::GetInputStrides(gert::Shape &shape, gert::Stride &inputStride, size_t idx)
{
    if (context_->InputIsView(idx)) {
        auto* strides = context_->GetInputStride(idx);
        if (strides == nullptr || strides->GetDimNum() == 0) {
            ContinuousStridesCompute(shape, inputStride);
        } else {
            inputStride = *strides;
        }
    } else {
        ContinuousStridesCompute(shape, inputStride);
    }
    return ge::GRAPH_SUCCESS
}

ge::graphStatus StemPrepKVTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const StemPrepKVCompileInfo *>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        totalCoreNum_ = compileInfoPtr->coreNum;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        totalCoreNum_ = static_cast<uint64_t>(ascendcPlatform.GetCoreNumAiv());
        if (totalCoreNum_ == 0UL) {
            OP_LOGE(context_->GetNodeName(), "coreNum is 0");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StemPrepKVTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE("StemOamPrepPagedKV", "context is null"), return ge::GRAPH_FAILED);
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    tilingData_ = context_->GetTilingData<StemPrepKVTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData_);

    lambdaMag_ = attrs->GetAttrValue<float>("lambdaMag");
    kvLayout_ = static_cast<uint32_t>(attrs->GetAttrValue<int64_t>("cacheLayout"));
    kvBlockSize_ = static_cast<uint32_t>(attrs->GetAttrValue<int64_t>("kvBlockSize"));
    stemBlockSize_ = static_cast<uint32_t>(attrs->GetAttrValue<int64_t>("stemBlockSize"));
    stemStride_ = static_cast<uint32_t>(attrs->GetAttrValue<int64_t>("stemStride"));

    rVal_ = stemBlockSize_ / stemStride_;

    const gert::StorageShape *kvSeqLensShape = context_->GetInputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kvSeqLensShape);
    batchSize_ = kvSeqLensShape->GetStorageShape().GetDim(0);

    const gert::StorageShape *kcacheShape = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kcacheShape);
    if (kvLayout_ == 0) {
        numKvHeads_ = kcacheShape->GetStorageShape().GetDim(2);
    } else {
        numKvHeads_ = kcacheShape->GetStorageShape().GetDim(1);
    }
    totalBlocks_ = kcacheShape->GetStorageShape().GetDim(0);
    dimQk_ = kcacheShape->GetStorageShape().GetDim(kcacheShape->GetStorageShape().GetDimNum() - 1);

    const gert::StorageShape *kvIndicesShape = context_->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kvIndicesShape);
    maxKvBlockSize_ = kvIndicesShape->GetStorageShape().GetDim(1);
    const gert::StorageShape *kScaleCacheShape = context_->GetInputShape(4);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kScaleCacheShape);
    const gert::StorageShape *kFlatOutShape = context_->GetOutputShape(0);
    maxKb_ = kFlatOutShape->GetStorageShape().GetDim(0);
    kflatDim_ = kFlatOutShape->GetStorageShape().GetDim(3);

    GetInputStrides(kcacheShape, kCacheStride_, 0);
    GetInputStrides(kcacheShape, vCacheStride_, 1);
    GetInputStrides(kScaleCacheShape, kScaleCacheStride_, 4);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StemPrepKVTiling::DoOpTiling()
{
    if (batchSize_ == 0 || numKvHeads_ == 0) {
        OP_LOGE(context_->GetNodeName(), "batchSize or numKvHeads is 0!");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StemPrepKVTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t StemPrepKVTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(0);
}

ge::graphStatus StemPrepKVTiling::GetWorkspaceSize()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    uint64_t maxKvPadded = MAX_KV_PADDED;
    uint64_t vFlatSize = static_cast<uint64_t>(batchSize_) *
                         static_cast<uint64_t>(numKvHeads_) *
                         (maxKvPadded / static_cast<uint64_t>(stemStride_)) *
                         sizeof(float);
    workspaceSize_ = vFlatSize;

    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_ + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StemPrepKVTiling::PostTiling()
{
    uint32_t usedCoreNum = static_cast<uint32_t>(totalCoreNum_);
    uint32_t totalBh = batchSize_ * numKvHeads_;

    for (int32_t i = 0; i < 4; i++) {
        tilingData_->kCacheStride[i] = kCacheStride_[i];
        tilingData_->vCacheStride[i] = vCacheStride_[i];
        tilingData_->kScaleCacheStride[i] = kScaleCacheStride_[i];
    }

    tilingData_->batchSize = batchSize_;
    tilingData_->numKvHeads = numKvHeads_;
    tilingData_->kflatDim = kflatDim_;
    tilingData_->maxKb = maxKb_;
    tilingData_->kvBlockSize = kvBlockSize_;
    tilingData_->dimQk = dimQk_;
    tilingData_->dimV = dimQk_;
    tilingData_->maxKvBlocks = maxKvBlockSize_;
    tilingData_->stemBlockSize = stemBlockSize_;
    tilingData_->stemStride = stemStride_;
    tilingData_->lambdaMag = lambdaMag_;
    tilingData_->kvLayout = kvLayout_;
    tilingData_->rVal = rVal_;

<<<<<<< HEAD

=======
>>>>>>> 4a93fae9e0d33ecab3ec842d5c38b49f32349b12
    context_->SetBlockDim(usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(StemOamPrepPagedKV, StemPrepKVTiling, 0);

} // namespace optiling
