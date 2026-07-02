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
 * \file stem_oam_prep_paged_kv_tiling.cpp
 * \brief StemOamPrepPagedKV arch35 Tiling implementation (split_mode=2)
 */
#include "stem_oam_prep_paged_kv_tiling.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

constexpr uint64_t SYS_WORKSPACE_SIZE = static_cast<uint64_t>(16 * 1024 * 1024);

bool StemPrepKVTiling::IsCapable()
{
    return true;
}

ge::graphStatus StemPrepKVTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const StemPrepKVCompileInfo *>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        totalCoreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        totalCoreNum_ = static_cast<uint64_t>(ascendcPlatform.GetCoreNumAiv());
        if (totalCoreNum_ == 0UL) {
            OP_LOGE(context_->GetNodeName(), "coreNum is 0");
            return ge::GRAPH_FAILED;
        }
        uint64_t ubSize = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        if (ubSize == static_cast<uint64_t>(0)) {
            OP_LOGE(context_->GetNodeName(), "ubSize is 0");
            return ge::GRAPH_FAILED;
        }
        ubSize_ = static_cast<uint64_t>(ubSize);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StemPrepKVTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE("StemOamPrepPagedKV", "context is null"), return ge::GRAPH_FAILED);
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    lambda_mag_ = attrs->GetAttrValue<float>("LambdaMag");
    kv_layout_ = static_cast<uint32_t>(attrs->GetAttrValue<int64_t>("CacheLayout"));
    kv_block_size_ = static_cast<uint32_t>(attrs->GetAttrValue<int64_t>("KvBlockSize"));
    stem_block_size_ = static_cast<uint32_t>(attrs->GetAttrValue<int64_t>("StemBlockSize"));
    stem_stride_ = static_cast<uint32_t>(attrs->GetAttrValue<int64_t>("StemStride"));

    const gert::StorageShape *kvSeqLensShape = context_->GetInputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kvSeqLensShape);
    batch_size_ = kvSeqLensShape->GetStorageShape().GetDim(0);

    const gert::StorageShape *kcacheShape = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kcacheShape);
    if (kv_layout_ == 0) {
        num_kv_heads_ = kcacheShape->GetStorageShape().GetDim(2);
    } else {
        num_kv_heads_ = kcacheShape->GetStorageShape().GetDim(1);
    }
    total_blocks_ = kcacheShape->GetStorageShape().GetDim(0);

    const gert::StorageShape *kvIndicesShape = context_->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kvIndicesShape);
    max_kv_block_size_ = kvIndicesShape->GetStorageShape().GetDim(1);

    const gert::Tensor *kvSeqLensTensor = context_->GetInputTensor(3);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kvSeqLensTensor);
    const int32_t *kvSeqLensData = kvSeqLensTensor->GetData<int32_t>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, kvSeqLensData);

    max_kv_len_ = 0;
    for (uint32_t b = 0; b < batch_size_; b++) {
        uint32_t kvLen = static_cast<uint32_t>(kvSeqLensData[b]);
        if (kvLen > max_kv_len_) {
            max_kv_len_ = kvLen;
        }
    }

    max_k_padded_ = ((max_kv_len_ + stem_block_size_ - 1) / stem_block_size_) * stem_block_size_;
    max_Kb_ = max_k_padded_ / stem_block_size_;
    vbias_stride_ = (max_Kb_ < 8) ? 8 : max_Kb_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StemPrepKVTiling::DoOpTiling()
{
    uint32_t N = static_cast<uint32_t>(totalCoreNum_);

    kv_blocks_per_stem_block_ = stem_block_size_ / kv_block_size_;
    total_stem_blocks_ = batch_size_ * num_kv_heads_ * max_Kb_;

    if (total_stem_blocks_ == 0) {
        OP_LOGE(context_->GetNodeName(), "total_stem_blocks is 0!");
        return ge::GRAPH_FAILED;
    }

    used_core_num_ = std::min(N, total_stem_blocks_);
    stem_blocks_per_core_ = (total_stem_blocks_ + used_core_num_ - 1) / used_core_num_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StemPrepKVTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t StemPrepKVTiling::GetTilingKey() const
{
    return 0;
}

ge::graphStatus StemPrepKVTiling::GetWorkspaceSize()
{
    workspaceSize_ = SYS_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StemPrepKVTiling::PostTiling()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;

    tilingData_.set_batch_size(batch_size_);
    tilingData_.set_num_kv_heads(num_kv_heads_);
    tilingData_.set_dim_qk(128);
    tilingData_.set_dim_v(128);
    tilingData_.set_stem_block_size(stem_block_size_);
    tilingData_.set_stem_stride(stem_stride_);
    tilingData_.set_R_val(stem_block_size_ / stem_stride_);
    tilingData_.set_kflat_dim(stem_stride_ * 128);
    tilingData_.set_kv_block_size(kv_block_size_);
    tilingData_.set_kv_layout(kv_layout_);
    tilingData_.set_lambda_mag(lambda_mag_);
    tilingData_.set_epsilon(1e-6f);
    tilingData_.set_total_blocks(total_blocks_);
    tilingData_.set_max_k_padded(max_k_padded_);
    tilingData_.set_max_Kb(max_Kb_);
    tilingData_.set_max_kv_blocks_size(max_kv_block_size_);
    tilingData_.set_vbias_stride(vbias_stride_);
    tilingData_.set_split_mode(split_mode_);
    tilingData_.set_cores_per_bh(0);
    tilingData_.set_total_tasks(used_core_num_);
    tilingData_.set_used_core_num(used_core_num_);
    tilingData_.set_total_stem_blocks(total_stem_blocks_);
    tilingData_.set_stem_blocks_per_core(stem_blocks_per_core_);
    tilingData_.set_kv_blocks_per_stem_block(kv_blocks_per_stem_block_);

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    context_->SetBlockDim(used_core_num_);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(StemOamPrepPagedKV, StemPrepKVTiling, 0);

} // namespace optiling