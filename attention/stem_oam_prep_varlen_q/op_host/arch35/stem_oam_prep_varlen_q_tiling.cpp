/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stem_oam_prep_varlen_q_tiling.h"
#include "../../op_kernel/arch35/stem_oam_prep_varlen_q_tiling_data.h"
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include <algorithm>

namespace optiling {

static constexpr int64_t DIM_QK_HOST = 128;
static constexpr int64_t MAX_BATCH_SIZE = 1024;

static void PrintTilingData(const char* nodeName, const StemPrepQTilingData* tiling)
{
    OP_LOGD(nodeName, "========== StemPrepQTilingData ==========");
    OP_LOGD(nodeName, "batch_size: %u", tiling->batch_size);
    OP_LOGD(nodeName, "num_q_heads: %u", tiling->num_q_heads);
    OP_LOGD(nodeName, "dim_qk: %u", tiling->dim_qk);
    OP_LOGD(nodeName, "stem_block_size: %u", tiling->stem_block_size);
    OP_LOGD(nodeName, "stem_stride: %u", tiling->stem_stride);
    OP_LOGD(nodeName, "R_val: %u", tiling->R_val);
    OP_LOGD(nodeName, "kflat_dim: %u", tiling->kflat_dim);
    OP_LOGD(nodeName, "max_Qb: %u", tiling->max_Qb);
    OP_LOGD(nodeName, "used_core_num: %u", tiling->used_core_num);
    OP_LOGD(nodeName, "blocks_per_core_base: %u", tiling->blocks_per_core_base);
    OP_LOGD(nodeName, "blocks_remainder: %u", tiling->blocks_remainder);
}

static ge::graphStatus ValidateInputs(gert::TilingContext* context,
                                       int64_t& stemBlockSize,
                                       int64_t& stemStride,
                                       int64_t& batch,
                                       int64_t& numQHeads)
{
    const char* nodeName = context->GetNodeName();
    
    auto compile_info = static_cast<const StemPrepQCompileInfo*>(context->GetCompileInfo());
    if (compile_info == nullptr || compile_info->core_num == 0) {
        OP_LOGE(nodeName, "Invalid compile info or core_num is 0");
        return ge::GRAPH_FAILED;
    }

    auto attrs = context->GetAttrs();
    const int64_t *stemBlockSizePtr = attrs->GetInt(0);
    const int64_t *stemStridePtr = attrs->GetInt(1);
    if (stemBlockSizePtr == nullptr || stemStridePtr == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "stemBlockSize/stemStride");
        return ge::GRAPH_FAILED;
    }
    stemBlockSize = *stemBlockSizePtr;
    stemStride = *stemStridePtr;

    if (stemBlockSize % 32 != 0 || stemBlockSize > 256 || stemBlockSize <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "stemBlockSize", 
            std::to_string(stemBlockSize).c_str(), "must be positive, <= 256, and multiple of 32");
        return ge::GRAPH_FAILED;
    }
    if (stemStride % 16 != 0 || stemStride > 64 || stemStride <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "stemStride", 
            std::to_string(stemStride).c_str(), "must be positive, <= 64, and multiple of 16");
        return ge::GRAPH_FAILED;
    }
    if (stemStride > stemBlockSize) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "stemStride", 
            std::to_string(stemStride).c_str(), 
            ("must be <= stemBlockSize (" + std::to_string(stemBlockSize) + ")").c_str());
        return ge::GRAPH_FAILED;
    }

    auto qShape = context->GetInputShape(0);
    auto qSeqLensShape = context->GetInputShape(1);
    if (qShape == nullptr || qSeqLensShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "q/qSeqLens");
        return ge::GRAPH_FAILED;
    }

    // Validate shape dimensions
    if (qShape->GetStorageShape().GetDimNum() != 3) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "q", 
            std::to_string(qShape->GetStorageShape().GetDimNum()).c_str(), "3");
        return ge::GRAPH_FAILED;
    }
    if (qSeqLensShape->GetStorageShape().GetDimNum() != 1) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "qSeqLens", 
            std::to_string(qSeqLensShape->GetStorageShape().GetDimNum()).c_str(), "1");
        return ge::GRAPH_FAILED;
    }

    // Validate dtypes
    auto qDesc = context->GetInputDesc(0);
    auto qSeqLensDesc = context->GetInputDesc(1);
    if (qDesc == nullptr || qSeqLensDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "q/qSeqLens descriptor");
        return ge::GRAPH_FAILED;
    }
    if (qDesc->GetDataType() != ge::DT_FLOAT8_E4M3FN) {
        OP_LOGE_WITH_INVALID_INPUT_DTYPE(nodeName, "q", 
            std::to_string(static_cast<int>(qDesc->GetDataType())).c_str(), "DT_FLOAT8_E4M3FN");
        return ge::GRAPH_FAILED;
    }
    if (qSeqLensDesc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_WITH_INVALID_INPUT_DTYPE(nodeName, "qSeqLens", 
            std::to_string(static_cast<int>(qSeqLensDesc->GetDataType())).c_str(), "DT_INT32");
        return ge::GRAPH_FAILED;
    }

    numQHeads = qShape->GetStorageShape().GetDim(1);
    int64_t D = qShape->GetStorageShape().GetDim(2);
    if (D != DIM_QK_HOST) {
        OP_LOGE_WITH_INVALID_INPUT_SHAPESIZE(nodeName, 0, 
            std::to_string(D).c_str(), std::to_string(DIM_QK_HOST).c_str());
        return ge::GRAPH_FAILED;
    }

    batch = qSeqLensShape->GetStorageShape().GetDim(0);
    if (batch <= 0 || batch > MAX_BATCH_SIZE || numQHeads <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "batch/numQHeads", 
            ("batch=" + std::to_string(batch) + ", numQHeads=" + std::to_string(numQHeads)).c_str(),
            ("batch must be in (0, " + std::to_string(MAX_BATCH_SIZE) + "], numQHeads must be > 0").c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalcSequenceInfo(gert::TilingContext* context,
                                         StemPrepQTilingData* tiling,
                                         uint32_t B,
                                         uint32_t& totalBlocks)
{
    const char* nodeName = context->GetNodeName();
    
    auto qSeqLensTensor = context->GetInputTensor(1);
    if (qSeqLensTensor == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "qSeqLens");
        return ge::GRAPH_FAILED;
    }
    int32_t* qSeqLensData = const_cast<int32_t*>(qSeqLensTensor->GetData<int32_t>());
    if (qSeqLensData == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "qSeqLens data");
        return ge::GRAPH_FAILED;
    }

    uint32_t max_q_len = 0;
    totalBlocks = 0;
    uint32_t H_q = tiling->num_q_heads;
    uint32_t batchU32 = tiling->batch_size;

    for (uint32_t b = 0; b < batchU32; b++) {
        uint32_t q_len_b = static_cast<uint32_t>(qSeqLensData[b]);
        uint32_t q_padded_b = ((q_len_b + B - 1) / B) * B;
        uint32_t num_Qb_b = q_padded_b / B;
        totalBlocks += H_q * num_Qb_b;
        if (q_len_b > max_q_len) {
            max_q_len = q_len_b;
        }
    }

    uint32_t max_q_padded = ((max_q_len + B - 1) / B) * B;
    tiling->max_Qb = max_q_padded / B;

    return ge::GRAPH_SUCCESS;
}

static void CalcCoreDistribution(gert::TilingContext* context,
                                  StemPrepQTilingData* tiling,
                                  uint32_t totalBlocks,
                                  uint32_t N)
{
    const char* nodeName = context->GetNodeName();
    
    if (totalBlocks == 0) {
        tiling->used_core_num = 0;
        tiling->blocks_per_core_base = 0;
        tiling->blocks_remainder = 0;
        context->SetTilingKey(0);
        context->SetBlockDim(1);
        OP_LOGI(nodeName, "No work to do, totalBlocks=0");
        return;
    }

    tiling->used_core_num = std::min(totalBlocks, N);
    tiling->blocks_per_core_base = totalBlocks / tiling->used_core_num;
    tiling->blocks_remainder = totalBlocks % tiling->used_core_num;

    context->SetTilingKey(0);
    context->SetBlockDim(tiling->used_core_num);
    context->SetScheduleMode(1);  // Batch mode
    
    OP_LOGI(nodeName, "Tiling: totalBlocks=%u, used_core_num=%u, blocks_per_core_base=%u, blocks_remainder=%u",
            totalBlocks, tiling->used_core_num, tiling->blocks_per_core_base, tiling->blocks_remainder);
}

ge::graphStatus Tiling4StemOamPrepVarlenQ(gert::TilingContext* context) {
    const char* nodeName = context->GetNodeName();
    int64_t stemBlockSize, stemStride, batch, numQHeads;

    if (ValidateInputs(context, stemBlockSize, stemStride, batch, numQHeads) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto compile_info = static_cast<const StemPrepQCompileInfo*>(context->GetCompileInfo());
    uint32_t N = static_cast<uint32_t>(compile_info->core_num);

    StemPrepQTilingData* tiling = context->GetTilingData<StemPrepQTilingData>();
    if (tiling == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "tiling");
        return ge::GRAPH_FAILED;
    }

    uint32_t B = static_cast<uint32_t>(stemBlockSize);
    uint32_t S = static_cast<uint32_t>(stemStride);

    tiling->stem_block_size = B;
    tiling->stem_stride = S;
    tiling->R_val = B / S;
    tiling->kflat_dim = S * DIM_QK_HOST;
    tiling->dim_qk = DIM_QK_HOST;
    tiling->batch_size = static_cast<uint32_t>(batch);
    tiling->num_q_heads = static_cast<uint32_t>(numQHeads);

    uint32_t totalBlocks = 0;
    if (CalcSequenceInfo(context, tiling, B, totalBlocks) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    CalcCoreDistribution(context, tiling, totalBlocks, N);

    // Calculate workspace size (this operator doesn't need workspace for computation)
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    size_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    if (workspaceSizes != nullptr) {
        workspaceSizes[0] = workspaceSize;
    }
    OP_LOGI(nodeName, "Workspace size: %zu bytes", workspaceSize);

    PrintTilingData(nodeName, tiling);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4StemOamPrepVarlenQ(gert::TilingParseContext* context) {
    auto compileInfo = context->GetCompiledInfo<StemPrepQCompileInfo>();
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->core_num = ascendcPlatform.GetCoreNumAiv();
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(StemOamPrepVarlenQ)
    .Tiling(Tiling4StemOamPrepVarlenQ)
    .TilingParse<StemPrepQCompileInfo>(TilingPrepare4StemOamPrepVarlenQ);

}  // namespace optiling
