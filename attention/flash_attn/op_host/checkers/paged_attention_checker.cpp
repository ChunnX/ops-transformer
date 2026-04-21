/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file paged_attention_checker.cpp
 * \brief Checker for PagedAttention parameters (文档约束: Paged Attention参数组)
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../flash_attn_tiling_constants.h"
#include "paged_attention_checker.h"

namespace optiling {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35FA;

ge::graphStatus PagedAttentionChecker::CheckSinglePara(const FaTilingInfo &faInfo)
{
    if (!faInfo.pageAttentionFlag) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(faInfo.blockSize > BLOCK_SIZE_MAX_FOR_NO_QUANT || faInfo.blockSize < BLOCK_SIZE_ALIGN_SIZE_16,
                OP_LOGE(faInfo.opName, "block_size only support [16, 1024], the current is %d.", faInfo.blockSize),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(faInfo.blockSize % BLOCK_SIZE_ALIGN_SIZE_16 != 0,
                OP_LOGE(faInfo.opName, "block_size must be 16 aligned, the current is %d.", faInfo.blockSize),
                return ge::GRAPH_FAILED);

    auto &blockTableTensor = faInfo.opParamInfo.blockTable.tensor;
    if (blockTableTensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const gert::CompileTimeTensorDesc *blockTableDesc = faInfo.opParamInfo.blockTable.desc;
    OP_CHECK_IF(blockTableDesc == nullptr, OP_LOGE(faInfo.opName, "blockTable desc is null pointer!"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(blockTableDesc->GetDataType() != ge::DT_INT32,
                OP_LOGE(faInfo.opName, "blockTable dtype must be INT32, but got %s",
                        DataTypeToSerialString(blockTableDesc->GetDataType()).c_str()),
                return ge::GRAPH_FAILED);

    uint32_t dimNum = blockTableTensor->GetStorageShape().GetDimNum();
    OP_CHECK_IF(dimNum != 2, OP_LOGE(faInfo.opName, "blockTable dim num must be 2, but got %u", dimNum),
                return ge::GRAPH_FAILED);

    for (uint32_t i = 0; i < dimNum; i++) {
        int64_t dim = blockTableTensor->GetStorageShape().GetDim(i);
        OP_CHECK_IF(dim <= 0, OP_LOGE(faInfo.opName, "blockTable dim %u value must be greater than 0, but got %ld", i, dim),
                    return ge::GRAPH_FAILED);
    }

    // TODO: tensor值校验需用户自行保证
    // 文档要求：block_table值只能为正整数、tensor_shape为(B, max_num_blocks_per_seq)
    // 由于tiling侧无法获取tensor的具体数值，此处不做值校验
    // 原代码已移除：
    // uint64_t blockTableSize = blockTableTensor->GetShapeSize();
    // for (uint64_t i = 0; i < blockTableSize; i++) {
    //     int64_t blockValue = blockTableTensor->GetData<int64_t>()[i];
    //     OP_CHECK_IF(blockValue <= 0,
    //                 OP_LOGE(faInfo.opName, "blockTable value[%lu](%ld) must be positive integer.", i, blockValue),
    //                 return ge::GRAPH_FAILED);
    // }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckParaExistence(const FaTilingInfo &faInfo)
{
    if (!faInfo.pageAttentionFlag) {
        return ge::GRAPH_SUCCESS;
    }

    auto &sequsedKvTensor = faInfo.opParamInfo.sequsedKv.tensor;
    OP_CHECK_IF(sequsedKvTensor == nullptr,
                OP_LOGE(faInfo.opName, "seqused_kv must be provided when PagedAttention is enabled."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckMultiPara(const FaTilingInfo &faInfo)
{
    if (!faInfo.pageAttentionFlag) {
        return ge::GRAPH_SUCCESS;
    }

    const std::vector<FaLayout> supportedKvLayouts = {FaLayout::PA_BBND, FaLayout::PA_BNBD, FaLayout::PA_Nz};
    OP_CHECK_IF(std::find(supportedKvLayouts.begin(), supportedKvLayouts.end(), faInfo.kvLayout) ==
                    supportedKvLayouts.end(),
                OP_LOGE(faInfo.opName,
                        "When PagedAttention is enabled, layout_kv must be one of PA_BBND/PA_BNBD/PA_Nz, but got %s",
                        LayoutToSerialString(faInfo.kvLayout).c_str()),
                return ge::GRAPH_FAILED);

    auto &blockTableTensor = faInfo.opParamInfo.blockTable.tensor;
    if (blockTableTensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    uint32_t dimNum = blockTableTensor->GetStorageShape().GetDimNum();
    OP_CHECK_IF(dimNum != 2, OP_LOGE(faInfo.opName, "blockTable dim num must be 2, but got %u", dimNum),
                return ge::GRAPH_FAILED);

    int64_t dim0 = blockTableTensor->GetStorageShape().GetDim(0);
    int64_t dim1 = blockTableTensor->GetStorageShape().GetDim(1);

    OP_CHECK_IF(
        dim0 != faInfo.bSize,
        OP_LOGE(faInfo.opName, "blockTable first dim(%ld) must be equal to batch size(%u).", dim0, faInfo.bSize),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(dim1 < faInfo.maxBlockNumPerBatch,
                OP_LOGE(faInfo.opName,
                        "blockTable second dim(%ld) must be greater than or equal to maxBlockNumPerBatch(%u).", dim1,
                        faInfo.maxBlockNumPerBatch),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling