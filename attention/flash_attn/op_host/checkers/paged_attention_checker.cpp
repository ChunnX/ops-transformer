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

    auto &blockTableTensor = faInfo.opParamInfo.blockTable.tensor;
    if (blockTableTensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const gert::CompileTimeTensorDesc *blockTableDesc = faInfo.opParamInfo.blockTable.desc;
    OP_CHECK_IF(blockTableDesc == nullptr,
        OP_LOGE(faInfo.opName, "blockTable desc is null pointer!"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(blockTableDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(faInfo.opName, "blockTable dtype must be INT32, but got %s",
                DataTypeToSerialString(blockTableDesc->GetDataType()).c_str()),
        return ge::GRAPH_FAILED);

    uint32_t dimNum = blockTableTensor->GetStorageShape().GetDimNum();
    OP_CHECK_IF(dimNum != 2,
        OP_LOGE(faInfo.opName, "blockTable dim num must be 2, but got %u", dimNum),
        return ge::GRAPH_FAILED);

    for (uint32_t i = 0; i < dimNum; i++) {
        int64_t dim = blockTableTensor->GetStorageShape().GetDim(i);
        OP_CHECK_IF(dim <= 0,
            OP_LOGE(faInfo.opName, "blockTable dim %u value cannot be 0, but got %ld", i, dim),
            return ge::GRAPH_FAILED);
    }

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

    auto &blockTableTensor = faInfo.opParamInfo.blockTable.tensor;
    if (blockTableTensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    uint32_t dimNum = blockTableTensor->GetStorageShape().GetDimNum();
    OP_CHECK_IF(dimNum != 2,
        OP_LOGE(faInfo.opName, "blockTable dim num must be 2, but got %u", dimNum),
        return ge::GRAPH_FAILED);

    int64_t dim0 = blockTableTensor->GetStorageShape().GetDim(0);
    int64_t dim1 = blockTableTensor->GetStorageShape().GetDim(1);

    OP_CHECK_IF(dim0 != faInfo.bSize,
        OP_LOGE(faInfo.opName, "blockTable first dim(%ld) must be equal to batch size(%u), but got %ld",
                    dim0, faInfo.bSize),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(dim1 < faInfo.maxBlockNumPerBatch,
        OP_LOGE(faInfo.opName, "blockTable second dim(%ld) must be greater than or equal to maxBlockNumPerBatch(%u), but got %ld",
                    dim1, faInfo.maxBlockNumPerBatch),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling