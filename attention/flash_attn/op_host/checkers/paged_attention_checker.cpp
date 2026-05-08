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
namespace flash_attn {
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
    // 这里和存在性校验冲突了，但是为了在单参数校验校验dtype需要先判空
    OP_CHECK_IF(blockTableTensor == nullptr, OP_LOGE(faInfo.opName, "block_table tensor is null pointer!"),
                return ge::GRAPH_FAILED);

    const gert::CompileTimeTensorDesc *blockTableDesc = faInfo.opParamInfo.blockTable.desc;
    OP_CHECK_IF(blockTableDesc == nullptr, OP_LOGE(faInfo.opName, "block_table desc is null pointer!"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(blockTableDesc->GetDataType() != ge::DT_INT32,
                OP_LOGE(faInfo.opName, "block_table dtype must be INT32, but got %s",
                        DataTypeToSerialString(blockTableDesc->GetDataType()).c_str()),
                return ge::GRAPH_FAILED);

    if (ge::GRAPH_SUCCESS != CheckFormatSupport(blockTableDesc, BLOCK_TABLE_NAME)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t dimNum = blockTableTensor->GetStorageShape().GetDimNum();
    OP_CHECK_IF(dimNum != 2, OP_LOGE(faInfo.opName, "block_table dim num must be 2, but got %u", dimNum),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckParaExistence(const FaTilingInfo &faInfo)
{
    if (!faInfo.pageAttentionFlag) {
        return ge::GRAPH_SUCCESS;
    }

    auto &blockTableTensor = faInfo.opParamInfo.blockTable.tensor;
    OP_CHECK_IF(blockTableTensor == nullptr,
                OP_LOGE(faInfo.opName, "block_table must be provided when PagedAttention is enabled."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckMultiPara(const FaTilingInfo &faInfo)
{
    if (!faInfo.pageAttentionFlag) {
        return ge::GRAPH_SUCCESS;
    }

    auto &blockTableTensor = faInfo.opParamInfo.blockTable.tensor;
    int64_t dim0 = blockTableTensor->GetStorageShape().GetDim(0);

    OP_CHECK_IF(
        dim0 != faInfo.bSize,
        OP_LOGE(faInfo.opName, "block_table first dim(%ld) must be equal to batch size(%ld).", dim0, faInfo.bSize),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

} // namespace flash_attn
} // namespace optiling