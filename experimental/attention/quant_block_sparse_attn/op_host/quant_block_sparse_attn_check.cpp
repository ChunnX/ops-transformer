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
 * \file quant_block_sparse_attn_check.cpp
 * \brief QuantBlockSparseAttn parameter validation implementation.
 */

#include "quant_block_sparse_attn_check.h"
#include "quant_block_sparse_attn_tiling.h"
#include "log/log.h"

namespace optiling {
namespace {
constexpr const char *kOpName = "QuantBlockSparseAttn";
} // namespace

QuantBlockSparseAttnCheck::QuantBlockSparseAttnCheck(const QuantBlockSparseAttnTilingInfo &tilingInfo)
    : tilingInfo_(tilingInfo)
{
}

ge::graphStatus QuantBlockSparseAttnCheck::CheckDtype() const
{
    if (tilingInfo_.qDtype != ge::DT_FLOAT8_E4M3FN) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "query", std::to_string(static_cast<int>(tilingInfo_.qDtype)),
                                  "FLOAT8_E4M3FN");
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo_.kvDtype != ge::DT_FLOAT8_E4M3FN) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "key/value", std::to_string(static_cast<int>(tilingInfo_.kvDtype)),
                                  "FLOAT8_E4M3FN");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnCheck::CheckBlockSize() const
{
    if (tilingInfo_.qBlockSizeVal != BSA_BLOCK_SIZE || tilingInfo_.kvBlockSizeVal != BSA_BLOCK_SIZE) {
        OP_LOGE_FOR_INVALID_VALUE(kOpName, "q_block_size/kv_block_size",
                                  std::to_string(tilingInfo_.qBlockSizeVal) + "/" +
                                      std::to_string(tilingInfo_.kvBlockSizeVal),
                                  std::to_string(BSA_BLOCK_SIZE));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnCheck::CheckShapeConsistency() const
{
    if (tilingInfo_.bSize == 0U || tilingInfo_.n1Size == 0U || tilingInfo_.n2Size == 0U || tilingInfo_.gSize == 0U) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "bSize/n1Size/n2Size/gSize",
            std::to_string(tilingInfo_.bSize) + "/" + std::to_string(tilingInfo_.n1Size) + "/" +
                std::to_string(tilingInfo_.n2Size) + "/" + std::to_string(tilingInfo_.gSize),
            "all dimensions must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo_.n1Size % tilingInfo_.n2Size != 0U) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "n1Size (query head num) ", std::to_string(tilingInfo_.n1Size),
                                              "must be divisible by n2Size (kv head num) " +
                                                  std::to_string(tilingInfo_.n2Size));
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo_.s1Size == 0U || tilingInfo_.s2Size == 0U) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "s1Size/s2Size", std::to_string(tilingInfo_.s1Size) + "/" + std::to_string(tilingInfo_.s2Size),
            "sequence lengths must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo_.qbMax == 0U || tilingInfo_.kbMax == 0U) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "qbMax/kbMax", std::to_string(tilingInfo_.qbMax) + "/" + std::to_string(tilingInfo_.kbMax),
            "block counts must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo_.dSize == 0U || tilingInfo_.dSizeV == 0U) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "dSize/dSizeV", std::to_string(tilingInfo_.dSize) + "/" + std::to_string(tilingInfo_.dSizeV),
            "head dimensions must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnCheck::CheckMaskMode() const
{
    if (tilingInfo_.maskModeVal > 4U) {
        OP_LOGE_WITH_INVALID_ATTR(kOpName, "mask_mode", std::to_string(tilingInfo_.maskModeVal), "0, 1, 2, 3 or 4");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnCheck::Process()
{
    if (CheckDtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckShapeConsistency() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckMaskMode() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
