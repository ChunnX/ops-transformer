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
 * \file quant_block_sparse_attention_check.cpp
 * \brief QuantBlockSparseAttention parameter validation implementation.
 */

#include "quant_block_sparse_attention_check.h"
#include "quant_block_sparse_attention_tiling.h"

namespace optiling {

QuantBlockSparseAttentionCheck::QuantBlockSparseAttentionCheck(const QuantBlockSparseAttentionTilingInfo &tilingInfo)
    : tilingInfo_(tilingInfo)
{
}

ge::graphStatus QuantBlockSparseAttentionCheck::CheckDtype() const
{
    if (tilingInfo_.qDtype != ge::DT_FLOAT8_E4M3FN && tilingInfo_.qDtype != ge::DT_HIFLOAT8) {
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo_.kvDtype != ge::DT_FLOAT8_E4M3FN && tilingInfo_.kvDtype != ge::DT_HIFLOAT8) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttentionCheck::CheckBlockSize() const
{
    if (tilingInfo_.qBlockSizeVal != BSA_BLOCK_SIZE || tilingInfo_.kvBlockSizeVal != BSA_BLOCK_SIZE) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttentionCheck::CheckShapeConsistency() const
{
    if (tilingInfo_.bSize == 0U || tilingInfo_.n1Size == 0U || tilingInfo_.n2Size == 0U || tilingInfo_.gSize == 0U) {
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo_.n1Size % tilingInfo_.n2Size != 0U) {
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo_.s1Size == 0U || tilingInfo_.s2Size == 0U) {
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo_.qbMax == 0U || tilingInfo_.kbMax == 0U) {
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo_.dSize == 0U || tilingInfo_.dSizeV == 0U) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttentionCheck::CheckMaskMode() const
{
    if (tilingInfo_.maskModeVal > 4U) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttentionCheck::Process()
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
