/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kv_quant_sparse_flash_attention_infershape.h
 * \brief KvQuantSparseFlashAttention 系列算子 infershape 共享定义
 */
#ifndef KV_QUANT_SPARSE_FLASH_ATTENTION_INFERSHAPE_H
#define KV_QUANT_SPARSE_FLASH_ATTENTION_INFERSHAPE_H

#include <string>
#include <graph/utils/type_utils.h>
#include <exe_graph/runtime/infer_shape_context.h>

namespace ops {
// Inputs Index
constexpr size_t QSFA_QUERY_INPUT_INDEX = 0;
// Attributes Index
constexpr uint32_t QSFA_LAYOUT_QUERY_ATTR_INDEX = 4;
constexpr uint32_t QSFA_ROPE_HEAD_DIM_ATTR_INDEX = 12;
// Dim Index
constexpr uint32_t QSFA_DIM_INDEX_0 = 0;
constexpr uint32_t QSFA_DIM_INDEX_1 = 1;
constexpr uint32_t QSFA_DIM_INDEX_2 = 2;
constexpr uint32_t QSFA_DIM_INDEX_3 = 3;
constexpr uint32_t QSFA_DIM_NUM_3 = 3;
constexpr uint32_t QSFA_DIM_NUM_4 = 4;

// 设置attention_out的shape: query shape去掉rope_head_dim维度
inline void SetQSFAAttentionOutShape(const gert::Shape *queryShape, gert::Shape *attentionOutShape,
                                     const std::string &inputLayoutQueryPtrStr, const int64_t ropeHeadDim)
{
    *attentionOutShape = *queryShape;
    if (inputLayoutQueryPtrStr == "BSND") {
        attentionOutShape->SetDimNum(QSFA_DIM_NUM_4);
        attentionOutShape->SetDim(QSFA_DIM_INDEX_0, queryShape->GetDim(QSFA_DIM_INDEX_0));
        attentionOutShape->SetDim(QSFA_DIM_INDEX_1, queryShape->GetDim(QSFA_DIM_INDEX_1));
        attentionOutShape->SetDim(QSFA_DIM_INDEX_2, queryShape->GetDim(QSFA_DIM_INDEX_2));
        if (queryShape->GetDim(QSFA_DIM_INDEX_3) != -1) {
            attentionOutShape->SetDim(QSFA_DIM_INDEX_3, queryShape->GetDim(QSFA_DIM_INDEX_3) - ropeHeadDim);
        }
    } else { // TND
        attentionOutShape->SetDimNum(QSFA_DIM_NUM_3);
        attentionOutShape->SetDim(QSFA_DIM_INDEX_0, queryShape->GetDim(QSFA_DIM_INDEX_0));
        attentionOutShape->SetDim(QSFA_DIM_INDEX_1, queryShape->GetDim(QSFA_DIM_INDEX_1));
        if (queryShape->GetDim(QSFA_DIM_INDEX_2) != -1) {
            attentionOutShape->SetDim(QSFA_DIM_INDEX_2, queryShape->GetDim(QSFA_DIM_INDEX_2) - ropeHeadDim);
        }
    }
}
} // namespace ops
#endif // KV_QUANT_SPARSE_FLASH_ATTENTION_INFERSHAPE_H
