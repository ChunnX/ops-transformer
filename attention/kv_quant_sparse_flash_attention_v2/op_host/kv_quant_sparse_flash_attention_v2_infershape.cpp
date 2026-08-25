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
 * \file kv_quant_sparse_flash_attention_v2_infershape.cpp
 * \brief
 */

#include "kv_quant_sparse_flash_attention_v2_infershape.h"

namespace ops {

ge::graphStatus InferShapeKvQuantSparseFlashAttentionV2(gert::InferShapeContext *context)
{
    return InferShapeKvQuantSparseFlashAttention(context);
}

ge::graphStatus InferDataTypeKvQuantSparseFlashAttentionV2(gert::InferDataTypeContext *context)
{
    return InferDataTypeKvQuantSparseFlashAttention(context);
}

IMPL_OP_INFERSHAPE(KvQuantSparseFlashAttentionV2)
    .InferShape(InferShapeKvQuantSparseFlashAttentionV2)
    .InferDataType(InferDataTypeKvQuantSparseFlashAttentionV2);

} // namespace ops
