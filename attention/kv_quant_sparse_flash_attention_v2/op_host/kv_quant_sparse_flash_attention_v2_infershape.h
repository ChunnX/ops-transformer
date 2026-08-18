/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KV_QUANT_SPARSE_FLASH_ATTENTION_V2_INFERSHAPE_H
#define KV_QUANT_SPARSE_FLASH_ATTENTION_V2_INFERSHAPE_H

#include <register/op_impl_registry.h>

namespace ops {

ge::graphStatus InferShapeKvQuantSparseFlashAttention(gert::InferShapeContext *context);
ge::graphStatus InferDataTypeKvQuantSparseFlashAttention(gert::InferDataTypeContext *context);

ge::graphStatus InferShapeKvQuantSparseFlashAttentionV2(gert::InferShapeContext *context);
ge::graphStatus InferDataTypeKvQuantSparseFlashAttentionV2(gert::InferDataTypeContext *context);

} // namespace ops

#endif // KV_QUANT_SPARSE_FLASH_ATTENTION_V2_INFERSHAPE_H
