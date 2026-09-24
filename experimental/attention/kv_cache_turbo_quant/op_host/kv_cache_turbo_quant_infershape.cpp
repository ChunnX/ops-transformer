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
 * \file kv_cache_turbo_quant_infershape.cpp
 * \brief KvCacheTurboQuant shape/dtype 推导
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/log.h"
#include "../op_kernel/kv_cache_turbo_quant_common.h"

using namespace ge;

namespace ops {
constexpr uint32_t KV_VECTORS_INDEX = 0;
constexpr uint32_t IDX_OUTPUT_INDEX = 0;
constexpr uint32_t QJL_OUTPUT_INDEX = 1;
constexpr uint32_t NORM_OUTPUT_INDEX = 2;
constexpr uint32_t GAMMA_OUTPUT_INDEX = 3;
constexpr size_t DIMS_LIMIT = 3;
constexpr size_t ATTR_MSE_BITS_INDEX = 0;
constexpr size_t ATTR_OUT_DTYPE_INDEX = 1;
constexpr int64_t HEAD_DIM = KvCacheTurboQuant::HEAD_DIM;   // 128，与 kernel 公共头单点定义
constexpr int64_t QJL_BYTES = KvCacheTurboQuant::QJL_BYTES; // qjl_dim=128, 1 bit/channel -> 16 bytes

// 构建期 opbuild 执行 infer 函数时不提供 attrs：取不到时回落 def 默认值（3 / 0），
// 非法显式值仍由后面的校验拒绝（运行期 attrs 恒存在，不影响严格性）。
static int64_t GetMseBits(gert::InferShapeContext *context)
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return 3;
    }
    const int64_t *pBits = attrs->GetAttrPointer<int64_t>(ATTR_MSE_BITS_INDEX);
    return (pBits != nullptr) ? *pBits : 3;
}

template <typename Ctx>
static int64_t GetOutDtype(Ctx *context)
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return 0;
    }
    const int64_t *pDtype = attrs->GetAttrPointer<int64_t>(ATTR_OUT_DTYPE_INDEX);
    return (pDtype != nullptr) ? *pDtype : 0;
}

ge::graphStatus InferShapeKvCacheTurboQuant(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("KvCacheTurboQuant", "InferShapeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    const gert::Shape *kvShape = context->GetInputShape(KV_VECTORS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, kvShape);
    OP_CHECK_IF(kvShape->GetDimNum() != DIMS_LIMIT,
                OP_LOGE("KvCacheTurboQuant", "kv_vectors must be a 3D tensor [num_tokens, num_kv_heads, head_dim]."),
                return ge::GRAPH_FAILED);
    // -1 为动态维度（构建期 infer 可能以动态 shape 探测），仅对确定值做约束校验
    const int64_t headDim = kvShape->GetDim(2);
    OP_CHECK_IF(headDim != -1 && headDim != HEAD_DIM,
                OP_LOGE("KvCacheTurboQuant", "head_dim only supports 128 in the first version."),
                return ge::GRAPH_FAILED);

    const int64_t mseBits = GetMseBits(context);
    OP_CHECK_IF(mseBits != 2 && mseBits != 3 && mseBits != 4,
                OP_LOGE("KvCacheTurboQuant", "mse_bits only supports 2/3/4."), return ge::GRAPH_FAILED);
    const int64_t outDtype = GetOutDtype(context);
    OP_CHECK_IF(outDtype != 0 && outDtype != 1,
                OP_LOGE("KvCacheTurboQuant", "out_dtype only supports 0(bfloat16)/1(float16)."),
                return ge::GRAPH_FAILED);

    const int64_t numTokens = kvShape->GetDim(0);
    const int64_t numHeads = kvShape->GetDim(1);
    // num_tokens>=1、num_kv_heads∈[4,32] 的范围校验放在 tiling（kv_cache_turbo_quant_tiling.cpp）：
    // 构建期 opbuild 会以 mock shape 执行本函数，范围校验在构建期会误伤；此处仅推导 shape。
    const int64_t idxBytes = HEAD_DIM * mseBits / 8; // 128*b/8 = 32/48/64

    gert::Shape *idxShape = context->GetOutputShape(IDX_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, idxShape);
    idxShape->SetDimNum(DIMS_LIMIT);
    idxShape->SetDim(0, numTokens);
    idxShape->SetDim(1, numHeads);
    idxShape->SetDim(2, idxBytes);

    gert::Shape *qjlShape = context->GetOutputShape(QJL_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qjlShape);
    qjlShape->SetDimNum(DIMS_LIMIT);
    qjlShape->SetDim(0, numTokens);
    qjlShape->SetDim(1, numHeads);
    qjlShape->SetDim(2, QJL_BYTES);

    for (uint32_t out : {NORM_OUTPUT_INDEX, GAMMA_OUTPUT_INDEX}) {
        gert::Shape *shape = context->GetOutputShape(out);
        OP_CHECK_NULL_WITH_CONTEXT(context, shape);
        shape->SetDimNum(2);
        shape->SetDim(0, numTokens);
        shape->SetDim(1, numHeads);
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeKvCacheTurboQuant(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("KvCacheTurboQuant", "InferDataTypeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    const ge::DataType normDtype = (GetOutDtype(context) == 1) ? ge::DT_FLOAT16 : ge::DT_BF16;
    context->SetOutputDataType(IDX_OUTPUT_INDEX, ge::DT_UINT8);
    context->SetOutputDataType(QJL_OUTPUT_INDEX, ge::DT_UINT8);
    context->SetOutputDataType(NORM_OUTPUT_INDEX, normDtype);
    context->SetOutputDataType(GAMMA_OUTPUT_INDEX, normDtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(KvCacheTurboQuant).InferShape(InferShapeKvCacheTurboQuant).InferDataType(InferDataTypeKvCacheTurboQuant);
} // namespace ops
