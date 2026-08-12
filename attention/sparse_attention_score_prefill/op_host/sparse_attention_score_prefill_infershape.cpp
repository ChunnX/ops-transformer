/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

/*
 * 本文件实现 SparseAttentionScorePrefill 算子的形状推断（InferShape）与数据类型推断（InferDataType）。
 * 这些回调由 GE 图编译期调用，用于在构图阶段确定输出 tensor 的形状与数据类型。
 */

#include <register/op_impl_registry.h>
#include "log/log.h"

using namespace ge;

namespace ops {

// 算子输入/输出的索引常量，便于在推断逻辑中按索引获取形状与数据类型。
static constexpr uint32_t QUERY_INDEX = 0;            // query 输入索引
static constexpr uint32_t ATTENTION_OUT_INDEX = 0;    // attentionOut 输出索引
static constexpr uint32_t TND_DIM_NUM = 3;            // TND 布局的维度数（T, N, D）
static constexpr int32_t UNKNOWN_DIMS = -2;           // 未知维度标记，用于动态形状场景

// InferShapeSparseAttentionScorePrefill: 输出形状推断函数。
// 推断规则：输出 attentionOut 的形状等于输入 query 的形状（TND 布局: [total_q_tokens, num_q_heads, D]）。
// 特殊处理：当 query 为未知维度（UNKNOWN_DIMS, 即 -2）时，输出也设为单维未知，用于图编译期的动态形状场景。
// 校验：query 必须为 3 维（TND_DIM_NUM），否则返回 GRAPH_FAILED。
static ge::graphStatus InferShapeSparseAttentionScorePrefill(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("SparseAttentionScorePrefill", "context is nullptr!");
        return ge::GRAPH_FAILED;
    }

    // 获取输入 query 的形状，作为输出形状推断的依据。
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);

    // 获取输出 attentionOut 的形状对象，待填充。
    gert::Shape *attentionOutShape = context->GetOutputShape(ATTENTION_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attentionOutShape);

    // 动态形状场景：query 整体未知时，输出同样标记为未知维度，直接返回成功。
    if (queryShape->GetDimNum() == 1 && queryShape->GetDim(0) == UNKNOWN_DIMS) {
        attentionOutShape->SetDimNum(1);
        (*attentionOutShape)[0] = UNKNOWN_DIMS;
        return ge::GRAPH_SUCCESS;
    }

    // 校验 query 必须为 TND 三维布局，否则报错。
    if (queryShape->GetDimNum() != TND_DIM_NUM) {
        OP_LOGE(context->GetNodeName(),
                "SparseAttentionScorePrefill only supports TND layout, queryDims(%zu) must be 3!",
                queryShape->GetDimNum());
        return ge::GRAPH_FAILED;
    }

    // 输出形状直接等于 query 形状（TND）。
    *attentionOutShape = *queryShape;
    return ge::GRAPH_SUCCESS;
}

// InferDataTypeSparseAttentionScorePrefill: 输出数据类型推断函数。
// 推断规则：输出 attentionOut 的数据类型等于输入 query 的数据类型（bf16）。
static ge::graphStatus InferDataTypeSparseAttentionScorePrefill(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // 取 query 的数据类型并赋给输出 attentionOut。
    auto dtype = context->GetInputDataType(QUERY_INDEX);
    context->SetOutputDataType(ATTENTION_OUT_INDEX, dtype);
    return ge::GRAPH_SUCCESS;
}

// 注册算子的 InferShape 与 InferDataType 回调到 GE。
IMPL_OP_INFERSHAPE(SparseAttentionScorePrefill)
    .InferShape(InferShapeSparseAttentionScorePrefill)
    .InferDataType(InferDataTypeSparseAttentionScorePrefill);

}  // namespace ops
