/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sparse_compression_checker.h"
#include "log/log.h"

namespace optiling {
namespace sparse_mla_checker {
namespace {
const char *Op(const CheckContext &context)
{
    return context.opName == nullptr ? "SparseMla" : context.opName;
}
} // namespace

ge::graphStatus SparseCompressionChecker::CheckIndex(const CheckContext &context, const TensorParam &param,
                                                      const char *name) const
{
    if (!param.present) {
        return ge::GRAPH_SUCCESS;
    }
    const size_t rank = context.qLayout == Layout::BSND ? 4U : 3U;
    if (CheckTensorDesc(context, param, name, {ge::DT_INT32}) != ge::GRAPH_SUCCESS ||
        CheckRank(context, param, name, {rank}) != ge::GRAPH_SUCCESS ||
        CheckNoEmptyDim(context, param, name) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseCompressionChecker::CheckTopkLength(const CheckContext &context, const TensorParam &param,
                                                           const char *name) const
{
    if (!param.present) {
        return ge::GRAPH_SUCCESS;
    }
    const size_t rank = context.qLayout == Layout::BSND ? 3U : 2U;
    if (CheckTensorDesc(context, param, name, {ge::DT_INT32}) != ge::GRAPH_SUCCESS ||
        CheckRank(context, param, name, {rank}) != ge::GRAPH_SUCCESS ||
        CheckNoEmptyDim(context, param, name) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseCompressionChecker::CheckSinglePara(const CheckContext &context) const
{
    OP_CHECK_IF(context.cmpRatio <= 0, OP_LOGE(Op(context), "cmp_ratio must be greater than 0."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.topkValueMode != 1, OP_LOGE(Op(context), "topk_value_mode only supports 1."),
                return ge::GRAPH_FAILED);
    if (CheckIndex(context, context.oriSparseIndices, "ori_sparse_indices") != ge::GRAPH_SUCCESS ||
        CheckIndex(context, context.cmpSparseIndices, "cmp_sparse_indices") != ge::GRAPH_SUCCESS ||
        CheckTopkLength(context, context.oriTopkLength, "ori_topk_length") != ge::GRAPH_SUCCESS ||
        CheckTopkLength(context, context.cmpTopkLength, "cmp_topk_length") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseCompressionChecker::CheckParaExistence(const CheckContext &context) const
{
    if (!context.cmpKv.present) {
        OP_CHECK_IF(context.cmpSparseIndices.present || context.cmpTopkLength.present ||
                        context.cmpResidualKv.present || context.cmpBlockTable.present ||
                        context.cuSeqlensCmpKv.present || context.sequsedCmpKv.present,
                    OP_LOGE(Op(context), "cmp-side optional tensors require cmp_kv."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(context.cmpRatio != 1, OP_LOGE(Op(context), "SWA mode requires cmp_ratio=1."),
                    return ge::GRAPH_FAILED);
    }

    if (context.variant == OperatorVariant::SPARSE) {
        OP_CHECK_IF(context.oriSparseIndices.present || context.oriTopkLength.present || context.cmpTopkLength.present,
                    OP_LOGE(Op(context),
                            "ori_sparse_indices, ori_topk_length and cmp_topk_length are not supported currently."),
                    return ge::GRAPH_FAILED);
    } else {
        const bool needOriLength = context.oriSparseIndices.present && context.oriMaskMode == 0;
        OP_CHECK_IF(needOriLength != context.oriTopkLength.present,
                    OP_LOGE(Op(context),
                            "ori_topk_length must be present exactly when ori_sparse_indices is used with no mask."),
                    return ge::GRAPH_FAILED);
        const bool needCmpLength = context.cmpKv.present && context.cmpSparseIndices.present &&
                                   context.cmpMaskMode == 0;
        OP_CHECK_IF(needCmpLength != context.cmpTopkLength.present,
                    OP_LOGE(Op(context),
                            "cmp_topk_length must be present exactly when cmp_sparse_indices is used with no mask."),
                    return ge::GRAPH_FAILED);
    }

    const bool needResidual = context.cmpKv.present && context.cmpMaskMode == 3 &&
                              (context.variant == OperatorVariant::SPARSE || context.cmpRatio != 1);
    OP_CHECK_IF(needResidual && !context.cmpResidualKv.present,
                OP_LOGE(Op(context), "cmp_residual_kv is required for the current compressed causal mode."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseCompressionChecker::CheckFeature(const CheckContext &context) const
{
    OP_CHECK_IF(context.cmpSparseIndices.present && !context.cmpKv.present,
                OP_LOGE(Op(context), "cmp_sparse_indices requires cmp_kv."), return ge::GRAPH_FAILED);
    if (context.cmpSparseIndices.present) {
        OP_CHECK_IF(context.cmpTopk <= 0,
                    OP_LOGE(Op(context), "CSA cmp_sparse_indices last dimension must be greater than 0."),
                    return ge::GRAPH_FAILED);
        if (context.variant == OperatorVariant::SPARSE && !context.isAscend950) {
            OP_CHECK_IF(context.cmpTopk != 512 && context.cmpTopk != 1024,
                        OP_LOGE(Op(context), "On Atlas A2/A3, CSA topK only supports 512 or 1024."),
                        return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseCompressionChecker::CheckIndexShape(const CheckContext &context, const TensorParam &param,
                                                           const char *name) const
{
    if (!param.present) {
        return ge::GRAPH_SUCCESS;
    }
    if (context.qLayout == Layout::BSND) {
        OP_CHECK_IF(GetDim(param, 0) != GetDim(context.q, 0) || GetDim(param, 1) != GetDim(context.q, 1) ||
                        GetDim(param, 2) != 1,
                    OP_LOGE(Op(context), "%s shape must be (b, q_s, kv_n, topk).", name),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(GetDim(param, 0) != GetDim(context.q, 0) || GetDim(param, 1) != 1,
                    OP_LOGE(Op(context), "%s shape must be (q_t, kv_n, topk).", name),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseCompressionChecker::CheckTopkLengthShape(const CheckContext &context,
                                                                const TensorParam &param, const char *name) const
{
    if (!param.present) {
        return ge::GRAPH_SUCCESS;
    }
    if (context.qLayout == Layout::BSND) {
        OP_CHECK_IF(GetDim(param, 0) != GetDim(context.q, 0) || GetDim(param, 1) != GetDim(context.q, 1) ||
                        GetDim(param, 2) != 1,
                    OP_LOGE(Op(context), "%s shape must be (b, q_s, kv_n).", name),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(GetDim(param, 0) != GetDim(context.q, 0) || GetDim(param, 1) != 1,
                    OP_LOGE(Op(context), "%s shape must be (q_t, kv_n).", name),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseCompressionChecker::CheckMultiPara(const CheckContext &context) const
{
    if (CheckIndexShape(context, context.oriSparseIndices, "ori_sparse_indices") != ge::GRAPH_SUCCESS ||
        CheckIndexShape(context, context.cmpSparseIndices, "cmp_sparse_indices") != ge::GRAPH_SUCCESS ||
        CheckTopkLengthShape(context, context.oriTopkLength, "ori_topk_length") != ge::GRAPH_SUCCESS ||
        CheckTopkLengthShape(context, context.cmpTopkLength, "cmp_topk_length") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace sparse_mla_checker
} // namespace optiling
