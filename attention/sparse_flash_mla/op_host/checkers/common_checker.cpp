/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common_checker.h"
#include <algorithm>
#include <cmath>
#include <vector>
#include "log/log.h"

namespace optiling {
namespace sparse_mla_checker {
namespace {
const char *Op(const CheckContext &context)
{
    return context.opName == nullptr ? "SparseMla" : context.opName;
}
} // namespace

ge::graphStatus CommonChecker::CheckQuery(const CheckContext &context) const
{
    if (!context.q.present) {
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus status = ge::GRAPH_FAILED;
    if (context.variant == OperatorVariant::SPARSE) {
        status = CheckTensorDesc(context, context.q, "q", {ge::DT_FLOAT16, ge::DT_BF16});
    } else if (context.variant == OperatorVariant::MIXED_QUANT) {
        status = CheckTensorDesc(context, context.q, "q", {ge::DT_BF16});
    } else {
        status = CheckTensorDesc(context, context.q, "q", {ge::DT_HIFLOAT8});
    }
    if (status != ge::GRAPH_SUCCESS || CheckNoEmptyDim(context, context.q, "q") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const size_t expectedRank = context.qLayout == Layout::BSND ? 4U : 3U;
    return CheckRank(context, context.q, "q", {expectedRank});
}

ge::graphStatus CommonChecker::CheckKv(const CheckContext &context, const TensorParam &kv, const char *name) const
{
    if (!kv.present) {
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus status = ge::GRAPH_FAILED;
    if (context.variant == OperatorVariant::SPARSE) {
        status = CheckTensorDesc(context, kv, name, {ge::DT_FLOAT16, ge::DT_BF16});
    } else if (context.variant == OperatorVariant::MIXED_QUANT) {
        status = CheckTensorDesc(context, kv, name, {ge::DT_FLOAT8_E4M3FN});
    } else {
        status = CheckTensorDesc(context, kv, name, {ge::DT_HIFLOAT8});
    }
    if (status != ge::GRAPH_SUCCESS || CheckNoEmptyDim(context, kv, name) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const size_t expectedRank = context.kvLayout == Layout::TND ? 3U : 4U;
    return CheckRank(context, kv, name, {expectedRank});
}

ge::graphStatus CommonChecker::CheckOutput(const CheckContext &context) const
{
    if (!context.attentionOut.present) {
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus status = ge::GRAPH_FAILED;
    if (context.variant == OperatorVariant::SPARSE) {
        status = CheckTensorDesc(context, context.attentionOut, "attention_out", {ge::DT_FLOAT16, ge::DT_BF16});
    } else {
        status = CheckTensorDesc(context, context.attentionOut, "attention_out", {ge::DT_BF16});
    }
    if (status != ge::GRAPH_SUCCESS ||
        CheckNoEmptyDim(context, context.attentionOut, "attention_out") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const size_t expectedRank = context.qLayout == Layout::BSND ? 4U : 3U;
    return CheckRank(context, context.attentionOut, "attention_out", {expectedRank});
}

ge::graphStatus CommonChecker::CheckSinglePara(const CheckContext &context) const
{
    OP_CHECK_IF(context.qLayout != Layout::BSND && context.qLayout != Layout::TND,
                OP_LOGE(Op(context), "layout_q only supports BSND and TND."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.kvLayout != Layout::BSND && context.kvLayout != Layout::TND &&
                    context.kvLayout != Layout::PA_BBND,
                OP_LOGE(Op(context), "layout_kv only supports BSND, TND and PA_BBND."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!std::isfinite(context.softmaxScale), OP_LOGE(Op(context), "softmax_scale must be finite."),
                return ge::GRAPH_FAILED);
    if (CheckQuery(context) != ge::GRAPH_SUCCESS || CheckKv(context, context.oriKv, "ori_kv") != ge::GRAPH_SUCCESS ||
        CheckKv(context, context.cmpKv, "cmp_kv") != ge::GRAPH_SUCCESS ||
        CheckOutput(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckParaExistence(const CheckContext &context) const
{
    OP_CHECK_IF(!context.q.present, OP_LOGE(Op(context), "q is required."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!context.oriKv.present, OP_LOGE(Op(context), "ori_kv is required in all supported modes."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!context.attentionOut.present, OP_LOGE(Op(context), "attention_out is required."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckFeature(const CheckContext &context) const
{
    OP_CHECK_IF(context.kvLayout != Layout::PA_BBND && context.qLayout != context.kvLayout,
                OP_LOGE(Op(context), "Non-PA layout_q and layout_kv must be the same."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckQueryAxes(const CheckContext &context) const
{
    const int64_t qSeq = GetDim(context.q, 0);
    int64_t qHeads = -1;
    int64_t qDim = -1;
    if (context.qLayout == Layout::BSND) {
        OP_CHECK_IF(GetDim(context.q, 0) <= 0 || GetDim(context.q, 1) <= 0,
                    OP_LOGE(Op(context), "q batch and sequence dimensions must be greater than 0."),
                    return ge::GRAPH_FAILED);
        qHeads = GetDim(context.q, 2);
        qDim = GetDim(context.q, 3);
    } else {
        OP_CHECK_IF(qSeq <= 0, OP_LOGE(Op(context), "q_t must be greater than 0."), return ge::GRAPH_FAILED);
        qHeads = GetDim(context.q, 1);
        qDim = GetDim(context.q, 2);
    }
    OP_CHECK_IF(qHeads <= 0 || qHeads > 128,
                OP_LOGE(Op(context), "q_n must be within [1, 128], but got %ld.", qHeads),
                return ge::GRAPH_FAILED);
    if (context.variant == OperatorVariant::SPARSE && !context.isAscend950) {
        const std::vector<int64_t> supported = {1, 2, 4, 8, 16, 32, 64, 128};
        OP_CHECK_IF(std::find(supported.begin(), supported.end(), qHeads) == supported.end(),
                    OP_LOGE(Op(context), "On Atlas A2/A3, q_n must be a power of two in [1, 128]."),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(qDim != 512, OP_LOGE(Op(context), "q head dimension must be 512, but got %ld.", qDim),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckKvAxes(const CheckContext &context, const TensorParam &kv,
                                           const char *name) const
{
    if (!kv.present) {
        return ge::GRAPH_SUCCESS;
    }
    int64_t numHeads = -1;
    int64_t headDim = -1;
    if (context.kvLayout == Layout::TND) {
        numHeads = GetDim(kv, 1);
        headDim = GetDim(kv, 2);
    } else {
        numHeads = GetDim(kv, 2);
        headDim = GetDim(kv, 3);
    }
    OP_CHECK_IF(numHeads != 1, OP_LOGE(Op(context), "%s kv_n must be 1, but got %ld.", name, numHeads),
                return ge::GRAPH_FAILED);
    int64_t expectedDim = 512;
    if (context.variant == OperatorVariant::MIXED_QUANT) {
        expectedDim = context.quantMode == 1 ? 608 : 584;
    }
    OP_CHECK_IF(headDim != expectedDim,
                OP_LOGE(Op(context), "%s head dimension must be %ld, but got %ld.", name, expectedDim, headDim),
                return ge::GRAPH_FAILED);
    if (context.kvLayout == Layout::PA_BBND) {
        const int64_t blockSize = GetDim(kv, 1);
        OP_CHECK_IF(blockSize <= 0 || blockSize > 1024,
                    OP_LOGE(Op(context), "%s block size must be within [1, 1024].", name),
                    return ge::GRAPH_FAILED);
        if (context.variant == OperatorVariant::SPARSE && !context.isAscend950) {
            OP_CHECK_IF(blockSize % 16 != 0,
                        OP_LOGE(Op(context), "%s block size must be 16-aligned on Atlas A2/A3.", name),
                        return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckMultiPara(const CheckContext &context) const
{
    if (CheckSameShape(context, context.q, "q", context.attentionOut, "attention_out") != ge::GRAPH_SUCCESS ||
        CheckQueryAxes(context) != ge::GRAPH_SUCCESS || CheckKvAxes(context, context.oriKv, "ori_kv") !=
            ge::GRAPH_SUCCESS || CheckKvAxes(context, context.cmpKv, "cmp_kv") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(context.variant == OperatorVariant::SPARSE &&
                    context.q.desc->GetDataType() != context.attentionOut.desc->GetDataType(),
                OP_LOGE(Op(context), "q and attention_out dtype must be the same."), return ge::GRAPH_FAILED);
    if (context.variant == OperatorVariant::SPARSE) {
        OP_CHECK_IF(context.q.desc->GetDataType() != context.oriKv.desc->GetDataType(),
                    OP_LOGE(Op(context), "q and ori_kv dtype must be the same."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(context.cmpKv.present && context.q.desc->GetDataType() != context.cmpKv.desc->GetDataType(),
                    OP_LOGE(Op(context), "q and cmp_kv dtype must be the same."), return ge::GRAPH_FAILED);
    }
    if (context.kvLayout == Layout::BSND) {
        const int64_t qBatch = GetDim(context.q, 0);
        OP_CHECK_IF(GetDim(context.oriKv, 0) != qBatch,
                    OP_LOGE(Op(context), "ori_kv batch must equal q batch."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(context.cmpKv.present && GetDim(context.cmpKv, 0) != qBatch,
                    OP_LOGE(Op(context), "cmp_kv batch must equal q batch."), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace sparse_mla_checker
} // namespace optiling
