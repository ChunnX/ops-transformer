/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "base_checker.h"
#include <algorithm>
#include "log/log.h"

namespace optiling {
namespace sparse_mla_checker {
namespace {
const char *GetOpName(const CheckContext &context)
{
    return context.opName == nullptr ? "SparseMla" : context.opName;
}
} // namespace

ge::graphStatus BaseChecker::CheckSinglePara(const CheckContext &context) const
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BaseChecker::CheckParaExistence(const CheckContext &context) const
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BaseChecker::CheckFeature(const CheckContext &context) const
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BaseChecker::CheckMultiPara(const CheckContext &context) const
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BaseChecker::CheckTensorDesc(const CheckContext &context, const TensorParam &param, const char *name,
                                             std::initializer_list<ge::DataType> dtypes) const
{
    if (!param.present) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(param.desc == nullptr, OP_LOGE(GetOpName(context), "%s desc is null.", name),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(param.desc->GetOriginFormat() != ge::FORMAT_ND,
                OP_LOGE(GetOpName(context), "%s format must be ND, but got %d.", name,
                        static_cast<int32_t>(param.desc->GetOriginFormat())),
                return ge::GRAPH_FAILED);
    const ge::DataType actual = param.desc->GetDataType();
    OP_CHECK_IF(std::find(dtypes.begin(), dtypes.end(), actual) == dtypes.end(),
                OP_LOGE(GetOpName(context), "%s dtype %d is not supported.", name, static_cast<int32_t>(actual)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BaseChecker::CheckRank(const CheckContext &context, const TensorParam &param, const char *name,
                                       std::initializer_list<size_t> ranks) const
{
    if (!param.present) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(param.shape == nullptr, OP_LOGE(GetOpName(context), "%s shape is null.", name),
                return ge::GRAPH_FAILED);
    const size_t actual = param.shape->GetDimNum();
    OP_CHECK_IF(std::find(ranks.begin(), ranks.end(), actual) == ranks.end(),
                OP_LOGE(GetOpName(context), "%s rank %zu is not supported.", name, actual),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BaseChecker::CheckNoEmptyDim(const CheckContext &context, const TensorParam &param,
                                             const char *name) const
{
    if (!param.present) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(param.shape == nullptr, OP_LOGE(GetOpName(context), "%s shape is null.", name),
                return ge::GRAPH_FAILED);
    for (size_t i = 0; i < param.shape->GetDimNum(); ++i) {
        OP_CHECK_IF(param.shape->GetDim(i) <= 0,
                    OP_LOGE(GetOpName(context), "%s dim %zu must be greater than 0, but got %ld.", name, i,
                            param.shape->GetDim(i)),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BaseChecker::CheckShape(const CheckContext &context, const TensorParam &param, const char *name,
                                        std::initializer_list<int64_t> expected) const
{
    if (!param.present) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(param.shape == nullptr || param.shape->GetDimNum() != expected.size(),
                OP_LOGE(GetOpName(context), "%s shape rank does not match the documented shape.", name),
                return ge::GRAPH_FAILED);
    size_t index = 0;
    for (const int64_t value : expected) {
        OP_CHECK_IF(param.shape->GetDim(index) != value,
                    OP_LOGE(GetOpName(context), "%s dim %zu must be %ld, but got %ld.", name, index, value,
                            param.shape->GetDim(index)),
                    return ge::GRAPH_FAILED);
        ++index;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BaseChecker::CheckSameShape(const CheckContext &context, const TensorParam &left,
                                            const char *leftName, const TensorParam &right,
                                            const char *rightName) const
{
    OP_CHECK_IF(left.shape == nullptr || right.shape == nullptr,
                OP_LOGE(GetOpName(context), "%s or %s shape is null.", leftName, rightName),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(left.shape->GetDimNum() != right.shape->GetDimNum(),
                OP_LOGE(GetOpName(context), "%s and %s rank must be the same.", leftName, rightName),
                return ge::GRAPH_FAILED);
    for (size_t i = 0; i < left.shape->GetDimNum(); ++i) {
        OP_CHECK_IF(left.shape->GetDim(i) != right.shape->GetDim(i),
                    OP_LOGE(GetOpName(context), "%s and %s dim %zu must be the same.", leftName, rightName, i),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

int64_t BaseChecker::GetDim(const TensorParam &param, size_t index) const
{
    if (param.shape == nullptr || index >= param.shape->GetDimNum()) {
        return -1;
    }
    return param.shape->GetDim(index);
}

size_t BaseChecker::GetRank(const TensorParam &param) const
{
    return param.shape == nullptr ? 0 : param.shape->GetDimNum();
}

bool BaseChecker::HasCmpKv(const CheckContext &context) const
{
    return context.cmpKv.present;
}

bool BaseChecker::IsAllSparse(const CheckContext &context) const
{
    return context.oriSparseIndices.present && context.cmpSparseIndices.present;
}

} // namespace sparse_mla_checker
} // namespace optiling
