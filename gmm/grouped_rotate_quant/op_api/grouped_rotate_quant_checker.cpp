/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "grouped_rotate_quant_checker.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include "aclnn_kernels/common/op_error_check.h"
#include "gmm/common/op_api/gmm_tensor_storage_check.h"
#include "opdev/format_utils.h"
#include "opdev/op_log.h"

namespace grq {
namespace {
constexpr char OP_NAME[] = "aclnnGroupedRotateQuantGetWorkspaceSize";
constexpr size_t MAX_X_RANK = 7;

aclnnStatus CheckTensor(const aclTensor *tensor, const char *name)
{
    CHECK_COND(tensor != nullptr, ACLNN_ERR_PARAM_NULLPTR, "%s: %s is null.", OP_NAME, name);
    CHECK_COND(tensor->GetStorageFormat() == op::Format::FORMAT_ND, ACLNN_ERR_PARAM_INVALID,
               "%s: %s must use ND storage.", OP_NAME, name);
    CHECK_COND(gmm::CheckTensorStorageBounds(tensor, OP_NAME, name), ACLNN_ERR_PARAM_INVALID,
               "%s: invalid storage for %s.", OP_NAME, name);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckList(const aclTensorList *list, const char *name)
{
    CHECK_COND(list != nullptr, ACLNN_ERR_PARAM_NULLPTR, "%s: %s list is null.", OP_NAME, name);
    CHECK_COND(list->Size() > 0 && list->Size() <= MAX_GROUP_NUM, ACLNN_ERR_PARAM_INVALID,
               "%s: %s list length must be in [1,1024].", OP_NAME, name);
    for (size_t i = 0; i < list->Size(); ++i) {
        const auto ret = CheckTensor((*list)[i], name);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGroupLayout(const Params &p, size_t &groupNum)
{
    CHECK_COND(p.groupType == NO_GROUP || p.groupType == GROUP_M, ACLNN_ERR_PARAM_INVALID,
               "%s: groupType must be -1 or 0; N/K grouping is unsupported.", OP_NAME);
    CHECK_COND(p.groupListType == 0 || p.groupListType == 1, ACLNN_ERR_PARAM_INVALID,
               "%s: groupListType must be 0 or 1.", OP_NAME);
    CHECK_COND(p.splitItem >= 0 && p.splitItem <= 3, ACLNN_ERR_PARAM_INVALID,
               "%s: splitItem must be in [0,3].", OP_NAME);
    groupNum = p.x->Size();
    if (p.groupType == NO_GROUP) {
        CHECK_COND(p.groupList == nullptr && p.splitItem < 2, ACLNN_ERR_PARAM_INVALID,
                   "%s: independent tensors require null groupList and splitItem 0/1.", OP_NAME);
    } else if (p.groupList != nullptr) {
        CHECK_RET(CheckTensor(p.groupList, "groupListOptional") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        const auto &shape = p.groupList->GetViewShape();
        CHECK_COND(p.groupList->GetDataType() == op::DataType::DT_INT64 && shape.GetDimNum() == 1,
                   ACLNN_ERR_PARAM_INVALID, "%s: groupList must be an INT64 vector.", OP_NAME);
        CHECK_COND(shape.GetDim(0) > 0 && shape.GetDim(0) <= static_cast<int64_t>(MAX_GROUP_NUM),
                   ACLNN_ERR_PARAM_INVALID, "%s: groupList length must be in [1,1024].", OP_NAME);
        groupNum = static_cast<size_t>(shape.GetDim(0));
    } else {
        CHECK_COND(p.x->Size() > 1, ACLNN_ERR_PARAM_NULLPTR,
                   "%s: a single M-grouped x requires groupListOptional.", OP_NAME);
    }
    CHECK_COND(p.x->Size() == 1 || p.x->Size() == groupNum, ACLNN_ERR_PARAM_INVALID,
               "%s: x list must contain one tensor or one tensor per group.", OP_NAME);
    CHECK_COND(p.rotation->Size() == 1 || p.rotation->Size() == groupNum, ACLNN_ERR_PARAM_INVALID,
               "%s: rotation list must contain one shared tensor or one tensor per group.", OP_NAME);
    const size_t outNum = p.splitItem < 2 ? groupNum : 1;
    CHECK_COND(p.y->Size() == outNum && p.scale->Size() == outNum, ACLNN_ERR_PARAM_INVALID,
               "%s: yOut and scaleOut list lengths must match splitItem and group count.", OP_NAME);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckQuantization(const Params &p)
{
    const auto xType = (*p.x)[0]->GetDataType();
    const auto yType = (*p.y)[0]->GetDataType();
    CHECK_COND(xType == op::DataType::DT_FLOAT16 || xType == op::DataType::DT_BF16, ACLNN_ERR_PARAM_INVALID,
               "%s: x must be FLOAT16 or BFLOAT16.", OP_NAME);
    const bool fp4 = yType == op::DataType::DT_FLOAT4_E2M1;
    const bool fp8 = yType == op::DataType::DT_FLOAT8_E4M3FN || yType == op::DataType::DT_FLOAT8_E5M2;
    CHECK_COND(fp4 || fp8, ACLNN_ERR_PARAM_INVALID, "%s: yOut must be MXFP4 or MXFP8.", OP_NAME);
    for (size_t i = 0; i < p.x->Size(); ++i) {
        CHECK_COND((*p.x)[i]->GetDataType() == xType, ACLNN_ERR_PARAM_INVALID,
                   "%s: all x tensors must use the same dtype.", OP_NAME);
    }
    for (size_t i = 0; i < p.rotation->Size(); ++i) {
        CHECK_COND((*p.rotation)[i]->GetDataType() == xType, ACLNN_ERR_PARAM_INVALID,
                   "%s: rotation dtype must match x.", OP_NAME);
    }
    for (size_t i = 0; i < p.y->Size(); ++i) {
        CHECK_COND((*p.y)[i]->GetDataType() == yType &&
                       (*p.scale)[i]->GetDataType() == op::DataType::DT_FLOAT8_E8M0,
                   ACLNN_ERR_PARAM_INVALID, "%s: inconsistent yOut dtype or non-E8M0 scaleOut.", OP_NAME);
    }
    const bool rint = std::strcmp(p.roundMode, "rint") == 0;
    CHECK_COND(rint || (fp4 && (std::strcmp(p.roundMode, "round") == 0 ||
                               std::strcmp(p.roundMode, "floor") == 0)),
               ACLNN_ERR_PARAM_INVALID, "%s: unsupported roundMode for the output dtype.", OP_NAME);
    CHECK_COND((fp4 && (p.scaleAlg == 0 || p.scaleAlg == 2)) ||
                   (fp8 && (p.scaleAlg == 0 || p.scaleAlg == 1)),
               ACLNN_ERR_PARAM_INVALID, "%s: unsupported scaleAlg for the output dtype.", OP_NAME);
    CHECK_COND(p.scaleAlg == 2 ? (p.dstTypeMax >= 6.0 && p.dstTypeMax <= 12.0) : p.dstTypeMax == 0.0,
               ACLNN_ERR_PARAM_INVALID, "%s: dstTypeMax must be [6,12] for scaleAlg=2, otherwise zero.", OP_NAME);
    CHECK_COND(!p.trans, ACLNN_ERR_PARAM_INVALID, "%s: output transpose is unsupported.", OP_NAME);
    if (p.alpha != nullptr) {
        const auto ret = CheckTensor(p.alpha, "alpha");
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        CHECK_COND(p.alpha->GetDataType() == op::DataType::DT_BF16 &&
                       p.alpha->GetViewShape() == op::Shape({1}),
                   ACLNN_ERR_PARAM_INVALID, "%s: alpha must be a BFLOAT16 tensor of shape [1].", OP_NAME);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckInputShapes(const Params &p, int64_t &totalM, int64_t &maxM)
{
    totalM = 0;
    maxM = 0;
    int64_t commonN = -1;
    for (size_t i = 0; i < p.x->Size(); ++i) {
        const auto &shape = (*p.x)[i]->GetViewShape();
        const auto rank = shape.GetDimNum();
        CHECK_COND(rank >= 1 && rank <= MAX_X_RANK, ACLNN_ERR_PARAM_INVALID,
                   "%s: x rank must be in [1,7].", OP_NAME);
        CHECK_COND(p.groupType != GROUP_M || rank == 2, ACLNN_ERR_PARAM_INVALID,
                   "%s: M grouping requires rank-2 x tensors.", OP_NAME);
        CHECK_COND(p.axis == -1 || p.axis == static_cast<int64_t>(rank) - 1, ACLNN_ERR_PARAM_INVALID,
                   "%s: axis must select the last dimension of every x tensor.", OP_NAME);
        const int64_t n = shape.GetDim(rank - 1);
        CHECK_COND(p.groupType != GROUP_M || commonN == -1 || commonN == n, ACLNN_ERR_PARAM_INVALID,
                   "%s: M-grouped x tensors must have the same N.", OP_NAME);
        commonN = n;
        int64_t m = 1;
        for (size_t dim = 0; dim + 1 < rank; ++dim) {
            const int64_t value = shape.GetDim(dim);
            CHECK_COND(value == 0 || m <= std::numeric_limits<int64_t>::max() / value,
                       ACLNN_ERR_PARAM_INVALID, "%s: flattened row count overflows int64.", OP_NAME);
            m *= value;
        }
        CHECK_COND(m <= std::numeric_limits<int64_t>::max() - totalM, ACLNN_ERR_PARAM_INVALID,
                   "%s: total row count overflows int64.", OP_NAME);
        totalM += m;
        maxM = std::max(maxM, m);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckRotationShapes(const Params &p, size_t groupNum)
{
    for (size_t i = 0; i < groupNum; ++i) {
        const auto &xShape = (*p.x)[p.x->Size() == 1 ? 0 : i]->GetViewShape();
        const auto &shape = (*p.rotation)[p.rotation->Size() == 1 ? 0 : i]->GetViewShape();
        const auto rank = shape.GetDimNum();
        CHECK_COND(rank == 2 || rank == 3, ACLNN_ERR_PARAM_INVALID,
                   "%s: each rotation must be [R,R] or [N/R,R,R].", OP_NAME);
        const int64_t r = shape.GetDim(rank - 1);
        const int64_t n = xShape.GetDim(xShape.GetDimNum() - 1);
        CHECK_COND(r == 32 || r == 64 || r == 128, ACLNN_ERR_PARAM_INVALID,
                   "%s: rotation block size R must be 32, 64 or 128.", OP_NAME);
        CHECK_COND(n % r == 0 && shape.GetDim(rank - 2) == r, ACLNN_ERR_PARAM_INVALID,
                   "%s: rotation must be square and N divisible by R.", OP_NAME);
        CHECK_COND(rank == 2 || shape.GetDim(0) == n / r, ACLNN_ERR_PARAM_INVALID,
                   "%s: rank-3 rotation must contain N/R column blocks.", OP_NAME);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckOutputShapes(const Params &p, int64_t totalM)
{
    int64_t outM = 0;
    for (size_t i = 0; i < p.y->Size(); ++i) {
        const auto &shape = (*p.y)[i]->GetViewShape();
        const auto rank = shape.GetDimNum();
        CHECK_COND(rank >= 1 && rank <= MAX_X_RANK, ACLNN_ERR_PARAM_INVALID,
                   "%s: yOut rank must be in [1,7].", OP_NAME);
        if (p.groupType == NO_GROUP) {
            CHECK_COND(shape == (*p.x)[i]->GetViewShape(), ACLNN_ERR_PARAM_INVALID,
                       "%s: independent yOut shape must equal its x shape.", OP_NAME);
        } else {
            CHECK_COND(rank == 2 && shape.GetDim(1) == (*p.x)[0]->GetViewShape().GetDim(1),
                       ACLNN_ERR_PARAM_INVALID, "%s: M-grouped yOut must have shape [M_i,N].", OP_NAME);
            if (p.x->Size() == p.y->Size()) {
                CHECK_COND(shape == (*p.x)[i]->GetViewShape(), ACLNN_ERR_PARAM_INVALID,
                           "%s: separate x and yOut must have matching per-group shapes.", OP_NAME);
            }
            CHECK_COND(shape.GetDim(0) <= std::numeric_limits<int64_t>::max() - outM,
                       ACLNN_ERR_PARAM_INVALID, "%s: output row count overflows int64.", OP_NAME);
            outM += shape.GetDim(0);
        }
        op::Shape expectedScale = shape;
        const int64_t n = shape.GetDim(rank - 1);
        expectedScale.SetDim(rank - 1, n / MX_SCALE_BLOCK + (n % MX_SCALE_BLOCK != 0));
        expectedScale.AppendDim(2);
        CHECK_COND((*p.scale)[i]->GetViewShape() == expectedScale, ACLNN_ERR_PARAM_INVALID,
                   "%s: scaleOut must replace N by ceil(N/64) and append 2.", OP_NAME);
    }
    CHECK_COND(p.groupType != GROUP_M || totalM == outM, ACLNN_ERR_PARAM_INVALID,
               "%s: input and output total row counts differ.", OP_NAME);
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus CheckParams(Params &p)
{
    auto ret = CheckList(p.x, "x");
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckList(p.rotation, "rotation");
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckList(p.y, "yOut");
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckList(p.scale, "scaleOut");
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    size_t groupNum = 0;
    ret = CheckGroupLayout(p, groupNum);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    p.roundMode = p.roundMode == nullptr ? "rint" : p.roundMode;
    ret = CheckQuantization(p);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    int64_t totalM = 0;
    int64_t maxM = 0;
    ret = CheckInputShapes(p, totalM, maxM);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckRotationShapes(p, groupNum);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckOutputShapes(p, totalM);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (p.tuningConfig != nullptr && p.tuningConfig->Size() > 0) {
        const int64_t expectedM = (*p.tuningConfig)[0];
        CHECK_COND(expectedM >= 0 && expectedM <= maxM, ACLNN_ERR_PARAM_INVALID,
                   "%s: tuningConfig[0] must be in [0,max(x[i].M)].", OP_NAME);
    }
    p.splitItem = p.splitItem < 2 ? (p.x->Size() == p.y->Size() ? 0 : 1) : (p.x->Size() == 1 ? 3 : 2);
    p.axis = -1;
    return ACLNN_SUCCESS;
}
} // namespace grq
