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
 * \file mat_mul_v3_infer.cpp
 * \brief
 */

#include <limits>
#include "common/error_util.h"
#include "common/log.h"
#include "register/op_impl_registry.h"

using namespace gert;
namespace {
constexpr size_t MATMUL_MIN_SHAPE_SIZE = 2;
constexpr size_t MATMUL_MAX_SHAPE_SIZE = 4;
constexpr size_t MATMUL_SLICE_SHAPE_SIZE = 3;
constexpr size_t MATMUL_BIAS_IDX = 2;
constexpr int64_t BLOCK_SIZE = 16;
constexpr int64_t UNKNOWN_DIM = -1;
constexpr int64_t UNKNOWN_DIM_NUM = -2;

void InferComplementedInput(
    Shape& shape_x1_new, Shape& shape_x2_new, bool& shape_x1_reshape_flag, bool& shape_x2_reshape_flag)
{
    if (shape_x1_new.GetDimNum() == 1 && shape_x1_new.GetDim(0) > 0) {
        shape_x1_reshape_flag = true;
        int64_t ori_dim = shape_x1_new.GetDim(0);
        shape_x1_new.SetDimNum(MATMUL_MIN_SHAPE_SIZE);
        shape_x1_new.SetDim(0, 1);
        shape_x1_new.SetDim(1, ori_dim);
    }

    if (shape_x2_new.GetDimNum() == 1 && shape_x2_new.GetDim(0) > 0) {
        shape_x2_reshape_flag = true;
        int64_t ori_dim = shape_x2_new.GetDim(0);
        shape_x2_new.SetDimNum(MATMUL_MIN_SHAPE_SIZE);
        shape_x2_new.SetDim(0, ori_dim);
        shape_x2_new.SetDim(1, 1);
    }
}

static void InferComplementedOutput(bool shape_x1_reshape_flag, bool shape_x2_reshape_flag, Shape& shape_out)
{
    size_t dim_num = shape_out.GetDimNum();
    if (dim_num >= MATMUL_MIN_SHAPE_SIZE) {
        if (shape_x1_reshape_flag && !shape_x2_reshape_flag) {
            shape_out.SetDim(dim_num - MATMUL_MIN_SHAPE_SIZE, shape_out.GetDim(dim_num - 1));
            shape_out.SetDimNum(dim_num - 1);
        }
        if (!shape_x1_reshape_flag && shape_x2_reshape_flag) {
            shape_out.SetDimNum(dim_num - 1);
        }
    }
}

bool CheckIsUnknownDimNum(const gert::Shape& shape)
{
    return shape.GetDimNum() == 1 && shape.GetDim(0) == UNKNOWN_DIM_NUM;
}

void UpdateUnknowDimNumToUnkownRank(gert::Shape& shape)
{
    if (CheckIsUnknownDimNum(shape)) {
        shape.SetDimNum(MATMUL_MIN_SHAPE_SIZE);
        shape.SetDim(0, UNKNOWN_DIM);
        shape.SetDim(1, UNKNOWN_DIM);
    }
}

bool UpdateOutputShapeByBias(const std::string& op_name, gert::Shape* shape_out, const gert::Shape* shape_bias)
{
    if (shape_bias != nullptr && shape_bias->GetDimNum() > 0) {
        Shape shape_bias_new(*shape_bias);
        UpdateUnknowDimNumToUnkownRank(shape_bias_new);
        int64_t bias_dim = shape_bias_new.GetDimNum();
        if (shape_bias_new.GetDim(bias_dim - 1) != UNKNOWN_DIM && shape_out->GetDim(1) != UNKNOWN_DIM) {
            OP_CHECK_IF(
                shape_bias_new.GetDim(bias_dim - 1) != shape_out->GetDim(1),
                OP_LOGE(
                    op_name, "The dimension of n [%ld] and bias [%ld] tensors must be the same", shape_out->GetDim(1),
                    shape_bias_new.GetDim(bias_dim - 1)),
                return false);
        }
        if (shape_bias_new.GetDim(bias_dim - 1) != UNKNOWN_DIM && shape_out->GetDim(1) == UNKNOWN_DIM) {
            shape_out->SetDim(1, shape_bias_new.GetDim(bias_dim - 1));
        }
    }
    return true;
}

static ge::graphStatus InferShapeForMatMulV3(InferShapeContext* context)
{
    auto op_name = context->GetNodeName();
    auto shape_x1 = context->GetInputShape(0);
    auto shape_x2 = context->GetInputShape(1);
    auto shape_out = context->GetOutputShape(0);
    auto tensor_x1 = context->GetInputDesc(0);
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(
        shape_x1 == nullptr || shape_x2 == nullptr || shape_out == nullptr || tensor_x1 == nullptr || attrs == nullptr,
        CUBE_INNER_ERR_REPORT(op_name, "shape or attrs is null"), return ge::GRAPH_FAILED);
    auto shape_bias = context->GetOptionalInputShape(MATMUL_BIAS_IDX);
    if (CheckIsUnknownDimNum(*shape_x1) && CheckIsUnknownDimNum(*shape_x2) && (shape_bias == nullptr || CheckIsUnknownDimNum(*shape_bias))) {
        shape_out->SetDimNum(1);
        shape_out->SetDim(0, UNKNOWN_DIM_NUM);
        return ge::GRAPH_SUCCESS;
    }

    const bool* trans_a = attrs->GetAttrPointer<bool>(0);
    const bool* trans_b = attrs->GetAttrPointer<bool>(1);
    OP_CHECK_IF(
        trans_a == nullptr || trans_b == nullptr, CUBE_INNER_ERR_REPORT(op_name, "attribute is null"),
        return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "transpose_x1: %d, transpose_x2: %d", *trans_a, *trans_b);

    ge::DataType dtype = tensor_x1->GetDataType();
    if (dtype == ge::DT_FLOAT) {
        OP_LOGW(op_name, "%s fp32 op has poor performance!", context->GetNodeName());
    }

    Shape shape_x1_new(*shape_x1);
    Shape shape_x2_new(*shape_x2);
    UpdateUnknowDimNumToUnkownRank(shape_x1_new);
    UpdateUnknowDimNumToUnkownRank(shape_x2_new);
    bool shape_x1_reshape_flag = false;
    bool shape_x2_reshape_flag = false;

    InferComplementedInput(shape_x1_new, shape_x2_new, shape_x1_reshape_flag, shape_x2_reshape_flag);

    OP_LOGD(op_name, "check the input shape length.");
    size_t dimNumX1 = shape_x1_new.GetDimNum();
    if (dimNumX1 < MATMUL_MIN_SHAPE_SIZE ||  dimNumX1 > MATMUL_MAX_SHAPE_SIZE) {
        CUBE_INNER_ERR_REPORT(op_name, "first input dim num[%zu] is not between [2, 4]!", shape_x1_new.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    size_t idx_m = 0;
    size_t idx_k_a = 1;
    size_t idx_k_b = 0;
    size_t idx_n = 1;

    if (*trans_a) {
        idx_m = 1;
        idx_k_a = 0;
    } else if (dimNumX1 == MATMUL_SLICE_SHAPE_SIZE) {
        idx_k_a = 2;
        idx_m = 1;
    }

    if (*trans_b) {
        idx_k_b = 1;
        idx_n = 0;
    }
    auto m = dimNumX1 == MATMUL_SLICE_SHAPE_SIZE ? shape_x1_new.GetDim(idx_m) * shape_x1_new.GetDim(0) :
                                                   shape_x1_new.GetDim(idx_m);
    auto n = shape_x2_new.GetDim(idx_n);
    if (shape_x1_new.GetDim(idx_k_a) != UNKNOWN_DIM && shape_x2_new.GetDim(idx_k_b) != UNKNOWN_DIM) {
        OP_CHECK_IF(
            shape_x1_new.GetDim(idx_k_a) != shape_x2_new.GetDim(idx_k_b),
            CUBE_INNER_ERR_REPORT(
                op_name, "The k-axis of a(%ld) and b(%ld) tensors must be the same", shape_x1_new.GetDim(idx_k_a),
                shape_x2_new.GetDim(idx_k_b)),
            return ge::GRAPH_FAILED);
    }

    shape_out->SetDimNum(MATMUL_MIN_SHAPE_SIZE);
    shape_out->SetDim(0, m);
    shape_out->SetDim(1, n);
    if (!UpdateOutputShapeByBias(op_name, shape_out, shape_bias)) {
        return ge::GRAPH_FAILED;
    }
    InferComplementedOutput(shape_x1_reshape_flag, shape_x2_reshape_flag, *shape_out);
    OP_LOGD(op_name, "end infershape.");
    return ge::GRAPH_SUCCESS;
}

} // namespace
namespace ops {
    IMPL_OP_INFERSHAPE(MatMulV3).InferShape(InferShapeForMatMulV3);
}
