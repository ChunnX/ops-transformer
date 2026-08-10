/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

/*!
 * \file matmul_ft_infershape.cpp
 * \brief MatmulFt shape and dtype inference.
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {
namespace {
constexpr size_t INPUT_A = 0;
constexpr size_t OUTPUT_COUNT = 18;
constexpr size_t OUTPUT_COMP_ROW = 4;
constexpr size_t OUTPUT_COMP_COL = 5;
constexpr size_t OUTPUT_C = 6;
constexpr size_t OUTPUT_B_MEAN_ABS = 8;
constexpr size_t OUTPUT_B_MEAN_SQUARE = 9;
constexpr size_t OUTPUT_B_VAR = 10;
constexpr size_t OUTPUT_BE = 11;
constexpr size_t OUTPUT_BE_FOR_AIV = 12;
constexpr size_t OUTPUT_B_MAX_SLICE = 13;
constexpr size_t OUTPUT_B_MIN_SLICE = 14;
constexpr size_t OUTPUT_A_MAX = 15;
constexpr size_t OUTPUT_A_MEAN = 16;
constexpr size_t OUTPUT_A_MIN = 17;

constexpr int64_t L1_TILE_N = 256;
constexpr int64_t COMP_PACK = 8;
constexpr int64_t FLOAT_ELEMENTS_PER_BLOCK = 8;

ge::graphStatus SetVectorShape(gert::InferShapeContext* context, size_t index, int64_t length)
{
    gert::Shape* shape = context->GetOutputShape(index);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape);
    shape->SetDimNum(1);
    shape->SetDim(0, length);
    return ge::GRAPH_SUCCESS;
}
} // namespace

static ge::graphStatus InferShapeMatmulFt(gert::InferShapeContext* context)
{
    const gert::Shape* aShape = context->GetInputShape(0);
    const gert::Shape* bShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, aShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, bShape);
    OP_CHECK_IF(aShape->GetDimNum() != 2 || bShape->GetDimNum() != 2,
        OP_LOGE(context->GetNodeName(), "MatmulFt expects rank-2 A and B"), return ge::GRAPH_FAILED);

    const int64_t m = aShape->GetDim(0);
    const int64_t k = aShape->GetDim(1);
    const int64_t bK = bShape->GetDim(0);
    const int64_t n = bShape->GetDim(1);
    OP_CHECK_IF(k != bK, OP_LOGE(context->GetNodeName(), "A.K must equal B.K"), return ge::GRAPH_FAILED);

    const int64_t splitN = (n + L1_TILE_N - 1) / L1_TILE_N;
    const int64_t splitBlockNum = (splitN + FLOAT_ELEMENTS_PER_BLOCK - 1) / FLOAT_ELEMENTS_PER_BLOCK;
    const int64_t bStatLength =
        splitBlockNum * FLOAT_ELEMENTS_PER_BLOCK + FLOAT_ELEMENTS_PER_BLOCK;
    const int64_t rowSplitLength = m * splitN;
    const int64_t beLength = k * splitN;

    for (size_t i = 0; i < OUTPUT_COUNT; ++i) {
        if (i == OUTPUT_C || i == OUTPUT_COMP_ROW || i == OUTPUT_COMP_COL ||
            (i >= OUTPUT_B_MEAN_ABS && i <= OUTPUT_B_MIN_SLICE) ||
            (i >= OUTPUT_A_MAX && i <= OUTPUT_A_MIN)) {
            continue;
        }
        OP_CHECK_IF(SetVectorShape(context, i, rowSplitLength) != ge::GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "failed to set output shape"), return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(SetVectorShape(context, OUTPUT_COMP_ROW, ((m + COMP_PACK - 1) / COMP_PACK) * splitN) !=
                    ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "failed to set comp_row shape"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(SetVectorShape(context, OUTPUT_COMP_COL, (n + COMP_PACK - 1) / COMP_PACK) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "failed to set comp_col shape"), return ge::GRAPH_FAILED);

    gert::Shape* cShape = context->GetOutputShape(OUTPUT_C);
    OP_CHECK_NULL_WITH_CONTEXT(context, cShape);
    cShape->SetDimNum(2);
    cShape->SetDim(0, m);
    cShape->SetDim(1, n);

    for (size_t i : {OUTPUT_B_MEAN_ABS, OUTPUT_B_MEAN_SQUARE, OUTPUT_B_VAR}) {
        OP_CHECK_IF(SetVectorShape(context, i, bStatLength) != ge::GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "failed to set B statistic shape"), return ge::GRAPH_FAILED);
    }
    for (size_t i : {OUTPUT_BE, OUTPUT_BE_FOR_AIV, OUTPUT_B_MAX_SLICE, OUTPUT_B_MIN_SLICE}) {
        OP_CHECK_IF(SetVectorShape(context, i, beLength) != ge::GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "failed to set BE shape"), return ge::GRAPH_FAILED);
    }
    for (size_t i : {OUTPUT_A_MAX, OUTPUT_A_MEAN, OUTPUT_A_MIN}) {
        OP_CHECK_IF(SetVectorShape(context, i, m) != ge::GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "failed to set A statistic shape"), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMatmulFt(gert::InferDataTypeContext* context)
{
    const ge::DataType precisionType = context->GetInputDataType(INPUT_A);
    for (size_t i = 0; i < OUTPUT_COUNT; ++i) {
        context->SetOutputDataType(i, ge::DT_FLOAT);
    }
    context->SetOutputDataType(OUTPUT_COMP_ROW, ge::DT_UINT8);
    context->SetOutputDataType(OUTPUT_COMP_COL, ge::DT_UINT8);
    context->SetOutputDataType(OUTPUT_BE, precisionType);
    context->SetOutputDataType(OUTPUT_BE_FOR_AIV,
        precisionType == ge::DT_FLOAT16 ? ge::DT_FLOAT16 : ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MatmulFt).InferShape(InferShapeMatmulFt).InferDataType(InferDataTypeMatmulFt);
} // namespace ops
