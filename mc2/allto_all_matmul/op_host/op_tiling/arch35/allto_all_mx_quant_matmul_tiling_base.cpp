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
 * \file allto_all_mx_quant_matmul_tiling_base.cpp
 * \brief
 */
#include "allto_all_mx_quant_matmul_tiling_base.h"

#include "../allto_all_matmul_tiling_base.h"
#include "common/utils/mc2_comm_utils.h"
#include "common/utils/op_mc2.h"
#include "mc2/matmul_allto_all/op_host/op_tiling/common/matmul_allto_all_util_tiling.h"
#include "mc2_log.h"

#include <cstdio>

using namespace Mc2Log;
using namespace AscendC;
using namespace Mc2Tiling;

namespace MC2Tiling {
/**
 * @brief AlltoAllMatmul MX量化的准入条件
 * 后续支持更多量化再进行修改
 *
 * @return true
 */
bool AllToAllMxQuantMatmulTilingBase::IsCapable()
{
    int64_t x2QuantMode = 0;
    int64_t x1QuantMode = 0;
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    if (const int64_t *ptr = attrs->GetAttrPointer<int64_t>(ATTR_X1_QUANTMODE_INDEX)) {
        x1QuantMode = *ptr;
    }
    if (const int64_t *ptr = attrs->GetAttrPointer<int64_t>(ATTR_X2_QUANTMODE_INDEX)) {
        x2QuantMode = *ptr;
    }
    if (x1QuantMode == X1_QUANTMODE_VALUES && x2QuantMode == X2_QUANTMODE_VALUES) {
        OP_LOGI(opName_, "Start with AlltoAllMxQuantMatmul tiling.");
        return true;
    }
    OP_LOGI(opName_, "Skip MxQuantMatmulAllToAll tiling when not MX_QUANT.");
    return false;
}

/**
 * @brief 校验输入信息是否合规:attr,Dtype,shape等，使用通用校验util中的check方法
 *
 * @return ge::graphStatus
 */
ge::graphStatus AllToAllMxQuantMatmulTilingBase::CheckOpInputInfo()
{
    OP_TILING_CHECK(MatmulAlltoAllTilingUtil::CheckAttrsInfo(context_, opName_, ALLTOALL_MATMUL_INDEX_SCHEMA) !=
                        ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check Attrs failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckMxTensorFormat(context_, opName_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check format failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckX2Transpose(context_, opName_, ALLTOALL_MATMUL_INDEX_SCHEMA) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check x2transpose failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckMatrixMulShapes(context_, opName_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check shape input and output shape failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckMxQuantTensorDataType(context_, opName_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check Dtype failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckMxQuantShapeInfo(context_, opName_, ALLTOALL_MATMUL_INDEX_SCHEMA) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check shape failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckAlltoAllOut(context_, opName_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check allToAllOut failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckGroupSize(context_, opName_, ALLTOALL_MATMUL_INDEX_SCHEMA) == ge::GRAPH_FAILED,
                    OP_LOGE(opName_, "Check groupsize failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验参数的format::是否为私有格式
 *
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName 算子名称
 * @return
 */
ge::graphStatus AllToAllMxQuantMatmulTilingBase::CheckMxTensorFormat(const gert::TilingContext *context,
                                                                     const char *opName)
{
    OP_TILING_CHECK(MatmulAlltoAllTilingUtil::CheckTensorFormat(context_, opName_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check format failed."), return ge::GRAPH_FAILED);
    auto x1ScaleTensorDesc = context->GetOptionalInputDesc(INPUT_X1_SCALE_INDEX);
    OP_TILING_CHECK((x1ScaleTensorDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "x1Scale"),
                    return ge::GRAPH_FAILED);
    ge::Format x1ScaleFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(x1ScaleTensorDesc->GetStorageFormat()));
    OP_TILING_CHECK(x1ScaleFormat != ge::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(opName, "x1Scale", Ops::Base::ToString(x1ScaleFormat).c_str(), "ND"),
                    return ge::GRAPH_FAILED);
    auto x2ScaleTensorDesc = context->GetOptionalInputDesc(INPUT_X2_SCALE_INDEX);
    OP_TILING_CHECK((x2ScaleTensorDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "x2Scale"),
                    return ge::GRAPH_FAILED);
    ge::Format x2ScaleFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(x2ScaleTensorDesc->GetStorageFormat()));
    OP_TILING_CHECK(x2ScaleFormat != ge::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(opName, "x2Scale", Ops::Base::ToString(x2ScaleFormat).c_str(), "ND"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 量化场景校验x2transpose是否一定为true
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName  算子名称
 * @param indexSchema 存放输入参数索引差别的结构体
 * @return
 */
ge::graphStatus AllToAllMxQuantMatmulTilingBase::CheckX2Transpose(const gert::TilingContext *context,
                                                                  const char *opName,
                                                                  const OpAttrIndexSchema &indexSchema)
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const bool *isTransX2 = attrs->GetAttrPointer<bool>(indexSchema.x2Transpose);
    OP_TILING_CHECK(isTransX2 == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName, "isTransX2"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!(*isTransX2), OP_LOGE_FOR_INVALID_VALUE(opName, "x2Transpose", "false", "true"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 量化场景校验groupsize
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName  算子名称
 * @param indexSchema 存放输入参数索引差别的结构体
 * @return
 */
ge::graphStatus AllToAllMxQuantMatmulTilingBase::CheckGroupSize(const gert::TilingContext *context, const char *opName,
                                                                const OpAttrIndexSchema &indexSchema)
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const int64_t *groupSizePtr = attrs->GetAttrPointer<int64_t>(ALLTOALLMATMUL_ATTR_GROUP_SIZE_INDEX);
    OP_TILING_CHECK(groupSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName, "groupSize"), return ge::GRAPH_FAILED);
    uint64_t groupSize = static_cast<uint64_t>(*groupSizePtr);
    OP_LOGI(opName, "groupSize=%lu", groupSize);
    mc2tiling::Mc2MatmulShapeInfo shapeInfo = {context_->GetInputShape(INPUT_X1_INDEX),
                                               context_->GetInputShape(INPUT_X2_INDEX),
                                               context_->GetOptionalInputShape(INPUT_X1_SCALE_INDEX),
                                               context_->GetOptionalInputShape(INPUT_X2_SCALE_INDEX),
                                               false,
                                               *context_->GetAttrs()->GetAttrPointer<bool>(indexSchema.x2Transpose),
                                               opName};
    uint64_t groupSizeK = groupSize & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeN = (groupSize >> GROUP_N_OFFSET) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeM = (groupSize >> GROUP_M_OFFSET) & GROUP_MNK_BIT_SIZE;
    shapeInfo.isMxfp = true;
    OP_TILING_CHECK(!mc2tiling::Mc2TilingUtils::InferGroupSize(shapeInfo, groupSizeM, groupSizeN, groupSizeK),
                    OP_LOGE(opName, "Failed to execute inferGroupSize in mx scene."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (groupSizeM != MX_SCALE_BLOCK_M) || (groupSizeN != MX_SCALE_BLOCK_N) || (groupSizeK != MX_SCALE_BLOCK_K),
        OP_LOGE_WITH_INVALID_ATTR(opName, "groupSize",
                                  (std::string("[groupSizeM=") + std::to_string(groupSizeM) + ", groupSizeN=" +
                                   std::to_string(groupSizeN) + ", groupSizeK=" + std::to_string(groupSizeK) + "]")
                                      .c_str(),
                                  "[groupSizeM=1, groupSizeN=1, groupSizeK=32]"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 工具函数：判断指定value是否存在于list中
 *
 * @param list: 有效值列表
 * @param value: 给定值
 * @return
 */
static bool IsContain(const std::vector<uint32_t> &list, uint32_t value)
{
    return std::count(list.begin(), list.end(), value) > 0;
}

/**
 * @brief 量化场景校验参数的DType
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName  算子名称
 * @return ge::graphStatus
 */
ge::graphStatus AllToAllMxQuantMatmulTilingBase::CheckMxQuantTensorDataType(const gert::TilingContext *context,
                                                                            const char *opName)
{
    // 获取并校验输入张量描述符
    auto x1TensorDesc = context->GetInputDesc(INPUT_X1_INDEX);
    auto x2TensorDesc = context->GetInputDesc(INPUT_X2_INDEX);
    OP_TILING_CHECK((x1TensorDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "x1"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((x2TensorDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "x2"), return ge::GRAPH_FAILED);
    // 获取数据类型并校验一致性与范围
    ge::DataType x1Dtype = x1TensorDesc->GetDataType();
    ge::DataType x2Dtype = x2TensorDesc->GetDataType();
    const std::vector<uint32_t> MX_QUANT_X_DTYPE_LIST = {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT4_E2M1};
    const std::vector<uint32_t> MX_QUANT_Y_DTYPE_LIST = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT};
    OP_TILING_CHECK(!IsContain(MX_QUANT_X_DTYPE_LIST, x1Dtype),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        opName, "x1", Ops::Base::ToString(x1Dtype).c_str(),
                        "The dtype of x1 must be within the range {float8_e4m3fn, float8_e5m2, float4_e2m1}."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!IsContain(MX_QUANT_X_DTYPE_LIST, x2Dtype),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        opName, "x2", Ops::Base::ToString(x2Dtype).c_str(),
                        "The dtype of x2 must be within the range {float8_e4m3fn, float8_e5m2, float4_e2m1}."),
                    return ge::GRAPH_FAILED);
    // 校验 bias 数据类型（如果存在）
    auto biasTensorDesc = context->GetOptionalInputDesc(INPUT_BIAS_INDEX);
    if (biasTensorDesc != nullptr) {
        ge::DataType biasDtype = biasTensorDesc->GetDataType();
        OP_TILING_CHECK((biasDtype != ge::DT_FLOAT),
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "bias", Ops::Base::ToString(biasDtype).c_str(),
                                                              "The dtype of bias must be FLOAT."),
                        return ge::GRAPH_FAILED);
    }
    auto x1ScaleTensorDesc = context->GetOptionalInputDesc(INPUT_X1_SCALE_INDEX);
    OP_TILING_CHECK((x1ScaleTensorDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "x1Scale"),
                    return ge::GRAPH_FAILED);
    ge::DataType x1scaleDtype = x1ScaleTensorDesc->GetDataType();
    OP_TILING_CHECK((x1scaleDtype != ge::DT_FLOAT8_E8M0),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "x1Scale", Ops::Base::ToString(x1scaleDtype).c_str(),
                                                          "The dtype of x1Scale must be DT_FLOAT8_E8M0."),
                    return ge::GRAPH_FAILED);
    auto x2ScaleTensorDesc = context->GetOptionalInputDesc(INPUT_X2_SCALE_INDEX);
    OP_TILING_CHECK((x2ScaleTensorDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "x2Scale"),
                    return ge::GRAPH_FAILED);
    ge::DataType x2scaleDtype = x2ScaleTensorDesc->GetDataType();
    OP_TILING_CHECK((x2scaleDtype != ge::DT_FLOAT8_E8M0),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "x2Scale", Ops::Base::ToString(x2scaleDtype).c_str(),
                                                          "The dtype of x2Scale must be DT_FLOAT8_E8M0."),
                    return ge::GRAPH_FAILED);
    // 校验输出张量数据类型
    auto yDesc = context->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_TILING_CHECK((yDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "y"), return ge::GRAPH_FAILED);
    ge::DataType yDtype = yDesc->GetDataType();
    OP_TILING_CHECK(
        !IsContain(MX_QUANT_Y_DTYPE_LIST, yDtype),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "y", Ops::Base::ToString(yDtype).c_str(),
                                              "The dtype of y must be within the range {float16, bfloat16, float}."),
        return ge::GRAPH_FAILED);
    if (x1Dtype == ge::DT_FLOAT4_E2M1 || x2Dtype == ge::DT_FLOAT4_E2M1) {
        isMxFp4_ = true;
        matmulQuantType_ = QuantType::MXFP4_QUANT;
        OP_TILING_CHECK((x1Dtype != x2Dtype),
                        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                            opName, "x1 and x2",
                            (Ops::Base::ToString(x1Dtype) + " and " + Ops::Base::ToString(x2Dtype)).c_str(),
                            "If mxfp4 quant mode is used, the dtypes of x1 and x2 must be float4_e2m1."),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验量化tiling输入的shape信息
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName 算子名称
 * @param indexSchema 存放输入参数索引差别的结构体
 * @return ge::graphStatus
 */
ge::graphStatus AllToAllMxQuantMatmulTilingBase::CheckMxQuantShapeInfo(const gert::TilingContext *context,
                                                                       const char *opName,
                                                                       const OpAttrIndexSchema &indexSchema)
{
    OP_TILING_CHECK(MatmulAlltoAllTilingUtil::CheckShapeInfo(context, opName, ALLTOALL_MATMUL_INDEX_SCHEMA) !=
                        ge::GRAPH_SUCCESS,
                    OP_LOGE(opName, "Tiling common info check shape failed."), return ge::GRAPH_FAILED);
    const gert::StorageShape *x1Shape = context->GetInputShape(INPUT_X1_INDEX);
    const gert::StorageShape *x2Shape = context->GetInputShape(INPUT_X2_INDEX);
    const gert::StorageShape *x1ScaleShape = context->GetOptionalInputShape(INPUT_X1_SCALE_INDEX);
    const gert::StorageShape *x2ScaleShape = context->GetOptionalInputShape(INPUT_X2_SCALE_INDEX);
    OP_TILING_CHECK((x1ScaleShape == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "x1Scale"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((x2ScaleShape == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "x2Scale"), return ge::GRAPH_FAILED);
    uint64_t x1Dim0 = x1Shape->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t x1Dim1 = x1Shape->GetStorageShape().GetDim(DIM_ONE);
    uint64_t x2Dim0 = x2Shape->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t x2Dim1 = x2Shape->GetStorageShape().GetDim(DIM_ONE);
    uint64_t x1ScaleDimNum = x1ScaleShape->GetStorageShape().GetDimNum();
    uint64_t x2ScaleDimNum = x2ScaleShape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK((x1Dim1 % MX_SCALE_ALIGN != 0),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "x1",
                                                          Ops::Base::ToString(x1Shape->GetStorageShape()).c_str(),
                                                          "The dim(k) of x1 must be divisible by 64."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((x2Dim1 % MX_SCALE_ALIGN != 0),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "x2",
                                                          Ops::Base::ToString(x2Shape->GetStorageShape()).c_str(),
                                                          "The dim(k) of x2 must be divisible by 64."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((x1ScaleDimNum != DIM_THREE),
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName, "x1Scale",
                                                             (std::to_string(x1ScaleDimNum) + "D").c_str(),
                                                             "The shape dim of x1Scale must be 3D."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((x2ScaleDimNum != DIM_THREE),
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName, "x2Scale",
                                                             (std::to_string(x2ScaleDimNum) + "D").c_str(),
                                                             "The shape dim of x2Scale must be 3D."),
                    return ge::GRAPH_FAILED);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const char *group = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    int64_t rankDim = 0;
    if (MatmulAlltoAllTilingUtil::GetAndValidateRankSize(context, opName, group, rankDim) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    uint64_t x1ScaleDim0 = x1ScaleShape->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t x1ScaleDim1 = x1ScaleShape->GetStorageShape().GetDim(DIM_ONE);
    uint64_t x1ScaleDim2 = x1ScaleShape->GetStorageShape().GetDim(DIM_TWO);
    OP_TILING_CHECK((x1ScaleDim0 != x1Dim0),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "x1Scale",
                                                          Ops::Base::ToString(x1ScaleShape->GetStorageShape()).c_str(),
                                                          "The shape [0] of x1Scale must be " + std::to_string(x1Dim0)),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((x1ScaleDim1 != x1Dim1 / MX_SCALE_ALIGN),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName, "x1Scale", Ops::Base::ToString(x1ScaleShape->GetStorageShape()).c_str(),
                        "The shape [1] of x1Scale must be " + std::to_string(x1Dim1 / MX_SCALE_ALIGN)),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((x1ScaleDim2 != DIM_TWO),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName, "x1Scale", Ops::Base::ToString(x1ScaleShape->GetStorageShape()).c_str(),
                        "The shape [2] of x1Scale must be " + std::to_string(DIM_TWO)),
                    return ge::GRAPH_FAILED);
    uint64_t x2ScaleDim0 = x2ScaleShape->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t x2ScaleDim1 = x2ScaleShape->GetStorageShape().GetDim(DIM_ONE);
    uint64_t x2ScaleDim2 = x2ScaleShape->GetStorageShape().GetDim(DIM_TWO);
    OP_TILING_CHECK((x2ScaleDim2 != DIM_TWO),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName, "x2Scale", Ops::Base::ToString(x2ScaleShape->GetStorageShape()).c_str(),
                        "The shape [2] of x2Scale must be " + std::to_string(DIM_TWO)),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((x2ScaleDim0 != x2Dim0),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "x2Scale",
                                                          Ops::Base::ToString(x2ScaleShape->GetStorageShape()).c_str(),
                                                          "The shape [0] of x2Scale must be " + std::to_string(x2Dim0)),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((x2ScaleDim1 != x2Dim1 / MX_SCALE_ALIGN),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName, "x2Scale", Ops::Base::ToString(x2ScaleShape->GetStorageShape()).c_str(),
                        "The shape [1] of x2Scale must be " + std::to_string(x2Dim1 / MX_SCALE_ALIGN)),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置量化的数据类型信息
 *
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName 算子名称
 * @param contextInfo 存储了tiling的过程信息
 * @return
 */
ge::graphStatus AllToAllMxQuantMatmulTilingBase::SetMxDataTypeInfo(const gert::TilingContext *context,
                                                                   const char *opName, TilingContextInfo &contextInfo)
{
    const gert::StorageShape *matrixBias = context->GetOptionalInputShape(INPUT_BIAS_INDEX);
    ge::DataType biasType;
    auto x1TensorDesc = context->GetInputDesc(INPUT_X1_INDEX);
    auto x2TensorDesc = context->GetInputDesc(INPUT_X2_INDEX);
    auto yTensorDesc = context->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_TILING_CHECK((x1TensorDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "x1"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((x2TensorDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "x2"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((yTensorDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "y"), return ge::GRAPH_FAILED);
    ge::DataType aType = x1TensorDesc->GetDataType();
    ge::DataType bType = x2TensorDesc->GetDataType();
    ge::DataType cType = yTensorDesc->GetDataType();

    bool isBias = true;
    if (matrixBias == nullptr) {
        isBias = false;
        biasType = cType;
    } else {
        auto biasTensorDesc = context->GetOptionalInputDesc(INPUT_BIAS_INDEX);
        OP_TILING_CHECK((biasTensorDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "bias"),
                        return ge::GRAPH_FAILED);
        biasType = biasTensorDesc->GetDataType();
    }

    contextInfo.args_.outputDtypeSize = mc2tiling::GetDataTypeSize(opName, cType);
    // 设置为x1的数据类型
    contextInfo.args_.inputDtypeSize = BIT_NUMBER;
    contextInfo.args_.isBias = isBias;
    contextInfo.args_.geCType = cType;
    contextInfo.args_.geBiasType = biasType;
    contextInfo.args_.geAType = aType;
    contextInfo.args_.geBType = bType;
    contextInfo.args_.aType = mc2tiling::ConvertGeTypeToMmType(opName, aType);
    contextInfo.args_.cType = mc2tiling::ConvertGeTypeToMmType(opName, cType);
    contextInfo.args_.bType = mc2tiling::ConvertGeTypeToMmType(opName, bType);
    contextInfo.args_.biasType = mc2tiling::ConvertGeTypeToMmType(opName, biasType);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 根据输入设置tiling参数
 *
 * @return ge::graphStatus
 */
ge::graphStatus AllToAllMxQuantMatmulTilingBase::InitTilingContextParameters()
{
    MC2_CHECK_LOG_RET(
        opName_, MatmulAlltoAllTilingUtil::SetAttrsInfo(context_, opName_, contextInfo_, ALLTOALL_MATMUL_INDEX_SCHEMA));
    MC2_CHECK_LOG_RET(opName_, SetMxDataTypeInfo(context_, opName_, contextInfo_));
    MC2_CHECK_LOG_RET(opName_, SetAlltoAllMatmulShapeInfo(context_, contextInfo_));
    return ge::GRAPH_SUCCESS;
}

CutResult AllToAllMxQuantMatmulTilingBase::GetCutResOfCommAndCompute()
{
    constexpr uint32_t COMM_RANKDIM_FOUR = 4;
    uint8_t commMode = mc2tiling::A5_AICPU_TS_ENGINE;
    if (MatmulAlltoAllTilingUtil::GetAndConvertCommMode(context_, opName_, contextInfo_,
                                                        ALLTOALL_MATMUL_INDEX_SCHEMA, commMode)
        != ge::GRAPH_SUCCESS) {
        OP_LOGD(opName_, "GetAndConvertCommMode failed, use default commMode AICPU for tiling.");
    }
    CutResult cutRes;
    if (contextInfo_.args_.rankDim == COMM_RANKDIM_FOUR) {
        AlltoAllMatmulFitBalanceTiling tiling(matmulQuantType_, contextInfo_.args_,
                                              TopoType::STANDARD_CARD, SocVersion::SOC950, commMode);
        cutRes = tiling.GetTiling();
    } else {
        OP_LOGD(opName_, "Falling back to formulaic tiling for arch35 tiling");
        AlltoAllMM alltoallMatmulTileFormulate(contextInfo_.args_, contextInfo_.args_.rankDim,
                                               KernelType::ALL_TO_ALL, SocVersion::SOC950, true);
        if (commMode == mc2tiling::A5_AICPU_TS_ENGINE) {
            constexpr uint64_t MX_AICPU_TILING_MAX_NUM = 3;
            alltoallMatmulTileFormulate.tilingM_.SetMaxTileCnt(MX_AICPU_TILING_MAX_NUM);
        }
        alltoallMatmulTileFormulate.GetTiling();
        cutRes = alltoallMatmulTileFormulate.tilingM_.cutRes;
    }
    AlignCutResForKernel(cutRes, contextInfo_.args_.mValue);
    return cutRes;
}

void AllToAllMxQuantMatmulTilingBase::AlignCutResForKernel(CutResult &cutRes, uint64_t mValue) const
{
    constexpr uint64_t KERNEL_ALIGN_M = 256;
    auto alignUp = [](uint64_t val, uint64_t align) { return (val + align - 1) / align * align; };

    uint64_t longTile = alignUp(cutRes.longTileLen, KERNEL_ALIGN_M);
    if (longTile >= mValue) {
        cutRes.longTileLen = mValue;
        cutRes.numLongTile = 1;
        cutRes.shortTileLen = 0;
        cutRes.numShortTile = 0;
    } else {
        cutRes.longTileLen = longTile;
        cutRes.numLongTile = mValue / longTile;
        uint64_t remainder = mValue % longTile;
        if (remainder > 0) {
            cutRes.shortTileLen = remainder;
            cutRes.numShortTile = 1;
        } else {
            cutRes.shortTileLen = 0;
            cutRes.numShortTile = 0;
        }
    }
    cutRes.shortTileAtBack = true;
    OP_LOGD(opName_, "Adjusted cutRes: longTile=%lu numLong=%lu shortTile=%lu numShort=%lu atBack=%d",
            cutRes.longTileLen, cutRes.numLongTile, cutRes.shortTileLen, cutRes.numShortTile,
            cutRes.shortTileAtBack);
}

/**
 * @brief 主要处理逻辑，设置hccl参数；进行通算切分, 获取mm tiling等
 *
 * @return ge::graphStatus
 */
ge::graphStatus AllToAllMxQuantMatmulTilingBase::DoOpTiling()
{
    // 输入参数的校验:Attrs,Dtype,Shape等
    MC2_CHECK_LOG_RET(opName_, CheckOpInputInfo());
    // 参数校验通过后赋值给全局上下文变量
    MC2_CHECK_LOG_RET(opName_, InitTilingContextParameters());
    // 进行通算切分
    MC2_CHECK_LOG_RET(opName_, TileCommAndCompute());
    if (contextInfo_.allToAllOutFlag) {
        // arch35 路径: all2all_out 非空，生成 AlltoAllQuantMatmulTilingData
        MC2_CHECK_LOG_RET(opName_, DoArch35MxQuantMMTiling());
    } else {
        // hcomm 路径: all2all_out 为空，生成 hcommAllToAllMatmulTilingData
        MC2_CHECK_LOG_RET(opName_, DoMxQuantMMTiling());
    }
    // hccl的tiling参数赋值处理
    MC2_CHECK_LOG_RET(opName_, SetHcclTiling());
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 进行通算切分之后单个块的MM Tiling
 *
 * @return ge::graphStatus
 */
ge::graphStatus AllToAllMxQuantMatmulTilingBase::DoMxQuantMMTiling()
{
    uint64_t perRankM = contextInfo_.args_.mValue;
    uint64_t perRankK = contextInfo_.args_.orgKValue;
    uint64_t nValue = contextInfo_.args_.nValue;
    bool transB = contextInfo_.args_.isBTrans;

    if (isMxFp4_) {
        QuantMatmulTilingSwat<mm::DataType::DT_FLOAT4_E2M1, mm::DataType::DT_FLOAT4_E2M1> tilingSwat;
        tilingSwat.GetTilingData(perRankM, nValue, perRankK, false, transB, localTilingData_.tileQbmmTilingData);
    } else {
        QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN> tilingSwat;
        tilingSwat.GetTilingData(perRankM, nValue, perRankK, false, transB, localTilingData_.tileQbmmTilingData);
    }
    localTilingData_.localMatmul = 0;
    OP_LOGD(opName_, "MxQuantMM tiling: perRankM=%lu perRankK=%lu N=%lu", perRankM, perRankK, nValue);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置hccl的config,进行hccl对应的通信任务设置
 *
 * @return ge::graphStatus
 */
ge::graphStatus AllToAllMxQuantMatmulTilingBase::SetHcclTiling()
{
    ge::DataType hcclDtype = ge::DT_UNDEFINED;
    if (isMxFp4_) {
        hcclDtype = ge::DT_UINT8;
    } else {
        hcclDtype = contextInfo_.args_.geAType;
    }
    OP_TILING_CHECK(
        mc2tiling::ConvertGeTypeToHcclType(opName_, hcclDtype) == mc2tiling::HcclDataType::HCCL_DATA_TYPE_RESERVED,
        OP_LOGE(opName_, "Cannot find HcclDataType according to ge datatype = %d.", static_cast<int32_t>(hcclDtype)),
        return ge::GRAPH_FAILED;);
    Mc2CcTilingConfigBuilder allToAllMatmulBuilder =
        Mc2CcTilingConfigBuilder::create(contextInfo_.group, mc2tiling::AicpuComType::HCCL_CMD_ALLTOALL,
                                         Mc2CcTilingConfigBuilder::AlgConfigType::ALL_TO_ALL);

    // 获取commMode
    uint8_t hcclServerEngine = 0;
    if (MatmulAlltoAllTilingUtil::GetAndConvertCommMode(context_, opName_, contextInfo_, ALLTOALL_MATMUL_INDEX_SCHEMA,
                                                        hcclServerEngine) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // reducetype接口附带的数据类型优先于调用通信接口传入的数据类型，因此这里需要设置
    AscendC::Mc2CcTilingConfig allToAllTilingConfig =
        allToAllMatmulBuilder.withReduceType(opName_, AscendC::HcclReduceOp::HCCL_REDUCE_SUM, hcclDtype, hcclDtype)
            .withCommEngine(hcclServerEngine)
            .build();
    if (!allToAllMatmulBuilder.isSuccess()) {
        OP_LOGE(opName_, "Allto All Matmul build hccl tiling config failed: %s",
                allToAllMatmulBuilder.errorMsg().c_str());
        return ge::GRAPH_FAILED;
    }
    if (contextInfo_.allToAllOutFlag) {
        allToAllTilingConfig.GetTiling(arch35TilingData_.mc2InitTiling);
        allToAllTilingConfig.GetTiling(arch35TilingData_.mc2CcTiling);
    } else {
        allToAllTilingConfig.GetTiling(localTilingData_.mc2InitTiling);
        allToAllTilingConfig.GetTiling(localTilingData_.mc2CcTiling);
    }
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 打印tilingInfo信息
 *
 * @param opName
 * @param tilingInfo
 */
void AllToAllMxQuantMatmulTilingBase::PrintAlltoAllMxQuantMatmulTilingInfo(const std::string &opName,
                                                                           CommTilingData &tilingInfo)
{
    OP_LOGD(opName, "TilingInfo.splitAxisTileSize: %u", tilingInfo.splitAxisTileSize);
    OP_LOGD(opName, "TilingInfo.splitAxisTileCnt: %u", tilingInfo.splitAxisTileCnt);
    OP_LOGD(opName, "TilingInfo.splitAxisTailSize: %u", tilingInfo.splitAxisTailSize);
    OP_LOGD(opName, "TilingInfo.splitAxisTailCnt: %u", tilingInfo.splitAxisTailCnt);
    OP_LOGD(opName, "TilingInfo.nonSplitAxisSize: %u", tilingInfo.nonSplitAxisSize);
}

/**
 * @brief 打印传递给kernel的tilingData
 *
 * @param outTilingData tilingData参数
 */
void AllToAllMxQuantMatmulTilingBase::PrintAlltoAllMxQuantMatmulTilingData(hcommAllToAllMatmulTilingData &outTilingData)
{
    PrintAlltoAllMxQuantMatmulTilingInfo(opName_, outTilingData.commTilingData);
    OP_LOGD(opName_, "tileQbmm: m=%u n=%u k=%u baseM=%u baseN=%u baseK=%u stepK=%u scaleKL1=%u dbL0c=%u localMatmul=%u",
            outTilingData.tileQbmmTilingData.m, outTilingData.tileQbmmTilingData.n,
            outTilingData.tileQbmmTilingData.k, outTilingData.tileQbmmTilingData.baseM,
            outTilingData.tileQbmmTilingData.baseN, outTilingData.tileQbmmTilingData.baseK,
            outTilingData.tileQbmmTilingData.stepK, outTilingData.tileQbmmTilingData.scaleKL1,
            outTilingData.tileQbmmTilingData.dbL0c, outTilingData.localMatmul);
}

/**
 * @brief 保存量化tiling数据到context
 *
 * @return ge::graphStatus
 */
ge::graphStatus AllToAllMxQuantMatmulTilingBase::PostTiling()
{
    context_->SetScheduleMode(1);
    size_t tilingBufCap = context_->GetRawTilingData()->GetCapacity();
    if (contextInfo_.allToAllOutFlag) {
        // arch35 路径: 输出 AlltoAllQuantMatmulTilingData
        SetArch35TilingInfo(arch35TilingData_.alltoAllQuantMatmulTilingInfo);
        AlltoAllQuantMatmulTilingData *outTilingData = context_->GetTilingData<AlltoAllQuantMatmulTilingData>();
        OP_TILING_CHECK((outTilingData == nullptr), OP_LOGE(opName_, "Fail to get arch35 tiling data from context"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK((tilingBufCap < sizeof(arch35TilingData_)),
                        OP_LOGE(opName_, "TilingBuffer capacity is too small, capacity = %zu, need = %zu.",
                                tilingBufCap, sizeof(arch35TilingData_)),
                        return ge::GRAPH_FAILED);
        errno_t ret = memcpy_s(outTilingData, tilingBufCap, &arch35TilingData_, sizeof(arch35TilingData_));
        if (ret != EOK) {
            OP_LOGE(opName_, "AlltoAllMxQuantMatmul postTiling: memcpy_s arch35 tiling data failed, ret=%d.", ret);
            return ge::GRAPH_FAILED;
        }
        OP_LOGD(opName_, "Final arch35 tiling data size=%zu and context capacity size=%zu.",
                sizeof(AlltoAllQuantMatmulTilingData), context_->GetRawTilingData()->GetCapacity());
        context_->GetRawTilingData()->SetDataSize(sizeof(AlltoAllQuantMatmulTilingData));
    } else {
        // hcomm 路径: 输出 hcommAllToAllMatmulTilingData
        SetTilingInfo(localTilingData_.commTilingData);
        hcommAllToAllMatmulTilingData *outTilingData = context_->GetTilingData<hcommAllToAllMatmulTilingData>();
        OP_TILING_CHECK((outTilingData == nullptr), OP_LOGE(opName_, "Fail to get tiling data from context"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK((tilingBufCap < sizeof(localTilingData_)),
                        OP_LOGE(opName_, "TilingBuffer capacity is too small, capacity = %zu, need = %zu.",
                                tilingBufCap, sizeof(localTilingData_)),
                        return ge::GRAPH_FAILED);
        errno_t ret = memcpy_s(outTilingData, tilingBufCap, &localTilingData_, sizeof(localTilingData_));
        if (ret != EOK) {
            OP_LOGE(opName_, "AlltoAllMxQuantMatmul postTiling: memcpy_s tiling data failed, ret=%d.", ret);
            return ge::GRAPH_FAILED;
        }
        OP_LOGD(opName_, "Final tiling data size=%zu and context capacity size=%zu.",
                sizeof(hcommAllToAllMatmulTilingData), context_->GetRawTilingData()->GetCapacity());
        context_->GetRawTilingData()->SetDataSize(sizeof(hcommAllToAllMatmulTilingData));
        PrintAlltoAllMxQuantMatmulTilingData(*outTilingData);
    }
    context_->SetBlockDim(contextInfo_.args_.aicCoreNum);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置tilingInfo结构体
 *
 * @param tilingInfo 目标结构体
 */
void AllToAllMxQuantMatmulTilingBase::SetTilingInfo(CommTilingData &tilingInfo) const
{
    // 基本字段拷贝
    tilingInfo.splitAxisTileSize = inferredInfo_.tileM;
    tilingInfo.splitAxisTileCnt = inferredInfo_.tileCnt;
    tilingInfo.splitAxisTailSize = inferredInfo_.tailM;
    tilingInfo.splitAxisTailCnt = inferredInfo_.tailCnt;
    tilingInfo.nonSplitAxisSize = contextInfo_.args_.orgKValue * contextInfo_.args_.rankDim;
}

/**
 * @brief 设置arch35路径的tilingInfo结构体
 *
 * @param tilingInfo 目标结构体
 */
void AllToAllMxQuantMatmulTilingBase::SetArch35TilingInfo(AlltoAllMatmulTilingInfo &tilingInfo) const
{
    tilingInfo.tileM = inferredInfo_.tileM;
    tilingInfo.tailM = inferredInfo_.tailM;
    tilingInfo.tileCnt = inferredInfo_.tileCnt;
    tilingInfo.tailCnt = inferredInfo_.tailCnt;
    tilingInfo.rankN = contextInfo_.args_.nValue;
    tilingInfo.rankM = contextInfo_.args_.orgMValue;
    tilingInfo.rankK = contextInfo_.args_.orgKValue;
    tilingInfo.commLen = inferredInfo_.commLen;
    tilingInfo.permuteLen = inferredInfo_.permuteLen;
    tilingInfo.rankDim = contextInfo_.args_.rankDim;
    tilingInfo.aicCoreNum = contextInfo_.args_.aicCoreNum;
    tilingInfo.hcclDataType = static_cast<uint64_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, contextInfo_.args_.geAType));
    tilingInfo.x1ScaleOptionalLen = inferredInfo_.commScaleLen;
    tilingInfo.dynamicExtraSpace = 0UL;
    tilingInfo.x1QuantDtype = 0UL;
    tilingInfo.biasLen = 0UL;
}

/**
 * @brief arch35路径: 生成DequantBmm tiling (头块+尾块)
 *
 * @return ge::graphStatus
 */
ge::graphStatus AllToAllMxQuantMatmulTilingBase::DoArch35MxQuantMMTiling()
{
    SetArch35TilingInfo(arch35TilingData_.alltoAllQuantMatmulTilingInfo);
    OP_LOGD(opName_, "Arch35MxQuantMM tiling: perRankM=%lu perRankK=%lu N=%lu", contextInfo_.args_.orgMValue,
            contextInfo_.args_.orgKValue, contextInfo_.args_.nValue);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 获取对应的tilingKey
 * 使用QUANT_MODE来区分tilingKey,此处的QUANT_MODE指的是x1,x2的QUANT模式组合，以x1为pergroup量化(G)，x2为pergroup量化(G)
 * 为例子，MX量化就代表一种组合
 *
 * @return uint64_t tilingKey结果
 */
uint64_t AllToAllMxQuantMatmulTilingBase::GetTilingKey() const
{
    // 按照量化组合模式，是否转置，bias数据类型进行展开
    bool x2TransposeFlag = contextInfo_.args_.isBTrans ? true : false;
    uint8_t biasDType = DTYPE_BIAS_FP32;
    // 获取commMode
    uint8_t hcclServerType = 0;
    if (MatmulAlltoAllTilingUtil::GetAndConvertCommMode(context_, opName_, contextInfo_, ALLTOALL_MATMUL_INDEX_SCHEMA,
                                                        hcclServerType) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    uint8_t commMode = (hcclServerType == mc2tiling::A5_CCU_ENGINE) ? ALL2ALL_COMM_TYPE_CCU : ALL2ALL_COMM_TYPE_AICPU;
    bool hasAll2AllOut = contextInfo_.allToAllOutFlag;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(MX_QUANT_MODE, x2TransposeFlag, biasDType, false, commMode,
                                                    hasAll2AllOut);
    OP_LOGD(opName_, "MXQUANTMODE,X2TRANSPOSE,DTYPEBIAS,ISSMALLK,COMMMODE,HAS_ALL2ALL_OUT: [%d,%d,%d,0,%d,%d], "
            "TilingKey is [%lu].", MX_QUANT_MODE, x2TransposeFlag, biasDType, commMode, hasAll2AllOut, tilingKey);
    return tilingKey;
}

/**
 * @brief 计算总共需要的workspace大小
 *
 */
ge::graphStatus AllToAllMxQuantMatmulTilingBase::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(opName_, "get workspace failed"), return ge::GRAPH_FAILED);
    SetUserWorkSpace();
    uint64_t workspaceSize = libApiWorkSpaceSize_ + inferredInfo_.commLen + inferredInfo_.permuteLen +
                             inferredInfo_.commScaleLen + inferredInfo_.permuteScaleLen;
    workspaces[0] = workspaceSize;
    OP_LOGD(opName_, "Workspaces[0] size=%zu, commlen=%zu, permuteLen=%zu", workspaces[0], inferredInfo_.commLen,
            inferredInfo_.permuteLen);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置额外需要的空间，包括通信结果地址，重排地址，偏移地址等
 *
 */
void AllToAllMxQuantMatmulTilingBase::SetUserWorkSpace()
{
    constexpr uint64_t alignAddrLen = 512;
    constexpr uint64_t mxGroupSize = 64;
    // AlltoAllMatmul先进行通信，需要有对应的空间先存放结果，假设x1(m,k),假设原始rank上X1的第0维为M，这里的m就是M/ranksize,
    // m已经在前面获取输入参数的时候进行过处理
    if (isMxFp4_) {
        inferredInfo_.commLen = mc2tiling::AlignUp(
            Ops::Base::CeilDiv(
                (contextInfo_.args_.mValue * contextInfo_.args_.kValue * contextInfo_.args_.inputDtypeSize), DIM_TWO),
            alignAddrLen);
    } else {
        inferredInfo_.commLen = mc2tiling::AlignUp(
            contextInfo_.args_.mValue * contextInfo_.args_.kValue * contextInfo_.args_.inputDtypeSize, alignAddrLen);
    }
    // 重排空间等于通信结果结果空间,如果存在alltoallout空间的话，不需要申请这块
    if (!contextInfo_.allToAllOutFlag) {
        inferredInfo_.permuteLen = inferredInfo_.commLen;
    }
    inferredInfo_.commScaleLen = mc2tiling::AlignUp(
        contextInfo_.args_.mValue * contextInfo_.args_.rankDim *
            Ops::Base::CeilDiv((contextInfo_.args_.kValue / contextInfo_.args_.rankDim), mxGroupSize) * DIM_TWO,
        alignAddrLen);
    inferredInfo_.permuteScaleLen = inferredInfo_.commScaleLen;
}

/**
 * @brief 构造函数，创建一个AllToAllMxQuantMatmulTilingBase对象
 *
 * @param context
 */
AllToAllMxQuantMatmulTilingBase::AllToAllMxQuantMatmulTilingBase(gert::TilingContext *context)
    : AllToAllMatmulTilingBase(context)
{
}

// 注册tiling类
REGISTER_TILING_TEMPLATE_WITH_ARCH(AlltoAllMatmul, AllToAllMxQuantMatmulTilingBase,
                                   static_cast<int32_t>(NpuArch::DAV_3510), 2);

} // namespace MC2Tiling
