/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTO_ALL_MATMUL_APACE_TILING_BASE_H
#define ALLTO_ALL_MATMUL_APACE_TILING_BASE_H

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <ascendc/host_api/tiling/template_argument.h>
#include "securec.h"
#include "apace/kernel/all_to_all_quant_matmul/all_to_all_matmul_tiling_data.h"
#include "apace/tiling/quant_matmul_tiling_swat.h"
#include "apace/tiling/comm_tiling_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "../../../op_kernel/arch35/allto_all_matmul_v2_tiling_key.h"

namespace MC2Tiling {

using namespace Ops::Transformer::OpTiling;

constexpr uint64_t WS_BYTES = 4096;

class AlltoAllMatmulV2TilingClass : public TilingBaseClass {
public:
    explicit AlltoAllMatmulV2TilingClass(gert::TilingContext *context)
        : TilingBaseClass(context)
    {}
    ~AlltoAllMatmulV2TilingClass() override = default;

protected:
    uint32_t usedCoreNum_ = 0; // 保存 tiling 实际使用的核数
    bool IsCapable() override { return true; }

    ge::graphStatus GetPlatformInfo() override
    {
        fe::PlatFormInfos *platformInfo = context_->GetPlatformInfo();
        if (platformInfo == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "platformInfo");
            return ge::GRAPH_FAILED;
        }
        platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
        aicoreParams_.aicNum = ascendcPlatform.GetCoreNumAic();
        return ge::GRAPH_SUCCESS;
    }

    static constexpr size_t IDX_INPUT_X1 = 1;
    static constexpr size_t IDX_INPUT_X2 = 2;
    static constexpr size_t IDX_INPUT_BIAS = 3;
    static constexpr size_t IDX_INPUT_X1_SCALE = 4;
    static constexpr size_t IDX_INPUT_X2_SCALE = 5;
    static constexpr size_t IDX_OUTPUT_Y = 0;
    static constexpr size_t IDX_ATTR_WORLD_SIZE = 1;
    static constexpr size_t IDX_ATTR_Y_DTYPE = 3;
    static constexpr size_t IDX_ATTR_X1_QUANT_MODE = 4;
    static constexpr size_t IDX_ATTR_X2_QUANT_MODE = 5;
    static constexpr size_t IDX_ATTR_TRANSPOSE_X2 = 7;
    static constexpr size_t IDX_ATTR_GROUP_SIZE = 8;
    static constexpr size_t IDX_ATTR_ALLTOALL_OUT_FLAG = 9;
    static constexpr size_t IDX_ATTR_COMM_MODE = 10;
    static constexpr size_t IDX_ATTR_PERF_MODE = 11;
    static constexpr uint64_t MX_SCALE_ALIGN = 64;
    static constexpr uint64_t SCALE_LAST_DIM = 2;
    static constexpr uint64_t SCALE_DIM_NUM = 3;
    static constexpr int64_t MAX_INT32_VAL = 2147483647;
    static constexpr uint64_t K_MAX_VAL = 65535;
    static constexpr uint64_t GROUP_MNK_BIT_SIZE = 0xFFFF;
    static constexpr uint64_t GROUP_M_OFFSET = 32;
    static constexpr uint64_t GROUP_N_OFFSET = 16;
    static constexpr uint64_t MX_GROUP_M = 1;
    static constexpr uint64_t MX_GROUP_N = 1;
    static constexpr uint64_t MX_GROUP_K = 32;

    ge::graphStatus CheckTensorAttrs()
    {
        const char *opName = context_->GetNodeName();

        auto *x1Desc = context_->GetInputDesc(IDX_INPUT_X1);
        auto *x2Desc = context_->GetInputDesc(IDX_INPUT_X2);
        auto *yDesc = context_->GetOutputDesc(IDX_OUTPUT_Y);
        if (!x1Desc || !x2Desc || !yDesc) {
            if (!x1Desc)
                OP_LOGE_WITH_INVALID_INPUT(opName, "x1");
            if (!x2Desc)
                OP_LOGE_WITH_INVALID_INPUT(opName, "x2");
            if (!yDesc)
                OP_LOGE_WITH_INVALID_INPUT(opName, "y");
            return ge::GRAPH_FAILED;
        }

        // ---- world_size validation ----
        auto *wsPtr = context_->GetAttrs()->GetAttrPointer<int64_t>(IDX_ATTR_WORLD_SIZE);
        if (!wsPtr) {
            OP_LOGE_WITH_INVALID_INPUT(opName, "world_size");
            return ge::GRAPH_FAILED;
        }
        if (*wsPtr <= 0) {
            OP_LOGE_WITH_INVALID_ATTR(opName, "world_size", std::to_string(*wsPtr).c_str(), "positive");
            return ge::GRAPH_FAILED;
        }
        uint64_t worldSize = static_cast<uint64_t>(*wsPtr);
        bool isPowerOfTwo = (worldSize & (worldSize - 1)) == 0;
        if (worldSize < 2 || worldSize > 16 || !isPowerOfTwo) {
            OP_LOGE_WITH_INVALID_ATTR(opName, "world_size", std::to_string(worldSize).c_str(), "2/4/8/16");
            return ge::GRAPH_FAILED;
        }

        // ---- format checks ----
        auto x1Fmt = static_cast<ge::Format>(ge::GetPrimaryFormat(x1Desc->GetStorageFormat()));
        auto x2Fmt = static_cast<ge::Format>(ge::GetPrimaryFormat(x2Desc->GetStorageFormat()));
        auto yFmt = static_cast<ge::Format>(ge::GetPrimaryFormat(yDesc->GetStorageFormat()));
        if (x1Fmt != ge::FORMAT_ND || x2Fmt != ge::FORMAT_ND) {
            OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
                opName, "x1,x2",
                (Ops::Base::ToString(x1Fmt) + "," + Ops::Base::ToString(x2Fmt)).c_str(),
                "x1 and x2 format must be ND");
            return ge::GRAPH_FAILED;
        }
        if (yFmt != ge::FORMAT_ND) {
            OP_LOGE_FOR_INVALID_FORMAT(opName, "y", Ops::Base::ToString(yFmt).c_str(), "ND");
            return ge::GRAPH_FAILED;
        }

        // ---- x1/x2 dtype checks ----
        auto x1Dtype = x1Desc->GetDataType();
        auto x2Dtype = x2Desc->GetDataType();
        if ((x1Dtype != ge::DT_FLOAT8_E4M3FN && x1Dtype != ge::DT_FLOAT8_E5M2 && x1Dtype != ge::DT_FLOAT4_E2M1) ||
            (x2Dtype != ge::DT_FLOAT8_E4M3FN && x2Dtype != ge::DT_FLOAT8_E5M2 && x2Dtype != ge::DT_FLOAT4_E2M1)) {
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                opName, "x1,x2",
                (Ops::Base::ToString(x1Dtype) + "," + Ops::Base::ToString(x2Dtype)).c_str(),
                "x1/x2 dtype must be FP8_E4M3/FP8_E5M2/FP4_E2M1");
            return ge::GRAPH_FAILED;
        }
        if ((x1Dtype == ge::DT_FLOAT4_E2M1) != (x2Dtype == ge::DT_FLOAT4_E2M1)) {
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                opName, "x1,x2",
                (Ops::Base::ToString(x1Dtype) + "," + Ops::Base::ToString(x2Dtype)).c_str(),
                "FP4 requires both x1 and x2 to be FP4_E2M1");
            return ge::GRAPH_FAILED;
        }

        // ---- y output dtype ----
        auto yDtype = yDesc->GetDataType();
        if (yDtype != ge::DT_BF16 && yDtype != ge::DT_FLOAT16) {
            OP_LOGE_FOR_INVALID_DTYPE(opName, "y", Ops::Base::ToString(yDtype).c_str(), "BF16/FP16");
            return ge::GRAPH_FAILED;
        }

        // ---- scale checks ----
        auto *x1ScaleDesc = context_->GetOptionalInputDesc(IDX_INPUT_X1_SCALE);
        auto *x2ScaleDesc = context_->GetOptionalInputDesc(IDX_INPUT_X2_SCALE);
        if (!x1ScaleDesc || !x2ScaleDesc) {
            if (!x1ScaleDesc)
                OP_LOGE_WITH_INVALID_INPUT(opName, "x1_scale");
            if (!x2ScaleDesc)
                OP_LOGE_WITH_INVALID_INPUT(opName, "x2_scale");
            return ge::GRAPH_FAILED;
        }
        auto x1ScaleFmt = static_cast<ge::Format>(ge::GetPrimaryFormat(x1ScaleDesc->GetStorageFormat()));
        auto x2ScaleFmt = static_cast<ge::Format>(ge::GetPrimaryFormat(x2ScaleDesc->GetStorageFormat()));
        if (x1ScaleFmt != ge::FORMAT_ND || x2ScaleFmt != ge::FORMAT_ND) {
            OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
                opName, "x1_scale,x2_scale",
                (Ops::Base::ToString(x1ScaleFmt) + "," + Ops::Base::ToString(x2ScaleFmt)).c_str(),
                "x1_scale and x2_scale format must be ND");
            return ge::GRAPH_FAILED;
        }
        auto x1ScaleDtype = x1ScaleDesc->GetDataType();
        auto x2ScaleDtype = x2ScaleDesc->GetDataType();
        if (x1ScaleDtype != ge::DT_FLOAT8_E8M0 || x2ScaleDtype != ge::DT_FLOAT8_E8M0) {
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                opName, "x1_scale,x2_scale",
                (Ops::Base::ToString(x1ScaleDtype) + "," + Ops::Base::ToString(x2ScaleDtype)).c_str(),
                "x1_scale and x2_scale must be FP8_E8M0");
            return ge::GRAPH_FAILED;
        }

        // ---- bias checks ----
        auto *biasDesc = context_->GetOptionalInputDesc(IDX_INPUT_BIAS);
        if (biasDesc != nullptr) {
            auto biasDtype = biasDesc->GetDataType();
            if (biasDtype != ge::DT_FLOAT) {
                OP_LOGE_FOR_INVALID_DTYPE(opName, "bias", Ops::Base::ToString(biasDtype).c_str(), "FLOAT");
                return ge::GRAPH_FAILED;
            }
            auto biasFmt = static_cast<ge::Format>(ge::GetPrimaryFormat(biasDesc->GetStorageFormat()));
            if (biasFmt != ge::FORMAT_ND) {
                OP_LOGE_FOR_INVALID_FORMAT(opName, "bias", Ops::Base::ToString(biasFmt).c_str(), "ND");
                return ge::GRAPH_FAILED;
            }
        }

        // ---- quant mode validation (MX quant only) ----
        auto *x1QmPtr = context_->GetAttrs()->GetAttrPointer<int64_t>(IDX_ATTR_X1_QUANT_MODE);
        auto *x2QmPtr = context_->GetAttrs()->GetAttrPointer<int64_t>(IDX_ATTR_X2_QUANT_MODE);
        int64_t x1QuantMode = x1QmPtr ? *x1QmPtr : static_cast<int64_t>(MX_QUANT_MODE);
        int64_t x2QuantMode = x2QmPtr ? *x2QmPtr : static_cast<int64_t>(MX_QUANT_MODE);
        if (x1QuantMode != static_cast<int64_t>(MX_QUANT_MODE) || x2QuantMode != static_cast<int64_t>(MX_QUANT_MODE)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                opName, "x1_quant_mode,x2_quant_mode",
                (std::to_string(x1QuantMode) + "," + std::to_string(x2QuantMode)).c_str(),
                "x1_quant_mode and x2_quant_mode must be MX_QUANT(6)");
            return ge::GRAPH_FAILED;
        }

        // ---- y_dtype attr validation ----
        auto *yDtypePtr = context_->GetAttrs()->GetAttrPointer<int64_t>(IDX_ATTR_Y_DTYPE);
        if (yDtypePtr != nullptr && *yDtypePtr != static_cast<int64_t>(ge::DT_BF16) &&
            *yDtypePtr != static_cast<int64_t>(ge::DT_FLOAT16)) {
            OP_LOGE_WITH_INVALID_ATTR(opName, "y_dtype", std::to_string(*yDtypePtr).c_str(), "BF16/FP16");
            return ge::GRAPH_FAILED;
        }

        // ---- attr checks ----
        auto *tx2 = context_->GetAttrs()->GetAttrPointer<bool>(IDX_ATTR_TRANSPOSE_X2);
        if (!tx2 || !(*tx2)) {
            OP_LOGE_WITH_INVALID_ATTR(opName, "transpose_x2",
                                      tx2 ? std::to_string(static_cast<int>(*tx2)).c_str() : "nil", "true");
            return ge::GRAPH_FAILED;
        }

        auto *commModePtr = context_->GetAttrs()->GetAttrPointer<char>(IDX_ATTR_COMM_MODE);
        if (!commModePtr) {
            OP_LOGE_WITH_INVALID_INPUT(opName, "comm_mode");
            return ge::GRAPH_FAILED;
        }
        if (std::strcmp(commModePtr, "urma") != 0) {
            OP_LOGE_WITH_INVALID_ATTR(opName, "comm_mode", commModePtr, "urma");
            return ge::GRAPH_FAILED;
        }

        auto *a2aFlag = context_->GetAttrs()->GetAttrPointer<bool>(IDX_ATTR_ALLTOALL_OUT_FLAG);
        if (!a2aFlag) {
            OP_LOGE_WITH_INVALID_INPUT(opName, "alltoall_out_flag");
            return ge::GRAPH_FAILED;
        }
        if (*a2aFlag) {
            OP_LOGE_WITH_INVALID_ATTR(opName, "alltoall_out_flag", "true", "false");
            return ge::GRAPH_FAILED;
        }

        // ---- group_size validation (MX quant: [1,1,32]) ----
        auto *gsPtr = context_->GetAttrs()->GetAttrPointer<int64_t>(IDX_ATTR_GROUP_SIZE);
        if (gsPtr) {
            uint64_t gs = static_cast<uint64_t>(*gsPtr);
            uint64_t gsK = gs & GROUP_MNK_BIT_SIZE;
            uint64_t gsN = (gs >> GROUP_N_OFFSET) & GROUP_MNK_BIT_SIZE;
            uint64_t gsM = (gs >> GROUP_M_OFFSET) & GROUP_MNK_BIT_SIZE;
            if (gsM != MX_GROUP_M || gsN != MX_GROUP_N || gsK != MX_GROUP_K) {
                OP_LOGE_WITH_INVALID_ATTR(
                    opName, "group_size",
                    ("[M=" + std::to_string(gsM) + ",N=" + std::to_string(gsN) + ",K=" + std::to_string(gsK) +
                     "]")
                        .c_str(),
                    ("[M=" + std::to_string(MX_GROUP_M) + ",N=" + std::to_string(MX_GROUP_N) + ",K=" +
                     std::to_string(MX_GROUP_K) + "]")
                        .c_str());
                return ge::GRAPH_FAILED;
            }
        }

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckTensorShapes()
    {
        const char *opName = context_->GetNodeName();

        auto *x1Shape = context_->GetInputShape(IDX_INPUT_X1);
        auto *x2Shape = context_->GetInputShape(IDX_INPUT_X2);
        auto *yShape = context_->GetOutputShape(IDX_OUTPUT_Y);
        if (!x1Shape || !x2Shape || !yShape) {
            if (!x1Shape)
                OP_LOGE_WITH_INVALID_INPUT(opName, "x1");
            if (!x2Shape)
                OP_LOGE_WITH_INVALID_INPUT(opName, "x2");
            if (!yShape)
                OP_LOGE_WITH_INVALID_INPUT(opName, "y");
            return ge::GRAPH_FAILED;
        }

        uint64_t x1DimNum = x1Shape->GetStorageShape().GetDimNum();
        uint64_t x2DimNum = x2Shape->GetStorageShape().GetDimNum();
        uint64_t yDimNum = yShape->GetStorageShape().GetDimNum();
        if (x1DimNum != 2 || x2DimNum != 2) {
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                opName, "x1,x2",
                (std::to_string(x1DimNum) + "D," + std::to_string(x2DimNum) + "D").c_str(),
                "x1 and x2 must be 2D");
            return ge::GRAPH_FAILED;
        }
        if (yDimNum != 2) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(opName, "y", std::to_string(yDimNum).c_str(), "2D");
            return ge::GRAPH_FAILED;
        }

        auto *wsPtr = context_->GetAttrs()->GetAttrPointer<int64_t>(IDX_ATTR_WORLD_SIZE);
        uint64_t worldSize = static_cast<uint64_t>(wsPtr ? *wsPtr : 1);

        uint64_t x1M = x1Shape->GetStorageShape().GetDim(0U);
        uint64_t x1K = x1Shape->GetStorageShape().GetDim(1U);
        uint64_t x2N = x2Shape->GetStorageShape().GetDim(0U);
        uint64_t x2K = x2Shape->GetStorageShape().GetDim(1U);
        uint64_t yM = yShape->GetStorageShape().GetDim(0U);
        uint64_t yN = yShape->GetStorageShape().GetDim(1U);

        if (x1M == 0 || x1K == 0 || x2N == 0 || x2K == 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                opName, "x1, x2",
                ("x1=[" + std::to_string(x1M) + "," + std::to_string(x1K) + "], x2=[" + std::to_string(x2N) + "," +
                 std::to_string(x2K) + "]")
                    .c_str(),
                "The dimensions of x1 and x2 must be non-zero.");
            return ge::GRAPH_FAILED;
        }
        if (x1M > static_cast<uint64_t>(MAX_INT32_VAL) || x2N > static_cast<uint64_t>(MAX_INT32_VAL)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                opName, "x1 dim0, x2 dim0",
                ("M=" + std::to_string(x1M) + ", N=" + std::to_string(x2N)).c_str(),
                "The dim0 of x1 and dim0 of x2 must not exceed INT32_MAX.");
            return ge::GRAPH_FAILED;
        }
        if (x1K > K_MAX_VAL || x2K > K_MAX_VAL) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                opName, "x1_k, x2_k",
                ("x1_k=" + std::to_string(x1K) + ", x2_k=" + std::to_string(x2K)).c_str(),
                ("The K dimension must not exceed " + std::to_string(K_MAX_VAL) + ".").c_str());
            return ge::GRAPH_FAILED;
        }

        if (x1M % worldSize != 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                opName, "x1 dim0",
                ("M=" + std::to_string(x1M)).c_str(),
                "The dim0 of x1 must be divisible by world_size.");
            return ge::GRAPH_FAILED;
        }
        if (x1K % MX_SCALE_ALIGN != 0 || x2K % MX_SCALE_ALIGN != 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                opName, "x1_k, x2_k",
                ("x1_k=" + std::to_string(x1K) + ", x2_k=" + std::to_string(x2K)).c_str(),
                ("The K dimension must be divisible by " + std::to_string(MX_SCALE_ALIGN) + ".").c_str());
            return ge::GRAPH_FAILED;
        }
        if (x2K != x1K * worldSize) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                opName, "x2_k",
                std::to_string(x2K).c_str(),
                ("x2.K must equal x1.K * world_size = " + std::to_string(x1K * worldSize) + ".").c_str());
            return ge::GRAPH_FAILED;
        }

        if (yM != x1M / worldSize || yN != x2N) {
            OP_LOGE_FOR_INVALID_SHAPE(
                opName, "y",
                ("[" + std::to_string(yM) + "," + std::to_string(yN) + "]").c_str(),
                ("[" + std::to_string(x1M / worldSize) + "," + std::to_string(x2N) + "]").c_str());
            return ge::GRAPH_FAILED;
        }

        // ---- bias shape ----
        auto *biasShape = context_->GetOptionalInputShape(IDX_INPUT_BIAS);
        if (biasShape != nullptr) {
            if (biasShape->GetStorageShape().GetDimNum() != 1) {
                OP_LOGE_FOR_INVALID_SHAPEDIM(opName, "bias",
                                             std::to_string(biasShape->GetStorageShape().GetDimNum()).c_str(), "1D");
                return ge::GRAPH_FAILED;
            }
            uint64_t biasDim0 = biasShape->GetStorageShape().GetDim(0U);
            if (biasDim0 != x2N) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    opName, "bias dim0",
                    std::to_string(biasDim0).c_str(),
                    ("bias dim0 must equal N = " + std::to_string(x2N) + ".").c_str());
                return ge::GRAPH_FAILED;
            }
        }

        // ---- scale shapes ----
        auto *x1ScaleShape = context_->GetOptionalInputShape(IDX_INPUT_X1_SCALE);
        auto *x2ScaleShape = context_->GetOptionalInputShape(IDX_INPUT_X2_SCALE);
        if (!x1ScaleShape || !x2ScaleShape) {
            if (!x1ScaleShape)
                OP_LOGE_WITH_INVALID_INPUT(opName, "x1_scale");
            if (!x2ScaleShape)
                OP_LOGE_WITH_INVALID_INPUT(opName, "x2_scale");
            return ge::GRAPH_FAILED;
        }

        uint64_t x1ScaleDimNum = x1ScaleShape->GetStorageShape().GetDimNum();
        uint64_t x2ScaleDimNum = x2ScaleShape->GetStorageShape().GetDimNum();
        if (x1ScaleDimNum != SCALE_DIM_NUM || x2ScaleDimNum != SCALE_DIM_NUM) {
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                opName, "x1_scale,x2_scale",
                (std::to_string(x1ScaleDimNum) + "D," + std::to_string(x2ScaleDimNum) + "D").c_str(),
                "x1_scale and x2_scale must be 3D");
            return ge::GRAPH_FAILED;
        }

        uint64_t x1ScaleM = x1ScaleShape->GetStorageShape().GetDim(0U);
        uint64_t x1ScaleK = x1ScaleShape->GetStorageShape().GetDim(1U);
        uint64_t x1ScaleLast = x1ScaleShape->GetStorageShape().GetDim(2U);
        uint64_t x2ScaleN = x2ScaleShape->GetStorageShape().GetDim(0U);
        uint64_t x2ScaleK = x2ScaleShape->GetStorageShape().GetDim(1U);
        uint64_t x2ScaleLast = x2ScaleShape->GetStorageShape().GetDim(2U);

        if (x1ScaleM != x1M || x1ScaleK != x1K / MX_SCALE_ALIGN || x1ScaleLast != SCALE_LAST_DIM) {
            OP_LOGE_FOR_INVALID_SHAPE(
                opName, "x1_scale",
                ("[m=" + std::to_string(x1ScaleM) + ",k=" + std::to_string(x1ScaleK) + ",last=" +
                 std::to_string(x1ScaleLast) + "]")
                    .c_str(),
                ("[m=" + std::to_string(x1M) + ",k=" + std::to_string(x1K / MX_SCALE_ALIGN) + ",last=" +
                 std::to_string(SCALE_LAST_DIM) + "]")
                    .c_str());
            return ge::GRAPH_FAILED;
        }
        if (x2ScaleN != x2N || x2ScaleK != x2K / MX_SCALE_ALIGN || x2ScaleLast != SCALE_LAST_DIM) {
            OP_LOGE_FOR_INVALID_SHAPE(
                opName, "x2_scale",
                ("[n=" + std::to_string(x2ScaleN) + ",k=" + std::to_string(x2ScaleK) + ",last=" +
                 std::to_string(x2ScaleLast) + "]")
                    .c_str(),
                ("[n=" + std::to_string(x2N) + ",k=" + std::to_string(x2K / MX_SCALE_ALIGN) + ",last=" +
                 std::to_string(SCALE_LAST_DIM) + "]")
                    .c_str());
            return ge::GRAPH_FAILED;
        }

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckOpInputInfo()
    {
        if (CheckTensorAttrs() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (CheckTensorShapes() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetShapeAttrsInfo() override
    {
        const char *opName = context_->GetNodeName();
        auto *x1sh = context_->GetInputShape(IDX_INPUT_X1);
        auto *x2sh = context_->GetInputShape(IDX_INPUT_X2);
        if (!x1sh || !x2sh) {
            if (!x1sh)
                OP_LOGE_WITH_INVALID_INPUT(opName, "x1");
            if (!x2sh)
                OP_LOGE_WITH_INVALID_INPUT(opName, "x2");
            return ge::GRAPH_FAILED;
        }
        auto &x1ss = x1sh->GetStorageShape();
        auto &x2ss = x2sh->GetStorageShape();
        if (x1ss.GetDimNum() < 2 || x2ss.GetDimNum() < 2) {
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                opName, "x1,x2",
                (std::to_string(x1ss.GetDimNum()) + "D," + std::to_string(x2ss.GetDimNum()) + "D").c_str(),
                "x1 and x2 must have at least 2D");
            return ge::GRAPH_FAILED;
        }
        // x1 shape: [M_total, Ka]; kernel expects M_per_rank = M_total / world_size
        auto *wsPtr = context_->GetAttrs()->GetAttrPointer<int64_t>(IDX_ATTR_WORLD_SIZE);
        if (!wsPtr || *wsPtr <= 0) {
            OP_LOGE_WITH_INVALID_INPUT(opName, "world_size");
            return ge::GRAPH_FAILED;
        }
        worldSize_ = static_cast<uint64_t>(*wsPtr);
        m_ = x1ss.GetDim(0U) / worldSize_;
        k_ = x1ss.GetDim(1U); // Ka (per-rank K)
        n_ = x2ss.GetDim(0U); // N (always dim0, x2 shape = [N, K_total])

        auto *pm = context_->GetAttrs()->GetAttrPointer<int64_t>(IDX_ATTR_PERF_MODE);
        perfMode_ = pm ? static_cast<uint32_t>(*pm) : 0;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoOpTiling() override
    {
        const char *opName = context_->GetNodeName();
        if (CheckOpInputInfo() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (m_ == 0 || k_ == 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                opName, "m, k",
                (std::to_string(m_) + ", " + std::to_string(k_)).c_str(),
                "The value of m and k must be non-zero.");
            return ge::GRAPH_FAILED;
        }
        uint64_t ka = k_; // Ka = per-rank K

        auto *rt = context_->GetRawTilingData();
        auto cap = rt->GetCapacity();
        if (!rt || cap < sizeof(allToAllMatmulTilingData)) {
            OP_LOGE_WITH_INVALID_INPUT(opName, "rawTilingData");
            return ge::GRAPH_FAILED;
        }
        memset_s(rt->GetData(), cap, 0, cap);

        auto *td = reinterpret_cast<allToAllMatmulTilingData *>(rt->GetData());

        QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN> tilingEngine;
        tilingEngine.GetTilingData(m_, n_, k_, false, true, td->tileQbmmTilingData);

        auto &mm = td->tileQbmmTilingData;
        usedCoreNum_ = mm.usedCoreNum;

        apace::CommTilingBase::GetCommTilingData(m_, n_, ka, mm, td->commTilingData, td->scaleCommTilingData);

        OP_LOGI(opName,
                "comm tiling: tileSize=%lu, tileCnt=%lu, tailSize=%lu, tailCnt=%lu, nonSplitSize=%lu",
                td->commTilingData.splitAxisTileSize, td->commTilingData.splitAxisTileCnt,
                td->commTilingData.splitAxisTailSize, td->commTilingData.splitAxisTailCnt,
                td->commTilingData.nonSplitAxisSize);
        OP_LOGI(opName,
                "scale comm tiling: tileSize=%lu, tileCnt=%lu, tailSize=%lu, tailCnt=%lu, nonSplitSize=%lu",
                td->scaleCommTilingData.splitAxisTileSize, td->scaleCommTilingData.splitAxisTileCnt,
                td->scaleCommTilingData.splitAxisTailSize, td->scaleCommTilingData.splitAxisTailCnt,
                td->scaleCommTilingData.nonSplitAxisSize);

        td->localMatmul = perfMode_;

        OP_LOGI(opName, "localMatmul perfMode=%u", perfMode_);

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoLibApiTiling() override { return ge::GRAPH_SUCCESS; }

    uint64_t GetTilingKey() const override
    {
        return GET_TPL_TILING_KEY(MX_QUANT_MODE, 1 /*x2Transpose*/, DTYPE_BIAS_FP32, 0 /*isSmallK*/,
                                  ALL2ALL_COMM_TYPE_UDMA);
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        platform_ascendc::PlatformAscendC ascendcPlatform(context_->GetPlatformInfo());
        workspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
        auto *wsBuf = context_->GetWorkspaceSizes(1);
        if (wsBuf != nullptr) {
            wsBuf[0] = workspaceSize_;
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PostTiling() override
    {
        context_->GetRawTilingData()->SetDataSize(context_->GetRawTilingData()->GetCapacity());
        context_->SetBlockDim(usedCoreNum_);
        return ge::GRAPH_SUCCESS;
    }

private:
    uint64_t m_{0}, k_{0}, n_{0};
    uint64_t worldSize_{1};
    uint32_t perfMode_{0};
};

} // namespace MC2Tiling

using AlltoAllMatmulV2TilingClass = MC2Tiling::AlltoAllMatmulV2TilingClass;
REGISTER_TILING_TEMPLATE_WITH_ARCH(AlltoAllMatmulV2, AlltoAllMatmulV2TilingClass,
                                   static_cast<int32_t>(NpuArch::DAV_3510), 1);

#endif
