/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_ALLTO_ALL_APACE_TILING_BASE_H
#define MATMUL_ALLTO_ALL_APACE_TILING_BASE_H

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <ascendc/host_api/tiling/template_argument.h>
#include "securec.h"
#include "apace/kernel/fusions/quant_matmul_all_to_all/matmul_all_to_all_tiling_data.h"
#include "apace/tiling/quant_matmul_tiling_swat.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_host/tiling_base.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "op_host/tiling_templates_registry.h"
#include "../../../op_kernel/arch35/matmul_allto_all_v2_tiling_key.h"

namespace MC2Tiling {

using namespace Ops::Transformer::OpTiling;

class MatmulAlltoAllV2TilingClass : public TilingBaseClass {
public:
    explicit MatmulAlltoAllV2TilingClass(gert::TilingContext *context)
        : TilingBaseClass(context)
    {}
    ~MatmulAlltoAllV2TilingClass() override = default;

protected:
    uint32_t usedCoreNum_ = 0; // 保存 tiling 实际使用的核数
    bool IsCapable() override
    {
        return true;
    }

    ge::graphStatus GetPlatformInfo() override
    {
        fe::PlatFormInfos *platformInfo = context_->GetPlatformInfo();
        if (platformInfo == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "platformInfo");
            return ge::GRAPH_FAILED;
        }
        platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
        aicoreParams_.aicNum = ascendcPlatform.GetCoreNumAic();

        // Cache the full platform profile from the tiling context so the apace
        // matmul engine can use it without querying the global singleton.
        quantPlatformInfo_.aicNum = ascendcPlatform.GetCoreNumAic();
        quantPlatformInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();
        quantPlatformInfo_.socVersion = ascendcPlatform.GetSocVersion();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, quantPlatformInfo_.ubSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, quantPlatformInfo_.l1Size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, quantPlatformInfo_.l0aSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, quantPlatformInfo_.l0bSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, quantPlatformInfo_.l0cSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, quantPlatformInfo_.l2Size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::BT, quantPlatformInfo_.btSize);
        return ge::GRAPH_SUCCESS;
    }

    static constexpr size_t IDX_INPUT_X1 = 1;
    static constexpr size_t IDX_INPUT_X2 = 2;
    static constexpr size_t IDX_INPUT_BIAS = 3;
    static constexpr size_t IDX_INPUT_X1_SCALE = 4;
    static constexpr size_t IDX_INPUT_X2_SCALE = 5;
    static constexpr size_t IDX_OUTPUT_Y = 0;
    static constexpr size_t IDX_ATTR_GROUP = 0;
    static constexpr size_t IDX_ATTR_WORLD_SIZE = 1;
    static constexpr size_t IDX_ATTR_HCCL_BUFFER_SIZE = 2;
    static constexpr size_t IDX_ATTR_Y_DTYPE = 3;
    static constexpr size_t IDX_ATTR_X1_QUANT_MODE = 4;
    static constexpr size_t IDX_ATTR_X2_QUANT_MODE = 5;
    static constexpr size_t IDX_ATTR_TRANSPOSE_X1 = 6;
    static constexpr size_t IDX_ATTR_TRANSPOSE_X2 = 7;
    static constexpr size_t IDX_ATTR_GROUP_SIZE = 8;
    static constexpr size_t IDX_ATTR_COMM_MODE = 9;
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
    static constexpr uint64_t MIN_WORLD_SIZE = 2;
    static constexpr uint64_t MAX_WORLD_SIZE = 16;
    static constexpr uint64_t MAX_GROUP_NAME_LEN = 128;

    ge::graphStatus CheckTensorAttrs()
    {
        const char *opName = context_->GetNodeName();
        if (context_->GetAttrs() == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT(opName, "attrs");
            return ge::GRAPH_FAILED;
        }
        if (!CheckInputDescs(opName) || !CheckWorldSizeAttr(opName) || !CheckGroupAttr(opName) ||
            !CheckTensorFormats(opName) || !CheckTensorDtypes(opName) || !CheckQuantAndYDtypeAttr(opName) ||
            !CheckOtherAttrs(opName)) {
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    bool CheckInputDescs(const char *opName)
    {
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
            return false;
        }
        return true;
    }

    bool CheckWorldSizeAttr(const char *opName)
    {
        auto *wsPtr = context_->GetAttrs()->GetAttrPointer<int64_t>(IDX_ATTR_WORLD_SIZE);
        if (!wsPtr) {
            OP_LOGE_WITH_INVALID_INPUT(opName, "world_size");
            return false;
        }
        if (*wsPtr <= 0) {
            OP_LOGE_WITH_INVALID_ATTR(opName, "world_size", std::to_string(*wsPtr).c_str(), "positive");
            return false;
        }
        uint64_t worldSize = static_cast<uint64_t>(*wsPtr);
        bool isPowerOfTwo = (worldSize & (worldSize - 1)) == 0;
        if (worldSize < MIN_WORLD_SIZE || worldSize > MAX_WORLD_SIZE || !isPowerOfTwo) {
            OP_LOGE_WITH_INVALID_ATTR(opName, "world_size", std::to_string(worldSize).c_str(), "2/4/8/16");
            return false;
        }
        return true;
    }

    bool CheckGroupAttr(const char *opName)
    {
        auto *groupPtr = context_->GetAttrs()->GetAttrPointer<char>(IDX_ATTR_GROUP);
        if (!groupPtr) {
            OP_LOGE_WITH_INVALID_INPUT(opName, "group");
            return false;
        }
        uint64_t groupLen = strnlen(groupPtr, MAX_GROUP_NAME_LEN);
        if (groupLen == 0 || groupLen == MAX_GROUP_NAME_LEN) {
            OP_LOGE_FOR_INVALID_VALUE(
                opName, "group", std::to_string(groupLen).c_str(),
                (std::string("length in (0, ") + std::to_string(MAX_GROUP_NAME_LEN) + ")").c_str());
            return false;
        }
        return true;
    }

    bool CheckTensorFormats(const char *opName)
    {
        auto *x1Desc = context_->GetInputDesc(IDX_INPUT_X1);
        auto *x2Desc = context_->GetInputDesc(IDX_INPUT_X2);
        auto *yDesc = context_->GetOutputDesc(IDX_OUTPUT_Y);
        auto x1Fmt = static_cast<ge::Format>(ge::GetPrimaryFormat(x1Desc->GetStorageFormat()));
        auto x2Fmt = static_cast<ge::Format>(ge::GetPrimaryFormat(x2Desc->GetStorageFormat()));
        auto yFmt = static_cast<ge::Format>(ge::GetPrimaryFormat(yDesc->GetStorageFormat()));
        if (x1Fmt != ge::FORMAT_ND || x2Fmt != ge::FORMAT_ND) {
            OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
                opName, "x1, x2", (Ops::Base::ToString(x1Fmt) + "," + Ops::Base::ToString(x2Fmt)).c_str(),
                "x1 and x2 format must be ND");
            return false;
        }
        if (yFmt != ge::FORMAT_ND) {
            OP_LOGE_FOR_INVALID_FORMAT(opName, "y", Ops::Base::ToString(yFmt).c_str(), "ND");
            return false;
        }
        return true;
    }

    bool CheckTensorDtypes(const char *opName)
    {
        auto *x1Desc = context_->GetInputDesc(IDX_INPUT_X1);
        auto *x2Desc = context_->GetInputDesc(IDX_INPUT_X2);
        auto *yDesc = context_->GetOutputDesc(IDX_OUTPUT_Y);
        auto x1Dtype = x1Desc->GetDataType();
        auto x2Dtype = x2Desc->GetDataType();
        if ((x1Dtype != ge::DT_FLOAT8_E4M3FN && x1Dtype != ge::DT_FLOAT8_E5M2 && x1Dtype != ge::DT_FLOAT4_E2M1) ||
            (x2Dtype != ge::DT_FLOAT8_E4M3FN && x2Dtype != ge::DT_FLOAT8_E5M2 && x2Dtype != ge::DT_FLOAT4_E2M1)) {
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                opName, "x1, x2", (Ops::Base::ToString(x1Dtype) + "," + Ops::Base::ToString(x2Dtype)).c_str(),
                "x1/x2 dtype must be FP8_E4M3/FP8_E5M2/FP4_E2M1");
            return false;
        }
        if ((x1Dtype == ge::DT_FLOAT4_E2M1) != (x2Dtype == ge::DT_FLOAT4_E2M1)) {
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                opName, "x1, x2", (Ops::Base::ToString(x1Dtype) + "," + Ops::Base::ToString(x2Dtype)).c_str(),
                "FP4 requires both x1 and x2 to be FP4_E2M1");
            return false;
        }
        auto yDtype = yDesc->GetDataType();
        if (yDtype != ge::DT_BF16 && yDtype != ge::DT_FLOAT16) {
            OP_LOGE_FOR_INVALID_DTYPE(opName, "y", Ops::Base::ToString(yDtype).c_str(), "BF16/FP16");
            return false;
        }
        return CheckScaleDesc(opName) && CheckBiasDesc(opName);
    }

    bool CheckScaleDesc(const char *opName)
    {
        auto *x1ScaleDesc = context_->GetOptionalInputDesc(IDX_INPUT_X1_SCALE);
        auto *x2ScaleDesc = context_->GetOptionalInputDesc(IDX_INPUT_X2_SCALE);
        if (!x1ScaleDesc || !x2ScaleDesc) {
            if (!x1ScaleDesc)
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    opName, "x1_scale", "nil", "only MXFP quantization is currently supported, x1_scale is required");
            if (!x2ScaleDesc)
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    opName, "x2_scale", "nil", "only MXFP quantization is currently supported, x2_scale is required");
            return false;
        }
        auto x1ScaleFmt = static_cast<ge::Format>(ge::GetPrimaryFormat(x1ScaleDesc->GetStorageFormat()));
        auto x2ScaleFmt = static_cast<ge::Format>(ge::GetPrimaryFormat(x2ScaleDesc->GetStorageFormat()));
        if ((x1ScaleFmt != ge::FORMAT_ND && x1ScaleFmt != ge::FORMAT_NCL) ||
            (x2ScaleFmt != ge::FORMAT_ND && x2ScaleFmt != ge::FORMAT_NCL)) {
            OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
                opName, "x1_scale, x2_scale",
                (Ops::Base::ToString(x1ScaleFmt) + "," + Ops::Base::ToString(x2ScaleFmt)).c_str(),
                "x1_scale and x2_scale format must be ND or NCL");
            return false;
        }
        auto x1ScaleDtype = x1ScaleDesc->GetDataType();
        auto x2ScaleDtype = x2ScaleDesc->GetDataType();
        if (x1ScaleDtype != ge::DT_FLOAT8_E8M0 || x2ScaleDtype != ge::DT_FLOAT8_E8M0) {
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                opName, "x1_scale, x2_scale",
                (Ops::Base::ToString(x1ScaleDtype) + "," + Ops::Base::ToString(x2ScaleDtype)).c_str(),
                "only MXFP quantization is currently supported, x1_scale and x2_scale must be FP8_E8M0");
            return false;
        }
        return true;
    }

    bool CheckBiasDesc(const char *opName)
    {
        auto *biasDesc = context_->GetOptionalInputDesc(IDX_INPUT_BIAS);
        if (biasDesc == nullptr) {
            return true;
        }
        auto biasDtype = biasDesc->GetDataType();
        if (biasDtype != ge::DT_FLOAT) {
            OP_LOGE_FOR_INVALID_DTYPE(opName, "bias", Ops::Base::ToString(biasDtype).c_str(), "FLOAT");
            return false;
        }
        auto biasFmt = static_cast<ge::Format>(ge::GetPrimaryFormat(biasDesc->GetStorageFormat()));
        if (biasFmt != ge::FORMAT_ND) {
            OP_LOGE_FOR_INVALID_FORMAT(opName, "bias", Ops::Base::ToString(biasFmt).c_str(), "ND");
            return false;
        }
        return true;
    }

    bool CheckQuantAndYDtypeAttr(const char *opName)
    {
        auto *x1QmPtr = context_->GetAttrs()->GetAttrPointer<int64_t>(IDX_ATTR_X1_QUANT_MODE);
        auto *x2QmPtr = context_->GetAttrs()->GetAttrPointer<int64_t>(IDX_ATTR_X2_QUANT_MODE);
        int64_t x1QuantMode = x1QmPtr ? *x1QmPtr : static_cast<int64_t>(MX_QUANT_MODE);
        int64_t x2QuantMode = x2QmPtr ? *x2QmPtr : static_cast<int64_t>(MX_QUANT_MODE);
        if (x1QuantMode != static_cast<int64_t>(MX_QUANT_MODE) || x2QuantMode != static_cast<int64_t>(MX_QUANT_MODE)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                opName, "x1_quant_mode, x2_quant_mode",
                (std::to_string(x1QuantMode) + "," + std::to_string(x2QuantMode)).c_str(),
                "x1_quant_mode and x2_quant_mode must be MX_QUANT(6)");
            return false;
        }
        auto *yDtypePtr = context_->GetAttrs()->GetAttrPointer<int64_t>(IDX_ATTR_Y_DTYPE);
        if (yDtypePtr != nullptr && *yDtypePtr != static_cast<int64_t>(ge::DT_BF16) &&
            *yDtypePtr != static_cast<int64_t>(ge::DT_FLOAT16)) {
            OP_LOGE_WITH_INVALID_ATTR(opName, "y_dtype", std::to_string(*yDtypePtr).c_str(), "BF16/FP16");
            return false;
        }
        return true;
    }

    bool CheckOtherAttrs(const char *opName)
    {
        auto *tx1 = context_->GetAttrs()->GetAttrPointer<bool>(IDX_ATTR_TRANSPOSE_X1);
        if (tx1 != nullptr && *tx1) {
            OP_LOGE_WITH_INVALID_ATTR(opName, "transpose_x1", "true", "false");
            return false;
        }
        auto *tx2 = context_->GetAttrs()->GetAttrPointer<bool>(IDX_ATTR_TRANSPOSE_X2);
        if (!tx2 || !(*tx2)) {
            OP_LOGE_WITH_INVALID_ATTR(opName, "transpose_x2",
                                      tx2 ? std::to_string(static_cast<int>(*tx2)).c_str() : "nil", "true");
            return false;
        }
        return CheckGroupSizeAttr(opName);
    }

    bool CheckGroupSizeAttr(const char *opName)
    {
        auto *gsPtr = context_->GetAttrs()->GetAttrPointer<int64_t>(IDX_ATTR_GROUP_SIZE);
        if (!gsPtr) {
            return true;
        }
        uint64_t gs = static_cast<uint64_t>(*gsPtr);
        uint64_t gsK = gs & GROUP_MNK_BIT_SIZE;
        uint64_t gsN = (gs >> GROUP_N_OFFSET) & GROUP_MNK_BIT_SIZE;
        uint64_t gsM = (gs >> GROUP_M_OFFSET) & GROUP_MNK_BIT_SIZE;

        // 自动推导：groupSize 中为 0 的维度根据 x1/x2/x1Scale/x2Scale 的 shape 推断。
        // MX 场景 groupSize=[1,1,32]，推断公式 groupSizeM=m/scaleM、groupSizeN=n/scaleN、groupSizeK=k/scaleK。
        mc2tiling::Mc2MatmulShapeInfo shapeInfo = {context_->GetInputShape(IDX_INPUT_X1),
                                                   context_->GetInputShape(IDX_INPUT_X2),
                                                   context_->GetOptionalInputShape(IDX_INPUT_X1_SCALE),
                                                   context_->GetOptionalInputShape(IDX_INPUT_X2_SCALE),
                                                   true, // isMxfp：本算子为 MX 量化
                                                   true, // isBTrans：transposeX2 恒为 true
                                                   opName};
        if (!mc2tiling::Mc2TilingUtils::InferGroupSize(shapeInfo, gsM, gsN, gsK)) {
            return false;
        }

        if (gsM != MX_GROUP_M || gsN != MX_GROUP_N || gsK != MX_GROUP_K) {
            OP_LOGE_WITH_INVALID_ATTR(
                opName, "group_size",
                ("[M=" + std::to_string(gsM) + ",N=" + std::to_string(gsN) + ",K=" + std::to_string(gsK) + "]").c_str(),
                ("[M=" + std::to_string(MX_GROUP_M) + ",N=" + std::to_string(MX_GROUP_N) +
                 ",K=" + std::to_string(MX_GROUP_K) + "]")
                    .c_str());
            return false;
        }
        return true;
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

        if (!CheckMatmulDimsAndBounds(opName) || !CheckMatmulDivisibility(opName) || !CheckBiasShape(opName) ||
            !CheckScaleShapes(opName)) {
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    bool CheckMatmulDimsAndBounds(const char *opName)
    {
        auto *x1Shape = context_->GetInputShape(IDX_INPUT_X1);
        auto *x2Shape = context_->GetInputShape(IDX_INPUT_X2);
        auto *yShape = context_->GetOutputShape(IDX_OUTPUT_Y);
        uint64_t x1DimNum = x1Shape->GetStorageShape().GetDimNum();
        uint64_t x2DimNum = x2Shape->GetStorageShape().GetDimNum();
        uint64_t yDimNum = yShape->GetStorageShape().GetDimNum();
        if (x1DimNum != 2 || x2DimNum != 2) {
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                opName, "x1, x2", (std::to_string(x1DimNum) + "D," + std::to_string(x2DimNum) + "D").c_str(),
                "x1 and x2 must be 2D");
            return false;
        }
        if (yDimNum != 2) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(opName, "y", std::to_string(yDimNum).c_str(), "2D");
            return false;
        }

        uint64_t x1M = x1Shape->GetStorageShape().GetDim(0U);
        uint64_t x1K = x1Shape->GetStorageShape().GetDim(1U);
        uint64_t x2N = x2Shape->GetStorageShape().GetDim(0U);
        uint64_t x2K = x2Shape->GetStorageShape().GetDim(1U);

        if (x1M == 0 || x1K == 0 || x2N == 0 || x2K == 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "x1, x2",
                                                  ("x1=[" + std::to_string(x1M) + "," + std::to_string(x1K) +
                                                   "], x2=[" + std::to_string(x2N) + "," + std::to_string(x2K) + "]")
                                                      .c_str(),
                                                  "The dimensions of x1 and x2 must be non-zero.");
            return false;
        }
        // y 的 dim0 = M * world_size，先取 world_size 再做上界比较
        auto *wsPtr = context_->GetAttrs()->GetAttrPointer<int64_t>(IDX_ATTR_WORLD_SIZE);
        uint64_t worldSize = static_cast<uint64_t>(wsPtr ? *wsPtr : 1);
        if (x1M * worldSize > static_cast<uint64_t>(MAX_INT32_VAL) || x2N > static_cast<uint64_t>(MAX_INT32_VAL)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "x1 dim0, x2 dim0",
                                                  ("M=" + std::to_string(x1M) + ", N=" + std::to_string(x2N)).c_str(),
                                                  "The dim0 of x1 * world_size and dim0 of x2 must not exceed "
                                                  "INT32_MAX.");
            return false;
        }
        if (x1K > K_MAX_VAL || x2K > K_MAX_VAL) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                opName, "x1_k, x2_k", ("x1_k=" + std::to_string(x1K) + ", x2_k=" + std::to_string(x2K)).c_str(),
                ("The K dimension must not exceed " + std::to_string(K_MAX_VAL) + ".").c_str());
            return false;
        }
        return true;
    }

    bool CheckMatmulDivisibility(const char *opName)
    {
        auto *x1Shape = context_->GetInputShape(IDX_INPUT_X1);
        auto *x2Shape = context_->GetInputShape(IDX_INPUT_X2);
        auto *yShape = context_->GetOutputShape(IDX_OUTPUT_Y);
        auto *wsPtr = context_->GetAttrs()->GetAttrPointer<int64_t>(IDX_ATTR_WORLD_SIZE);
        uint64_t worldSize = static_cast<uint64_t>(wsPtr ? *wsPtr : 1);

        uint64_t x1M = x1Shape->GetStorageShape().GetDim(0U);
        uint64_t x1K = x1Shape->GetStorageShape().GetDim(1U);
        uint64_t x2N = x2Shape->GetStorageShape().GetDim(0U);
        uint64_t x2K = x2Shape->GetStorageShape().GetDim(1U);
        uint64_t yM = yShape->GetStorageShape().GetDim(0U);
        uint64_t yN = yShape->GetStorageShape().GetDim(1U);

        if (x1K % MX_SCALE_ALIGN != 0 || x2K % MX_SCALE_ALIGN != 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                opName, "x1_k, x2_k", ("x1_k=" + std::to_string(x1K) + ", x2_k=" + std::to_string(x2K)).c_str(),
                ("The K dimension must be divisible by " + std::to_string(MX_SCALE_ALIGN) + ".").c_str());
            return false;
        }
        if (x2K != x1K) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "x2_k", std::to_string(x2K).c_str(),
                                                  ("x2.K must equal x1.K = " + std::to_string(x1K) + ".").c_str());
            return false;
        }
        if (x2N % worldSize != 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "x2 dim0", ("N=" + std::to_string(x2N)).c_str(),
                                                  "The dim0 of x2 must be divisible by world_size.");
            return false;
        }
        if (yM != x1M * worldSize || yN != x2N / worldSize) {
            OP_LOGE_FOR_INVALID_SHAPE(
                opName, "y", ("[" + std::to_string(yM) + "," + std::to_string(yN) + "]").c_str(),
                ("[" + std::to_string(x1M * worldSize) + "," + std::to_string(x2N / worldSize) + "]").c_str());
            return false;
        }
        return true;
    }

    bool CheckBiasShape(const char *opName)
    {
        auto *biasShape = context_->GetOptionalInputShape(IDX_INPUT_BIAS);
        if (biasShape == nullptr) {
            return true;
        }
        if (biasShape->GetStorageShape().GetDimNum() != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(opName, "bias",
                                         std::to_string(biasShape->GetStorageShape().GetDimNum()).c_str(), "1D");
            return false;
        }
        uint64_t biasDim0 = biasShape->GetStorageShape().GetDim(0U);
        uint64_t x2N = context_->GetInputShape(IDX_INPUT_X2)->GetStorageShape().GetDim(0U);
        if (biasDim0 != x2N) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "bias dim0", std::to_string(biasDim0).c_str(),
                                                  ("bias dim0 must equal N = " + std::to_string(x2N) + ".").c_str());
            return false;
        }
        return true;
    }

    bool CheckScaleShapes(const char *opName)
    {
        auto *x1ScaleShape = context_->GetOptionalInputShape(IDX_INPUT_X1_SCALE);
        auto *x2ScaleShape = context_->GetOptionalInputShape(IDX_INPUT_X2_SCALE);
        if (!x1ScaleShape || !x2ScaleShape) {
            if (!x1ScaleShape)
                OP_LOGE_WITH_INVALID_INPUT(opName, "x1_scale");
            if (!x2ScaleShape)
                OP_LOGE_WITH_INVALID_INPUT(opName, "x2_scale");
            return false;
        }

        uint64_t x1ScaleDimNum = x1ScaleShape->GetStorageShape().GetDimNum();
        uint64_t x2ScaleDimNum = x2ScaleShape->GetStorageShape().GetDimNum();
        if (x1ScaleDimNum != SCALE_DIM_NUM || x2ScaleDimNum != SCALE_DIM_NUM) {
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                opName, "x1_scale, x2_scale",
                (std::to_string(x1ScaleDimNum) + "D," + std::to_string(x2ScaleDimNum) + "D").c_str(),
                "x1_scale and x2_scale must be 3D");
            return false;
        }

        uint64_t x1ScaleM = x1ScaleShape->GetStorageShape().GetDim(0U);
        uint64_t x1ScaleK = x1ScaleShape->GetStorageShape().GetDim(1U);
        uint64_t x1ScaleLast = x1ScaleShape->GetStorageShape().GetDim(2U);
        uint64_t x2ScaleN = x2ScaleShape->GetStorageShape().GetDim(0U);
        uint64_t x2ScaleK = x2ScaleShape->GetStorageShape().GetDim(1U);
        uint64_t x2ScaleLast = x2ScaleShape->GetStorageShape().GetDim(2U);

        uint64_t x1M = context_->GetInputShape(IDX_INPUT_X1)->GetStorageShape().GetDim(0U);
        uint64_t x1K = context_->GetInputShape(IDX_INPUT_X1)->GetStorageShape().GetDim(1U);
        uint64_t x2N = context_->GetInputShape(IDX_INPUT_X2)->GetStorageShape().GetDim(0U);
        uint64_t x2K = context_->GetInputShape(IDX_INPUT_X2)->GetStorageShape().GetDim(1U);

        if (x1ScaleM != x1M || x1ScaleK != x1K / MX_SCALE_ALIGN || x1ScaleLast != SCALE_LAST_DIM) {
            OP_LOGE_FOR_INVALID_SHAPE(opName, "x1_scale",
                                      ("[m=" + std::to_string(x1ScaleM) + ",k=" + std::to_string(x1ScaleK) +
                                       ",last=" + std::to_string(x1ScaleLast) + "]")
                                          .c_str(),
                                      ("[m=" + std::to_string(x1M) + ",k=" + std::to_string(x1K / MX_SCALE_ALIGN) +
                                       ",last=" + std::to_string(SCALE_LAST_DIM) + "]")
                                          .c_str());
            return false;
        }
        if (x2ScaleN != x2N || x2ScaleK != x2K / MX_SCALE_ALIGN || x2ScaleLast != SCALE_LAST_DIM) {
            OP_LOGE_FOR_INVALID_SHAPE(opName, "x2_scale",
                                      ("[n=" + std::to_string(x2ScaleN) + ",k=" + std::to_string(x2ScaleK) +
                                       ",last=" + std::to_string(x2ScaleLast) + "]")
                                          .c_str(),
                                      ("[n=" + std::to_string(x2N) + ",k=" + std::to_string(x2K / MX_SCALE_ALIGN) +
                                       ",last=" + std::to_string(SCALE_LAST_DIM) + "]")
                                          .c_str());
            return false;
        }
        return true;
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
        // world_size 已在 CheckWorldSizeAttr 校验，此处仍做判空防御
        auto *wsPtr = context_->GetAttrs()->GetAttrPointer<int64_t>(IDX_ATTR_WORLD_SIZE);
        if (!wsPtr || *wsPtr <= 0) {
            OP_LOGE_WITH_INVALID_INPUT(opName, "world_size");
            return ge::GRAPH_FAILED;
        }
        worldSize_ = static_cast<uint64_t>(*wsPtr);
        m_ = x1ss.GetDim(0U);  // M（本算子 x1 的 M 即 per-rank 完整 M，不除 world_size）
        k_ = x1ss.GetDim(1U);  // K（完整 K）
        n_ = x2ss.GetDim(0U);  // N（x2 = [N, K]，dim0 恒为 N，transpose_x2 强制 true）
        np_ = n_ / worldSize_; // Np = N / world_size
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoOpTiling() override
    {
        const char *opName = context_->GetNodeName();
        if (CheckOpInputInfo() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        auto *hccBufPtr = context_->GetAttrs()->GetAttrPointer<int64_t>(IDX_ATTR_HCCL_BUFFER_SIZE);
        OP_LOGI(opName, "[hcclBufferSize] builder-assigned hcclBufferSize = %ld",
                hccBufPtr ? static_cast<long>(*hccBufPtr) : -1L);

        auto *rt = context_->GetRawTilingData();
        if (rt == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT(opName, "rawTilingData");
            return ge::GRAPH_FAILED;
        }
        auto cap = rt->GetCapacity();
        if (cap < sizeof(MatmulAllToAllTilingData)) {
            OP_LOGE_WITH_INVALID_INPUT(opName, "rawTilingData");
            return ge::GRAPH_FAILED;
        }
        memset_s(rt->GetData(), cap, 0, cap);

        auto *td = reinterpret_cast<MatmulAllToAllTilingData *>(rt->GetData());

        // 按 x1/x2 dtype 选择 QuantMatmulTilingSwat 引擎模板（5 分支，见下方正式 tiling 段）。
        auto *x1Desc = context_->GetInputDesc(IDX_INPUT_X1);
        auto *x2Desc = context_->GetInputDesc(IDX_INPUT_X2);
        if (x1Desc == nullptr || x2Desc == nullptr) {
            if (!x1Desc)
                OP_LOGE_WITH_INVALID_INPUT(opName, "x1 desc");
            if (!x2Desc)
                OP_LOGE_WITH_INVALID_INPUT(opName, "x2 desc");
            return ge::GRAPH_FAILED;
        }
        auto x1Dtype = x1Desc->GetDataType();
        auto x2Dtype = x2Desc->GetDataType();

        // M 轴切分（临时硬编码策略，对齐 ST cases.csv 全部用例的 tileCnt=4 形态）：
        // - m 可被 4 整除：4 个等大 tile（tileM = m/4），无 tail；
        // - 不可整除：前 3 个 tile 等大（tileM = floor(m/4)），余量整体作为 1 个 tail tile
        //   （tail = m - 3*floor(m/4)，与 head 至多相差 3 行，尽量接近）；
        // - m < 4：无法切出非零 head，整体作为单个 tail tile。
        constexpr uint64_t FIXED_TILE_CNT = 4;
        const bool mDivisible = (m_ % FIXED_TILE_CNT == 0);
        uint64_t headMSize = m_ / FIXED_TILE_CNT;
        uint64_t headTileCnt = mDivisible ? FIXED_TILE_CNT : (FIXED_TILE_CNT - 1);
        if (headMSize == 0) {
            headTileCnt = 0;
        }
        uint64_t tailMSize = mDivisible ? 0 : (m_ - headTileCnt * headMSize);
        uint64_t tailTileCnt = (tailMSize > 0) ? 1 : 0;

        // hccl_buffer_size 容量校验（win 区精确大小）：每个 rank 的 win 区需容纳所有 rank 写来的
        // y 分片，即 totalTileCnt * worldSize * maxTileM * np * 2（y 为 BF16/FP16，2 字节）+ 2MB 预留。
        uint64_t maxTileM = std::max(headMSize, tailMSize);
        uint64_t totalTileCnt = headTileCnt + tailTileCnt;
        constexpr uint64_t HCCL_BUFFER_RESERVED_BYTES = 2UL * 1024UL * 1024UL;
        uint64_t needBytes = totalTileCnt * worldSize_ * maxTileM * np_ * 2UL + HCCL_BUFFER_RESERVED_BYTES;
        if (hccBufPtr != nullptr && static_cast<uint64_t>(*hccBufPtr) < needBytes) {
            OP_LOGE_WITHOUT_REPORT(opName,
                                   "[hcclBufferSize] hcclBufferSize(%ld) is less than "
                                   "totalTileCnt(%lu) * worldSize(%lu) * maxTileM(%lu) * np(%lu) * 2 + "
                                   "reserved(%lu) = %lu bytes",
                                   static_cast<long>(*hccBufPtr), totalTileCnt, worldSize_, maxTileM, np_,
                                   HCCL_BUFFER_RESERVED_BYTES, needBytes);
            return ge::GRAPH_FAILED;
        }

        // 正式 QBMM tiling：LocalDelay 固定开启，remoteBatch = worldSize - 1；
        // localQbmm 覆盖本 rank 完整 M 的延迟计算，batch = 1。
        uint64_t remoteBatch = worldSize_ - 1;
        if (x1Dtype == ge::DT_FLOAT4_E2M1 && x2Dtype == ge::DT_FLOAT4_E2M1) {
            QuantMatmulTilingSwat<mm::DataType::DT_FLOAT4_E2M1, mm::DataType::DT_FLOAT4_E2M1> tilingEngine;
            tilingEngine.SetPlatformInfoPtr(context_->GetPlatformInfo());
            tilingEngine.GetTilingData(headMSize, np_, k_, false, true, td->tileQbmmTilingData, remoteBatch);
            if (tailMSize > 0) {
                tilingEngine.GetTilingData(tailMSize, np_, k_, false, true, td->tailQbmmTilingData, remoteBatch);
            }
            tilingEngine.GetTilingData(m_, np_, k_, false, true, td->localQbmmTilingData, 1);
        } else if (x1Dtype == ge::DT_FLOAT8_E4M3FN && x2Dtype == ge::DT_FLOAT8_E4M3FN) {
            QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN> tilingEngine;
            tilingEngine.SetPlatformInfoPtr(context_->GetPlatformInfo());
            tilingEngine.GetTilingData(headMSize, np_, k_, false, true, td->tileQbmmTilingData, remoteBatch);
            if (tailMSize > 0) {
                tilingEngine.GetTilingData(tailMSize, np_, k_, false, true, td->tailQbmmTilingData, remoteBatch);
            }
            tilingEngine.GetTilingData(m_, np_, k_, false, true, td->localQbmmTilingData, 1);
        } else if (x1Dtype == ge::DT_FLOAT8_E5M2 && x2Dtype == ge::DT_FLOAT8_E5M2) {
            QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E5M2, mm::DataType::DT_FLOAT8_E5M2> tilingEngine;
            tilingEngine.SetPlatformInfoPtr(context_->GetPlatformInfo());
            tilingEngine.GetTilingData(headMSize, np_, k_, false, true, td->tileQbmmTilingData, remoteBatch);
            if (tailMSize > 0) {
                tilingEngine.GetTilingData(tailMSize, np_, k_, false, true, td->tailQbmmTilingData, remoteBatch);
            }
            tilingEngine.GetTilingData(m_, np_, k_, false, true, td->localQbmmTilingData, 1);
        } else if (x1Dtype == ge::DT_FLOAT8_E4M3FN && x2Dtype == ge::DT_FLOAT8_E5M2) {
            QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E5M2> tilingEngine;
            tilingEngine.SetPlatformInfoPtr(context_->GetPlatformInfo());
            tilingEngine.GetTilingData(headMSize, np_, k_, false, true, td->tileQbmmTilingData, remoteBatch);
            if (tailMSize > 0) {
                tilingEngine.GetTilingData(tailMSize, np_, k_, false, true, td->tailQbmmTilingData, remoteBatch);
            }
            tilingEngine.GetTilingData(m_, np_, k_, false, true, td->localQbmmTilingData, 1);
        } else if (x1Dtype == ge::DT_FLOAT8_E5M2 && x2Dtype == ge::DT_FLOAT8_E4M3FN) {
            QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E5M2, mm::DataType::DT_FLOAT8_E4M3FN> tilingEngine;
            tilingEngine.SetPlatformInfoPtr(context_->GetPlatformInfo());
            tilingEngine.GetTilingData(headMSize, np_, k_, false, true, td->tileQbmmTilingData, remoteBatch);
            if (tailMSize > 0) {
                tilingEngine.GetTilingData(tailMSize, np_, k_, false, true, td->tailQbmmTilingData, remoteBatch);
            }
            tilingEngine.GetTilingData(m_, np_, k_, false, true, td->localQbmmTilingData, 1);
        } else {
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                opName, "x1, x2", (Ops::Base::ToString(x1Dtype) + "," + Ops::Base::ToString(x2Dtype)).c_str(),
                "x1/x2 dtype combination is not supported");
            return ge::GRAPH_FAILED;
        }

        // matmul 实际分配核数必须 >= worldSize（kernel 侧通信切分依赖每 rank 至少一个核），
        // 否则部分 rank 无核参与 alltoall 导致死锁。
        usedCoreNum_ = td->tileQbmmTilingData.usedCoreNum;
        if (static_cast<uint64_t>(usedCoreNum_) < worldSize_) {
            OP_LOGE_WITHOUT_REPORT(opName,
                                   "[coreNum] usedCoreNum(%u) is less than rankSize(%lu), "
                                   "matmul tiling 分配核数不足以覆盖所有 rank",
                                   usedCoreNum_, worldSize_);
            return ge::GRAPH_FAILED;
        }

        // 可用核数 = min(aicNum, aivNum)，必须 >= worldSize（通信切分依赖每 rank 至少一个核）
        uint64_t availCoreNum = std::min(quantPlatformInfo_.aicNum, quantPlatformInfo_.aivNum);
        if (availCoreNum < worldSize_) {
            OP_LOGE_WITHOUT_REPORT(opName, "[coreNum] availCoreNum(%lu) is less than rankSize(%lu)", availCoreNum,
                                   worldSize_);
            return ge::GRAPH_FAILED;
        }

        td->commTilingData.splitAxisTileSize = headMSize;
        td->commTilingData.splitAxisTileCnt = headTileCnt;
        td->commTilingData.splitAxisTailSize = tailMSize;
        td->commTilingData.splitAxisTailCnt = tailTileCnt;
        td->commTilingData.nonSplitAxisSize = np_;

        OP_LOGI(opName, "comm tiling: tileSize=%lu, tileCnt=%lu, tailSize=%lu, tailCnt=%lu, nonSplitSize=%lu",
                td->commTilingData.splitAxisTileSize, td->commTilingData.splitAxisTileCnt,
                td->commTilingData.splitAxisTailSize, td->commTilingData.splitAxisTailCnt,
                td->commTilingData.nonSplitAxisSize);

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    uint64_t GetTilingKey() const override
    {
        return GET_TPL_TILING_KEY(MX_QUANT_MODE, 1 /*x2Transpose*/, DTYPE_BIAS_FP32, 0 /*isSmallK*/,
                                  ALL2ALL_COMM_TYPE_UDMA);
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        auto *platformInfo = context_->GetPlatformInfo();
        if (platformInfo == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "platformInfo");
            return ge::GRAPH_FAILED;
        }
        platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
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
    uint64_t m_{0}, k_{0}, n_{0}, np_{0};
    uint64_t worldSize_{1};
    QuantMatmulPlatformInfo quantPlatformInfo_;
};

} // namespace MC2Tiling

using MatmulAlltoAllV2TilingClass = MC2Tiling::MatmulAlltoAllV2TilingClass;
REGISTER_TILING_TEMPLATE_WITH_ARCH(MatmulAlltoAllV2, MatmulAlltoAllV2TilingClass,
                                   static_cast<int32_t>(NpuArch::DAV_3510), 1);

#endif
