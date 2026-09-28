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
 * \file grouped_matmul_swiglu_quant_v2_weight_quant_checker.h
 * \brief
 */

#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <type_traits>
#include <exe_graph/runtime/tiling_context.h>
#include <graph/utils/type_utils.h>
#include <graph/types.h>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "err/ops_err.h"
#include "../grouped_matmul_swiglu_quant_v2_tiling.h"
#include "../../../op_kernel/arch35/weight_quant_asw/grouped_matmul_swiglu_quant_v2_weight_quant_asw_tiling_data.h"
#include "../../../../grouped_matmul/op_host/grouped_matmul_host_util.h"

namespace {
static constexpr const char *OP_NAME = "GroupedMatmulSwigluQuantV2";
}

namespace optiling {

namespace GMMSQWeightQuantChecker {

using namespace GroupedMatmulSwigluQuantV2Tiling;

struct GMMSQWeightQuantInputParams {
    bool wTrans = false;
    ge::Format wFormat = ge::FORMAT_ND;
    int64_t groupListType = -1;
    int64_t dequantMode = -1;
    int64_t dequantDtype = -1;
    int64_t quantMode = -1;
    int64_t quantDtype = -1;
    int64_t mSize = -1;
    int64_t kSize = -1;
    int64_t nSize = -1;
    int64_t groupNum = -1;
};

constexpr uint32_t AIC_AIV_CORE_RATIO_2 = 2;
constexpr uint32_t AIC_AIV_CORE_RATIO_1 = 1;
constexpr uint32_t MX_SCALE_BLOCK_SIZE = 64;
constexpr uint32_t MX_LAST_DIM_2 = 2;
constexpr uint32_t MX_LAST_DIM_1 = 1;
constexpr uint32_t ALIGN_SIZE_K = 32;
constexpr uint32_t ALIGN_SIZE_N = 128;
constexpr int64_t MIN_K_SIZE = 32;
constexpr int64_t MIN_N_SIZE = 128;
constexpr int64_t MAX_GROUP_NUM = 1024;
constexpr int64_t MX_QUANT_MODE = 2;

constexpr uint64_t DIM_NUM_X = 2;
constexpr uint64_t DIM_NUM_WEIGHT_ND = 3;

constexpr int64_t C0_SIZE = 32;
constexpr int64_t CUBE_BLOCK = 16;

template <typename T>
constexpr T GetOrDefault(const T *ptr, T defaultValue)
{
    return ptr == nullptr ? defaultValue : *ptr;
}

template <typename T>
std::string ToVectorString(const std::vector<T> &vec)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < vec.size(); i++) {
        if constexpr (std::is_same<T, ge::DataType>::value) {
            oss << ge::TypeUtils::DataTypeToSerialString(vec[i]);
        } else {
            oss << ge::TypeUtils::FormatToSerialString(vec[i]);
        }
        if (i != vec.size() - 1) {
            oss << ", ";
        }
    }
    oss << "]";
    return oss.str();
}

bool CheckCoreNum(NpuArch npuArch, uint32_t aicNum, uint32_t aivNum);
bool CheckDynamicInputShapeNotEmpty(const gert::TilingContext &context, uint32_t inputIndex, const char *inputName);
bool IsDynamicInputEmpty(const gert::TilingContext &context, uint32_t index);
bool CheckTilingDataCapacity(gert::TilingContext &context, uint64_t tilingDataSize);
bool CheckContext(gert::TilingContext &context, uint64_t tilingDataSize);
bool GetAttrs(GMMSQWeightQuantInputParams &params, const gert::TilingContext &context);
bool CheckAttrs(const GMMSQWeightQuantInputParams &params);
bool CheckDtypes(const gert::TilingContext &context, NpuArch npuArch);
bool GetInputs(GMMSQWeightQuantInputParams &params, const gert::TilingContext &context);
bool CheckInputFormats(const gert::TilingContext &context, const GMMSQWeightQuantInputParams &params, NpuArch npuArch);
std::string ToShapeString(const std::vector<int64_t> &shape);
bool CheckInputShape(const char *variableName, const gert::Shape &shape, const std::vector<int64_t> &expectedShape);
bool IsInputShapeValid(const GMMSQWeightQuantInputParams &params);
int64_t GetC0Size(ge::Format format, ge::DataType dType);
bool CheckInputs(const gert::TilingContext &context, const GMMSQWeightQuantInputParams &params, NpuArch npuArch);
ge::graphStatus GetShapeAttrsInfo(gert::TilingContext &context, GMMSQWeightQuantInputParams &params, NpuArch npuArch);

} // namespace GMMSQWeightQuantChecker
} // namespace optiling