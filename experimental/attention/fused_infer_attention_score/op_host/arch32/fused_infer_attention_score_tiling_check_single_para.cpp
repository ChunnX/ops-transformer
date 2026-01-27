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
 * \file fused_infer_attention_score_tiling_check_single_para.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "fused_infer_attention_score_tiling_check.h"

using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
namespace optiling {
const std::map<std::string, std::vector<ge::DataType>> DTYPE_SUPPORT_MAP = {
    {QUERY_NAME,                  {ge::DT_FLOAT16, ge::DT_BF16}},
    {KEY_NAME,                    {ge::DT_FLOAT16, ge::DT_BF16}},
    {VALUE_NAME,                  {ge::DT_FLOAT16, ge::DT_BF16}},
    {QUERY_ROPE_NAME,             {ge::DT_FLOAT16, ge::DT_BF16}},
    {KEY_ROPE_NAME,               {ge::DT_FLOAT16, ge::DT_BF16}},
    {ATTEN_OUT_NAME,              {ge::DT_FLOAT16, ge::DT_BF16}},
};

const std::map<std::string, std::vector<FiaLayout>> LAYOUT_SUPPORT_MAP = {
    {QUERY_NAME,      {FiaLayout::BSH, FiaLayout::BSND, FiaLayout::BNSD, FiaLayout::TND, FiaLayout::NTD}},
    {KEY_NAME,        {FiaLayout::BSH, FiaLayout::BSND, FiaLayout::BNSD, FiaLayout::TND, FiaLayout::NTD, FiaLayout::NZ, FiaLayout::BnBsH, FiaLayout::BnNBsD}},
    {VALUE_NAME,      {FiaLayout::BSH, FiaLayout::BSND, FiaLayout::BNSD, FiaLayout::TND, FiaLayout::NTD, FiaLayout::NZ, FiaLayout::BnBsH, FiaLayout::BnNBsD}},
    {ATTEN_OUT_NAME,  {FiaLayout::BSH, FiaLayout::BSND, FiaLayout::BNSD, FiaLayout::TND, FiaLayout::NTD, FiaLayout::NBSD}},
};

const std::set<ge::Format> FORMAT_SUPPORT_SET = {
    ge::FORMAT_ND,
    ge::FORMAT_NCHW,
    ge::FORMAT_NHWC,
    ge::FORMAT_NCDHW
};

void FiaTilingCheck::LogErrorDtypeSupport(const std::vector<ge::DataType> &expectDtypeList,
    const ge::DataType &actualDtype, const std::string &name) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectDtypeList.size(); ++i) {
        oss << FusedDataTypeToSerialString(expectDtypeList[i]);
        if (i < expectDtypeList.size() - 1) {
            oss << ", ";
        }
    }
    OP_LOGE(opName_, "Tensor %s only supports dtype %s, but got %s",
        name.c_str(), oss.str().c_str(), FusedDataTypeToSerialString(actualDtype).c_str());
}

ge::graphStatus FiaTilingCheck::CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc,
    const std::string &name) const
{
    if (desc != nullptr) {
        const auto& it = DTYPE_SUPPORT_MAP.find(name);
        OP_CHECK_IF(it == DTYPE_SUPPORT_MAP.end(),
            OP_LOGE(opName_, "%s datatype support list should be specify in DTYPE_SUPPORT_MAP", name.c_str()),
            return ge::GRAPH_FAILED);
        auto &expectDtypeList = it->second;
        OP_CHECK_IF(std::find(
            expectDtypeList.begin(), expectDtypeList.end(), desc->GetDataType()) == expectDtypeList.end(),
            LogErrorDtypeSupport(expectDtypeList, desc->GetDataType(), name),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFormatSupport(const gert::CompileTimeTensorDesc *desc,
    const std::string &name) const
{
    if (desc != nullptr) {
        auto format = desc->GetOriginFormat();
        OP_CHECK_IF(
            (FORMAT_SUPPORT_SET.find(format) == FORMAT_SUPPORT_SET.end()),
            OP_LOGE(opName_, "%s format only supports ND/NCHW/NHWC/NCDHW, but got %d",
                name.c_str(), static_cast<int32_t>(format)),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
void FiaTilingCheck::LogErrorNumberSupport(const std::vector<T> &expectNumberList,
    const T &actualValue, const std::string &name, const std::string subName) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectNumberList.size(); ++i) {
        oss << std::to_string(expectNumberList[i]);
        if (i < expectNumberList.size() - 1) {
            oss << ", ";
        }
    }

    OP_LOGE(opName_, "%s %s only supports %s, but got %s",
              name.c_str(), subName.c_str(), oss.str().c_str(), std::to_string(actualValue).c_str());
}

template <typename T>
void FiaTilingCheck::LogErrorDimNumSupport(const std::vector<T> &expectNumberList,
    const T &actualValue, const std::string &name) const
{
    LogErrorNumberSupport(expectNumberList, actualValue, name, "dim num");
}

template <typename T>
void FiaTilingCheck::LogErrorShapeNumSupport(const std::vector<T> &expectNumberList,
    const T &actualValue, const std::string &name) const
{
    LogErrorNumberSupport(expectNumberList, actualValue, name, "shape num");
}

template <typename T>
void FiaTilingCheck::LogErrorAttrValueSupport(const std::vector<T> &expectNumberList,
    const T &actualValue, const std::string &name) const
{
    LogErrorNumberSupport(expectNumberList, actualValue, name, "attr value");
}

ge::graphStatus FiaTilingCheck::CheckDimNumSupport(const gert::StorageShape *shape,
    const std::vector<size_t> &expectDimNumList, const std::string &name) const
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (std::find(expectDimNumList.begin(), expectDimNumList.end(),
        shape->GetStorageShape().GetDimNum()) == expectDimNumList.end()) {
        LogErrorDimNumSupport(expectDimNumList, shape->GetStorageShape().GetDimNum(), name);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckDimNumSupport(const gert::Tensor *tensor,
    const std::vector<size_t> &expectDimNumList, const std::string &name) const
{
    if (tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (std::find(expectDimNumList.begin(), expectDimNumList.end(),
        tensor->GetStorageShape().GetDimNum()) == expectDimNumList.end()) {
        LogErrorDimNumSupport(expectDimNumList, tensor->GetStorageShape().GetDimNum(), name);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckShapeSupport(const gert::Tensor *tensor,
    const std::vector<int64_t> &expectShapeList, const std::string &name) const
{
    if (tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (std::find(expectShapeList.begin(), expectShapeList.end(),
        tensor->GetShapeSize()) == expectShapeList.end()) {
        LogErrorShapeNumSupport(expectShapeList, tensor->GetShapeSize(), name);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

void FiaTilingCheck::LogErrorLayoutSupport(const std::vector<FiaLayout> &expectLayoutList,
    const FiaLayout &actualLayout, const std::string &name) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectLayoutList.size(); ++i) {
        oss << LayoutToSerialString(expectLayoutList[i]);
        if (i < expectLayoutList.size() - 1) {
            oss << ", ";
        }
    }
    OP_LOGE(opName_, "%s only supports layout %s, but got %s",
        name.c_str(), oss.str().c_str(), LayoutToSerialString(actualLayout).c_str());
}

ge::graphStatus FiaTilingCheck::CheckLayoutSupport(const FiaLayout &actualLayout, const std::string &name) const
{
    const auto& it = LAYOUT_SUPPORT_MAP.find(name);
    OP_CHECK_IF(it == LAYOUT_SUPPORT_MAP.end(),
        OP_LOGE(opName_, "%s layout support list should be specify in LAYOUT_SUPPORT_MAP", name.c_str()),
        return ge::GRAPH_FAILED);
    auto &expectLayoutList = it->second;
    OP_CHECK_IF(std::find(
        expectLayoutList.begin(), expectLayoutList.end(), actualLayout) == expectLayoutList.end(),
        LogErrorLayoutSupport(expectLayoutList, actualLayout, name),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaQuery() const
{
    const std::vector<size_t> queryDimNumList = {DIM_NUM_THREE, DIM_NUM_FOUR};
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.query.desc, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(qLayout_, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(opParamInfo_.query.shape, queryDimNumList, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckFormatSupport(opParamInfo_.query.desc, QUERY_NAME)) {
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(opParamInfo_.query.shape->GetStorageShape().GetShapeSize() == 0,
        OP_LOGE(opName_, "%s shapesize is 0.", QUERY_NAME.c_str()), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaKey() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.key.desc, KEY_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(kvLayout_, KEY_NAME) ||
        ge::GRAPH_SUCCESS != CheckFormatSupport(opParamInfo_.key.desc, KEY_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaValue() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.value.desc, VALUE_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(kvLayout_, VALUE_NAME) ||
        ge::GRAPH_SUCCESS != CheckFormatSupport(opParamInfo_.value.desc, VALUE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaActualSeqLengthsQ() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaActualSeqLengths() const
{
    return ge::GRAPH_SUCCESS;
}


ge::graphStatus FiaTilingCheck::CheckSingleParaQueryRope() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.queryRope.desc, QUERY_ROPE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaKeyRope() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.keyRope.desc, KEY_ROPE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaAttenOut() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.attenOut.desc, ATTEN_OUT_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(outLayout_, ATTEN_OUT_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaScaleValue() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaKvHeadNums() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaLayout() const
{
    const std::vector<std::string> inputLayoutList = {
        "BSH", "BSND", "BNSD", "TND", "NTD", "BSH_NBSD", "BSND_NBSD", "BNSD_NBSD", "TND_NTD", "NTD_TND", "BSH_BNSD", "BSND_BNSD", "BNSD_BSND"
    };
    std::string inputLayout = opParamInfo_.layOut;
    if (std::find(inputLayoutList.begin(), inputLayoutList.end(), inputLayout) == inputLayoutList.end()) {
        OP_LOGE(opName_,
            "input layout only supports BSH, BSND, BNSD, TND, NTD, BSH_NBSD, BSND_NBSD, BNSD_NBSD, TND_NTD, NTD_TND, BSH_BNSD, BSND_BNSD, BNSD_BSND, but got %s",
            inputLayout.c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSinglePara() const
{
    if (ge::GRAPH_SUCCESS != CheckSingleParaQuery() ||
        ge::GRAPH_SUCCESS != CheckSingleParaKey() ||
        ge::GRAPH_SUCCESS != CheckSingleParaValue() ||
        ge::GRAPH_SUCCESS != CheckSingleParaActualSeqLengthsQ() ||
        ge::GRAPH_SUCCESS != CheckSingleParaActualSeqLengths() ||
        ge::GRAPH_SUCCESS != CheckSingleParaQueryRope() ||
        ge::GRAPH_SUCCESS != CheckSingleParaKeyRope() ||
        ge::GRAPH_SUCCESS != CheckSingleParaAttenOut() ||
        ge::GRAPH_SUCCESS != CheckSingleParaScaleValue() ||
        ge::GRAPH_SUCCESS != CheckSingleParaKvHeadNums() ||
        ge::GRAPH_SUCCESS != CheckSingleParaLayout()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
