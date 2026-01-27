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
 * \file fused_infer_attention_score_tiling_check_existence.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "fused_infer_attention_score_tiling_check.h"

using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
namespace optiling {

ge::graphStatus FiaTilingCheck::CheckRopeExistence() const
{
    OP_CHECK_IF((opParamInfo_.queryRope.tensor != nullptr && opParamInfo_.keyRope.tensor == nullptr),
        OP_LOGE(opName_, "%s is null, but queryRope exists, they should be both null or exist.", KEY_ROPE_NAME.c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((opParamInfo_.queryRope.tensor == nullptr && opParamInfo_.keyRope.tensor != nullptr),
        OP_LOGE(opName_, "%s is null, but keyRope exists, they should be both null or exist.", QUERY_ROPE_NAME.c_str()),
        return ge::GRAPH_FAILED);

    if (ropeMode_ == RopeMode::ROPE_SPLIT) {
        OP_CHECK_IF(opParamInfo_.keyRope.desc == nullptr || opParamInfo_.queryRope.desc == nullptr,
            OP_LOGE(opName_, "In %s situation and rope exsists, desc of %s and %s should not be null",
                QuantModeToSerialString(quantMode_).c_str(),
                KEY_ROPE_NAME.c_str(), QUERY_ROPE_NAME.c_str()),
            return ge::GRAPH_FAILED);
    } else if (ropeMode_ == RopeMode::ROPE_COMBINE) {
        OP_CHECK_IF(opParamInfo_.keyRope.desc != nullptr || opParamInfo_.queryRope.desc != nullptr,
            OP_LOGE(opName_, "In %s situation and rope exsists, desc of %s and %s should be null",
                QuantModeToSerialString(quantMode_).c_str(),
                KEY_ROPE_NAME.c_str(), QUERY_ROPE_NAME.c_str()),
            return ge::GRAPH_FAILED);
    }

    OP_LOGI(opName_, "rope mode is %s", RopeModeToSerialString(ropeMode_).c_str());
    return ge::GRAPH_SUCCESS;
}

static std::string DtypeListToStr(const std::vector<DataType> &dtypeList)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < dtypeList.size(); ++i) {
        oss << FusedDataTypeToSerialString(dtypeList[i]);
        if (i < dtypeList.size() - 1) {
            oss << ", ";
        }
    }
    oss << "]";

    return oss.str();
}

static std::string DtypeDoubleListToStr(const std::vector<std::vector<DataType>> &dtypeDoubleList)
{
    std::ostringstream oss;
    for (size_t i = 0; i < dtypeDoubleList.size(); ++i) {
        oss << DtypeListToStr(dtypeDoubleList[i]);
        if (i < dtypeDoubleList.size() - 1) {
            oss << ", ";
        }
    }
    return oss.str();
}

ge::graphStatus FiaTilingCheck::CheckDtypeAndSetQuantFlagMla()
{
    const std::vector<std::vector<ge::DataType>> mlaNoquantDtypeList = {
        // queryDtype,   kvDtype,        queryRopeDtype, keyRopeDtype
        {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
        {ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16},
    };

    ge::DataType queryRopeDtype = opParamInfo_.queryRope.desc->GetDataType();
    ge::DataType keyRopeDtype = opParamInfo_.keyRope.desc->GetDataType();
    std::vector<ge::DataType> actualDtypeList = {
        inputQType_, inputKvType_, queryRopeDtype, keyRopeDtype
    };
    if (VecContains(mlaNoquantDtypeList, actualDtypeList)) {
        quantMode_ = FiaQuantMode::NO_QUANT;
    } else {
        OP_LOGE(opName_, "In %s situation and rope exsists, only supports [query_dtype, kv_dtype, query_rope_dtype, key_rope_dtype] as %s, but got %s",
            QuantModeToSerialString(quantMode_).c_str(),
            DtypeDoubleListToStr(mlaNoquantDtypeList).c_str(),
            DtypeListToStr(actualDtypeList).c_str());
        return ge::GRAPH_FAILED;
    }

    OP_LOGI(opName_, "quant mode is %s", QuantModeToSerialString(quantMode_).c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckDtypeAndSetQuantFlagGqa()
{
    const std::vector<std::vector<ge::DataType>> gqaNoquantDtypeList = {
        // queryDtype,   kvDtype
        {ge::DT_FLOAT16, ge::DT_FLOAT16},
        {ge::DT_BF16,    ge::DT_BF16},
    };

    std::vector<ge::DataType> actualDtypeList = {
        inputQType_, inputKvType_
    };
    if (VecContains(gqaNoquantDtypeList, actualDtypeList)) {
        quantMode_ = FiaQuantMode::NO_QUANT;
    } else {
        OP_LOGE(opName_, "In %s situation, only supports [query_dtype] as %s, but got %s",
               QuantModeToSerialString(quantMode_).c_str(),
               DtypeDoubleListToStr(gqaNoquantDtypeList).c_str(),
               DtypeListToStr(actualDtypeList).c_str());
        return ge::GRAPH_FAILED;
    }

    OP_LOGI(opName_, "quant mode is %s", QuantModeToSerialString(quantMode_).c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckDtypeAndSetQuantFlag()
{
    if (ropeMode_ == RopeMode::ROPE_SPLIT) {
        return CheckDtypeAndSetQuantFlagMla();
    } else {
        return CheckDtypeAndSetQuantFlagGqa();
    }
}

ge::graphStatus FiaTilingCheck::CheckParaExistence()
{
    printf("the rope is %d\n", ropeMode_);
    if (ge::GRAPH_SUCCESS != CheckRopeExistence() ||
        ge::GRAPH_SUCCESS != CheckDtypeAndSetQuantFlag()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
