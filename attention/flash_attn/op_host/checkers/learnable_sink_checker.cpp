/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file learnable_sink_checker.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../flash_attn_tiling_constants.h"
#include "learnable_sink_checker.h"

namespace optiling {
using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
using namespace arch35FA;

// check sink dtype
ge::graphStatus LearnableSinkChecker::CheckSinkDtypeSupport(const FaTilingInfo &faInfo)
{
    if (!faInfo.learnableSinkFlag) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::CompileTimeTensorDesc *learnableSinkDesc = faInfo.opParamInfo.learnableSink.desc;
    if (learnableSinkDesc != nullptr) {
        OP_CHECK_IF(learnableSinkDesc->GetDataType() != ge::DT_BF16 && learnableSinkDesc->GetDataType() != ge::DT_FLOAT16,
            OP_LOGE(faInfo.opName, "When learnable sink enable, the datatype(%s) of sink only support BF16/FP16.",
                DataTypeToSerialString(learnableSinkDesc->GetDataType()).c_str()),
        return ge::GRAPH_FAILED);

        OP_CHECK_IF(learnableSinkDesc->GetDataType() != faInfo.inputQType,
            OP_LOGE(faInfo.opName, "When learnable sink enable, the datatype(%s) of sink should be equal to query(%s).",
                DataTypeToSerialString(learnableSinkDesc->GetDataType()).c_str(), 
                DataTypeToSerialString(faInfo.inputQType).c_str()),
        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// CheckFeature
ge::graphStatus LearnableSinkChecker::CheckFeatureSupport(const FaTilingInfo &faInfo)
{
    if (!faInfo.learnableSinkFlag) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(faInfo.qPaddingSizeFlag || faInfo.kvPaddingSizeFlag,
        OP_LOGE(faInfo.opName,
                "When learnable sink enable, left padding is not supported."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(faInfo.sysPrefixFlag,
        OP_LOGE(faInfo.opName,
                "When learnable sink enable, system prefix is not supported."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(faInfo.enableAlibiPse,
        OP_LOGE(faInfo.opName, "When learnable sink enable, pseType = 2/3 is not supported."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(faInfo.pseShiftFlag,
        OP_LOGE(faInfo.opName,
                "When learnable sink enable, pse is not supported."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(faInfo.isOutQuantEnable,
        OP_LOGE(faInfo.opName, "When learnable sink enable, post qunat is not supported."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(faInfo.innerPrecise != HIGH_PRECISION,
        OP_LOGE(faInfo.opName,
                "When learnable sink enable, innerPrecise(%u) only support %u.", faInfo.innerPrecise, HIGH_PRECISION),
        return ge::GRAPH_FAILED);

    // 仅支持GQA 非量化
    OP_CHECK_IF(!(enableNonQuant_ && faInfo.mlaMode != MlaMode::ROPE_SPLIT_D512),
        OP_LOGE(faInfo.opName,
                "Learnable sink only supports no-quantized GQA mode."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// check sink shape
ge::graphStatus LearnableSinkChecker::CheckSinkShapeSupport(const FaTilingInfo &faInfo)
{
    if (!faInfo.learnableSinkFlag) {
        return ge::GRAPH_SUCCESS;
    }
    uint32_t sinkDimNum = faInfo.opParamInfo.learnableSink.tensor->GetStorageShape().GetDimNum();
    uint32_t sinkDim0 = faInfo.opParamInfo.learnableSink.tensor->GetStorageShape().GetDim(0);
    OP_CHECK_IF(sinkDimNum != 1,
        OP_LOGE(faInfo.opName, " When learnable sink enable, the dimension(%u) of sink must be 1!", sinkDimNum),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(sinkDim0 != faInfo.n1Size, OP_LOGE(faInfo.opName,
        "When learnable sink enable, the shape([%u]) of sink must be equal to axis N(%u) of query.",
        sinkDim0, faInfo.n1Size),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// check axis
ge::graphStatus LearnableSinkChecker::CheckAxisSupport(const FaTilingInfo &faInfo)
{
    if (!faInfo.learnableSinkFlag) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(faInfo.qkHeadDim != NUM_192 && faInfo.qkHeadDim != NUM_128 && faInfo.qkHeadDim != NUM_64,
        OP_LOGE(faInfo.opName, " When learnable sink enable, the head dim(%u) of query should be in range of "
            "{192, 128, 64}.", faInfo.qkHeadDim),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LearnableSinkChecker::CheckSinglePara(const FaTilingInfo &faInfo)
{
    if (ge::GRAPH_SUCCESS != CheckSinkDtypeSupport(faInfo) ||
        ge::GRAPH_SUCCESS != CheckSinkShapeSupport(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LearnableSinkChecker::CheckParaExistence(const FaTilingInfo &faInfo)
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LearnableSinkChecker::CheckFeature(const FaTilingInfo &faInfo)
{
    if (ge::GRAPH_SUCCESS != CheckFeatureSupport(faInfo) ||
        ge::GRAPH_SUCCESS != CheckAxisSupport(faInfo)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LearnableSinkChecker::CheckMultiPara(const FaTilingInfo &faInfo)
{
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
