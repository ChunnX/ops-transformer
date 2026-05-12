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
 * \file sinks_checker.cpp
 * \brief Checker for sinks parameter (文档参数名: sinks, 文档约束: LearnableSink参数组)
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../flash_attn_tiling_constants.h"
#include "sinks_checker.h"

namespace optiling {
namespace flash_attn {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35FA;

ge::graphStatus SinksChecker::CheckSinglePara(const FaTilingInfo &faInfo)
{
    auto &sinksTensor = faInfo.opParamInfo.sinks.tensor;
    if (sinksTensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const gert::CompileTimeTensorDesc *sinksDesc = faInfo.opParamInfo.sinks.desc;
    OP_CHECK_IF(sinksDesc == nullptr, OP_LOGE(faInfo.opName, "sinks desc is null pointer!"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(sinksDesc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE(faInfo.opName, "sinks dtype must be FP32, but got %s",
                        DataTypeToSerialString(sinksDesc->GetDataType()).c_str()),
                return ge::GRAPH_FAILED);

    if (ge::GRAPH_SUCCESS != CheckFormatSupport(sinksDesc, SINKS_NAME)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t dimNum = sinksTensor->GetStorageShape().GetDimNum();
    OP_CHECK_IF(dimNum != 1, OP_LOGE(faInfo.opName, "sinks dim num must be 1, but got %u", dimNum),
                return ge::GRAPH_FAILED);

    int64_t dim0 = sinksTensor->GetStorageShape().GetDim(0);
    OP_CHECK_IF(dim0 != faInfo.n1Size,
                OP_LOGE(faInfo.opName, "sinks shape(%ld) must be equal to N_Q(%ld)", dim0, faInfo.n1Size),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

} // namespace flash_attn
} // namespace optiling