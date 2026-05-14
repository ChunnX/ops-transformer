/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You can not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file attr_checker.cpp
 * \brief Checker for scalar attr parameters (deterministic, etc.)
 */

#include <map>
#include <numeric>
#include <vector>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../fa_tiling_info.h"
#include "attr_checker.h"

namespace optiling {
namespace flash_attn {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35FA;

ge::graphStatus AttrChecker::CheckSingleParaDeterministic(const FaTilingInfo &faInfo)
{
    const int64_t *deterministic = faInfo.opParamInfo.deterministic;
    if (deterministic == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const std::vector<int64_t> validList = {0, 1};
    OP_CHECK_IF(ge::GRAPH_SUCCESS != CheckValueSupport(*deterministic, validList),
                OP_LOGE(faInfo.opName, "deterministic only supports 0 or 1, but got %ld", *deterministic),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AttrChecker::CheckSinglePara(const FaTilingInfo &faInfo)
{
    if (CheckSingleParaDeterministic(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace flash_attn
} // namespace optiling
