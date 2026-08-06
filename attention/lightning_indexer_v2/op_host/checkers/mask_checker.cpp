/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "log/log.h"
#include "mask_checker.h"
namespace optiling {
namespace lightning_indexer_v2 {
ge::graphStatus MaskChecker::CheckSinglePara(const IndexerCheckerInfo &info)
{
    OP_CHECK_IF(info.maskMode != 0 && info.maskMode != 3,
                OP_LOGE_FOR_INVALID_VALUE(info.opName, "mask_mode", std::to_string(info.maskMode).c_str(), "0 or 3"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
} // namespace lightning_indexer_v2
} // namespace optiling
