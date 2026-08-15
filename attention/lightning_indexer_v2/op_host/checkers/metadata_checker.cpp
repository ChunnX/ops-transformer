/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "metadata_checker.h"
namespace optiling {
namespace lightning_indexer_v2 {
ge::graphStatus MetadataChecker::CheckParaExistence(const IndexerCheckerInfo &info)
{
    return CheckRequiredTensor(info.metadata, "metadata", info);
}
ge::graphStatus MetadataChecker::CheckSinglePara(const IndexerCheckerInfo &info)
{
    if (CheckDtype(info.metadata, "metadata", {ge::DT_INT32}, info) != ge::GRAPH_SUCCESS ||
        CheckFormat(info.metadata, "metadata", info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return CheckShape(info.metadata, "metadata", {1024}, info);
}
} // namespace lightning_indexer_v2
} // namespace optiling
