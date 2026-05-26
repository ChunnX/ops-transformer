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
 * \file indexer_quant_cache.cpp
 * \brief L0 operator implementation for IndexerQuantCache
 */

#include "indexer_quant_cache.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(IndexerQuantCache);

const aclTensor *IndexerQuantCache(
    aclTensor *indexerCompressCache,
    aclTensor *indexerCompressCacheScale,
    const aclTensor *x,
    const aclTensor *slotMapping,
    int64_t quantMode,
    bool roundScale,
    float scale,
    aclOpExecutor *executor)
{
    L0_DFX(IndexerQuantCache,
           indexerCompressCache, indexerCompressCacheScale,
           x, slotMapping,
           quantMode, roundScale, scale);

    ADD_TO_LAUNCHER_LIST_AICORE(IndexerQuantCache,
                                OP_INPUT(indexerCompressCache, indexerCompressCacheScale,
                                         x, slotMapping),
                                OP_OUTPUT(indexerCompressCache, indexerCompressCacheScale),
                                OP_ATTR(quantMode, roundScale, scale));

    return indexerCompressCache;
}

}  // namespace l0op