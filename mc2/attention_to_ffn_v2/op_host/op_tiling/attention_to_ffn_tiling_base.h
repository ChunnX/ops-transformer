/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATTENTION_TO_FFN_TILING_BASE_H
#define ATTENTION_TO_FFN_TILING_BASE_H

#include <cstdint>
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

namespace MC2Tiling {

struct AttentionToFFNTilingConfig {
    uint32_t contextIndex;
    uint32_t xIndex;
    uint32_t sessionIdIndex;
    uint32_t microBatchIdIndex;
    uint32_t layerIdIndex;
    uint32_t expertIdsIndex;
    uint32_t expertRankTableIndex;
    uint32_t scalesIndex;
    uint32_t activeMaskIndex;

    uint32_t attrWorldSizeIndex;
    uint32_t attrCclBufferSizeIndex;
    uint32_t attrFfnTokenInfoTableShapeIndex;
    uint32_t attrFfnTokenDataShapeIndex;
    uint32_t attrAttnTokenInfoTableShapeIndex;
    uint32_t attrMoeExpertNumIndex;
    uint32_t attrQuantModeIndex;
    uint32_t attrSyncFlagIndex;
    uint32_t attrFfnStartRankIdIndex;

    bool isExplicitContext;
};

ge::graphStatus AttentionToFFNTilingV3FuncBase(gert::TilingContext *context,
                                               AttentionToFFNTilingConfig &config);

} // namespace MC2Tiling

#endif // ATTENTION_TO_FFN_TILING_BASE_H
