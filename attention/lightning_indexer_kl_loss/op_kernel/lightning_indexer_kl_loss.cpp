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
 * \file lightning_indexer_kl_loss.cpp
 * \brief
 */

#include "lightning_indexer_kl_loss.h"

enum class LightningIndexerKlLossTilingKey : uint32_t {
    TILING_KEY_EXAMPLE_FLOAT = 0,
    TILING_KEY_EXAMPLE_INT32 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void lightning_indexer_kl_loss(GM_ADDR targetScore, GM_ADDR indexProbs, GM_ADDR loss, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LightningIndexerKlLossTilingData);
    GET_TILING_DATA_WITH_STRUCT(LightningIndexerKlLossTilingData, tilingData, tiling);

    // 场景1
    if constexpr (schMode == static_cast<uint32_t>(LightningIndexerKlLossTilingKey::TILING_KEY_EXAMPLE_FLOAT)) {
        //
    }

    // 场景2
    if constexpr (schMode == static_cast<uint32_t>(LightningIndexerKlLossTilingKey::TILING_KEY_EXAMPLE_INT32)) {
        //
    }
}
