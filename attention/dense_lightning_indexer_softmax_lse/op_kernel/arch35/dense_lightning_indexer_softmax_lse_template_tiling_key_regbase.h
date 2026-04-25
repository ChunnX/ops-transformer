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
 * \file dense_lightning_indexer_softmax_lse_template_tiling_key_regbase.h
 * \brief Arch35 tiling key template definitions
 */

#ifndef DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_TEMPLATE_TILING_KEY_REGBASE_H
#define DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_TEMPLATE_TILING_KEY_REGBASE_H

#include <cstdint>

namespace optiling {

enum class LayoutTypeRegbase : uint32_t {
    BSND = 0,
    TND = 1
};

using DenseLISoftmaxLseLayoutRegbase = LayoutTypeRegbase;

inline uint64_t GetDenseLISoftmaxLseTilingKeyRegbase(LayoutTypeRegbase layoutType)
{
    uint64_t tilingKey = 0;
    tilingKey |= static_cast<uint64_t>(layoutType);
    return tilingKey + 1000;
}

} // namespace optiling

#endif // DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_TEMPLATE_TILING_KEY_REGBASE_H