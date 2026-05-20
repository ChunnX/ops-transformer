/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMBINE_NORMAL_TILING_KEY_H
#define COMBINE_NORMAL_TILING_KEY_H

#define TILINGKEY_COMBINE_NORMAL_A3_SINGLE 13000
#define TILINGKEY_COMBINE_NORMAL_A3_MULTI 13001
#define TILINGKEY_COMBINE_NORMAL_A5_SINGLE 15000
#define TILINGKEY_COMBINE_NORMAL_A5_MULTI 15001

namespace Mc2Tiling {
inline uint64_t GetCombineNormalTilingKey(bool multiRound, bool isA5)
{
    if (isA5) {
        return multiRound ? TILINGKEY_COMBINE_NORMAL_A5_MULTI : TILINGKEY_COMBINE_NORMAL_A5_SINGLE;
    }
    return multiRound ? TILINGKEY_COMBINE_NORMAL_A3_MULTI : TILINGKEY_COMBINE_NORMAL_A3_SINGLE;
}
} // namespace Mc2Tiling

#endif // COMBINE_NORMAL_TILING_KEY_H
