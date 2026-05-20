/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DISPATCH_LAYOUT_TILING_KEY_H
#define DISPATCH_LAYOUT_TILING_KEY_H

namespace Mc2Tiling {
constexpr int TILINGKEY_DISPATCH_LAYOUT_INT = 23;
constexpr int TILINGKEY_DISPATCH_LAYOUT_A2_TYPE = 100;

inline int GetDispatchLayoutTilingKey(bool isA2Machine)
{
    return TILINGKEY_DISPATCH_LAYOUT_INT + (isA2Machine ? TILINGKEY_DISPATCH_LAYOUT_A2_TYPE : 0);
}
} // namespace Mc2Tiling

#endif // DISPATCH_LAYOUT_TILING_KEY_H
