/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NOTIFY_DISPATCH_TILING_KEY_H
#define NOTIFY_DISPATCH_TILING_KEY_H

#define TILINGKEY_NOTIFY_DISPATCH_INT 23
#define TILINGKEY_NOTIFY_DISPATCH_A2_TYPE 100
#define TILINGKEY_NOTIFY_DISPATCH_A5_TYPE 200
#define TILINGKEY_NOTIFY_DISPATCH_A2 123
#define TILINGKEY_NOTIFY_DISPATCH_A5 223

namespace Mc2Tiling {
inline int GetNotifyDispatchTilingKey(bool isA2Machine, bool isA5Machine)
{
    if (isA5Machine) {
        return TILINGKEY_NOTIFY_DISPATCH_A5;
    }
    if (isA2Machine) {
        return TILINGKEY_NOTIFY_DISPATCH_A2;
    }
    return TILINGKEY_NOTIFY_DISPATCH_INT;
}
} // namespace Mc2Tiling

#endif // NOTIFY_DISPATCH_TILING_KEY_H
