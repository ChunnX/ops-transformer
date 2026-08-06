/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

namespace LightningAttn::Tiling
{

template <typename T> inline T CeilDiv(const T a, const T b)
{
    if (b == static_cast<T>(0))
    {
        return 0;
    }
    return (a + b - 1) / b;
}

template <typename T> inline T Align(const T a, const T b)
{
    return CeilDiv(a, b) * b;
}

} // namespace LightningAttn::Tiling