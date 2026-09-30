/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Stage4：TopK 与可选还原。mask 只决定候选长度。
//
// TopKExpandPartial（Qwen）：Expand 之后拼接当前 query 未完成的 micro-block
//   { r * floor((i+1)/r), ..., i }，不是 key 序列尾残余。
// TopKExpandKeyTail（PKI，预留）：拼接 key 尾残余。

#ifndef GLI_STAGE_SELECT_H
#define GLI_STAGE_SELECT_H

#include <cstdint>

namespace gli {

struct TopK {
    static constexpr uint32_t id = 0;
    static constexpr bool expand = false;
    static constexpr bool appendPartialBlock = false;
    static constexpr bool appendKeyTail = false;
};

struct TopKCmp {
    static constexpr uint32_t id = 1;
    static constexpr bool expand = false;
    static constexpr bool appendPartialBlock = false;
    static constexpr bool appendKeyTail = false;
};

struct TopKExpandKeyTail {
    static constexpr uint32_t id = 2;
    static constexpr bool expand = true;
    static constexpr bool appendPartialBlock = false;
    static constexpr bool appendKeyTail = true;
};

struct TopKExpandPartial {
    static constexpr uint32_t id = 3;
    static constexpr bool expand = true;
    static constexpr bool appendPartialBlock = true;
    static constexpr bool appendKeyTail = false;
};

struct SelectNone {
    static constexpr uint32_t id = 4;
    static constexpr bool expand = false;
    static constexpr bool appendPartialBlock = false;
    static constexpr bool appendKeyTail = false;
};

} // namespace gli

#endif
