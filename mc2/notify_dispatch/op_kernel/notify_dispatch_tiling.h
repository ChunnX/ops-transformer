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
 * \file notify_dispatch_tiling.h
 * \brief
 */

#ifndef NOTIFY_DISPATCH_TILING_H
#define NOTIFY_DISPATCH_TILING_H

#include "kernel_tiling/kernel_tiling.h"

struct NotifyDispatchInfo {
    uint32_t rankSize;
    uint32_t rankId;
    uint32_t localRankSize;
    uint32_t localRankId;
    uint32_t sendCount;
    uint32_t numTokens;
    uint32_t round;
    uint32_t perRoundTokens;
    uint32_t aivNum;
    uint64_t totalUbSize;
    uint64_t totalWinSize;
};

struct NotifyDispatchTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling1;
    NotifyDispatchInfo notifyDispatchInfo;
};

#endif
