/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LOCAL_MS_RW_H
#define LOCAL_MS_RW_H

#include <cstdint>

void local_ms_rw_demo(uint32_t mode0, uint32_t enableWarmup, uint32_t msId, uint32_t loopCount,
                      uint64_t dataSize, uint32_t blockDim, void *stream, uint8_t *msAddr0, uint8_t *msAddr1,
                      uint8_t *msAddr2, uint8_t *msAddr3, uint8_t *timeAddr);

#endif // LOCAL_MS_RW_H
