/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LOCAL_XNCKE_RW_H
#define LOCAL_XNCKE_RW_H

#include <cstdint>

void local_xncke_rw_demo(uint32_t mode, uint32_t resKind, uint32_t enableWarmup, uint32_t loopCount,
                         uint64_t pattern, uint32_t xnNum, uint32_t blockDim, void *stream,
                         uint8_t *xnAddr, uint8_t *ckeAddr, uint8_t *timeAddr);

#endif // LOCAL_XNCKE_RW_H
