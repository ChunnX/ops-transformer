/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef VLLM_A5_LI_MANAGE_C8_VICTIM_SCAN_H
#define VLLM_A5_LI_MANAGE_C8_VICTIM_SCAN_H

#include <cstdint>

namespace vllm_a5_li_manage_c8_victim_scan {

// Seed from this invocation's actual KV length, NOT the block-aligned
// durable-prefix length. Candidate length still bounds the victim scan.
// Independent of request-pool placement and stateless across graph replays;
// equal final lengths intentionally give equal seeds.
__aicore__ inline uint32_t HashVictimScanSeed(uint32_t finalLen)
{
    uint32_t value = finalLen;
    value ^= value >> 16U;
    value *= 0x7feb352dU;
    value ^= value >> 15U;
    value *= 0x846ca68bU;
    value ^= value >> 16U;
    return value;
}

} // namespace vllm_a5_li_manage_c8_victim_scan

#endif
