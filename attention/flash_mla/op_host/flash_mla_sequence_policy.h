/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_MLA_SEQUENCE_POLICY_H
#define FLASH_MLA_SEQUENCE_POLICY_H

#include "../op_kernel/flash_mla_schedule_config.h"

namespace FlashMla {
inline bool IsMlaQuerySeqIgnored(uint32_t queryDataType, bool isTnd)
{
    // Original IFA uses the stored Q extent for non-TND full-quant MLA.
    // Neither the supplied query sequence values nor their item count is used.
    return queryDataType == kMlaInt8 && !isTnd;
}
} // namespace FlashMla
#endif
