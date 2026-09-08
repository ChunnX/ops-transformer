/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#ifdef TEST_AIC
#define MegaMoeImpl InputPrepareAic
#else
#define MegaMoeImpl InputPrepareAiv
#endif
#include "../../../../op_kernel/arch35/common/mega_moe_input_prepare_sync.h"
#include "../../../../op_kernel/arch35/stage/mega_moe_shared_expert_input.h"

namespace MegaMoeImpl {
void Notify()
{
    NotifyMteInputPrepared();
}
void Wait()
{
    WaitForMteInputPrepared();
}
void ReleaseUb()
{
    ProtectMteInputUbReuse();
}
void Store(GM_ADDR data, GM_ADDR scales, uint32_t token, uint32_t bytes, uint32_t scaleCount, uint32_t scaleOffset,
           uint8_t *source)
{
    StoreSharedExpertInputToken<uint8_t, uint8_t>(data, scales, token, bytes, scaleCount, scaleOffset, {source});
}
} // namespace MegaMoeImpl
