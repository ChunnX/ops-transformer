/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "moe_shared_jetty_adapter.h"
#include "hcomm_host.h"

namespace Mc2Api {
// Keep hcomm_host.h's function-pointer typedefs out of translation units using
// hccl_common.h, whose public function-pointer variables have the same names.
HcclResult CreateMoeMultiChannelHandle(HcclComm comm, const char *queueTag, const HcclChannelDesc *descs,
                                     uint32_t channelNum, uint64_t *handle)
{
    return MakeMultiChannelHandle(comm, queueTag, descs, channelNum, handle);
}
} // namespace Mc2Api
