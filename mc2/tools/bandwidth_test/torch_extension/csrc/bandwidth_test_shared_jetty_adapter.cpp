/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdint>

// Keep the public asc-comm Host adapter out of the context translation unit: its internal loader typedefs
// use the same identifiers as the existing hccl_common.h function-pointer variables.
#include "hcomm/hcomm_host.h"

namespace op_api {
HcclResult BuildSharedJettyMultiChannelHandle(HcclComm comm, const char *sharedQueueTag,
                                              const HcclChannelDesc *channelDescs, uint32_t channelNum,
                                              uint64_t *multiChannel)
{
    if (multiChannel == nullptr) {
        return HCCL_E_PTR;
    }
    MultiChannelHandle handle = 0;
    HcclResult ret = MakeMultiChannelHandle(comm, sharedQueueTag, channelDescs, channelNum, &handle);
    *multiChannel = static_cast<uint64_t>(handle);
    return ret;
}
} // namespace op_api
