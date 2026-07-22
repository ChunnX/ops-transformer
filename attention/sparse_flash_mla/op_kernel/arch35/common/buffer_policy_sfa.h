/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file buffer_policy_sfa.h
 * \brief
 */
#ifndef BUFFER_POLICY_SFA_H
#define BUFFER_POLICY_SFA_H

#if __has_include("../../../common/op_kernel/buffers_policy.h")
#include "../../../common/op_kernel/buffers_policy.h"
#elif __has_include("../../common/op_kernel/buffers_policy.h")
#include "../../common/op_kernel/buffers_policy.h"
#else
#include "../common/op_kernel/buffers_policy.h"
#endif

namespace fa_base_matmul {
template <BufferType bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC,
          SyncMode syncMode = SyncMode::SET_WAIT_FLAG, IdSource idSource = IdSource::INTERNAL>
class BuffersPolicy3buffSFA : public BuffersPolicy3buff<bufferType, syncType, syncMode, idSource> {
public:
    using BuffersPolicy3buff<bufferType, syncType, syncMode, idSource>::Get;

    __aicore__ inline Buffer<bufferType, syncType, syncMode> &Get(uint32_t id)
    {
        uint32_t flag = id % NUM_3;
        if (flag == 0) {
            return this->a_;
        } else if (flag == 1) {
            return this->b_;
        } else {
            return this->c_;
        }
    }
};
} // namespace fa_base_matmul
#endif
