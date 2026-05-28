/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "thread_local_context.h"
#include "bridge_pool.h"

namespace op {
namespace internal {

void *Allocate(size_t size)
{
    int32_t id = op::internal::GetThreadLocalContext().poolIndex_;
    if (id != op::kInvalidHugeMemIndexId) {
        return GetAddr(id, size);
    } else {
        return op::internal::BlockCache::CacheAlloc(size);
    }
}

void DeAllocate(void *addr)
{
    OP_CHECK(addr != nullptr, 
            OP_LOGW("deAllocate addr is nullptr."),
            return);
    if (op::internal::BlockPool::InHugeMemRange(addr)) {
        // since huge mem pool use offset, so free just a dummy operation
    } else {
        op::internal::BlockCache::CacheFree(addr);
    }
}

int32_t GetPoolIndex()
{
    return op::internal::GetThreadLocalContext().poolIndex_;
}

void UpdateHugeMemIndex(int32_t id)
{
    OP_LOGI("Hugemem trace: update pool index: %d", id);
    op::internal::GetThreadLocalContext().poolIndex_ = id;
}

} // namespace internal
} // namespace op
