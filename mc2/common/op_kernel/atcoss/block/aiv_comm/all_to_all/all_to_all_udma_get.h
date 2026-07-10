/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file all_to_all_udma_get.h
 * \brief AllToAll Get模式实现 - 四段式集合通信
 */

#pragma once

#include "kernel_operator.h"
#include "shmem.h"
#include "../collective_comm_base.h"
#include "../barrier/barrier_ubmem.h"

namespace Dista {
namespace AivComm {

using namespace AscendC;

template<typename Dtype>
class AllToAllGetImpl : public CollectiveCommBase<AllToAllGetImpl<Dtype>, Dtype> {
    friend class CollectiveCommBase<AllToAllGetImpl<Dtype>, Dtype>;
    
private:
    __aicore__ inline void PostInit() {
    }
    
    __aicore__ inline void DoCommit(uint32_t targetRankId, uint64_t tileOffset, uint64_t tileSize) {
        Barrier barrier;
        barrier.Execute(threadCnt_, threadId_);
        
        const uint32_t myRankId = ctx_->myRankId;
        
        const uint64_t remoteChunkOffsetBytes = myRankId * chunkBytes_ + 
                                                 tileOffset * nonSplitAxisBytes_;
        const uint64_t localChunkOffsetBytes = targetRankId * chunkBytes_ + 
                                                tileOffset * nonSplitAxisBytes_;
        const uint64_t tileSizeBytes = tileSize * nonSplitAxisBytes_;
        
        GM_ADDR remoteWinAddr = aclshmem_ptr(ctx_->shmemSpace, targetRankId);
        aclshmem_getmem(localAddr_ + localChunkOffsetBytes, 
                        remoteWinAddr + remoteChunkOffsetBytes, 
                        tileSizeBytes);
    }
    
    __aicore__ inline void DoWait() {
        aclshmem_quiet();
    }
    
    __aicore__ inline void DoFinalize() {
        Barrier barrier;
        barrier.Execute(threadCnt_, threadId_);
    }
};

} // namespace AivComm
} // namespace Dista