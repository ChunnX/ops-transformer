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
 * \file collective_comm_base.h
 * \brief Collective communication base class - CRTP pattern + unified state management
 */

#pragma once

#include "kernel_operator.h"
#include "shmem.h"
#include "../../utils/comm_tiling_data.h"

namespace Dista {
namespace AivComm {

using namespace AscendC;
using Utils::CommTilingData;

enum class CollectiveOp {
    AllToAll,
    AllGather,
    ReduceScatter
};

enum class CommMode {
    PUT,
    GET
};

struct CommContext {
    __gm__ void* shmemSpace;
    uint32_t rankSize;
    uint32_t myRankId;
};

template<typename Impl, typename Dtype>
class CollectiveCommBase {
protected:
    const CommContext* ctx_;
    GM_ADDR localAddr_;
    const CommTilingData* tilingData_;  // tileSize/tailSize: 元素个数
    
    uint64_t chunkSize_;                // 单个块的总大小（元素个数）
    uint64_t nonSplitAxisBytes_;        // 预计算：nonSplitAxisSize * sizeof(Dtype)
    uint64_t chunkBytes_;               // 预计算：chunkSize * nonSplitAxisBytes
    
    uint64_t totalTiles_;
    uint64_t currentTileIdx_;
    uint64_t currentTileOffset_;        // 当前切片偏移（元素个数）
    uint64_t remainingChunkSize_;       // 剩余块大小（元素个数）
    
    uint32_t targetRankStart_;          // 首个目标 rankId
    uint32_t targetRankCnt_;            // 连续的目标 rank 数量
    
    uint32_t threadCnt_;                // 总参与核数
    uint32_t threadId_;                 // 当前核的统一编号
    
public:
    __aicore__ inline CollectiveCommBase() {}
    
    __aicore__ inline uint64_t Init(
        const CommContext* ctx,
        GM_ADDR localAddr,
        const CommTilingData& tilingData,
        uint32_t threadCnt,
        uint32_t threadId) {
        
        ctx_ = ctx;
        localAddr_ = localAddr;
        tilingData_ = &tilingData;  // 仅指针赋值，避免结构体拷贝
        
        threadCnt_ = threadCnt;
        threadId_ = threadId;
        
        uint32_t rankSize = ctx->rankSize;
        
        uint32_t targetRankPerCore = (rankSize + threadCnt - 1) / threadCnt;
        uint32_t targetRankStart = threadId * targetRankPerCore;
        
        uint32_t targetRankCnt;
        if (targetRankStart + targetRankPerCore <= rankSize) {
            targetRankCnt = targetRankPerCore;
        } else if (targetRankStart < rankSize) {
            targetRankCnt = rankSize - targetRankStart;
        } else {
            targetRankCnt = 0;
        }
        
        targetRankStart_ = targetRankStart;
        targetRankCnt_ = targetRankCnt;
        
        chunkSize_ = tilingData.splitAxisTileSize * tilingData.splitAxisTileCnt + 
                     tilingData.splitAxisTailSize * tilingData.splitAxisTailCnt;
        
        // 预计算常用值，避免重复计算
        nonSplitAxisBytes_ = tilingData.nonSplitAxisSize * sizeof(Dtype);
        chunkBytes_ = chunkSize_ * nonSplitAxisBytes_;
        
        totalTiles_ = tilingData_->splitAxisTileCnt + tilingData_->splitAxisTailCnt;
        
        currentTileIdx_ = 0;
        currentTileOffset_ = 0;
        remainingChunkSize_ = chunkSize_;
        
        static_cast<Impl*>(this)->PostInit();
        return totalTiles_;
    }
    
    __aicore__ inline void Commit() {
        if (remainingChunkSize_ > 0) {
            uint64_t currentTileSize;
            
            if (currentTileIdx_ < tilingData_->splitAxisTileCnt) {
                currentTileSize = tilingData_->splitAxisTileSize;
            } else {
                currentTileSize = tilingData_->splitAxisTailSize;
            }
            
            if (currentTileSize > remainingChunkSize_) {
                currentTileSize = remainingChunkSize_;
            }
            
            const uint32_t myRankId = ctx_->myRankId;
            const uint32_t targetRankStart = targetRankStart_;
            const uint32_t targetRankCnt = targetRankCnt_;
            
            if (targetRankCnt == 1) {
                uint32_t targetRankId = targetRankStart;
                if (targetRankId != myRankId) {
                    static_cast<Impl*>(this)->DoCommit(
                        targetRankId, currentTileOffset_, currentTileSize);
                }
            } else {
                for (uint32_t i = 0; i < targetRankCnt; i++) {
                    uint32_t targetRankId = targetRankStart + i;
                    if (targetRankId != myRankId) {
                        static_cast<Impl*>(this)->DoCommit(
                            targetRankId, currentTileOffset_, currentTileSize);
                    }
                }
            }
            
            currentTileIdx_++;
            currentTileOffset_ += currentTileSize;
            remainingChunkSize_ -= currentTileSize;
        }
    }
    
    __aicore__ inline void Wait() {
        static_cast<Impl*>(this)->DoWait();
    }
    
    __aicore__ inline void Finalize() {
        static_cast<Impl*>(this)->DoFinalize();
    }
};

} // namespace AivComm
} // namespace Dista