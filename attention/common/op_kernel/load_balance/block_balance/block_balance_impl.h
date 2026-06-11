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
 * \file block_balance_impl.h
 * \brief BlockBalance具体实现
 */

#ifndef BLOCK_BALANCE_IMPL_H
#define BLOCK_BALANCE_IMPL_H

#include <algorithm>
#include "block_balance_def.h"

namespace load_balance {

class BlockBalanceImpl {
public:
    inline BlockBalanceResult Compute();
    inline uint32_t SetParam(const BlockBalanceParam &param);

private:
    struct GridInfo {
        std::vector<uint32_t> mBaseNum {};
        std::vector<std::vector<uint32_t>> s2BaseNum {};
        bool isEmpty { true };

        explicit GridInfo(uint32_t batchSize)
            : mBaseNum(batchSize),
              s2BaseNum(batchSize) {}
    };

    struct CostInfo {
        std::vector<uint64_t> bN2BlockOfEachBatch {};
        uint64_t blockNum { 0U };

        explicit CostInfo(uint32_t batchSize)
            : bN2BlockOfEachBatch(batchSize) {}
    };

    struct CoreCache {
        uint64_t blockLimit { 0U };
        uint64_t block { 0U };
    };

    struct AssignContext {
        uint32_t curCoreIdx { 0U };
        uint32_t curBIdx { 0U };
        uint32_t curBN2Idx { 0U };
        uint32_t curS1Idx { 0U };
        uint32_t curS2Idx { 0U };
        uint64_t unassignedBlock { 0U };
        uint64_t bN2Block { 0U }; // 当前BN2从curS1Idx开始的剩余基本块数
        CoreCache coreCache {};
        bool isFinished { false };
    };

private:
    inline uint32_t GetRowBlockNum(const GridInfo &gridInfo, uint32_t bIdx, uint32_t s1Idx) const;
    inline void CalcGridInfo(GridInfo &gridInfo);
    inline void CalcCostInfo(const GridInfo &gridInfo, CostInfo &costInfo);
    inline BlockBalanceResult ScheduleSection(const GridInfo &gridInfo, const CostInfo &costInfo);
    inline void ScheduleFa(const GridInfo &gridInfo, const CostInfo &costInfo, BlockBalanceResult &result);
    inline void AssignByBatch(const GridInfo &gridInfo, const CostInfo &costInfo, AssignContext &assignContext);
    inline void AssignByRow(const GridInfo &gridInfo, const CostInfo &costInfo, AssignContext &assignContext);

private:
    BlockBalanceParam m_param {};
};

inline uint32_t BlockBalanceImpl::SetParam(const BlockBalanceParam &param)
{
    if (param.batchSize == 0U || param.kvHeadNum == 0U || param.aicCoreNum == 0U ||
        param.qBlockMax == 0U || param.sparseSeqLen == nullptr) {
        return BLOCK_BALANCE_ERROR_INVALID_PARAM;
    }
    m_param = param;
    return BLOCK_BALANCE_SUCCESS;
}

inline BlockBalanceResult BlockBalanceImpl::Compute()
{
    GridInfo gridInfo(m_param.batchSize);
    CalcGridInfo(gridInfo);
    CostInfo costInfo(m_param.batchSize);
    CalcCostInfo(gridInfo, costInfo);
    return ScheduleSection(gridInfo, costInfo);
}

inline uint32_t BlockBalanceImpl::GetRowBlockNum(const GridInfo &gridInfo, uint32_t bIdx, uint32_t s1Idx) const
{
    if (bIdx >= gridInfo.s2BaseNum.size() || s1Idx >= gridInfo.s2BaseNum[bIdx].size()) {
        return 0U;
    }
    return gridInfo.s2BaseNum[bIdx][s1Idx];
}

inline void BlockBalanceImpl::CalcGridInfo(GridInfo &gridInfo)
{
    gridInfo.s2BaseNum.assign(m_param.batchSize, std::vector<uint32_t>(m_param.qBlockMax, 0U));
    gridInfo.isEmpty = true;

    for (uint32_t bIdx = 0U; bIdx < m_param.batchSize; ++bIdx) {
        for (uint32_t s1Idx = 0U; s1Idx < m_param.qBlockMax; ++s1Idx) {
            uint32_t idx = bIdx * m_param.qBlockMax + s1Idx;
            uint32_t rowBlock = 0U;
            if (m_param.sparseSeqLen[idx] > 0) {
                rowBlock = static_cast<uint32_t>(m_param.sparseSeqLen[idx]);
            }
            gridInfo.s2BaseNum[bIdx][s1Idx] = rowBlock;
            if (rowBlock > 0U) {
                gridInfo.mBaseNum[bIdx] = s1Idx + 1U;
                gridInfo.isEmpty = false;
            }
        }
    }
}

inline void BlockBalanceImpl::CalcCostInfo(const GridInfo &gridInfo, CostInfo &costInfo)
{
    costInfo.blockNum = 0U;
    std::fill(costInfo.bN2BlockOfEachBatch.begin(), costInfo.bN2BlockOfEachBatch.end(), 0U);
    for (uint32_t bIdx = 0U; bIdx < m_param.batchSize; ++bIdx) {
        uint64_t batchBlock = 0U;
        for (uint32_t s1Idx = 0U; s1Idx < gridInfo.mBaseNum[bIdx]; ++s1Idx) {
            batchBlock += GetRowBlockNum(gridInfo, bIdx, s1Idx);
        }
        costInfo.bN2BlockOfEachBatch[bIdx] = batchBlock;
        costInfo.blockNum += batchBlock * m_param.kvHeadNum;
    }
}

inline BlockBalanceResult BlockBalanceImpl::ScheduleSection(const GridInfo &gridInfo, const CostInfo &costInfo)
{
    BlockBalanceResult result(m_param.aicCoreNum);
    if (gridInfo.isEmpty || costInfo.blockNum == 0U) {
        result.usedCoreNum = 1U;
        result.bN2End[0] = m_param.batchSize * m_param.kvHeadNum;
        result.s1End[0] = 0U;
        result.s2End[0] = 0U;
        result.firstFdDataWorkspaceIdx[0] = 0U;
        return result;
    }
    ScheduleFa(gridInfo, costInfo, result);
    return result;
}

inline void BlockBalanceImpl::AssignByBatch(const GridInfo &gridInfo, const CostInfo &costInfo,
                                            AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    while (!assignContext.isFinished &&
           assignContext.curBN2Idx < m_param.batchSize * m_param.kvHeadNum &&
           assignContext.curS1Idx == 0U) {
        if (assignContext.bN2Block == 0U) {
            assignContext.curBN2Idx++;
            if (assignContext.curBN2Idx == m_param.batchSize * m_param.kvHeadNum) {
                assignContext.curS1Idx = 0U;
                assignContext.curS2Idx = 0U;
                assignContext.isFinished = true;
                return;
            }
            assignContext.curBIdx = assignContext.curBN2Idx / m_param.kvHeadNum;
            assignContext.curS1Idx = 0U;
            assignContext.curS2Idx = 0U;
            assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
            continue;
        }
        if (assignContext.coreCache.block + assignContext.bN2Block > assignContext.coreCache.blockLimit) {
            break;
        }
        assignContext.coreCache.block += assignContext.bN2Block;
        assignContext.curBN2Idx++;
        assignContext.curS1Idx = 0U;
        assignContext.curS2Idx = 0U;
        if (assignContext.curBN2Idx == m_param.batchSize * m_param.kvHeadNum) {
            assignContext.isFinished = true;
            return;
        }
        assignContext.curBIdx = assignContext.curBN2Idx / m_param.kvHeadNum;
        assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
    }
}

inline void BlockBalanceImpl::AssignByRow(const GridInfo &gridInfo, const CostInfo &costInfo,
                                          AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    while (!assignContext.isFinished && assignContext.curBN2Idx < m_param.batchSize * m_param.kvHeadNum) {
        while (assignContext.curS1Idx < gridInfo.mBaseNum[assignContext.curBIdx] &&
               GetRowBlockNum(gridInfo, assignContext.curBIdx, assignContext.curS1Idx) == 0U) {
            assignContext.curS1Idx++;
        }
        while (assignContext.curS1Idx >= gridInfo.mBaseNum[assignContext.curBIdx]) {
            assignContext.curBN2Idx++;
            assignContext.curS1Idx = 0U;
            assignContext.curS2Idx = 0U;
            if (assignContext.curBN2Idx == m_param.batchSize * m_param.kvHeadNum) {
                assignContext.isFinished = true;
                return;
            }
            assignContext.curBIdx = assignContext.curBN2Idx / m_param.kvHeadNum;
            assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
            while (assignContext.curS1Idx < gridInfo.mBaseNum[assignContext.curBIdx] &&
                   GetRowBlockNum(gridInfo, assignContext.curBIdx, assignContext.curS1Idx) == 0U) {
                assignContext.curS1Idx++;
            }
        }

        uint32_t rowBlock = GetRowBlockNum(gridInfo, assignContext.curBIdx, assignContext.curS1Idx);
        if (rowBlock == 0U) {
            assignContext.curS1Idx++;
            continue;
        }
        if (assignContext.coreCache.block + rowBlock > assignContext.coreCache.blockLimit) {
            break;
        }
        assignContext.coreCache.block += rowBlock;
        assignContext.bN2Block = assignContext.bN2Block > rowBlock ? assignContext.bN2Block - rowBlock : 0U;
        assignContext.curS1Idx++;
    }
}

inline void BlockBalanceImpl::ScheduleFa(const GridInfo &gridInfo, const CostInfo &costInfo,
                                         BlockBalanceResult &result)
{
    AssignContext assignContext {};
    assignContext.unassignedBlock = costInfo.blockNum;
    assignContext.curBIdx = 0U;
    assignContext.curBN2Idx = 0U;
    assignContext.curS1Idx = 0U;
    assignContext.curS2Idx = 0U;
    assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
    while (assignContext.curS1Idx < gridInfo.mBaseNum[assignContext.curBIdx] &&
           GetRowBlockNum(gridInfo, assignContext.curBIdx, assignContext.curS1Idx) == 0U) {
        assignContext.curS1Idx++;
    }

    for (uint32_t i = 0U; i < m_param.aicCoreNum; ++i) {
        if (assignContext.isFinished || assignContext.unassignedBlock == 0U) {
            break;
        }
        while (!assignContext.isFinished) {
            while (assignContext.curS1Idx < gridInfo.mBaseNum[assignContext.curBIdx] &&
                   GetRowBlockNum(gridInfo, assignContext.curBIdx, assignContext.curS1Idx) == 0U) {
                assignContext.curS1Idx++;
            }
            if (assignContext.curS1Idx < gridInfo.mBaseNum[assignContext.curBIdx]) {
                break;
            }
            assignContext.curBN2Idx++;
            assignContext.curS1Idx = 0U;
            assignContext.curS2Idx = 0U;
            if (assignContext.curBN2Idx == m_param.batchSize * m_param.kvHeadNum) {
                assignContext.isFinished = true;
                break;
            }
            assignContext.curBIdx = assignContext.curBN2Idx / m_param.kvHeadNum;
            assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
        }
        if (assignContext.isFinished) {
            break;
        }
        assignContext.curCoreIdx = i;

        assignContext.coreCache = {};
        uint32_t remainingCore = m_param.aicCoreNum - i;
        assignContext.coreCache.blockLimit =
            (assignContext.unassignedBlock + remainingCore - 1U) / remainingCore;
        uint32_t rowBlock = GetRowBlockNum(gridInfo, assignContext.curBIdx, assignContext.curS1Idx);
        assignContext.coreCache.blockLimit = std::max(assignContext.coreCache.blockLimit,
                                                      static_cast<uint64_t>(rowBlock));

        AssignByBatch(gridInfo, costInfo, assignContext);
        AssignByRow(gridInfo, costInfo, assignContext);

        result.bN2End[i] = assignContext.curBN2Idx;
        result.s1End[i] = assignContext.curS1Idx;
        result.s2End[i] = 0U;
        result.firstFdDataWorkspaceIdx[i] = 0U;
        result.maxBlock = std::max(result.maxBlock, assignContext.coreCache.block);
        assignContext.unassignedBlock = assignContext.unassignedBlock > assignContext.coreCache.block ?
                                        assignContext.unassignedBlock - assignContext.coreCache.block : 0U;
        result.usedCoreNum = i + 1U;
    }
    result.usedCoreNum = std::max(result.usedCoreNum, 1U);
}

} // namespace load_balance

#endif
