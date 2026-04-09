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
 * \file section_stream_k_impl.h
 * \brief 
 */

#ifndef SECTION_STREAM_K_IMPL_H
#define SECTION_STREAM_K_IMPL_H

#include "load_balance_common.h"
#include "section_stream_k.h"

namespace load_balance {

class SectionStreamKImpl {
public:
    void Compute(const DeviceInfo &deviceInfo, const IBaseInfo &baseInfo, SectionStreamKResult &result);
    void SetParam(const SectionStreamKParam &param);
private:
    enum BlockType : uint32_t {
        NORMAL_BLOCK = 0,
        TAIL_BLOCK,
        BLOCK_MAX_TYPE
    };

    struct GridInfo {
        uint32_t sectionNum { 0U };                             // section
        std::vector<uint32_t> sectionBn2Idx {};                 // BN2方向，section的切分点
        std::vector<uint32_t> mBaseNum {};                      // S1G方向，切了多少个基本块
        std::vector<uint32_t> s2BaseNum {};                     // S2方向，切了多少个基本块
        std::vector<uint32_t> mTailSize {};                     // S1G方向，尾块size
        std::vector<uint32_t> s2TailSize {};                    // S2方向，尾块size
        bool isEmpty { true };

        explicit GridInfo(uint32_t batchSize) :
            mBaseNum(batchSize),
            s2BaseNum(batchSize),
            mTailSize(batchSize),
            s2TailSize(batchSize) {}
    };

    struct CostInfo {
        std::vector<int64_t> bN2CostOfEachBatch {};             // 整个batch的开销
        std::vector<uint32_t> bN2BlockOfEachBatch {};           // 整个batch的开销
        std::vector<int64_t> bN2LastBlockCostOfEachBatch {};    // batch最后一块的开销
        std::vector<uint32_t> sectionBlockNum {};
        std::vector<uint32_t> sectionCost {};

        explicit CostInfo(uint32_t batchSize) :
            bN2CostOfEachBatch(batchSize),
            bN2BlockOfEachBatch(batchSize),
            bN2LastBlockCostOfEachBatch(batchSize) {}
    };

    struct ComputeContext {
        const DeviceInfo &deviceInfo;
        const IBaseInfo &baseInfo;
        GridInfo gridInfo;
        CostInfo costInfo;

        explicit ComputeContext(const DeviceInfo &dInfo, const IBaseInfo &bInfo) :
            deviceInfo(dInfo),
            baseInfo(bInfo),
            gridInfo(bInfo.GetBatchSize()),
            costInfo(bInfo.GetBatchSize()) {}
    };

    // 分核功能模块内部使用：记录batch相关的临时信息
    struct BatchCache {
        uint32_t bIdx { 0U };
        uint32_t s1Size { 0U };
        uint32_t s2Size { 0U };
        int64_t preTokenLeftUp { 0 };
        int64_t nextTokenLeftUp { 0 };
        SquareTable<int64_t, BLOCK_MAX_TYPE> typeCost {};
    };

    // 分核功能模块内部使用：记录当前行（S1G）的临时信息
    struct S1GCache {
        uint32_t bIdx { 0U };
        uint32_t s1GIdx { 0U };
        uint32_t s2Start { 0U };
        uint32_t s2End { 0U };
        int64_t s1GCost { 0 };
        int64_t s1GLastBlockCost { 0 };
        uint32_t s1GBlock { 0U };
        int64_t s1GNormalBlockCost { 0 };
    };

    struct CacheResult {
        uint32_t aiCoreNum { 0U };
        uint32_t aiVecNum { 0U };
        int64_t maxCost { 0 };                                  // 慢核开销
        uint32_t bn2Start { 0U };

        // FA
        uint32_t usedCoreNum { 0U };                            // 使用的核数量
        std::vector<uint32_t> bN2End {};                        // 每个核处理数据的BN2结束点
        std::vector<uint32_t> gS1End {};                        // 每个核处理数据的GS1结束点
        std::vector<uint32_t> s2End {};                         // 每个核处理数据的S2结束点
        uint32_t maxS2SplitNum { 0U };                          // 单个归约任务最大分核数量
        uint32_t fdTaskNum { 0U };                              // 归约任务数量
        std::vector<uint32_t> firstFdDataWorkspaceIdx {};       // 每个核第一份归约任务的存放地址

        // FD
        uint32_t usedVecNum { 0U };                             // 归约过程使用的vector数量
        // 1、归约任务的索引信息
        std::vector<uint32_t> bN2Idx {};            // 每个归约任务的BN2索引，脚标为归约任务的序号，最大为核数-1
        std::vector<uint32_t> mIdx {};              // 每个归约任务的GS1索引，脚标为归约任务的序号
        std::vector<uint32_t> workspaceIdx {};      // 每个归约任务在workspace中的存放位置
        std::vector<uint32_t> s2SplitNum {};        // 每个归约任务的S2核间切分份数，脚标为归约任务的序号
        std::vector<uint32_t> mSize {};             // 每个归约任务的M轴大小，脚标为归约任务的序号
        // 2、FD kernel阶段，归约任务的分核信息
        std::vector<uint32_t> taskIdx {};           // 每个vector处理的归约任务对应ID
        std::vector<uint32_t> mStart {};            // 每个vector处理的归约任务的M轴起点
        std::vector<uint32_t> mLen {};              // 每个vector处理的归约任务的M轴行数

        explicit CacheResult(uint32_t aicNum, uint32_t aivNum) :
            aiCoreNum(aicNum),
            aiVecNum(aivNum),
            bN2End(aicNum),
            gS1End(aicNum),
            s2End(aicNum),
            firstFdDataWorkspaceIdx(aicNum),
            bN2Idx(aicNum),
            mIdx(aicNum),
            workspaceIdx(aicNum),
            s2SplitNum(aicNum),
            mSize(aicNum),
            taskIdx(aivNum),
            mStart(aivNum),
            mLen(aivNum) {}

        void Clear()
        {
            maxCost = INT64_MIN;
            usedCoreNum = 0U;
            maxS2SplitNum = 0U;
            fdTaskNum = 0U;
            bn2Start = 0U;
            std::fill(bN2End.begin(), bN2End.end(), 0U);
            std::fill(gS1End.begin(), gS1End.end(), 0U);
            std::fill(s2End.begin(), s2End.end(), 0U);
            std::fill(firstFdDataWorkspaceIdx.begin(), firstFdDataWorkspaceIdx.end(), 0U);

            usedVecNum = 0U;
            std::fill(bN2Idx.begin(), bN2Idx.end(), 0U);
            std::fill(mIdx.begin(), mIdx.end(), 0U);
            std::fill(workspaceIdx.begin(), workspaceIdx.end(), 0U);
            std::fill(s2SplitNum.begin(), s2SplitNum.end(), 0U);
            std::fill(mSize.begin(), mSize.end(), 0U);
            std::fill(taskIdx.begin(), taskIdx.end(), 0U);
            std::fill(mStart.begin(), mStart.end(), 0U);
            std::fill(mLen.begin(), mLen.end(), 0U);
        }

        void CopyTo(CacheResult &other) const
        {
            other.maxCost = maxCost;
            other.bn2Start = bn2Start;
            other.usedCoreNum = usedCoreNum;
            other.maxS2SplitNum = maxS2SplitNum;
            other.fdTaskNum = fdTaskNum;
            other.usedVecNum = usedVecNum;
            other.bN2End = bN2End;
            other.gS1End = gS1End;
            other.s2End = s2End;
            other.firstFdDataWorkspaceIdx = firstFdDataWorkspaceIdx;
            other.bN2Idx = bN2Idx;
            other.mIdx = mIdx;
            other.workspaceIdx = workspaceIdx;
            other.s2SplitNum = s2SplitNum;
            other.mSize = mSize;
            other.taskIdx = taskIdx;
            other.mStart = mStart;
            other.mLen = mLen;
        }

        SectionStreamKFaResult ToSectionStreamKFaResult() const
        {
            SectionStreamKFaResult tmp(aiCoreNum);
            tmp.usedCoreNum = usedCoreNum;
            tmp.bN2End = bN2End;
            tmp.gS1End = gS1End;
            tmp.s2End = s2End;
            tmp.firstFdDataWorkspaceIdx = firstFdDataWorkspaceIdx;
            return tmp;
        }

        SectionStreamKFdResult ToSectionStreamKFdResult() const
        {
            SectionStreamKFdResult tmp(aiCoreNum, aiVecNum);
            tmp.usedVecNum = usedVecNum;
            tmp.bN2Idx = bN2Idx;
            tmp.gS1Idx = mIdx;
            tmp.workspaceIdx = workspaceIdx;
            tmp.s2SplitNum = s2SplitNum;
            tmp.taskIdx = taskIdx;
            tmp.mStart = mStart;
            tmp.mLen = mLen;
            return tmp;
        }
    };

    // 分核功能模块内部使用：记录分配过程中，当前核的负载信息
    struct CoreCache {
        int64_t costLimit { 0 };  // 负载上限
        int64_t cost { 0 };       // 已分配负载
        uint32_t block { 0U };      // 已分配块数
    };

    struct AssignContext {
        uint32_t curSectionIdx { 0U };
        uint32_t curBIdx { 0U };
        uint32_t curBN2Idx { 0U };
        uint32_t curS1GIdx { 0U };
        uint32_t curS2Idx { 0U };
        uint32_t curCoreIdx { 0U };
        int64_t unassignedCost { 0 };
        uint32_t usedCoreNum { 0U };
        uint32_t curKvSplitPart { 1U };
        uint32_t preFdDataNum { 0U };
        int64_t bN2Cost { 0 };
        uint32_t bN2Block { 0U };
        bool isFinished { false };
        BatchCache batchCache {};
        S1GCache s1GCache {};
        CoreCache coreCache {};
    };

    struct FaConfig {
        bool FdOn { true };
        uint32_t coreNum { 0U };
        int64_t coreLimit { 0U };
        int64_t costLimit { INT64_MIN };
        uint32_t sectionIdx { 0U };
    };

private:
    static int64_t CalcCost(uint32_t basicM, uint32_t basicS2);
    static SquareTable<int64_t, BLOCK_MAX_TYPE> CalcCostTable(uint32_t s1NormalSize, uint32_t s2NormalSize,
        uint32_t s1GTailSize, uint32_t s2TailSize);
    static void ClearResult(SectionStreamKResult &result);
    Range<uint32_t> CalcS2Range(uint32_t s1GIdx, const IBaseInfo &baseInfo, const BatchCache &batchCache);
    void CalcGridInfo(ComputeContext &computeContext);
    void CalcCostInfo(ComputeContext &computeContext);
    void CalcBatchCost(uint32_t bIdx, const ComputeContext &computeContext, CostInfo &costInfo);
    void CalcBatchCache(uint32_t bIdx, const ComputeContext &splitContext, BatchCache &batchCache);
    void CalcS1GCache(uint32_t s1GIdx, const ComputeContext &computeContext, const BatchCache &batchCache,
        S1GCache &s1GCache);
    void ScheduleFa(const FaConfig &faConfig, const ComputeContext &computeContext, CacheResult &result);
    void ScheduleFd(CacheResult &result);
    bool IsNeedRecordFDInfo(const AssignContext &assignContext, const CacheResult &result);
    void RecordFDInfo(const ComputeContext &computeContext, const AssignContext &assignContext, CacheResult &result);

    // assign
    void UpdateCursor(const ComputeContext &computeContext, AssignContext &assignContext);
    void AssignByBatch(const ComputeContext &computeContext, AssignContext &assignContext);
    void AssignByRow(const ComputeContext &computeContext, AssignContext &assignContext);
    void AssignByBlock(const ComputeContext &computeContext, AssignContext &assignContext);
    void ForceAssign(const ComputeContext &computeContext, AssignContext &assignContext);
private:
    SectionStreamKParam m_param {};
};

}
#endif