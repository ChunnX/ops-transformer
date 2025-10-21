/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file split_core.h
 * \brief
 */

#ifndef SPLIT_CORE_H
#define SPLIT_CORE_H


namespace optiling {

constexpr uint32_t MAX_SPLIT_RATIO = 2;

struct BaseInfo {
    uint32_t bSize = 0U;
    uint32_t n2Size = 0U;
    uint32_t gSize = 0U;
    uint32_t s1Size = 0U;
    uint32_t s2Size = 0U;
    bool isAccumSeqS1 = false;
    bool isAccumSeqS2 = false;
    bool slidingFlag = false;
    const int64_t *actualSeqS1Size = nullptr;
    const int64_t *actualSeqS2Size = nullptr;
    uint32_t actualLenQDims = 0U;
    uint32_t actualLenKvDims = 0U;
    int64_t preToken = 0;
    int64_t nextToken = 0;
};

struct InnerSplitParams {
    uint32_t s1GBaseSize = 1U;
    uint32_t s2BaseSize = 1U;
};

struct OuterSplitParams {
    uint32_t *bN2End = nullptr;
    uint32_t *gS1End = nullptr;
    uint32_t *s2End = nullptr;
};

struct FlashDecodeParams {
    uint32_t *bN2IdxOfFdHead = nullptr;
    uint32_t *gS1IdxOfFdHead = nullptr;
    uint32_t *s2SplitNumOfFdHead = nullptr;
    uint32_t *s2SplitStartIdxOfCore = nullptr;
    uint32_t gS1BaseSizeOfFd = 0U;
    uint32_t *gS1SplitNumOfFdHead = nullptr;
    uint32_t *gS1LastPartSizeOfFdHead = nullptr;
    uint32_t *gS1IdxEndOfFdHead = nullptr;
    uint32_t *gS1IdxEndOfFdHeadSplit = nullptr;
};

struct SplitCoreRes {
    uint32_t numOfFdHead = 0U;
    uint32_t maxS2SplitNum = 0U;
    uint32_t usedCoreNum = 0U;
    uint32_t usedVecNumOfFd = 0U;
};

struct SplitBatchInfo {
    std::vector<uint32_t> s1GBaseNum;           // S1G方向，切了多少个基本块
    std::vector<uint32_t> s2BaseNum;            // S2方向，切了多少个基本块
    std::vector<uint32_t> s1GTailSize;          // S1G方向，尾块size
    std::vector<uint32_t> s2TailSize;           // S2方向，尾块size
    std::vector<uint32_t> batchTotalCost;       // 整个batch的开销
    std::vector<uint32_t> normalS1GTotalCost;   // 满行的开销
    std::vector<uint32_t> tialS1GTotalCost;     // 尾行的开销
    std::vector<uint32_t> batchLastBlockCost;   // batch最后一块的开销
    std::vector<uint32_t> s1GLastBlockCost;     // M轴不满、S2轴满的块的开销
    std::vector<uint32_t> s2LastBlockCost;      // M轴满、S2轴不满的块的开销
    uint32_t totalBlockNum = 0U;
    uint32_t totalCost = 0U;
    uint32_t normalBlockCost = 0U;
    uint32_t lastValidBIdx = 0U;

    SplitBatchInfo(uint32_t batchSize) 
    :   s1GBaseNum(batchSize),
        s2BaseNum(batchSize),
        s1GTailSize(batchSize),
        s2TailSize(batchSize),
        batchTotalCost(batchSize),
        normalS1GTotalCost(batchSize),
        tialS1GTotalCost(batchSize),
        batchLastBlockCost(batchSize),
        s1GLastBlockCost(batchSize),
        s2LastBlockCost(batchSize)
    {}
};

struct AssignInfo {
    uint32_t curCoreIdx = 0U;
    uint32_t curCostOnCore = 0U;
    uint32_t curBlockOnCore = 0U;
    uint32_t costLimit = 0U;
    uint32_t unassignedCost = 0U;
    uint32_t bIdx = 0U;
    uint32_t s1GIdx = 0U;
    uint32_t s2Idx = 0U;
    uint32_t batchLeftCost = 0U;
    uint32_t s1GLeftCost = 0U;
    uint32_t batchLeftBlock = 0U;
    uint32_t s1GLeftBlock = 0U;
    bool isComplete = false;
};

enum  BlockType {
    NORMAL_BLOCK = 0,
    TAIL_BLOCK = 1,
    BLOCK_MAX_TYPE = 2,
};

using BlockCost = std::array<std::array<uint32_t, static_cast<uint32_t>(BLOCK_MAX_TYPE)>, static_cast<uint32_t>(BLOCK_MAX_TYPE)>;

void RecordFDInfo(const BaseInfo &baseInfo, const InnerSplitParams &innerSplitParams, const SplitBatchInfo &splitBatchInfo, 
                const OuterSplitParams &outerSplitParams, FlashDecodeParams &fDParams, SplitCoreRes &res, uint32_t curCoreIdx, uint32_t currKvSplitPart);
void SplitCoreOfBand(const BaseInfo &baseInfo, const InnerSplitParams &innerSplitParams, uint32_t coreNum, OuterSplitParams outerSplitParams, FlashDecodeParams fDParams, SplitCoreRes &res);
void CaclCostTable(BlockCost &typeCost, uint32_t s1NomralSize, uint32_t s2NormalSize, uint32_t s1GTailSize, uint32_t s2TailSize);
int64_t ClipSInnerToken(int64_t sInnerToken, int64_t minValue, int64_t maxValue);
void SplitCore(const BaseInfo &baseInfo, const InnerSplitParams &innerSplitParams, uint32_t coreNum, OuterSplitParams &outerSplitParams, FlashDecodeParams &fDParams, SplitCoreRes &res);
void GetBlockNumOnCore(const SplitBatchInfo &splitBatchInfo, uint32_t coreNum, std::vector<uint32_t> &blockNumOnCore, uint32_t &coreUse, uint32_t &maxCost);
void CaclAllCost(const BlockCost &typeCost, SplitBatchInfo &splitBatchInfo, uint32_t bIdx);
void GetSqeSize(const BaseInfo &baseInfo, uint32_t &s1Size, uint32_t &s2Size, uint32_t bIdx);
void CalSplitBatchInfo(const BaseInfo &baseInfo, const InnerSplitParams &innerSplitParams, SplitBatchInfo &splitBatchInfo, uint32_t coreNum);
void GetPreNextTokenLeftUp(const BaseInfo &baseInfo, int64_t s1Size, int64_t s2Size,
                           int64_t &preTokenLeftUp, int64_t &nextTokenLeftUp);
void SplitFD(SplitCoreRes &res, FlashDecodeParams &fDParams, uint32_t coreNum);
void CalBasicCost(uint32_t basicM, uint32_t basicS2, uint32_t &cost);
void AssignByRow(const SplitBatchInfo &splitBatchInfo, AssignInfo &assignInfo);
bool IsSpaceEnough(uint32_t spaceLimit, uint32_t spaceOccupied, uint32_t spaceTolerance, uint32_t newOccupancy);
void ReconSplitPlan(const BaseInfo &baseInfo, const InnerSplitParams &innerSplitParams, const SplitBatchInfo &splitBatchInfo, const std::vector<uint32_t> &blockNumOnCore, OuterSplitParams &outerSplitParams, FlashDecodeParams &fDParams, SplitCoreRes &res);
void UpdateSInnerLoop(const BaseInfo &baseInfo, const InnerSplitParams &innerSplitParams, uint32_t &s2Start, uint32_t &s2End,
                      uint32_t s1Idx, int64_t s2Size, int64_t preTokenLeftUp, int64_t nextTokenLeftUp, bool seqZeroFlag);
void AssignByBlock(const SplitBatchInfo &splitBatchInfo, AssignInfo &assignInfo);
void AssignByBatch(const SplitBatchInfo &splitBatchInfo, AssignInfo &assignInfo);
uint32_t GetCalcBlockNumOneHead(const BaseInfo &baseInfo, const InnerSplitParams &innerSplitParams, uint32_t s1GBaseNum,
                                uint32_t s2BaseNum, uint32_t s2Size, int64_t preTokenLeftUp, int64_t nextTokenLeftUp, bool seqZeroFlag);
                                
int64_t ClipSInnerToken(int64_t sInnerToken, int64_t minValue, int64_t maxValue) {
    sInnerToken = sInnerToken > minValue ? sInnerToken : minValue;
    sInnerToken = sInnerToken < maxValue ? sInnerToken : maxValue;
    return sInnerToken;
}

void GetPreNextTokenLeftUp(const BaseInfo &baseInfo, int64_t s1Size, int64_t s2Size,
                           int64_t &preTokenLeftUp, int64_t &nextTokenLeftUp) {
    if (baseInfo.slidingFlag) {
        preTokenLeftUp = baseInfo.preToken - s2Size + s1Size;
        nextTokenLeftUp = baseInfo.nextToken + s2Size - s1Size;
    }
}

void UpdateSInnerLoop(const BaseInfo &baseInfo, const InnerSplitParams &innerSplitParams, uint32_t &s2Start, uint32_t &s2End,
                      uint32_t s1Idx, int64_t s2Size, int64_t preTokenLeftUp, int64_t nextTokenLeftUp, bool seqZeroFlag) {
    if (!baseInfo.slidingFlag || seqZeroFlag) {
        return;
    }
    uint32_t s1BaseSize = innerSplitParams.s1GBaseSize / baseInfo.gSize; // 暂时只支持BSND
    int64_t s1Offset = static_cast<int64_t>(s1Idx) * static_cast<int64_t>(s1BaseSize);
    int64_t s2FirstToken = ClipSInnerToken(s1Offset - preTokenLeftUp, 0, s2Size);
    s2Start = static_cast<uint32_t>(s2FirstToken / static_cast<int64_t>(innerSplitParams.s2BaseSize));

    int64_t s2LastToken = ClipSInnerToken(s1Offset +
        nextTokenLeftUp + static_cast<int64_t>(s1BaseSize), 0, s2Size);
    s2End = (s2LastToken + static_cast<int64_t>(innerSplitParams.s2BaseSize) - 1) / static_cast<int64_t>(innerSplitParams.s2BaseSize);
}

uint32_t GetCalcBlockNumOneHead(const BaseInfo &baseInfo, const InnerSplitParams &innerSplitParams, uint32_t s1GBaseNum,
                                uint32_t s2BaseNum, uint32_t s2Size, int64_t preTokenLeftUp, int64_t nextTokenLeftUp, bool seqZeroFlag) {
    if (!baseInfo.slidingFlag || seqZeroFlag) {
        return s1GBaseNum * s2BaseNum;
    }
    uint32_t totalBlockNum = 0;
    for (uint32_t s1Idx = 0; s1Idx < s1GBaseNum; s1Idx++) {
        uint32_t s2Start = 0;
        uint32_t s2End = s2BaseNum;
        UpdateSInnerLoop(baseInfo, innerSplitParams, s2Start, s2End,
                         s1Idx, s2Size, preTokenLeftUp, nextTokenLeftUp, seqZeroFlag);
        totalBlockNum +=  s2End - s2Start;           
    }
    return totalBlockNum;
}

void CalBasicCost(uint32_t basicM, uint32_t basicS2, uint32_t &cost)
{
    uint32_t alignCoefM = 16;
    uint32_t alignCoefS2 = 64;
    uint32_t alignBasicM = (basicM + alignCoefM - 1U) >> 4;      // 按alignCoefM对齐，向上取整，4：移位操作实现除16
    uint32_t alignBasicS2 = (basicS2 + alignCoefS2 - 1U) >> 6;   // 按alignCoefS2对齐，向上取整，6：移位操作实现除64
    cost = 6U * alignBasicM + 10U * alignBasicS2;                 // 6：M轴系数，10：S2轴系数
}

void CaclCostTable(BlockCost &typeCost, uint32_t s1NomralSize, uint32_t s2NormalSize, uint32_t s1GTailSize, uint32_t s2TailSize)
{
    if (s1GTailSize != 0U) {
        CalBasicCost(s1GTailSize, s2NormalSize, typeCost[TAIL_BLOCK][NORMAL_BLOCK]);
    }
    if (s2TailSize != 0U) {
        CalBasicCost(s1NomralSize, s2TailSize, typeCost[NORMAL_BLOCK][TAIL_BLOCK]);
    }
    if (s1GTailSize != 0U && s2TailSize != 0U) {
        CalBasicCost(s1GTailSize, s2TailSize, typeCost[TAIL_BLOCK][TAIL_BLOCK]);
    }
}

void CaclAllCost(const BlockCost &typeCost, SplitBatchInfo &splitBatchInfo, uint32_t bIdx)
{
    BlockType s1Type = NORMAL_BLOCK;
    BlockType s2Type = NORMAL_BLOCK;
    splitBatchInfo.batchTotalCost[bIdx] = 0;
    uint32_t tailS1GNum = splitBatchInfo.s1GTailSize[bIdx] != 0 ? 1 : 0;
    uint32_t tailS2Num = splitBatchInfo.s2TailSize[bIdx] != 0 ? 1 : 0;
    uint32_t normalS1GNum = splitBatchInfo.s1GBaseNum[bIdx] - tailS1GNum;
    uint32_t normalS2Num = splitBatchInfo.s2BaseNum[bIdx] - tailS2Num;

    // |————|#|
    // |****|@|
    // 计算满块负载
    s1Type = NORMAL_BLOCK;
    s2Type = NORMAL_BLOCK;
    splitBatchInfo.batchTotalCost[bIdx] += typeCost[NORMAL_BLOCK][NORMAL_BLOCK] * normalS1GNum * normalS2Num;           // 累加满块开销
    splitBatchInfo.batchTotalCost[bIdx] += typeCost[TAIL_BLOCK][NORMAL_BLOCK] * tailS1GNum * normalS2Num;               // 累加尾行除最后一块开销
    splitBatchInfo.batchTotalCost[bIdx] += typeCost[NORMAL_BLOCK][TAIL_BLOCK] * normalS1GNum * tailS2Num;               // 累加尾列除最后一块开销
    splitBatchInfo.batchTotalCost[bIdx] += typeCost[TAIL_BLOCK][TAIL_BLOCK] * tailS1GNum * tailS2Num;                   // 累加最后一块开销

    splitBatchInfo.normalS1GTotalCost[bIdx] = normalS1GNum > 0 ? (typeCost[NORMAL_BLOCK][NORMAL_BLOCK] * normalS2Num + typeCost[NORMAL_BLOCK][TAIL_BLOCK] * tailS2Num) : 0;     // 求M轴满行对应的开销
    splitBatchInfo.tialS1GTotalCost[bIdx] = (typeCost[TAIL_BLOCK][NORMAL_BLOCK] * normalS2Num + typeCost[TAIL_BLOCK][TAIL_BLOCK] * tailS2Num) * tailS1GNum;                     // 求M轴尾行对应的开销

    s1Type = tailS1GNum > 0u ? TAIL_BLOCK : NORMAL_BLOCK;
    s2Type = tailS2Num > 0u ? TAIL_BLOCK : NORMAL_BLOCK;
    splitBatchInfo.batchLastBlockCost[bIdx] = typeCost[s1Type][s2Type];     // batch的最后一块开销，需要考虑两个方向的尾块

    splitBatchInfo.s1GLastBlockCost[bIdx] = normalS2Num > 0 ? typeCost[s1Type][NORMAL_BLOCK] : typeCost[s1Type][s2Type];    // 第一列的S1G方向的最后一块开销，需要考虑S2方向是否只有一块
    splitBatchInfo.s2LastBlockCost[bIdx] = normalS1GNum > 0 ? typeCost[NORMAL_BLOCK][s2Type] : typeCost[s1Type][s2Type];    // 第一行的S2方向的最后一块开销，需要考虑S1G方向是否只有一块
}

void GetSqeSize(const BaseInfo &baseInfo, uint32_t &s1Size, uint32_t &s2Size, uint32_t bIdx)
{
    s1Size = baseInfo.s1Size;
    s2Size = baseInfo.s2Size;
    if (baseInfo.actualSeqS1Size != nullptr) {
        if (baseInfo.actualLenQDims == 1U) {
            s1Size = static_cast<uint32_t>(baseInfo.actualSeqS1Size[0]);
        } else {
            if (baseInfo.isAccumSeqS1 && bIdx > 0U) {
                s1Size = static_cast<uint32_t>(baseInfo.actualSeqS1Size[bIdx] - baseInfo.actualSeqS1Size[bIdx - 1U]);
            } else {
                s1Size = static_cast<uint32_t>(baseInfo.actualSeqS1Size[bIdx]);
            } 
        }
    }
    if (baseInfo.actualSeqS2Size != nullptr) {
        if (baseInfo.actualLenKvDims == 1U) {
            s2Size = static_cast<uint32_t>(baseInfo.actualSeqS2Size[0]);
        } else {
            if (baseInfo.isAccumSeqS2 && bIdx > 0U) {
                s2Size = static_cast<uint32_t>(baseInfo.actualSeqS2Size[bIdx] - baseInfo.actualSeqS2Size[bIdx - 1U]);
            } else {
                s2Size = static_cast<uint32_t>(baseInfo.actualSeqS2Size[bIdx]);
            }
        } 
    }
}

bool IsSpaceEnough(uint32_t spaceLimit, uint32_t spaceOccupied, uint32_t spaceTolerance, uint32_t newOccupancy)
{
    return spaceLimit + spaceTolerance >= spaceOccupied + newOccupancy;
}

void RecordFDInfo(const BaseInfo &baseInfo, const InnerSplitParams &innerSplitParams, const SplitBatchInfo &splitBatchInfo, 
                const OuterSplitParams &outerSplitParams, FlashDecodeParams &fDParams, SplitCoreRes &res, uint32_t curCoreIdx, uint32_t currKvSplitPart)
{
    uint32_t s1Size = 0U;
    uint32_t s2Size = 0U;
    uint32_t splitBIdx = static_cast<uint32_t>(outerSplitParams.bN2End[curCoreIdx-1U]);
    uint32_t splitS1GIdx = static_cast<uint32_t>(outerSplitParams.gS1End[curCoreIdx-1U]);
    GetSqeSize(baseInfo, s1Size, s2Size, splitBIdx);
    // 计算归约数据的FD均衡划分信息
    uint32_t currFdS1gSize = (splitS1GIdx == splitBatchInfo.s1GBaseNum[splitBIdx] - 1) ? 
                            (s1Size * baseInfo.gSize - splitS1GIdx * innerSplitParams.s1GBaseSize) : innerSplitParams.s1GBaseSize;
    uint32_t currFdS1gSplitPart = (currFdS1gSize + fDParams.gS1BaseSizeOfFd - 1U) / fDParams.gS1BaseSizeOfFd;
    uint32_t currFdS1gLastPartSize = currFdS1gSize % fDParams.gS1BaseSizeOfFd;
    if (currFdS1gLastPartSize == 0U) {
        currFdS1gLastPartSize = fDParams.gS1BaseSizeOfFd;
    }
    res.maxS2SplitNum = std::max(res.maxS2SplitNum, currKvSplitPart);
    fDParams.bN2IdxOfFdHead[res.numOfFdHead] =  splitBIdx;  // 若存在头归约，则切分点一定为上一个核结束的位置
    fDParams.gS1IdxOfFdHead[res.numOfFdHead] = splitS1GIdx;
    fDParams.s2SplitNumOfFdHead[res.numOfFdHead] = currKvSplitPart;
    fDParams.gS1SplitNumOfFdHead[res.numOfFdHead] = currFdS1gSplitPart;
    fDParams.gS1LastPartSizeOfFdHead[res.numOfFdHead] = currFdS1gLastPartSize;
    res.numOfFdHead += 1U;
}

void AssignByBatch(const SplitBatchInfo &splitBatchInfo, AssignInfo &assignInfo)
{
    if (assignInfo.isComplete) {
        return;
    }

    // 1、按整batch分配
    while (IsSpaceEnough(assignInfo.costLimit, assignInfo.curCostOnCore, splitBatchInfo.batchLastBlockCost[assignInfo.bIdx] / 2, assignInfo.batchLeftCost)) {// 2: 当前batch分配给当前核后，超出部分小于最后一块的一半（对齐按块分配的标准），则可以分配
        assignInfo.curCostOnCore += assignInfo.batchLeftCost;
        assignInfo.curBlockOnCore += assignInfo.batchLeftBlock;
        if (assignInfo.bIdx >= splitBatchInfo.lastValidBIdx) {  // 所有负载全部分配完
            assignInfo.isComplete = true;
            return;
        }

        // 当前batch全部分配给当前核了，更新未分配负载的起始地址
        assignInfo.bIdx ++;
        assignInfo.s1GIdx = 0U;
        assignInfo.s2Idx = 0U;

        assignInfo.batchLeftCost = splitBatchInfo.batchTotalCost[assignInfo.bIdx];   //更新batch剩余，不是残留负载，所以当前batch剩下的负载即为batch负载总和
        assignInfo.s1GLeftCost = (assignInfo.s1GIdx == splitBatchInfo.s1GBaseNum[assignInfo.bIdx] - 1 && splitBatchInfo.s1GTailSize[assignInfo.bIdx] != 0) ? 
                        splitBatchInfo.tialS1GTotalCost[assignInfo.bIdx] : splitBatchInfo.normalS1GTotalCost[assignInfo.bIdx];    // 更新行剩余，需要考虑是不是尾行
        assignInfo.batchLeftBlock = splitBatchInfo.s1GBaseNum[assignInfo.bIdx] * splitBatchInfo.s2BaseNum[assignInfo.bIdx];
        assignInfo.s1GLeftBlock = splitBatchInfo.s2BaseNum[assignInfo.bIdx];
    }
}

void AssignByRow(const SplitBatchInfo &splitBatchInfo, AssignInfo &assignInfo)
{
    if (assignInfo.isComplete) {
        return;
    }

    // 2、按行分配
    uint32_t curTailBlockCost = (assignInfo.s1GIdx == splitBatchInfo.s1GBaseNum[assignInfo.bIdx] - 1 && splitBatchInfo.s1GTailSize[assignInfo.bIdx] != 0) ? 
                                splitBatchInfo.batchLastBlockCost[assignInfo.bIdx] : splitBatchInfo.s2LastBlockCost[assignInfo.bIdx];
    while (IsSpaceEnough(assignInfo.costLimit, assignInfo.curCostOnCore, curTailBlockCost / 2U, assignInfo.s1GLeftCost)) {  // 2: 当前行分配给当前核后，超出部分小于最后一块的一半（对齐按块分配的标准），则可以分配
        assignInfo.curCostOnCore += assignInfo.s1GLeftCost;
        assignInfo.curBlockOnCore += assignInfo.s1GLeftBlock;

        assignInfo.s1GIdx ++;
        assignInfo.s2Idx = 0U;
        assignInfo.batchLeftCost = assignInfo.batchLeftCost > assignInfo.s1GLeftCost ? assignInfo.batchLeftCost - assignInfo.s1GLeftCost : 0U; // 当前batch被分配一行出去，更新剩余负载 
        assignInfo.s1GLeftCost = (assignInfo.s1GIdx == splitBatchInfo.s1GBaseNum[assignInfo.bIdx] - 1 && splitBatchInfo.s1GTailSize[assignInfo.bIdx] != 0) ? 
                        splitBatchInfo.tialS1GTotalCost[assignInfo.bIdx] : splitBatchInfo.normalS1GTotalCost[assignInfo.bIdx];   // 更新行剩余，需要考虑是不是尾行
        assignInfo.batchLeftBlock = assignInfo.batchLeftBlock > assignInfo.s1GLeftBlock ? assignInfo.batchLeftBlock - assignInfo.s1GLeftBlock : 0U;
        assignInfo.s1GLeftBlock = splitBatchInfo.s2BaseNum[assignInfo.bIdx];
        if (assignInfo.s1GIdx >= splitBatchInfo.s1GBaseNum[assignInfo.bIdx] - 1) {  // 根据AssignByBatch的规则，最后一行一定不会被分配
            return;
        }
    }
}

void AssignByBlock(const SplitBatchInfo &splitBatchInfo, AssignInfo &assignInfo)
{
    if (assignInfo.isComplete) {
        return;
    }

    // 3、按块分配
    // 获取当前行S2维度满块的负载大小，需要考虑是否为尾行
    // 当前块分配给当前核后，超出部分小于该块的一半（对齐按块分配的标准），则可以分配
    // 使用当前行S2维度满块作为度量的原因是，按行分配的流程走完后，最后一块一定不能被分配，否则在行分配流程中就可以分配了
    uint32_t normalCost = splitBatchInfo.normalBlockCost;
    uint32_t tailCost = splitBatchInfo.s2LastBlockCost[assignInfo.bIdx];
    if (assignInfo.s1GIdx == (splitBatchInfo.s1GBaseNum[assignInfo.bIdx] - 1U) && splitBatchInfo.s1GTailSize[assignInfo.bIdx] != 0U) {
        normalCost = splitBatchInfo.s1GLastBlockCost[assignInfo.bIdx];
        tailCost = splitBatchInfo.batchLastBlockCost[assignInfo.bIdx];
    }
    uint32_t curCost = (assignInfo.s2Idx == splitBatchInfo.s2BaseNum[assignInfo.bIdx] - 1) ? tailCost : normalCost;
    while (IsSpaceEnough(assignInfo.costLimit, assignInfo.curCostOnCore, curCost / 2, curCost) || assignInfo.curBlockOnCore == 0) {      // (maxCost - curCostOnCore) * 2 > s1GLeftCost
        assignInfo.curCostOnCore += curCost;
        assignInfo.curBlockOnCore += 1U;
        assignInfo.s2Idx ++;
        assignInfo.batchLeftCost = assignInfo.batchLeftCost > curCost ? assignInfo.batchLeftCost - curCost : 0U;   // 当前batch被分配一块出去，更新剩余负载
        assignInfo.s1GLeftCost = assignInfo.s1GLeftCost > curCost ? assignInfo.s1GLeftCost - curCost : 0U;     // 当前行被分配一块出去，更新剩余负载
        assignInfo.batchLeftBlock --;
        assignInfo.s1GLeftBlock --;
        if (assignInfo.s2Idx == splitBatchInfo.s2BaseNum[assignInfo.bIdx] - 1) { // 根据AssignByRow的规则，最后一块一定不会被分配
            curCost = tailCost;
        } else if (assignInfo.s2Idx >= splitBatchInfo.s2BaseNum[assignInfo.bIdx]) {
            return;
        }
    }
}

void GetBlockNumOnCore(const SplitBatchInfo &splitBatchInfo, uint32_t coreNum, std::vector<uint32_t> &blockNumOnCore, uint32_t &coreUse, uint32_t &maxCost) 
{
    // 初始化负载上限
    if (coreNum == 0U) {
        return;
    }
    maxCost = 0U;
    coreUse = 0U;
    AssignInfo assignInfo;
    assignInfo.unassignedCost = splitBatchInfo.totalCost;
    // 初始化待分配负载
    assignInfo.batchLeftCost = splitBatchInfo.batchTotalCost[0];
    assignInfo.s1GLeftCost = (0 == splitBatchInfo.s1GBaseNum[0] - 1 && splitBatchInfo.s1GTailSize[0] != 0) ? 
                            splitBatchInfo.tialS1GTotalCost[0] : splitBatchInfo.normalS1GTotalCost[0];
    assignInfo.batchLeftBlock = splitBatchInfo.s1GBaseNum[0] * splitBatchInfo.s2BaseNum[0];
    assignInfo.s1GLeftBlock = splitBatchInfo.s2BaseNum[0];

    while (assignInfo.unassignedCost > 0U && coreNum > assignInfo.curCoreIdx) {
        // 更新当前核的负载上限
        assignInfo.costLimit = assignInfo.unassignedCost / (coreNum - assignInfo.curCoreIdx);
        assignInfo.curCostOnCore = 0U;
        assignInfo.curBlockOnCore = 0U;
        // 1、按整batch分配
        AssignByBatch(splitBatchInfo, assignInfo);

        // 2、按行分配
        AssignByRow(splitBatchInfo, assignInfo);

        // 3、按块分配
        AssignByBlock(splitBatchInfo, assignInfo);

        blockNumOnCore[assignInfo.curCoreIdx] = assignInfo.curBlockOnCore;
        maxCost = std::max(maxCost, assignInfo.curCostOnCore);
        assignInfo.curCoreIdx ++;
        assignInfo.unassignedCost = assignInfo.unassignedCost > assignInfo.curCostOnCore ? assignInfo.unassignedCost - assignInfo.curCostOnCore : 0U;
    }
    coreUse = assignInfo.curCoreIdx;
}

void CalSplitBatchInfo(const BaseInfo &baseInfo, const InnerSplitParams &innerSplitParams, SplitBatchInfo &splitBatchInfo, uint32_t coreNum)
{
    BlockCost typeCost = {{{10, 10}, {10, 10}}};   // 10：初始化开销列表，每个块被视作相同权重，同时为了避免按50%算空间容忍度时被舍掉，造成基本块耗时被忽略，无限叠加在同一核上

    // 计算分块信息
    uint32_t s1Size;
    uint32_t s2Size;
    bool kvseqZeroFlag = true;
    for (uint32_t bIdx = 0; bIdx < baseInfo.bSize; bIdx++) {
        GetSqeSize(baseInfo, s1Size, s2Size, bIdx);
        splitBatchInfo.s1GBaseNum[bIdx] = (s1Size * baseInfo.gSize + (innerSplitParams.s1GBaseSize - 1U)) / innerSplitParams.s1GBaseSize;
        splitBatchInfo.s1GTailSize[bIdx] = (s1Size * baseInfo.gSize) % innerSplitParams.s1GBaseSize; 
        splitBatchInfo.s2BaseNum[bIdx] = (s2Size + innerSplitParams.s2BaseSize - 1U) / innerSplitParams.s2BaseSize;
        splitBatchInfo.s2TailSize[bIdx] = s2Size % innerSplitParams.s2BaseSize;
        if (splitBatchInfo.s2BaseNum[bIdx] != 0) {
            kvseqZeroFlag = false;
        }
    }

    for (uint32_t bIdx = 0; bIdx < baseInfo.bSize; bIdx++) {
        if (kvseqZeroFlag) {
            splitBatchInfo.s1GBaseNum[bIdx] = 1;
            splitBatchInfo.s2BaseNum[bIdx] = 1;
        }
        splitBatchInfo.totalBlockNum += splitBatchInfo.s1GBaseNum[bIdx] * splitBatchInfo.s2BaseNum[bIdx] * baseInfo.n2Size;
        if (splitBatchInfo.s1GBaseNum[bIdx] > 0) {
            splitBatchInfo.lastValidBIdx = bIdx;
        }
    }

    bool isNeedCostBase = (splitBatchInfo.totalBlockNum > coreNum); // 当前基本块数少于等于核数时，分配结果为为一核一块，所以无需计算基本块开销，默认为1

    if (isNeedCostBase) {
        // 计算总基本块数
        CalBasicCost(innerSplitParams.s1GBaseSize, innerSplitParams.s2BaseSize, typeCost[NORMAL_BLOCK][NORMAL_BLOCK]);  // 计算满块的开销，每个batch无需重复计算
    }

    for (uint32_t bIdx = 0; bIdx < baseInfo.bSize; bIdx++) {
        if (isNeedCostBase) {
            CaclCostTable(typeCost, innerSplitParams.s1GBaseSize, innerSplitParams.s2BaseSize, splitBatchInfo.s1GTailSize[bIdx], splitBatchInfo.s2TailSize[bIdx]);
        }
        CaclAllCost(typeCost, splitBatchInfo, bIdx);  // 每个batch需要更新尾块的开销
        splitBatchInfo.totalCost += splitBatchInfo.batchTotalCost[bIdx];
    }
    splitBatchInfo.normalBlockCost = typeCost[NORMAL_BLOCK][NORMAL_BLOCK];
}

void ReconSplitPlan(const BaseInfo &baseInfo, const InnerSplitParams &innerSplitParams, const SplitBatchInfo &splitBatchInfo, const std::vector<uint32_t> &blockNumOnCore, OuterSplitParams &outerSplitParams, FlashDecodeParams &fDParams, SplitCoreRes &res)
{
    res.numOfFdHead = 0U;
    res.maxS2SplitNum = 1U;
    fDParams.s2SplitStartIdxOfCore[0] = 0U; //每核头块所处当前线段被切的第几部分

    AssignInfo assignInfo;
    assignInfo.batchLeftBlock = splitBatchInfo.s1GBaseNum[0] * splitBatchInfo.s2BaseNum[0];
    assignInfo.s1GLeftBlock = splitBatchInfo.s2BaseNum[0];
    uint32_t currKvSplitPart = 1U;
    uint32_t bN2End = assignInfo.bIdx;
    uint32_t gS1End = assignInfo.s1GIdx;
    uint32_t s2End = assignInfo.s2Idx;

    for (uint32_t coreIdx = 0U; coreIdx < res.usedCoreNum; coreIdx++) {
        // 记录上一个核的尾归约的信息
        if (coreIdx > 0U && s2End < splitBatchInfo.s2BaseNum[bN2End] - 1U) {    // 只有切到S2的中间位置，才涉及规约，将currKvSplitPart加1
            currKvSplitPart += 1U;
            fDParams.s2SplitStartIdxOfCore[coreIdx] = currKvSplitPart - 1U;   // 下一个核的起始归约数据的序号
        } else {
            fDParams.s2SplitStartIdxOfCore[coreIdx] = 0U;                     // 当前行刚好被分配完，下一个核的起始归约数据的序号为0
        }

        assignInfo.curBlockOnCore = 0U;
        // 1、按batch数块
        while (blockNumOnCore[coreIdx] >= assignInfo.curBlockOnCore + assignInfo.batchLeftBlock) {
            assignInfo.curBlockOnCore += assignInfo.batchLeftBlock;
            if (assignInfo.bIdx == splitBatchInfo.lastValidBIdx) {  // 所有负载全部分配完
                // 如果最后一个核存在头归约，则根据上一个核的结束位置，定位归约数据的索引
                if (currKvSplitPart > 1) {
                    RecordFDInfo(baseInfo, innerSplitParams, splitBatchInfo, outerSplitParams, fDParams, res, coreIdx, currKvSplitPart);
                }
                return;
            }
            
            bN2End = assignInfo.bIdx;
            gS1End = splitBatchInfo.s1GBaseNum[assignInfo.bIdx] - 1;
            s2End = splitBatchInfo.s2BaseNum[assignInfo.bIdx] - 1;

            assignInfo.bIdx ++;
            assignInfo.s1GIdx = 0;
            assignInfo.s2Idx = 0;
            assignInfo.batchLeftBlock = splitBatchInfo.s1GBaseNum[assignInfo.bIdx] * splitBatchInfo.s2BaseNum[assignInfo.bIdx];
            assignInfo.s1GLeftBlock = splitBatchInfo.s2BaseNum[assignInfo.bIdx];
        }

        // 2、按行数块
        while (blockNumOnCore[coreIdx] >= assignInfo.curBlockOnCore + assignInfo.s1GLeftBlock) {
            assignInfo.curBlockOnCore += assignInfo.s1GLeftBlock;
            bN2End = assignInfo.bIdx;
            gS1End = assignInfo.s1GIdx;
            s2End = splitBatchInfo.s2BaseNum[assignInfo.bIdx] - 1U;

            assignInfo.s1GIdx ++;
            assignInfo.s2Idx = 0U;
            assignInfo.batchLeftBlock -= assignInfo.s1GLeftBlock;
            assignInfo.s1GLeftBlock = splitBatchInfo.s2BaseNum[assignInfo.bIdx];
        }

        // 3、按块数，当前核一定无法完全存放当前行，否则就属于按行分配的
        if (blockNumOnCore[coreIdx] > assignInfo.curBlockOnCore) {
            uint32_t remainSpace = blockNumOnCore[coreIdx] - assignInfo.curBlockOnCore;
            assignInfo.curBlockOnCore += remainSpace;
            bN2End = assignInfo.bIdx;
            gS1End = assignInfo.s1GIdx;
            s2End = assignInfo.s2Idx + remainSpace - 1;
            assignInfo.s2Idx += remainSpace;
            assignInfo.batchLeftBlock -= remainSpace;
            assignInfo.s1GLeftBlock -= remainSpace;
        }

        outerSplitParams.bN2End[coreIdx] = bN2End;
        outerSplitParams.gS1End[coreIdx] = gS1End;
        outerSplitParams.s2End[coreIdx] = s2End;

        // 初始行是否处理完：1、刚好处理完初始行还未跨行；2、已经跨行
        if (coreIdx > 0U && (bN2End != outerSplitParams.bN2End[coreIdx-1U] || gS1End != outerSplitParams.gS1End[coreIdx-1U])) { 
            // 当前位置和上次切分点不在同一行，则将需要考虑当前核是否存在头归约
            if (currKvSplitPart > 1U) {
                RecordFDInfo(baseInfo, innerSplitParams, splitBatchInfo, outerSplitParams, fDParams, res, coreIdx, currKvSplitPart);
                currKvSplitPart = 1U;    // 处理完头归约，S2轴的切分份数初始化为1
            }
        }
    }
}

void SplitCore(const BaseInfo &baseInfo, const InnerSplitParams &innerSplitParams, uint32_t coreNum, OuterSplitParams &outerSplitParams, FlashDecodeParams &fDParams, SplitCoreRes &res) {
    // 1、划分基本块，统计信息
    SplitBatchInfo splitBatchInfo(baseInfo.bSize);
    CalSplitBatchInfo(baseInfo, innerSplitParams, splitBatchInfo, coreNum);

    // 2、获取每个核的分配方案  
    std::vector<uint32_t> blockNumOnCore(coreNum);
    uint32_t maxCore = std::min(coreNum, splitBatchInfo.totalBlockNum);
    uint32_t minCore = static_cast<uint32_t>(std::sqrt(static_cast<float>(splitBatchInfo.totalBlockNum) + 0.25) + 0.5);
    minCore = std::min(minCore, maxCore);
    uint32_t coreUse = 0U;
    uint32_t minMaxCost = UINT32_MAX;
    
    uint32_t tmpMaxCost = 0U;
    uint32_t tmpCoreUse = 0U;
    std::vector<uint32_t> tmpBlockNumOnCore(coreNum);

    for (uint32_t i = minCore; i <= maxCore; ++i) {
        GetBlockNumOnCore(splitBatchInfo, i, tmpBlockNumOnCore, tmpCoreUse, tmpMaxCost);
        if (minMaxCost > tmpMaxCost) {
            minMaxCost = tmpMaxCost;
            coreUse = tmpCoreUse;
            blockNumOnCore.assign(tmpBlockNumOnCore.begin(), tmpBlockNumOnCore.end());
        }
    }
    coreUse = std::max(coreUse, 1U);    // 至少使用1个核，用于刷0
    res.usedCoreNum = coreUse;
    // 3、根据每个核的分配数量重建分核方案，获取切分点、记录归约信息等
    ReconSplitPlan(baseInfo, innerSplitParams, splitBatchInfo, blockNumOnCore, outerSplitParams, fDParams, res);
    
    // 4、刷新每个核各轴结束位置，结束位置为开区间
    for (uint32_t i = 0; i < coreUse - 1; i++) {
        uint32_t s1GCarry = 0U;
        uint32_t bN2Carry = 0U;
        uint32_t curBEnd = outerSplitParams.bN2End[i] / baseInfo.n2Size;

        outerSplitParams.s2End[i] += 1U;
        if (outerSplitParams.s2End[i] == splitBatchInfo.s2BaseNum[curBEnd]) {
            s1GCarry = 1U;
            outerSplitParams.s2End[i] = 0U;
        }
        outerSplitParams.gS1End[i] += s1GCarry;
        if (outerSplitParams.gS1End[i] == splitBatchInfo.s1GBaseNum[curBEnd]) {
            bN2Carry = 1U;
            outerSplitParams.gS1End[i] = 0U;
        }
        outerSplitParams.bN2End[i] += bN2Carry;
    }

    outerSplitParams.bN2End[coreUse - 1] = baseInfo.bSize * baseInfo.n2Size;
    outerSplitParams.s2End[coreUse - 1] = 0U;
    outerSplitParams.gS1End[coreUse - 1] = 0U;
}

void SplitCoreOfBand(const BaseInfo &baseInfo, const InnerSplitParams &innerSplitParams, uint32_t coreNum, OuterSplitParams outerSplitParams, FlashDecodeParams fDParams, SplitCoreRes &res) {
    // 计算总基本块数
    SplitBatchInfo splitBatchInfo(baseInfo.bSize);
    uint32_t totalBaseNum = 0;
    uint32_t s1Size = 0U;
    uint32_t s2Size = 0U;
    bool seqZeroFlag = true;
    for (uint32_t bIdx = 0; bIdx < baseInfo.bSize; bIdx++) {
        GetSqeSize(baseInfo, s1Size, s2Size, bIdx);
        splitBatchInfo.s1GBaseNum[bIdx] = (s1Size * baseInfo.gSize + (innerSplitParams.s1GBaseSize - 1)) / innerSplitParams.s1GBaseSize;
        splitBatchInfo.s2BaseNum[bIdx] = (s2Size + innerSplitParams.s2BaseSize - 1) / innerSplitParams.s2BaseSize;
        if (s1Size != 0 && s2Size != 0) {
           seqZeroFlag = false;
        }
    }
    for (uint32_t bIdx = 0; bIdx < baseInfo.bSize; bIdx++) {
        GetSqeSize(baseInfo, s1Size, s2Size, bIdx);
        if (seqZeroFlag) {
           splitBatchInfo.s1GBaseNum[bIdx] = 1;
           splitBatchInfo.s2BaseNum[bIdx] = 1;
        }
        int64_t preTokenLeftUp = 0;
        int64_t nextTokenLeftUp = 0;
        GetPreNextTokenLeftUp(baseInfo, static_cast<int64_t>(s1Size), static_cast<int64_t>(s2Size), preTokenLeftUp, nextTokenLeftUp);
        totalBaseNum += GetCalcBlockNumOneHead(baseInfo, innerSplitParams, splitBatchInfo.s1GBaseNum[bIdx], splitBatchInfo.s2BaseNum[bIdx],
                                               s2Size, preTokenLeftUp, nextTokenLeftUp, seqZeroFlag) * baseInfo.n2Size;
    }
    uint32_t avgBaseNum = 1;
    if (totalBaseNum > coreNum) {
        avgBaseNum = (totalBaseNum + coreNum - 1U) / coreNum;
    }

    uint32_t accumBaseNum = 0;       // 当前累积的基本块数
    uint32_t targetBaseNum = 0;
    uint32_t currCoreIdx = 0;
    uint32_t lastValidBN2Idx = 0;
    res.numOfFdHead = 0U;
    res.maxS2SplitNum = 1U;
    fDParams.s2SplitStartIdxOfCore[0] = 0U; //每核头块所处当前线段被切的第几部分
    //分核流程，保存分核数据
    for (uint32_t bN2Idx = 0; bN2Idx < baseInfo.bSize * baseInfo.n2Size; bN2Idx++) { 
        uint32_t bIdx = bN2Idx / baseInfo.n2Size;
        GetSqeSize(baseInfo, s1Size, s2Size, bIdx);
        int64_t preTokenLeftUp = 0;
        int64_t nextTokenLeftUp = 0;
        GetPreNextTokenLeftUp(baseInfo, static_cast<int64_t>(s1Size), static_cast<int64_t>(s2Size), preTokenLeftUp, nextTokenLeftUp);
        for (uint32_t s1GIdx = 0; s1GIdx < splitBatchInfo.s1GBaseNum[bIdx]; s1GIdx++) {
            uint32_t s2Start = 0;
            uint32_t s2End = splitBatchInfo.s2BaseNum[bIdx];
            UpdateSInnerLoop(baseInfo, innerSplitParams, s2Start, s2End,
                         s1GIdx, s2Size, preTokenLeftUp, nextTokenLeftUp, seqZeroFlag);
            uint32_t currKvSplitPart = 1;           // [B,N2,S1]确定后，S2被切了几份
            
            // 计算当前gS1轴被分为多少行，作为FD负载均衡的基本单位
            uint32_t currFdS1gSize = (s1GIdx == splitBatchInfo.s1GBaseNum[bIdx] - 1) ? 
                                    (s1Size * baseInfo.gSize - s1GIdx * innerSplitParams.s1GBaseSize) : innerSplitParams.s1GBaseSize;
            uint32_t currFdS1gSplitPart = (currFdS1gSize + fDParams.gS1BaseSizeOfFd - 1U) / fDParams.gS1BaseSizeOfFd;
            uint32_t currFdS1gLastPartSize = currFdS1gSize % fDParams.gS1BaseSizeOfFd;
            if (currFdS1gLastPartSize == 0U) {
                currFdS1gLastPartSize = fDParams.gS1BaseSizeOfFd;
            }
            for (uint32_t s2Idx = s2Start; s2Idx < s2End; s2Idx++) {
                accumBaseNum += 1U;
                targetBaseNum = (currCoreIdx + 1U) * avgBaseNum;         // 计算当前的目标权重
                if (accumBaseNum >= targetBaseNum) {
                    // 更新当前核的End分核信息
                    outerSplitParams.bN2End[currCoreIdx] = bN2Idx;
                    outerSplitParams.gS1End[currCoreIdx] = s1GIdx;
                    outerSplitParams.s2End[currCoreIdx] = s2Idx;
                    uint32_t s1GCarry = 0U;
                    uint32_t bN2Carry = 0U;

                    outerSplitParams.s2End[currCoreIdx] += 1U;
                    if (outerSplitParams.s2End[currCoreIdx] == s2End) {
                        s1GCarry = 1U;
                        outerSplitParams.s2End[currCoreIdx] = 0U;
                    }
                    outerSplitParams.gS1End[currCoreIdx] += s1GCarry;
                    if (outerSplitParams.gS1End[currCoreIdx] == splitBatchInfo.s1GBaseNum[bIdx]) {
                        bN2Carry = 1U;
                        outerSplitParams.gS1End[currCoreIdx] = 0U;
                    }
                    outerSplitParams.bN2End[currCoreIdx] += bN2Carry;
                    currCoreIdx += 1U;
                    if (s2Idx < s2End - 1U) {    // 只有切到S2的中间位置，才涉及规约，将currKvSplitPart加1
                        currKvSplitPart += 1U;
                        fDParams.s2SplitStartIdxOfCore[currCoreIdx] = currKvSplitPart - 1U;
                    } else {
                        fDParams.s2SplitStartIdxOfCore[currCoreIdx] = 0U;
                    }
                }
            }
            res.maxS2SplitNum = std::max(res.maxS2SplitNum, currKvSplitPart);
            if (currKvSplitPart > 1U) {
                // S2被切过了，需要规约，记录[B,N,S1]三根轴的idx和切分份数，用于规约
                fDParams.bN2IdxOfFdHead[res.numOfFdHead] = bN2Idx;
                fDParams.gS1IdxOfFdHead[res.numOfFdHead] = s1GIdx;
                fDParams.s2SplitNumOfFdHead[res.numOfFdHead] = currKvSplitPart;
                fDParams.gS1SplitNumOfFdHead[res.numOfFdHead] = currFdS1gSplitPart;
                fDParams.gS1LastPartSizeOfFdHead[res.numOfFdHead] = currFdS1gLastPartSize;
                res.numOfFdHead += 1U;
            }
        }
        if ((splitBatchInfo.s1GBaseNum[bIdx] > 0) && (splitBatchInfo.s2BaseNum[bIdx] > 0)) {
            lastValidBN2Idx = bN2Idx;
        }
    }
    if (accumBaseNum < targetBaseNum) {
        // 更新最后一个核的End分核信息
        outerSplitParams.bN2End[currCoreIdx] = lastValidBN2Idx + 1;
        outerSplitParams.gS1End[currCoreIdx] = 0U;
        outerSplitParams.s2End[currCoreIdx] = 0U;
        currCoreIdx += 1U;
    }
    res.usedCoreNum = currCoreIdx;
}

void SplitFD(SplitCoreRes &res, FlashDecodeParams &fDParams, uint32_t coreNum)
{ 
    uint32_t totalFDLoad = 0;
    uint32_t totalFDHeadSplit = 0;
    // 计算FD的总数据量
    for (uint32_t i = 0; i <  res.numOfFdHead; i++) {
        totalFDLoad += fDParams.s2SplitNumOfFdHead[i] * fDParams.gS1SplitNumOfFdHead[i];
        totalFDHeadSplit += fDParams.gS1SplitNumOfFdHead[i];
    }

    // 基于FA开核数量，计算每个Vector需要计算的FD数据量
    uint32_t maxVectorNum = std::min(totalFDHeadSplit, coreNum * 2U);  // FD均衡的最小单位为一个归约任务的一个split，所以最多占用totalFDHeadSplit个vector
    double loadThrOfVector = static_cast<double>(totalFDLoad) / static_cast<double>(maxVectorNum);  // 初始化vector的负载上限
    int64_t loadOfCurVector = 0;
    uint32_t curCoreIndex = 0;
    uint32_t preTmpFDIndexEndOfFdHead = 0;
    uint32_t preTmpFDIndexEndOfFdHeadSplit = 0;
    for (uint32_t i = 0; i <  res.numOfFdHead; i++) {
        uint32_t fDKVSplitNum = fDParams.s2SplitNumOfFdHead[i];
        for (uint32_t gS1SplitIdx = 0; gS1SplitIdx < fDParams.gS1SplitNumOfFdHead[i]; gS1SplitIdx++) {
            double remainSpace = loadThrOfVector - loadOfCurVector;  // 计算当前vector剩余负载空间
            // 判断是否放在当前vector的标准是剩余空间是否能容纳一半当前归约块
            if (fDKVSplitNum > remainSpace * MAX_SPLIT_RATIO) {
                fDParams.gS1IdxEndOfFdHead[curCoreIndex] = preTmpFDIndexEndOfFdHead;
                fDParams.gS1IdxEndOfFdHeadSplit[curCoreIndex] = preTmpFDIndexEndOfFdHeadSplit;
                curCoreIndex += 1U;
                totalFDLoad -= static_cast<uint32_t>(loadOfCurVector);  // 当前未分配的总负载
                loadThrOfVector = static_cast<double>(totalFDLoad) / static_cast<double>(maxVectorNum - curCoreIndex);  // 根据剩余负载和剩余可用vector更新负载上限，保证最后一个vector能分配所有负载
                loadOfCurVector = 0;
            }
            loadOfCurVector += fDKVSplitNum;
            preTmpFDIndexEndOfFdHead = i;
            preTmpFDIndexEndOfFdHeadSplit = gS1SplitIdx;
        }
    }
    fDParams.gS1IdxEndOfFdHead[curCoreIndex] = preTmpFDIndexEndOfFdHead;
    fDParams.gS1IdxEndOfFdHeadSplit[curCoreIndex] = preTmpFDIndexEndOfFdHeadSplit;
    res.usedVecNumOfFd = curCoreIndex + 1;
}

}
#endif
