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
 * \file split_core_v1.h
 * \brief
 */

#ifndef SPLIT_CORE_H
#define SPLIT_CORE_H

namespace optiling {

struct BaseInfo {
    uint32_t bSize;
    uint32_t n2Size;
    uint32_t gSize;
    uint32_t s1Size = 0;
    uint32_t s2Size = 0;
    bool isAccumSeqS1 = false;
    bool isAccumSeqS2 = false;
    const int64_t *actualSeqS1Size = nullptr;
    const int64_t *actualSeqS2Size = nullptr;
    uint32_t actualLenQDims = 0;
    uint32_t actualLenKvDims = 0;
    int64_t preToken = 0;
    int64_t nextToken = 0;
    bool slidingFlag = false;
};

struct InnerSplitParams {
    uint32_t s1GBaseSize = 1;
    uint32_t s2BaseSize = 1;
};

struct OuterSplitParams {
    uint32_t *bN2End;
    uint32_t *gS1End;
    uint32_t *s2End;
};

struct FlashDecodeParams {
    uint32_t *bN2IdxOfFdHead;
    uint32_t *gS1IdxOfFdHead;
    uint32_t *s2SplitNumOfFdHead;
    uint32_t *s2SplitStartIdxOfCore;
    uint32_t gS1BaseSizeOfFd;
    uint32_t *gS1SplitNumOfFdHead;
    uint32_t *gS1LastPartSizeOfFdHead;
    uint32_t *gS1IdxEndOfFdHead;
    uint32_t *gS1IdxEndOfFdHeadSplit;
};

struct SplitCoreRes {
    uint32_t numOfFdHead;
    uint32_t maxS2SplitNum;
    uint32_t usedCoreNum;
    uint32_t usedVecNumOfFd;
};

#if 1
static void SplitCore(const BaseInfo &baseInfo, const InnerSplitParams &innerSplitParams, uint32_t coreNum, OuterSplitParams outerSplitParams, FlashDecodeParams fDParams, SplitCoreRes &res) {
    std::vector<uint32_t> s1GBaseNum(baseInfo.bSize);       // S1G方向，切了多少个基本块
    std::vector<uint32_t> s2BaseNum(baseInfo.bSize);        // S2方向，切了多少个基本块
     // 计算总基本块数
    uint32_t totalBaseNum = 0;
    bool seqZeroFlag = true;
    for (uint32_t bIdx = 0; bIdx < baseInfo.bSize; bIdx++) {
        uint32_t s1Size = baseInfo.s1Size;
        uint32_t s2Size = baseInfo.s2Size;
        if (baseInfo.actualSeqS1Size != nullptr) {
            if (baseInfo.actualLenQDims == 1) {
                s1Size = baseInfo.actualSeqS1Size[0];
            } else {
                if (baseInfo.isAccumSeqS1 && bIdx > 0) {
                    s1Size = baseInfo.actualSeqS1Size[bIdx] - baseInfo.actualSeqS1Size[bIdx - 1];
                } else {
                    s1Size = baseInfo.actualSeqS1Size[bIdx];
                } 
            }
        }
        if (baseInfo.actualSeqS2Size != nullptr) {
            if (baseInfo.actualLenKvDims == 1) {
                s2Size = baseInfo.actualSeqS2Size[0];
            } else {
                if (baseInfo.isAccumSeqS2 && bIdx > 0) {
                    s2Size = baseInfo.actualSeqS2Size[bIdx] - baseInfo.actualSeqS2Size[bIdx - 1];
                } else {
                    s2Size = baseInfo.actualSeqS2Size[bIdx];
                }
            }
        }
        s1GBaseNum[bIdx] = (s1Size * baseInfo.gSize + (innerSplitParams.s1GBaseSize - 1)) / innerSplitParams.s1GBaseSize;
        s2BaseNum[bIdx] = (s2Size + innerSplitParams.s2BaseSize - 1) / innerSplitParams.s2BaseSize;
        if (s1GBaseNum[bIdx] != 0 && s2BaseNum[bIdx] != 0) {
            seqZeroFlag = false;
        }
    }
    for (uint32_t bIdx = 0; bIdx < baseInfo.bSize; bIdx++) {
        if (seqZeroFlag) {
            s1GBaseNum[bIdx] = 1;
            s2BaseNum[bIdx] = 1;
        }
        totalBaseNum += s1GBaseNum[bIdx] * s2BaseNum[bIdx] * baseInfo.n2Size;
    }

    uint32_t avgBaseNum = 1;
    if (totalBaseNum > coreNum) {
        if (coreNum != 0) {
            avgBaseNum = (totalBaseNum + coreNum - 1) / coreNum;
        }
    }

    uint32_t accumBaseNum = 0;       // 当前累积的基本块数
    uint32_t targetBaseNum = 0;
    uint32_t currCoreIdx = 0;
    uint32_t lastValidBIdx = 0;
    res.numOfFdHead = 0;
    res.maxS2SplitNum = 1;
    fDParams.s2SplitStartIdxOfCore[0] = 0; //每核头块所处当前线段被切的第几部分
    //分核流程，保存分核数据
    for (uint32_t bN2Idx = 0; bN2Idx < baseInfo.bSize * baseInfo.n2Size; bN2Idx++) { 
        uint32_t bIdx = bN2Idx / baseInfo.n2Size;
        uint32_t s1Size = baseInfo.s1Size;
        if (baseInfo.actualSeqS1Size != nullptr) {
            if (baseInfo.actualLenQDims == 1) {
                s1Size = baseInfo.actualSeqS1Size[0];
            } else {
                if (baseInfo.isAccumSeqS1 && bIdx > 0) {
                    s1Size = baseInfo.actualSeqS1Size[bIdx] - baseInfo.actualSeqS1Size[bIdx - 1];
                } else {
                    s1Size = baseInfo.actualSeqS1Size[bIdx];
                } 
            }
        }
        for (uint32_t s1GIdx = 0; s1GIdx < s1GBaseNum[bIdx]; s1GIdx++) {
            uint32_t currKvSplitPart = 1;           // [B,N2,S1]确定后，S2被切了几份
            
            // 计算当前gS1轴被分为多少行，作为FD负载均衡的基本单位
            uint32_t currFdS1gSize = (s1GIdx == s1GBaseNum[bIdx] - 1) ? 
                                    (s1Size * baseInfo.gSize - s1GIdx * innerSplitParams.s1GBaseSize) : innerSplitParams.s1GBaseSize;
            uint32_t currFdS1gSplitPart = (currFdS1gSize + fDParams.gS1BaseSizeOfFd - 1) / fDParams.gS1BaseSizeOfFd;
            uint32_t currFdS1gLastPartSize = currFdS1gSize % fDParams.gS1BaseSizeOfFd;
            if (currFdS1gLastPartSize == 0) {
                currFdS1gLastPartSize = fDParams.gS1BaseSizeOfFd;
            }
            for (uint32_t s2Idx = 0; s2Idx < s2BaseNum[bIdx]; s2Idx++) {
                accumBaseNum += 1;
                targetBaseNum = (currCoreIdx + 1) * avgBaseNum;         // 计算当前的目标权重
                if (accumBaseNum >= targetBaseNum) {
                    // 更新当前核的End分核信息
                    outerSplitParams.bN2End[currCoreIdx] = bN2Idx;
                    outerSplitParams.gS1End[currCoreIdx] = s1GIdx;
                    outerSplitParams.s2End[currCoreIdx] = s2Idx;
                    currCoreIdx += 1;
                    if (s2Idx < s2BaseNum[bIdx] - 1) {    // 只有切到S2的中间位置，才涉及规约，将currKvSplitPart加1
                        currKvSplitPart += 1;
                        fDParams.s2SplitStartIdxOfCore[currCoreIdx] = currKvSplitPart - 1;
                    } else {
                        fDParams.s2SplitStartIdxOfCore[currCoreIdx] = 0;
                    }
                }
            }
            res.maxS2SplitNum = std::max(res.maxS2SplitNum, currKvSplitPart);
            if (currKvSplitPart > 1) {
                // S2被切过了，需要规约，记录[B,N,S1]三根轴的idx和切分份数，用于规约
                fDParams.bN2IdxOfFdHead[res.numOfFdHead] = bN2Idx;
                fDParams.gS1IdxOfFdHead[res.numOfFdHead] = s1GIdx;
                fDParams.s2SplitNumOfFdHead[res.numOfFdHead] = currKvSplitPart;
                fDParams.gS1SplitNumOfFdHead[res.numOfFdHead] = currFdS1gSplitPart;
                fDParams.gS1LastPartSizeOfFdHead[res.numOfFdHead] = currFdS1gLastPartSize;
                res.numOfFdHead += 1;
            }
        }
        if ((s1GBaseNum[bIdx] > 0) && (s2BaseNum[bIdx] > 0)) {
            lastValidBIdx = bIdx;
        }
    }
    if (accumBaseNum < targetBaseNum) {
        // 更新最后一个核的End分核信息
        outerSplitParams.bN2End[currCoreIdx] = (lastValidBIdx + 1) * baseInfo.n2Size - 1;
        outerSplitParams.gS1End[currCoreIdx] = s1GBaseNum[lastValidBIdx] - 1;
        outerSplitParams.s2End[currCoreIdx] = s2BaseNum[lastValidBIdx] - 1;;
        currCoreIdx += 1;
    }
    res.usedCoreNum = currCoreIdx;

    // 更新outerSplitParams
    for (uint32_t i = 0; i < currCoreIdx; i++) {
        uint32_t s1GCarry = 0;
        uint32_t bn2Carry = 0;
        uint32_t curBIdxEnd = outerSplitParams.bN2End[i] / baseInfo.n2Size;
        outerSplitParams.s2End[i] += 1;
        if (outerSplitParams.s2End[i] == s2BaseNum[curBIdxEnd]) {
            s1GCarry = 1;
            outerSplitParams.s2End[i] = 0;
        }
        outerSplitParams.gS1End[i] += s1GCarry;
        if (outerSplitParams.gS1End[i] == s1GBaseNum[curBIdxEnd]) {
            bn2Carry = 1;
            outerSplitParams.gS1End[i] = 0;
        }
        outerSplitParams.bN2End[i] += bn2Carry;
    }
    outerSplitParams.bN2End[currCoreIdx-1] = baseInfo.bSize * baseInfo.n2Size;
    outerSplitParams.gS1End[currCoreIdx-1] = 0;
    outerSplitParams.s2End[currCoreIdx-1] = 0;
}
#endif


static void SplitFD(SplitCoreRes &res, FlashDecodeParams fDParams, uint32_t coreNum)
{ 
    uint64_t totalFDLoad = 0;
    uint32_t totalFDHeadSplit = 0;
    // 计算FD的总数据量
    for (uint32_t i = 0; i <  res.numOfFdHead; i++) {
        totalFDLoad += fDParams.s2SplitNumOfFdHead[i] * fDParams.gS1SplitNumOfFdHead[i];
        totalFDHeadSplit += fDParams.gS1SplitNumOfFdHead[i];
    }

    // 基于FA开核数量，计算每个Vector需要计算的FD数据量
    uint32_t maxVectorNum = std::min(totalFDHeadSplit, coreNum * 2);  // FD均衡的最小单位为一个归约任务的一个split，所以最多占用totalFDHeadSplit个vector
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
            uint32_t spaceMulti = 2;
            if (fDKVSplitNum > remainSpace * spaceMulti) {
                fDParams.gS1IdxEndOfFdHead[curCoreIndex] = preTmpFDIndexEndOfFdHead;
                fDParams.gS1IdxEndOfFdHeadSplit[curCoreIndex] = preTmpFDIndexEndOfFdHeadSplit;
                curCoreIndex += 1;
                totalFDLoad -= loadOfCurVector;  // 当前未分配的总负载
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
