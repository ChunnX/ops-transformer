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
 * \file mhc_sinkhorn_backward_tiling.cpp
 * \brief mhc_sinkhorn_backward_tiling
 */

#include <vector>
#include <iostream>
#include "util/platform_util.h"
#include "util/shape_util.h"
#include "platform/platform_info.h"
#include "log/log.h"
#include "mhc_sinkhorn_base_tiling.h"

using namespace AscendC;
namespace optiling {

constexpr int64_t BUFFER_NUM = 3;
constexpr int64_t DOUBLE_SIZE = 2;
constexpr int64_t VL_SIZE = 256;
constexpr int64_t MASK_BUFFER = 64;
constexpr int64_t MAX_BUFFER_BACKWARD = 256 * BUFFER_NUM;
constexpr int64_t MAX_BUFFER_FORWARD = 256 * BUFFER_NUM;

static void PrintSplitInfo(const char *tag, const MhcSinkhornSplitCoreInfo &splitInfo)
{
    std::cout << "[MhcSinkhornBaseTiling][" << tag << "]"
              << " tNormCore=" << splitInfo.tNormCore
              << ", usedCoreNum=" << splitInfo.usedCoreNum
              << ", tTailCore=" << splitInfo.tTailCore
              << ", tUbFactor=" << splitInfo.tUbFactor
              << ", tNormCoreLoop=" << splitInfo.tNormCoreLoop
              << ", tTailCoreLoop=" << splitInfo.tTailCoreLoop
              << ", tUbFactorTail=" << splitInfo.tUbFactorTail
              << ", tUbTailTail=" << splitInfo.tUbTailTail << std::endl;
}

MhcSinkhornSplitCoreInfo MhcSinkhornBaseTiling::BaseSplitCores(int64_t tSize, int64_t nSize,
                                                               int64_t xDtypeSize, int64_t outFlag)
{
    MhcSinkhornSplitCoreInfo forwardInfo;
    MhcSinkhornSplitCoreInfo backwardInfo;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    int64_t totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    std::cout << "\n[MhcSinkhornBaseTiling][BaseSplitCores] begin"
              << " tSize=" << tSize
              << ", nSize=" << nSize
              << ", xDtypeSize=" << xDtypeSize
              << ", outFlag=" << outFlag
              << ", totalCoreNum=" << totalCoreNum
              << ", ubSize=" << ubSize << std::endl;

    BackwardSplitCore(ubSize, totalCoreNum, tSize, nSize, xDtypeSize, backwardInfo);
    ForwardSplitCore(ubSize, totalCoreNum, tSize, nSize, xDtypeSize, outFlag, forwardInfo);

    PrintSplitInfo("BackwardSplitResult", backwardInfo);
    PrintSplitInfo("ForwardSplitResult", forwardInfo);

    bool useForward = forwardInfo.tUbFactor <= backwardInfo.tUbFactor;
    std::cout << "[MhcSinkhornBaseTiling][BaseSplitCores] selected="
              << (useForward ? "forwardInfo" : "backwardInfo")
              << ", forward.tUbFactor=" << forwardInfo.tUbFactor
              << ", backward.tUbFactor=" << backwardInfo.tUbFactor << std::endl;

    return useForward ? forwardInfo : backwardInfo;
}

int64_t MhcSinkhornBaseTiling::BackwardCalOccupySize(int64_t ubFactor, int64_t nSize, int64_t xDtypeSize)
{
    int64_t ubBlock = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
    int64_t dataBlockNum = Ops::Base::FloorDiv(ubBlock, xDtypeSize);
    int64_t nAlignSize = Ops::Base::CeilAlign(nSize, dataBlockNum);

    int64_t inQueSize = Ops::Base::CeilAlign(ubFactor * nSize * nSize, dataBlockNum);
    int64_t outQueSize = Ops::Base::CeilAlign(ubFactor * nSize * nSize, dataBlockNum);
    int64_t normOutSize = Ops::Base::CeilAlign(2 * ubFactor * nSize * nAlignSize, dataBlockNum);
    int64_t sumOutSize = Ops::Base::CeilAlign(2 * ubFactor * nAlignSize, dataBlockNum);
    int64_t occupySize = (inQueSize + outQueSize + normOutSize + sumOutSize) * xDtypeSize;

    std::cout << "[MhcSinkhornBaseTiling][BackwardCalOccupySize]"
              << " ubFactor=" << ubFactor
              << ", nSize=" << nSize
              << ", xDtypeSize=" << xDtypeSize
              << ", ubBlock=" << ubBlock
              << ", dataBlockNum=" << dataBlockNum
              << ", nAlignSize=" << nAlignSize
              << ", inQueSizeElem=" << inQueSize
              << ", outQueSizeElem=" << outQueSize
              << ", normOutSizeElem=" << normOutSize
              << ", sumOutSizeElem=" << sumOutSize
              << ", occupySizeBytes=" << occupySize << std::endl;

    return occupySize;
}

void MhcSinkhornBaseTiling::BackwardSplitCore(uint64_t ubSize, int64_t totalCoreNum, int64_t tSize, int64_t nSize,
                                              int64_t xDtypeSize, MhcSinkhornSplitCoreInfo &splitInfo)
{
    std::cout << "\n[MhcSinkhornBaseTiling][BackwardSplitCore] begin"
              << " ubSize=" << ubSize
              << ", totalCoreNum=" << totalCoreNum
              << ", tSize=" << tSize
              << ", nSize=" << nSize
              << ", xDtypeSize=" << xDtypeSize << std::endl;

    splitInfo.tNormCore = Ops::Base::CeilDiv(tSize, totalCoreNum);
    splitInfo.usedCoreNum = Ops::Base::CeilDiv(tSize, splitInfo.tNormCore);
    splitInfo.tTailCore = tSize - splitInfo.tNormCore * (splitInfo.usedCoreNum - 1);
    int64_t availableUbSize = ubSize - MASK_BUFFER - MAX_BUFFER_BACKWARD;
    int64_t singleUbSize = availableUbSize / DOUBLE_SIZE;
    int64_t occupySize = BackwardCalOccupySize(splitInfo.tNormCore, nSize, xDtypeSize);

    std::cout << "[MhcSinkhornBaseTiling][BackwardSplitCore] basicSplit"
              << " tNormCore=" << splitInfo.tNormCore
              << ", usedCoreNum=" << splitInfo.usedCoreNum
              << ", tTailCore=" << splitInfo.tTailCore
              << ", availableUbSize=" << availableUbSize
              << ", singleUbSize=" << singleUbSize
              << ", occupySizeForNormCore=" << occupySize << std::endl;

    if (occupySize <= singleUbSize) {
        splitInfo.tUbFactor = splitInfo.tNormCore;
        std::cout << "[MhcSinkhornBaseTiling][BackwardSplitCore] occupySize <= singleUbSize, tUbFactor=tNormCore="
                  << splitInfo.tUbFactor << std::endl;
    } else {
        int64_t onePiceSize = BackwardCalOccupySize(1, nSize, xDtypeSize);
        splitInfo.tUbFactor = singleUbSize / onePiceSize;
        int64_t tVlFactor = Ops::Base::FloorDiv(VL_SIZE, (8 * xDtypeSize));
        std::cout << "[MhcSinkhornBaseTiling][BackwardSplitCore] occupySize > singleUbSize"
                  << ", onePiceSize=" << onePiceSize
                  << ", rawTUbFactor=" << splitInfo.tUbFactor
                  << ", tVlFactor=" << tVlFactor << std::endl;
        splitInfo.tUbFactor = Ops::Base::FloorAlign(splitInfo.tUbFactor, tVlFactor);
        std::cout << "[MhcSinkhornBaseTiling][BackwardSplitCore] floorAlignedTUbFactor="
                  << splitInfo.tUbFactor << std::endl;
    }

    splitInfo.tNormCoreLoop = splitInfo.tUbFactor == 0 ? 1 :
                              Ops::Base::CeilDiv(splitInfo.tNormCore, splitInfo.tUbFactor);
    splitInfo.tTailCoreLoop = splitInfo.tUbFactor == 0 ? 1 :
                              Ops::Base::CeilDiv(splitInfo.tTailCore, splitInfo.tUbFactor);

    splitInfo.tUbFactorTail = splitInfo.tNormCore - (splitInfo.tNormCoreLoop - 1) * splitInfo.tUbFactor;
    splitInfo.tUbTailTail = splitInfo.tTailCore - (splitInfo.tTailCoreLoop - 1) * splitInfo.tUbFactor;

    PrintSplitInfo("BackwardSplitCoreFinal", splitInfo);
}

int64_t MhcSinkhornBaseTiling::ForwardCalOccupySize(int64_t ubFactor, int64_t nSize,
                                                    int64_t xDtypeSize, int64_t outFlag)
{
    int64_t ubBlock = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
    int64_t dataBlockNum = ubBlock / xDtypeSize;

    int64_t nnSize = nSize * nSize;
    int64_t nAlign = Ops::Base::CeilAlign(nSize, dataBlockNum);

    int64_t inQueSize = Ops::Base::CeilAlign(ubFactor * nnSize, dataBlockNum);
    int64_t outQueSize = Ops::Base::CeilAlign(ubFactor * nnSize, dataBlockNum);
    int64_t normOutSize = Ops::Base::CeilAlign(nSize * ubFactor * nAlign, dataBlockNum);
    int64_t sumColSize = Ops::Base::CeilAlign(ubFactor * nAlign, dataBlockNum);
    int64_t sumRowSize = Ops::Base::CeilAlign(ubFactor * nAlign, dataBlockNum);

    int64_t occupySize = (inQueSize + outQueSize + normOutSize + sumColSize) * xDtypeSize;
    if (outFlag == 1) {
        occupySize = (inQueSize + outQueSize + normOutSize * 2 + sumColSize + sumRowSize) * xDtypeSize;
    }

    std::cout << "[MhcSinkhornBaseTiling][ForwardCalOccupySize]"
              << " ubFactor=" << ubFactor
              << ", nSize=" << nSize
              << ", xDtypeSize=" << xDtypeSize
              << ", outFlag=" << outFlag
              << ", ubBlock=" << ubBlock
              << ", dataBlockNum=" << dataBlockNum
              << ", nnSize=" << nnSize
              << ", nAlign=" << nAlign
              << ", inQueSizeElem=" << inQueSize
              << ", outQueSizeElem=" << outQueSize
              << ", normOutSizeElem=" << normOutSize
              << ", sumColSizeElem=" << sumColSize
              << ", sumRowSizeElem=" << sumRowSize
              << ", occupySizeBytes=" << occupySize << std::endl;

    return occupySize;
}

void MhcSinkhornBaseTiling::ForwardSplitCore(uint64_t ubSize, int64_t totalCoreNum, int64_t tSize, int64_t nSize,
                                             int64_t xDtypeSize, int64_t outFlag, MhcSinkhornSplitCoreInfo &splitInfo)
{
    std::cout << "\n[MhcSinkhornBaseTiling][ForwardSplitCore] begin"
              << " ubSize=" << ubSize
              << ", totalCoreNum=" << totalCoreNum
              << ", tSize=" << tSize
              << ", nSize=" << nSize
              << ", xDtypeSize=" << xDtypeSize
              << ", outFlag=" << outFlag << std::endl;

    splitInfo.tNormCore = Ops::Base::CeilDiv(tSize, totalCoreNum);
    splitInfo.usedCoreNum = Ops::Base::CeilDiv(tSize, splitInfo.tNormCore);
    splitInfo.tTailCore = tSize - splitInfo.tNormCore * (splitInfo.usedCoreNum - 1);
    auto ubSizeUsed = ubSize - MASK_BUFFER - MAX_BUFFER_FORWARD;
    auto availableUbSize = ubSizeUsed / DOUBLE_SIZE;
    
    splitInfo.tUbFactor = Ops::Base::CeilAlign(splitInfo.tNormCore, static_cast<int64_t>(8));
    int64_t occupySize = ForwardCalOccupySize(splitInfo.tUbFactor, nSize, xDtypeSize, outFlag);

    std::cout << "[MhcSinkhornBaseTiling][ForwardSplitCore] basicSplit"
              << " tNormCore=" << splitInfo.tNormCore
              << ", usedCoreNum=" << splitInfo.usedCoreNum
              << ", tTailCore=" << splitInfo.tTailCore
              << ", ubSizeUsed=" << ubSizeUsed
              << ", availableUbSize=" << availableUbSize
              << ", ceilAlignTUbFactor=" << splitInfo.tUbFactor
              << ", occupySizeForTUbFactor=" << occupySize << std::endl;

    if (occupySize > availableUbSize) {
        auto onePiceSize = ForwardCalOccupySize(1, nSize, xDtypeSize, outFlag);
        splitInfo.tUbFactor = availableUbSize / onePiceSize;
        std::cout << "[MhcSinkhornBaseTiling][ForwardSplitCore] occupySize > availableUbSize"
                  << ", onePiceSize=" << onePiceSize
                  << ", rawTUbFactor=" << splitInfo.tUbFactor << std::endl;
        splitInfo.tUbFactor = Ops::Base::FloorAlign(splitInfo.tUbFactor, static_cast<int64_t>(8));
        std::cout << "[MhcSinkhornBaseTiling][ForwardSplitCore] floorAlignedTUbFactor="
                  << splitInfo.tUbFactor << std::endl;
    }
    splitInfo.tUbFactor = splitInfo.tUbFactor == 0 ? 8 : splitInfo.tUbFactor;

    splitInfo.tNormCoreLoop = Ops::Base::CeilDiv(splitInfo.tNormCore, splitInfo.tUbFactor);
    splitInfo.tTailCoreLoop = Ops::Base::CeilDiv(splitInfo.tTailCore, splitInfo.tUbFactor);

    splitInfo.tUbFactorTail = splitInfo.tNormCore - (splitInfo.tNormCoreLoop - 1) * splitInfo.tUbFactor;
    splitInfo.tUbTailTail = splitInfo.tTailCore - (splitInfo.tTailCoreLoop - 1) * splitInfo.tUbFactor;

    PrintSplitInfo("ForwardSplitCoreFinal", splitInfo);
}

} // namespace optiling