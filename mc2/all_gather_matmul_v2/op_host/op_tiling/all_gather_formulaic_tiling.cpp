/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file all_gather_formulaic_tiling.cpp
 * \brief
 */
#include <iostream>
#include "mc2_log.h"
#include "all_gather_formulaic_tiling.h"

void AllGatherPlusMMV2::PrintEstimateKernelTimeResult(double totalMatmulTime, double totalTpTime)
{
    OPS_LOG_D("AllGatherMatmul", "Input shape {M, N, K} = {%lu, %lu, %lu}, cubeUtil_ %f, "
        "totalMatmulTime %f, totalCommTime %f, minTileSize %lu, mAlignLen %lu, commTimeFactor_ %f, "
        "rankDim_ %lu, rankTile %lu",
        clusterInfo_.mValue, clusterInfo_.nValue, clusterInfo_.kValue, matmulPerf_.cubeUtil_,
        totalMatmulTime, totalTpTime, tilingM_.GetMinLen(), tilingM_.GetAlignLength(), commPerf_.commTimeFactor_,
        rankDim_, rankTileNum_);
}


void AllGatherPlusMMV2::EstimateKernelTime()
{
    SetCommTimeFactor();
    
    // 预测计算、通信任务耗时
    double totalMatmulTime = EstimateTotalMatmulTime();
    double totalTpTime = EstimateTotalCommTime();
    ratioCalcComm_ = (std::max(totalTpTime, totalMatmulTime) / std::min(totalTpTime, totalMatmulTime));
    double frontUtil = matmulPerf_.FindCubeUtilByL2Usage(clusterInfo_.mValue, 1); // Local部分matmul耗时预测
    frontMMTime_ = matmulPerf_.MatmulTime(clusterInfo_.mValue, 1) * matmulPerf_.cubeUtil_ / frontUtil;
    if (totalMatmulTime >= totalTpTime) {
        tilingM_.cutRes.shortTileAtBack = true;
    }

    // 根据通信、计算耗时比例，调整切分约束
    // 2x compute time
    strongTpBound_ = (totalTpTime > totalMatmulTime * 2U) && (clusterInfo_.kValue >= LARGE_K_BOUNDARY);
    // 2x matmulMinTileSize
    bool smallMFlag = (clusterInfo_.mValue < tilingM_.GetMinLen() * 2U) &&
                      (rankTileNum_ > MatmulPerformance::SMALL_RANKTILE);
    bool allowMoreCuts = (!tilingM_.cutRes.shortTileAtBack && smallMFlag) ||
        (strongTpBound_ && clusterInfo_.socType == SocVersion::SOC910_B);
    bool reduceAlignLen = clusterInfo_.nValue > SMALL_N_BOUNDARY && clusterInfo_.mValue <= TINY_M;
    if (allowMoreCuts) {
        if (reduceAlignLen) {
            tilingM_.SetAlignLength(tilingM_.GetAlignLength() / TWO);  // 0.5 is half of mAlignLen
        }
        tilingM_.SetMinLenByMin(tilingM_.GetAlignLength());
    }
    noCutFlag_ = totalTpTime < frontMMTime_ && clusterInfo_.mValue <= tilingM_.tileArgs.maxTileLen;
    if (clusterInfo_.kValue > HUGE_K_BOUNDARY) { // 特大shpae切分后matmul性能好一点
        noCutFlag_ = false;
    }
    PrintEstimateKernelTimeResult(totalMatmulTime, totalTpTime);
}

void AllGatherPlusMMV2::SetCommTimeFactorForA5()
{
    commPerf_.ChangeCommTimeFactorByDivision(ALLGATHERMM_COMMTIME_FACTOR); // 2x time of factor
}

void AllGatherPlusMMV2::SetCommTimeFactorForOther()
{
    // 通算并行时通信有膨胀，大K大N场景膨胀明显，做特殊处理
    bool medianMFlag = (clusterInfo_.mValue > SMALL_M) && (clusterInfo_.mValue <= MEDIAN_M);
    bool bwGrowthByUtil = matmulPerf_.cubeUtil_ < PART_L2_UTIL;
    bool bwGrowthByShape = clusterInfo_.kValue >= LARGE_K_BOUNDARY && clusterInfo_.nValue > LARGE_N_BOUNDARY;
    bool smallDim = rankDim_ <= MIN_COMM_RANKDIM || rankTileNum_ <= MatmulPerformance::SMALL_RANKTILE;
    bwGrowthByUtil = bwGrowthByUtil && !smallDim;
    bwGrowthByShape = bwGrowthByShape && !smallDim;
    if (bwGrowthByUtil) { // 0.85 is max cube utilizaion rate
        if (!medianMFlag) {
            tilingM_.SetMaxTileCnt(8U); // 8 is max tile cnt of M
        }
        commPerf_.ChangeCommTimeFactorByDivision(gatherLargerNKCommGrowRatio1); // 3x time of factor
    } else if (bwGrowthByShape) {
        if (!medianMFlag) {
            tilingM_.SetMaxTileCnt(8); // 8 is max tile cnt of M
        }
        commPerf_.ChangeCommTimeFactorByDivision(gatherLargerNKCommGrowRatio2); // 1.5x time of factor
    }
    commPerf_.ChangeCommTimeFactorByDivision(commGrowRatio); // 1.15x time of factor
	if (clusterInfo_.socType == SocVersion::SOC910_93) {
		commPerf_.ChangeCommTimeFactorByDivision(0.6);   // 0.6x time of factor
	}
}

void AllGatherPlusMMV2::SetCommTimeFactor()
{
	if (clusterInfo_.socType == SocVersion::SOC950) {
		SetCommTimeFactorForA5();
	}else{
        SetCommTimeFactorForOther();
    }
}

void AllGatherPlusMMV2::SelectTilingMethod()
{    
    if (tilingM_.SetShortTileLen(noCutFlag_)) { // 如果shape太小就不切
        return;
    }
    // 流水配平，找到理论切分长度
    if (tilingM_.cutRes.shortTileAtBack) {
        bool smallFront = rankDim_ <= MIN_COMM_RANKDIM; // 2p时，front占了一半的计算量
        if (!smallFront) {
            tilingM_.cutRes.longTileLen = commPerf_.InverseCommTime(frontMMTime_);
        } else {
            ShortAtEndCalcBoundBalancing();
        }
    } else {
        ShortAtFrontCommBoundBalancing();
        if (strongTpBound_) {
            tilingM_.cutRes.longTileLen = std::min(tilingM_.cutRes.longTileLen, tilingM_.cutRes.shortTileLen);
        }
    }

    // 生成切分
    bool smallDimAlignUp = (rankDim_ <= MIN_COMM_RANKDIM) && tilingM_.cutRes.shortTileAtBack &&
        (clusterInfo_.nValue < SMALL_SHAPE_BAR || clusterInfo_.kValue < SMALL_SHAPE_BAR); // 2p，计算Bound，且N轴或者K轴很小
    bool goodLinearityShape = (clusterInfo_.kValue * clusterInfo_.nValue >= LARGE_NK_BAR_BASE * ONE_MBYTE);
    tilingM_.FitTileLengthDiscrete(smallDimAlignUp, goodLinearityShape, hasLocalAtFront_);
    OPS_LOG_D("AllGatherMatmul", "Final cut: shortTileAtBack %d, longTileLen %lu"
        ", numLongTile %lu, shortTileLen %lu, numShortTile %lu",
        tilingM_.cutRes.shortTileAtBack, tilingM_.cutRes.longTileLen, tilingM_.cutRes.numLongTile,
        tilingM_.cutRes.shortTileLen, tilingM_.cutRes.numShortTile);
}

void AllGatherPlusMMV2::CheckHCCLLimit()
{
    // 计算单次通信量（假设整个M作为一块）
    uint64_t tileSize = clusterInfo_.mValue * clusterInfo_.kValue *
                        clusterInfo_.inMatrixADtypeSize * rankDim_;

    if (tileSize <= HCCL_MEM_LIMIT) {
        hcclLimited_ = false;
        hcclMaxTileM_ = UINT64_MAX;
        OPS_LOG_D("AllGatherMatmul", "HCCL limit not triggered. totalSize=%lu", tileSize);
        return;
    }

    hcclLimited_ = true;
    // 计算满足HCCL限制的最大tileM
    hcclMaxTileM_ = HCCL_MEM_LIMIT /
                    (clusterInfo_.kValue * clusterInfo_.inMatrixADtypeSize * rankDim_);

    OPS_LOG_I("AllGatherMatmul", "HCCL limit triggered. totalSize=%lu, maxTileM=%lu, limit=%lu",
              tileSize, hcclMaxTileM_, HCCL_MEM_LIMIT);
}

void AllGatherPlusMMV2::AdjustForHCCLLimit()
{
    if (!hcclLimited_) {
        return;
    }

    uint64_t mValue = clusterInfo_.mValue;
    uint64_t minTileLen = tilingM_.GetMinLen();

    OPS_LOG_I("AllGatherMatmul", "Before HCCL adjust: tileM=%lu, tileCnt=%lu, tailM=%lu, shortTileAtBack=%d",
              tilingM_.cutRes.longTileLen, tilingM_.cutRes.numLongTile,
              tilingM_.cutRes.shortTileLen, tilingM_.cutRes.shortTileAtBack);

    if (tilingM_.cutRes.shortTileAtBack) {
        // 计算Bound：短块后置，保持 tailM <= minTileLen
        AdjustForCalcBoundHCCL(mValue, minTileLen);
    } else {
        // 通信Bound：短块前置，保持切分均衡
        AdjustForCommBoundHCCL(mValue, minTileLen);
    }

    OPS_LOG_I("AllGatherMatmul", "After HCCL adjust: tileM=%lu, tileCnt=%lu, tailM=%lu",
              tilingM_.cutRes.longTileLen, tilingM_.cutRes.numLongTile,
              tilingM_.cutRes.shortTileLen);
}

void AllGatherPlusMMV2::AdjustForCalcBoundHCCL(uint64_t mValue, uint64_t minTileLen)
{
    // 目标：tileM <= hcclMaxTileM_，tailM <= minTileLen
    // 计算最少需要的轮次
    uint64_t minTileCnt = Ops::Base::CeilDiv(mValue, hcclMaxTileM_);
    uint64_t tileCnt = minTileCnt;
    uint64_t tileM = Ops::Base::FloorDiv(mValue - minTileLen, tileCnt);

    // 确保 tileM <= hcclMaxTileM_
    if (tileM > hcclMaxTileM_) {
        tileM = hcclMaxTileM_;
        tileCnt = Ops::Base::CeilDiv(mValue - minTileLen, tileM);
    }

    // 确保 tileM >= minTileLen
    if (tileM < minTileLen) {
        tileM = minTileLen;
        tileCnt = Ops::Base::CeilDiv(mValue, tileM);
    }

    // 计算tailM
    uint64_t tailM = mValue - tileCnt * tileM;
    // 验证tailM约束
    if (tailM > minTileLen) {
        // tailM太大，增加轮次
        tileCnt++;
        tileM = Ops::Base::FloorDiv(mValue - minTileLen, tileCnt);
        tileM = std::min(tileM, hcclMaxTileM_);
        tileM = std::max(tileM, minTileLen);
        tailM = mValue - tileCnt * tileM;
    }

    // 更新切分结果
    if (tailM == 0) {
        tilingM_.cutRes.numShortTile = 0;
        tilingM_.cutRes.shortTileLen = 0;
    } else {
        tilingM_.cutRes.numShortTile = 1;
        tilingM_.cutRes.shortTileLen = tailM;
    }

    tilingM_.cutRes.longTileLen = tileM;
    tilingM_.cutRes.numLongTile = tileCnt;
    tilingM_.cutRes.totalTileCnt = tileCnt + tilingM_.cutRes.numShortTile;
}

void AllGatherPlusMMV2::AdjustForCommBoundHCCL(uint64_t mValue, uint64_t minTileLen)
{
    // 目标：tileM <= hcclMaxTileM_，切分尽量均衡
    // 计算最少需要的轮次
    uint64_t minTileCnt = Ops::Base::CeilDiv(mValue, hcclMaxTileM_);
    uint64_t tileCnt = minTileCnt;
    uint64_t tileM = Ops::Base::CeilDiv(mValue, tileCnt);
    tileM = std::min(tileM, hcclMaxTileM_);

    // 计算tailM
    uint64_t actualTileCnt = Ops::Base::FloorDiv(mValue, tileM);
    uint64_t tailM = mValue - actualTileCnt * tileM;

    // 如果tailM过大（超过tileM的一半），增加轮次使其更均衡
    if (tailM > tileM / 2 && tailM < tileM) {
        tileCnt++;
        tileM = Ops::Base::CeilDiv(mValue, tileCnt);
        tileM = std::min(tileM, hcclMaxTileM_);
        actualTileCnt = Ops::Base::FloorDiv(mValue, tileM);
        tailM = mValue - actualTileCnt * tileM;
    }

    // 更新切分结果（通信Bound场景短块前置）
    if (tailM == 0) {
        tilingM_.cutRes.numShortTile = 0;
        tilingM_.cutRes.shortTileLen = 0;
        tilingM_.cutRes.numLongTile = actualTileCnt;
    } else {
        tilingM_.cutRes.numShortTile = 1;
        tilingM_.cutRes.shortTileLen = tailM;
        tilingM_.cutRes.numLongTile = actualTileCnt;
    }

    tilingM_.cutRes.longTileLen = tileM;
    tilingM_.cutRes.shortTileAtBack = false; // 通信Bound：短块前置
    tilingM_.cutRes.totalTileCnt = tilingM_.cutRes.numLongTile + tilingM_.cutRes.numShortTile;
}

void AllGatherPlusMMV2::GetTiling()
{
    // 先检查HCCL限制
    CheckHCCLLimit();

    // 如果有HCCL限制，更新maxTileLen约束
    if (hcclLimited_) {
        tilingM_.tileArgs.maxTileLen = std::min(tilingM_.tileArgs.maxTileLen, hcclMaxTileM_);
    }

    // 调用父类的标准流程
    matmulPerf_.FindCubeUtil(tilingM_.GetMinLen(), rankTileNum_, inferFlag_,
                             &tilingM_.tileArgs.maxTileLen);
    matmulPerf_.GetMatmulGradient();

    EstimateKernelTime();
    SelectTilingMethod();

    // HCCL限制修正（如果切分仍然超限）
    AdjustForHCCLLimit();

    // 归一化处理（长块短块一样长时合并）
    if (tilingM_.cutRes.shortTileLen == tilingM_.cutRes.longTileLen) {
        tilingM_.cutRes.shortTileLen = 0U;
        tilingM_.cutRes.numShortTile = 0U;
        tilingM_.cutRes.numLongTile++;
        tilingM_.cutRes.totalTileCnt = tilingM_.cutRes.numLongTile;
    }
}