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
 * \file all_to_all_matmul_kernel.h
 * \brief AllToAll + Matmul 融合算子实现（先通信后计算）
 * 
 * 核心设计：
 * - Win区布局（方案1）：每个rank的所有tile连续排列
 * - FragmentTensor D1轴切分：虚拟拼接所有rank的K轴
 * - 通信计算解耦：流水线模式（等待tile通信完成）
 * - ScaleA提前通信：数据量小，一次性通信
 * - Blaze组件复用：BlockScheduler + BlockMmad
 */

#pragma once

#include "kernel_operator.h"
#include "shmem.h"

#include "../../block/aiv_comm/collective_comm_udma_api.h"
#include "../../basic/fragment_tensor/fragment_tensor_api.h"
#include "../../utils/constant.h"

#include "../../block/blaze_ext/block/block_mmad_qbmm_mx_frag.h"
#include "../../block/blaze_ext/policy/dispatch_policy_frag.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/gemm/utils/common_utils.h"

namespace Dista {
namespace Kernel {

using namespace AscendC;
using namespace Dista::AivComm;
using namespace Dista::Utils;
using namespace Blaze::Gemm;
using namespace Blaze::Gemm::Block;

constexpr uint32_t FLAG_TILE_READY  = 0x2;  // tile数据就绪

struct AllToAllMatmulTilingData {
    uint64_t axisM;
    uint64_t axisK;
    uint32_t axisN;
    uint32_t rankSize;
    
    CommTilingData commTilingData;
    
    // FragmentTensor元信息（Host侧预计算）
    Dista::Utils::FragmentTilingData<2> fragTensorTilingA;      // A矩阵
    Dista::Utils::FragmentTilingData<2> fragScaleTilingA;        // ScaleA
    
    uint32_t baseM;
    uint32_t baseN;
    uint32_t baseK;
    uint32_t stepK;
    uint32_t scaleKL1;
    uint32_t dbL0c;
    
    uint32_t mTailTile;
    uint32_t nTailTile;
    uint32_t mBaseTailSplitCnt;
    uint32_t nBaseTailSplitCnt;
    uint32_t mTailMain;
    uint32_t nTailMain;
};

using AType = fp8_e4m3fn_t;
using BType = fp8_e4m3fn_t;
using CType = bfloat16_t;
using ScaleType = fp8_e8m0_t;

using LayoutA = AscendC::Te::NDExtLayoutPtn;
using LayoutB = AscendC::Te::DNExtLayoutPtn;
using LayoutC = AscendC::Te::NDExtLayoutPtn;
using LayoutScaleA = AscendC::Te::ScaleANDLayoutPtn;
using LayoutScaleB = AscendC::Te::ScaleBNDLayoutPtn;

using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMxFrag<0, 0>;
using BlockMmadType = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC, float, LayoutC>;

using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
using BlockCoord = AscendC::Te::Coord<int64_t, int64_t, int64_t, int64_t>;

using BlockSchedulerType = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
    ProblemShape, 0, LayoutA, LayoutB, AType>;

using BlockMmadParams = typename BlockMmadType::Params;
using L1Params = typename BlockMmadType::L1Params;
using BlockSchedulerParams = typename BlockSchedulerType::Params;

template<typename AType, typename BType, typename CType>
class AllToAllMatmulKernel {
public:
    __aicore__ inline AllToAllMatmulKernel() {}
    
    __aicore__ inline void Init(
        const CommContext* ctx,
        GM_ADDR aGM, GM_ADDR aScaleGM,
        GM_ADDR bGM, GM_ADDR bScaleGM,
        GM_ADDR cGM,
        const AllToAllMatmulTilingData* tilingData);
    
    __aicore__ inline void Process();
    
private:
    __aicore__ inline void InitBaseParams(const AllToAllMatmulTilingData* tilingData);
    __aicore__ inline void CreateTensors();
    __aicore__ inline void CreateFragmentTensor();
    __aicore__ inline void CreateScaleFragmentTensor();
    
    __aicore__ inline uint64_t GetCurMSize(uint32_t tileIdx) const;
    __aicore__ inline uint64_t CalcMOffset(uint32_t tileIdx) const;
    __aicore__ inline uint32_t CalcDependTileIdx(uint64_t mPos) const;
    
    CollectiveComm<CollectiveOp::AllToAll, CommMode::PUT, AType> allToAll_;
    CollectiveComm<CollectiveOp::AllToAll, CommMode::PUT, ScaleType> allToAllScale_;
    const CommContext* ctx_;
    
    decltype(AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(nullptr),
        AscendC::Te::MakeLayout(
            AscendC::Te::MakeShape(0, 0),
            AscendC::Te::MakeStride(0, 0)))) tensorB_;
    
    decltype(AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(nullptr),
        AscendC::Te::MakeLayout(
            AscendC::Te::MakeShape(0, 0),
            AscendC::Te::MakeStride(0, 0)))) tensorScaleB_;
    
    decltype(AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(nullptr),
        AscendC::Te::MakeLayout(
            AscendC::Te::MakeShape(0, 0),
            AscendC::Te::MakeStride(0, 0)))) tensorC_;
    
    decltype(AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(nullptr),
        AscendC::Te::MakeLayout(
            AscendC::Te::MakeShape(0, 0),
            AscendC::Te::MakeStride(0, 0)))) tensorBias_;
    
    // FragmentTensor 成员（直接声明类型）
    Dista::Basic::FragmentTensor<2, MAX_FRAGMENT_COUNT> fragTensorA_;
    Dista::Basic::FragmentTensor<2, MAX_FRAGMENT_COUNT> fragScaleA_;
    
    GM_ADDR aGM_, aScaleGM_;
    GM_ADDR bGM_, bScaleGM_, cGM_;
    
    uint32_t rankId_;
    uint32_t rankSize_;
    uint64_t axisM_;
    uint64_t axisK_;
    uint32_t axisN_;
    
    uint64_t chunkSize_;
    uint64_t axisKTotal_;
    uint64_t scaleK_;
    uint64_t scaleKTotal_;
    
    uint32_t baseM_;
    uint32_t baseN_;
    uint32_t baseK_;
    uint32_t dbL0c_;
    
    L1Params l1Params_;
    BlockSchedulerParams schParams_;
    
    const AllToAllMatmulTilingData* tilingData_;
};

template<typename AType, typename BType, typename CType>
__aicore__ inline void AllToAllMatmulKernel<AType, BType, CType>::InitBaseParams(
    const AllToAllMatmulTilingData* tilingData)
{
    rankSize_ = aclshmem_n_pes();
    rankId_ = aclshmem_my_pe();
    
    axisM_ = tilingData->axisM;
    axisK_ = tilingData->axisK;
    axisN_ = tilingData->axisN;
    
    baseM_ = tilingData->baseM;
    baseN_ = tilingData->baseN;
    baseK_ = tilingData->baseK;
    dbL0c_ = tilingData->dbL0c;
    
    schParams_ = {
        baseM_,
        baseN_,
        tilingData->mTailTile,
        tilingData->nTailTile,
        tilingData->mBaseTailSplitCnt,
        tilingData->nBaseTailSplitCnt,
        tilingData->mTailMain,
        tilingData->nTailMain
    };
    
    l1Params_ = {
        baseK_ * tilingData->stepK,
        tilingData->scaleKL1,
        2
    };
}

template<typename AType, typename BType, typename CType>
__aicore__ inline void AllToAllMatmulKernel<AType, BType, CType>::CreateTensors()
{
    auto layoutB = AscendC::Te::MakeLayout(
        AscendC::Te::MakeShape(axisKTotal_, axisN_),
        AscendC::Te::MakeStride(axisN_, 1));
    tensorB_ = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(bGM_), layoutB);
    
    auto layoutScaleB = AscendC::Te::MakeLayout(
        AscendC::Te::MakeShape(scaleKTotal_, axisN_),
        AscendC::Te::MakeStride(axisN_ * MXFP_MULTI_BASE_SIZE, MXFP_MULTI_BASE_SIZE));
    tensorScaleB_ = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(bScaleGM_), layoutScaleB);
    
    auto layoutC = AscendC::Te::MakeLayout(
        AscendC::Te::MakeShape(axisM_, axisN_),
        AscendC::Te::MakeStride(axisN_, 1));
    tensorC_ = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(cGM_), layoutC);
    
    auto layoutBias = AscendC::Te::MakeLayout(
        AscendC::Te::MakeShape(1, axisN_),
        AscendC::Te::MakeStride(axisN_, 1));
    tensorBias_ = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(nullptr), layoutBias);
}

template<typename AType, typename BType, typename CType>
__aicore__ inline void AllToAllMatmulKernel<AType, BType, CType>::CreateFragmentTensor()
{
    GM_ADDR winAddr = aclshmem_ptr(ctx_->shmemSpace, rankId_);
    
    GM_ADDR addrList[MAX_FRAGMENT_COUNT];
    for (uint32_t rank = 0; rank < rankSize_; ++rank) {
        addrList[rank] = winAddr + rank * tilingData_->fragTensorTilingA.fragmentSize *
                         tilingData_->fragTensorTilingA.fragmentStride[0] * sizeof(AType);
    }
    
    fragTensorA_ = Dista::Basic::MakeFragmentTensor<2>(
        tilingData_->fragTensorTilingA, addrList);
}

template<typename AType, typename BType, typename CType>
__aicore__ inline void AllToAllMatmulKernel<AType, BType, CType>::CreateScaleFragmentTensor()
{
    GM_ADDR winAddr = aclshmem_ptr(ctx_->shmemSpace, rankId_);
    
    GM_ADDR addrScaleList[MAX_FRAGMENT_COUNT];
    for (uint32_t rank = 0; rank < rankSize_; ++rank) {
        addrScaleList[rank] = winAddr + rank * tilingData_->fragScaleTilingA.fragmentSize *
                             tilingData_->fragScaleTilingA.fragmentStride[0] * sizeof(ScaleType);
    }
    
    fragScaleA_ = Dista::Basic::MakeFragmentTensor<2>(
        tilingData_->fragScaleTilingA, addrScaleList);
}

template<typename AType, typename BType, typename CType>
__aicore__ inline uint64_t AllToAllMatmulKernel<AType, BType, CType>::GetCurMSize(
    uint32_t tileIdx) const
{
    uint64_t headTileCnt = tilingData_->commTilingData.splitAxisTileCnt;
    
    if (tileIdx < headTileCnt) {
        return tilingData_->commTilingData.splitAxisTileSize;
    }
    return tilingData_->commTilingData.splitAxisTailSize;
}

template<typename AType, typename BType, typename CType>
__aicore__ inline uint64_t AllToAllMatmulKernel<AType, BType, CType>::CalcMOffset(
    uint32_t tileIdx) const
{
    uint64_t headTileCnt = tilingData_->commTilingData.splitAxisTileCnt;
    uint64_t headTileSize = tilingData_->commTilingData.splitAxisTileSize;
    uint64_t tailTileSize = tilingData_->commTilingData.splitAxisTailSize;
    
    if (tileIdx < headTileCnt) {
        return static_cast<uint64_t>(tileIdx) * headTileSize;
    }
    return static_cast<uint64_t>(headTileCnt) * headTileSize + 
           static_cast<uint64_t>(tileIdx - headTileCnt) * tailTileSize;
}

template<typename AType, typename BType, typename CType>
__aicore__ inline uint32_t AllToAllMatmulKernel<AType, BType, CType>::CalcDependTileIdx(
    uint64_t mPos) const
{
    uint64_t headTileSize = tilingData_->commTilingData.splitAxisTileSize;
    uint32_t totalTiles = tilingData_->commTilingData.splitAxisTileCnt + 
                          tilingData_->commTilingData.splitAxisTailCnt;
    
    uint32_t tileIdx = static_cast<uint32_t>(mPos / headTileSize);
    
    if (tileIdx >= totalTiles) {
        tileIdx = totalTiles - 1;
    }
    
    return tileIdx;
}

template<typename AType, typename BType, typename CType>
__aicore__ inline void AllToAllMatmulKernel<AType, BType, CType>::Init(
    const CommContext* ctx,
    GM_ADDR aGM, GM_ADDR aScaleGM,
    GM_ADDR bGM, GM_ADDR bScaleGM,
    GM_ADDR cGM,
    const AllToAllMatmulTilingData* tilingData)
{
    ctx_ = ctx;
    tilingData_ = tilingData;
    
    aGM_ = aGM;
    aScaleGM_ = aScaleGM;
    bGM_ = bGM;
    bScaleGM_ = bScaleGM;
    cGM_ = cGM;
    
    InitBaseParams(tilingData);
    
    chunkSize_ = (axisM_ / rankSize_) * axisK_;
    axisKTotal_ = axisK_ * rankSize_;
    scaleK_ = CeilDiv(axisK_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
    scaleKTotal_ = CeilDiv(axisKTotal_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
    
    uint32_t blockDim = GetBlockDim();
    uint32_t blockIdx = GetBlockIdx();
    
    uint32_t threadCnt = blockDim;
    uint32_t threadId = blockIdx;
    
    allToAll_.Init(ctx, aGM, tilingData->commTilingData, 
                   threadCnt, threadId);
    
    uint64_t scaleChunkSize = axisM_ / rankSize_;
    CommTilingData scaleTilingData = {
        scaleChunkSize,
        1,
        0,
        0,
        scaleK_
    };
    
    allToAllScale_.Init(ctx, aScaleGM, scaleTilingData, 
                        threadCnt, threadId);
    
    CreateTensors();
    CreateFragmentTensor();
    CreateScaleFragmentTensor();
}

template<typename AType, typename BType, typename CType>
__aicore__ inline void AllToAllMatmulKernel<AType, BType, CType>::Process()
{
    uint32_t totalTiles = tilingData_->commTilingData.splitAxisTileCnt + 
                          tilingData_->commTilingData.splitAxisTailCnt;
    
    if ASCEND_IS_AIV {
        allToAllScale_.Commit();
        allToAll_.Commit();
        allToAllScale_.Wait();
        allToAll_.Wait();
        CrossCoreSetFlag<FLAG_TILE_READY, PIPE_MTE3>(0);
        
        for (uint32_t tileIdx = 1; tileIdx < totalTiles; ++tileIdx) {
            allToAll_.Commit();
            allToAll_.Wait();
            CrossCoreSetFlag<FLAG_TILE_READY, PIPE_MTE3>(tileIdx);
        }
        
        allToAll_.Finalize();
        allToAllScale_.Finalize();
    }
    
    if ASCEND_IS_AIC {
        ProblemShape problemShape{axisM_, axisN_, axisKTotal_, 1};
        BlockSchedulerType sch(problemShape, schParams_);
        
        BlockShape l0TileShape{baseM_, baseN_, baseK_, 0};
        bool enableL0CPingPong = (dbL0c_ > 1);
        
        BlockMmadType mmadOp;
        mmadOp.Init(problemShape, l0TileShape, l1Params_, false, enableL0CPingPong);
        
        BlockCoord blockIdx;
        uint32_t lastTileIdx = -1;
        
        while (sch.GetTileIdx(blockIdx)) {
            BlockShape singleShape = sch.template GetBlockShape<
                QuantMode::MX_PERGROUP_MODE, QuantMode::MX_PERGROUP_MODE, false>(blockIdx);
            
            if (AscendC::Te::Get<IDX_M_TILEIDX>(singleShape) <= 0 || 
                AscendC::Te::Get<IDX_N_TILEIDX>(singleShape) <= 0) {
                break;
            }
            
            int64_t mPos = 0, nPos = 0;
            sch.GetTileCoord(blockIdx, mPos, nPos);
            
            uint32_t dependTileIdx = CalcDependTileIdx(mPos);
            
            if (dependTileIdx != lastTileIdx) {
                CrossCoreWaitFlag<FLAG_TILE_READY, PIPE_MTE2>(dependTileIdx);
                lastTileIdx = dependTileIdx;
            }
            
            auto mmadBlockA = fragTensorA_.Slice(
                AscendC::Te::MakeCoord(mPos, 0),
                AscendC::Te::MakeShape(
                    AscendC::Te::Get<IDX_M_TILEIDX>(singleShape),
                    axisKTotal_));
            
            auto mmadScaleA = fragScaleA_.Slice(
                AscendC::Te::MakeCoord(mPos, 0),
                AscendC::Te::MakeShape(
                    AscendC::Te::Get<IDX_M_TILEIDX>(singleShape),
                    scaleKTotal_));
            
            auto mmadBlockB = tensorB_.Slice(
                AscendC::Te::MakeCoord(0, nPos),
                AscendC::Te::MakeShape(
                    axisKTotal_,
                    AscendC::Te::Get<IDX_N_TILEIDX>(singleShape)));
            
            auto mmadScaleB = tensorScaleB_.Slice(
                AscendC::Te::MakeCoord(0, nPos),
                AscendC::Te::MakeShape(
                    scaleKTotal_,
                    AscendC::Te::Get<IDX_N_TILEIDX>(singleShape)));
            
            auto mmadBlockC = tensorC_.Slice(
                AscendC::Te::MakeCoord(mPos, nPos),
                AscendC::Te::MakeShape(
                    AscendC::Te::Get<IDX_M_TILEIDX>(singleShape),
                    AscendC::Te::Get<IDX_N_TILEIDX>(singleShape)));
            
            mmadOp(mmadBlockA, mmadBlockB, mmadScaleA, mmadScaleB, 
                   tensorBias_, mmadBlockC, singleShape);
        }
    }
}

__global__ __aicore__ void AllToAllMatmulKernelEntry(
    const CommContext* ctx,
    GM_ADDR aGM, GM_ADDR aScaleGM,
    GM_ADDR bGM, GM_ADDR bScaleGM,
    GM_ADDR cGM,
    AllToAllMatmulTilingData tilingData)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    
    AllToAllMatmulKernel<fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t> impl;
    impl.Init(ctx, aGM, aScaleGM, bGM, bScaleGM, cGM, &tilingData);
    impl.Process();
}

} // namespace Kernel
} // namespace Dista