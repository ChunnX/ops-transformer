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
 * \file kernel_gmm_swiglu_mxquant.h
 * \brief
 */

#ifndef KERNEL_GMM_SWIGLU_MXQUANT_H
#define KERNEL_GMM_SWIGLU_MXQUANT_H

#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"

#include "gmmsq_base.h"
#include "tuple_utils.h"
#include "block_epilogue_swiglu_mx_quant.h"
#include "workspace_info.h"
#include "hccl_shmem.h"
#include "mega_moe_tiling_def.h"
#include "moe_init_routing_v3_apt.h"
#include "moe_finalize_routing_v2_apt.h"
#include "argsort_simt.h"
using namespace Catlass;

template <class BlockMmad_, class BlockMmad2_, class BlockEpilogue_, class BlockScheduler_>
class KernelGmmSwiGluMixOnlineDynamic {
public:
    CATLASS_DEVICE KernelGmmSwiGluMixOnlineDynamic() {}

    CATLASS_DEVICE ~KernelGmmSwiGluMixOnlineDynamic() {}

    using BlockEpilogue = BlockEpilogue_;
    using BlockScheduler = BlockScheduler_;
    using DataTypeOut = typename BlockEpilogue::DataTypeOut;
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementMxScaleA = typename BlockMmad::TileCopy::ElementMxScaleA;
    using LayoutMxScaleA = typename BlockMmad::TileCopy::LayoutMxScaleA;
    using ElementMxScaleB = typename BlockMmad::TileCopy::ElementMxScaleB;
    using LayoutMxScaleB = typename BlockMmad::TileCopy::LayoutMxScaleB;
    using ElementMMadOut = typename BlockMmad::ElementC;
    using LayoutMMadOut = typename BlockMmad::LayoutC;
    using LayoutTagA = typename BlockMmad::TileCopy::LayoutTagA;
    using LayoutTagB = typename BlockMmad::TileCopy::LayoutTagB;
    using LayoutTagC = typename BlockMmad::TileCopy::LayoutTagC;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    using BlockMmad2 = BlockMmad2_;
    using L1TileShape2 = typename BlockMmad2::L1TileShape;
    using ElementA2 = typename BlockMmad2::ElementA;
    using LayoutA2 = typename BlockMmad2::LayoutA;
    using ElementB2 = typename BlockMmad2::ElementB;
    using LayoutB2 = typename BlockMmad2::LayoutB;
    using ElementMxScaleA2 = typename BlockMmad2::TileCopy::ElementMxScaleA;
    using LayoutMxScaleA2 = typename BlockMmad2::TileCopy::LayoutMxScaleA;
    using ElementMxScaleB2 = typename BlockMmad2::TileCopy::ElementMxScaleB;
    using LayoutMxScaleB2 = typename BlockMmad2::TileCopy::LayoutMxScaleB;
    using ElementMMadOut2 = typename BlockMmad2::ElementC;
    using LayoutMMadOut2 = typename BlockMmad2::LayoutC;
    using LayoutTagA2 = typename BlockMmad2::TileCopy::LayoutTagA;
    using LayoutTagB2 = typename BlockMmad2::TileCopy::LayoutTagB;
    using LayoutTagC2 = typename BlockMmad2::TileCopy::LayoutTagC;

    static constexpr uint32_t L1_TILE_M2 = tla::get<0>(L1TileShape2{});
    static constexpr uint32_t L1_TILE_N2 = tla::get<1>(L1TileShape2{});
    static constexpr uint32_t L1_TILE_K2 = tla::get<2>(L1TileShape2{});

    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    // a, b, x1scale, x2scale, bias, y, yscale, groupFlag, b2, b2Scale, y2, m
    using BlockOffset = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t,
                                       int64_t, int64_t, int64_t, int64_t, int64_t,
                                       int64_t, int64_t>;

    AscendC::GlobalTensor<ElementA> aGlobal;
    AscendC::GlobalTensor<ElementB> bGlobal;
    AscendC::GlobalTensor<ElementMxScaleA> aScaleGlobal;
    AscendC::GlobalTensor<ElementMxScaleB> bScaleGlobal;
    AscendC::GlobalTensor<int32_t> groupListGm;
    AscendC::GlobalTensor<int32_t> groupFlagListGm;
    AscendC::GlobalTensor<int32_t> groupFlagListGm2;
    AscendC::LocalTensor<ElementMMadOut> l0cOutUbFirst{AscendC::TPosition::VECIN, 0, MAX_SINGLE_MN};
    AscendC::LocalTensor<ElementMMadOut> l0cOutUbSecond{AscendC::TPosition::VECIN, MAX_SINGLE_MN * sizeof(ElementMMadOut), MAX_SINGLE_MN};

    uint16_t GMM2_PINGPONG_IDX = 0;
    static constexpr uint32_t GMM2_PINGPONG_COUNT = 2;
    static constexpr uint32_t GMM2_BLOCK_SIZE = L1_TILE_M2 * L1_TILE_N2;
    AscendC::GlobalTensor<ElementA2> a2Global; // gmm1的输出
    AscendC::GlobalTensor<ElementB2> b2Global;
    AscendC::GlobalTensor<ElementMxScaleA2> a2ScaleGlobal;
    AscendC::GlobalTensor<ElementMxScaleB2> b2ScaleGlobal;
    AscendC::LocalTensor<ElementMMadOut2> l0cOutUbGMM2First{AscendC::TPosition::VECIN, 0, GMM2_BLOCK_SIZE};
    AscendC::LocalTensor<ElementMMadOut2> l0cOutUbGMM2Second{AscendC::TPosition::VECIN, GMM2_BLOCK_SIZE * sizeof(ElementMMadOut2), GMM2_BLOCK_SIZE};
    AscendC::GlobalTensor<ElementMMadOut2> outputGm; // gmm2的输出
    AscendC::GlobalTensor<int32_t> preSumBeforeRank;
    AscendC::GlobalTensor<int32_t> tokenPerExpert;
    AscendC::GlobalTensor<int32_t> cumsumMM;

    BlockEpilogue epilogueOp;
    Arch::Resource<ArchTag> resource;

    TupleShape problemShape{};
    BlockOffset baseOffset{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    Layout3D tokenPerExpertLayout;
    uint64_t preOffset = 0;

    bool isVecSetSyncCom = false;
    uint32_t VecSetSyncCom2 = 0;
    uint32_t startBlockIdx = 0;
    uint32_t blockNum = AscendC::GetBlockNum();
    uint32_t aicBlockNum = blockNum / AscendC::GetTaskRation();
    uint32_t blockIdx = AscendC::GetBlockIdx() / AscendC::GetTaskRation();
    uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
    uint32_t aivCoreIdx = AscendC::GetBlockIdx();
    int32_t prevSum = 0;
    uint32_t dispatchBlockNumPerEP = 2;

    struct Params {
        GM_ADDR aGmAddr;
        GM_ADDR expertIdxGmAddr;
        GM_ADDR bGmAddr;
        GM_ADDR bScaleGmAddr;
        GM_ADDR b2GmAddr;
        GM_ADDR b2ScaleGmAddr;
        GM_ADDR y2GmAddr;
        GM_ADDR probsGmAddr;
        WorkspaceInfo workspaceInfo;
        PeermemInfo peermemInfo;
        HcclShmem shmem;
        const MegaMoeTilingData *tilingData;
        Params() = default;
    };

    CATLASS_DEVICE void NotifyCube(uint16_t value = 0)
    {
        AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIV_SYNC_AIC_FLAG + value);
    }
    CATLASS_DEVICE void WaitForVector(uint16_t value = 0)
    {
        AscendC::CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG + value);
        //AscendC::CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG + FLAG_ID_MAX);
    }
    CATLASS_DEVICE void NotifyVector(uint16_t value = 0)
    {
        AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG + value);
        //AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG + FLAG_ID_MAX);
    }
    CATLASS_DEVICE void WaitForCube(uint16_t value = 0)
    {
        AscendC::CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIC_SYNC_AIV_FLAG + value);
    }

    CATLASS_DEVICE void EndSync(bool isVecSetSync)
    {
        if (!isVecSetSync) {
            return;
        }
        if constexpr(g_coreType == AscendC::AIC) {
            WaitForVector();
        }
    }

    CATLASS_DEVICE void EndGMM2Sync(uint32_t syncNum)
    {
        if constexpr(g_coreType == AscendC::AIV) {
            return;
        }
        if (syncNum <= 0) {
            return;
        } else if (syncNum == 1) {
            WaitForVector();
        } else {
            WaitForVector(GMM2_PINGPONG_IDX);
            WaitForVector(1 - GMM2_PINGPONG_IDX);
        }
    }

    CATLASS_DEVICE int32_t GetSplitValueFromGroupList(uint32_t groupIdx, uint8_t groupListType)
    {
        int32_t splitValue = 0;
        if (groupListType == 0) {
            int32_t offset = static_cast<int32_t>(groupListGm.GetValue(groupIdx));
            splitValue = offset - preOffset;
            preOffset = offset;
        } else {
            splitValue = static_cast<uint64_t>(groupListGm.GetValue(groupIdx));
        }
        return splitValue;
    }

    CATLASS_DEVICE void UpdateGlobalBuffer(const Params& params)
    {
        aGlobal.SetGlobalBuffer((__gm__ ElementA*)params.workspaceInfo.ptrA + Get<IDX_A_OFFSET>(baseOffset));
        aScaleGlobal.SetGlobalBuffer((__gm__ ElementMxScaleA*)params.workspaceInfo.ptrAScale + Get<IDX_A_SCALE_OFFSET>(baseOffset));
        if constexpr(g_coreType == AscendC::AIC) {
            bGlobal.SetGlobalBuffer((__gm__ ElementB*)params.bGmAddr + Get<IDX_B_OFFSET>(baseOffset));
            bScaleGlobal.SetGlobalBuffer((__gm__ ElementMxScaleB*)params.bScaleGmAddr + Get<IDX_B_SCALE_OFFSET>(baseOffset));
            a2Global.SetGlobalBuffer((__gm__ ElementA2*)params.workspaceInfo.ptrA2 + Get<IDX_C_OFFSET>(baseOffset));
            a2ScaleGlobal.SetGlobalBuffer((__gm__ ElementMxScaleA2*)params.workspaceInfo.ptrA2Scale + Get<IDX_C_SCALE_OFFSET>(baseOffset));
            b2Global.SetGlobalBuffer((__gm__ ElementB2*)params.b2GmAddr + Get<IDX_B2_OFFSET>(baseOffset));
            b2ScaleGlobal.SetGlobalBuffer((__gm__ ElementMxScaleB2*)params.b2ScaleGmAddr + Get<IDX_B2_SCALE_OFFSET>(baseOffset));
        }
        if constexpr(g_coreType == AscendC::AIV) {
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                Get<IDX_C_OFFSET>(baseOffset), Get<IDX_C_SCALE_OFFSET>(baseOffset),
                Get<IDX_FLAG_OFFSET>(baseOffset), 0L, 0L, 0L};
            epilogueOp.UpdateGlobalAddr(vecBaseOffset);
            outputGm.SetGlobalBuffer((__gm__ ElementMMadOut2*)params.y2GmAddr + Get<IDX_Y2_OFFSET>(baseOffset));
        }
        groupFlagListGm.SetGlobalBuffer((__gm__ int32_t*)params.workspaceInfo.ptrFlagList + Get<IDX_FLAG_OFFSET>(baseOffset) * INT_CACHELINE);
        groupFlagListGm2.SetGlobalBuffer((__gm__ int32_t*)params.workspaceInfo.ptrFlagList2 + Get<IDX_FLAG_OFFSET>(baseOffset) * INT_CACHELINE);
    }

    CATLASS_DEVICE void UpdateOffset(uint32_t groupIdx)
    {
        // baseOffset is 0 when groupIdx = 0
        if (groupIdx == 0) {
            return;
        }
        uint64_t m = Get<M_VALUE>(problemShape);
        uint64_t n = Get<N_VALUE>(problemShape);
        uint64_t k = Get<K_VALUE>(problemShape);
        Get<IDX_A_OFFSET>(baseOffset) += m * k;
        Get<IDX_B_OFFSET>(baseOffset) += n * k;
        // only splitM
        auto scaleK = CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        // scaleAAxisBaseOffset (m, ceil(k,64), 2)
        Get<IDX_A_SCALE_OFFSET>(baseOffset) += m * scaleK;
        // scaleBAxisBaseOffset (g, n, ceil(k,64), 2) or (g, ceil(k,64), n, 2)
        Get<IDX_B_SCALE_OFFSET>(baseOffset) += n * scaleK;
        Get<IDX_C_OFFSET>(baseOffset) += m * n / SWIGLU_N_HALF;
        Get<IDX_C_SCALE_OFFSET>(baseOffset) +=
            m * CeilDiv(n / SWIGLU_N_HALF, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_FLAG_OFFSET>(baseOffset) += 1;
        Get<IDX_B2_OFFSET>(baseOffset) += k * n / SWIGLU_N_HALF;
        Get<IDX_B2_SCALE_OFFSET>(baseOffset) += k * CeilDiv(n / SWIGLU_N_HALF, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_Y2_OFFSET>(baseOffset) += m * k;
        Get<IDX_M_OFFSET>(baseOffset) += m;
    }

    CATLASS_DEVICE bool UpdateGroupParams(const Params& params, uint32_t groupIdx)
    {
        UpdateOffset(groupIdx);
        int32_t splitValue = GetSplitValueFromGroupList(groupIdx, params.tilingData->groupListType);
        Get<M_VALUE>(problemShape) = splitValue;
        // split_m，when m=0, skip
        if (Get<M_VALUE>(problemShape) == 0) {
            return false;
        }
        return true;
    }

    template<typename T, typename ElementMxScaleA>
    CATLASS_DEVICE void CopyGMToGMPerToken(
        AscendC::GlobalTensor<T> dst,
        AscendC::GlobalTensor<ElementMxScaleA> dstScale,
        AscendC::GlobalTensor<T> src,
        int32_t rows,
        int32_t hiddenSize
    )
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
 
        constexpr int32_t BufferNum = 2;
        AscendC::LocalTensor<T> tmpBuffer1{AscendC::TPosition::VECIN, 0, 0};
        AscendC::LocalTensor<T> tmpBuffer2{AscendC::TPosition::VECIN, 96 * 1024, 0};
        uint32_t scaleLen = CeilDiv(hiddenSize, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        uint32_t copyInNum = hiddenSize + scaleLen;
        // [ReduceScatter] 2. Pre Interface Sync
        int pingpongId = 0;
        for (uint32_t processIndex = 0; processIndex < rows; ++processIndex) {
            AscendC::TEventID EVENT_ID = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
            AscendC::LocalTensor<T> buf = pingpongId == 0 ? tmpBuffer1 : tmpBuffer2;
            AscendC::LocalTensor<ElementMxScaleA> bufScale = buf[hiddenSize].template ReinterpretCast<ElementMxScaleA>();
            auto inputOffset = processIndex * copyInNum;
            auto outputOffset = processIndex * hiddenSize;
            auto outputScaleOffset = processIndex * scaleLen;
            // [ReduceScatter] 2. Pre Interface Sync
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            // [ReduceScatter] 3. Start shmem_mte_get_mem_nbi
            AscendC::DataCopy(buf, src[inputOffset], copyInNum);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
            AscendC::DataCopy(dst[outputOffset], buf, hiddenSize);
            AscendC::DataCopyPad(dstScale[outputScaleOffset], bufScale, {1, scaleLen, 0, 0, 0});
 
            // [ReduceScatter] 4. Post Interface Sync
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            pingpongId = (pingpongId + 1) % BufferNum;
        }
        // [ReduceScatter] 4. Post Interface Sync
 
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    template<typename T, typename ElementMxScaleA>
    CATLASS_DEVICE void CopyGMToGMPerTokenV2(
        AscendC::GlobalTensor<T> dst,
        AscendC::GlobalTensor<ElementMxScaleA> dstScale,
        AscendC::GlobalTensor<T> src,
        int32_t rows,
        int32_t hiddenSize,
        int32_t ubMoveNum,
        int32_t& pingpongId
    ) {
 
        constexpr int32_t BufferNum = 2;
        AscendC::LocalTensor<T> tmpBuffer1{AscendC::TPosition::VECIN, 0, 0};
        AscendC::LocalTensor<T> tmpBuffer2{AscendC::TPosition::VECIN, 96 * 1024, 0};
        uint32_t scaleLen = CeilDiv(hiddenSize, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        uint32_t copyInNum = hiddenSize + scaleLen;
        auto processCount = CeilDiv(rows, ubMoveNum);
        for (uint32_t processIndex = 0; processIndex < processCount; ++processIndex) {
            pingpongId = (pingpongId + 1) % BufferNum;
            AscendC::TEventID EVENT_ID = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
            AscendC::LocalTensor<T> buf = pingpongId == 0 ? tmpBuffer1 : tmpBuffer2;
            AscendC::LocalTensor<ElementMxScaleA> bufScale = buf[hiddenSize].template ReinterpretCast<ElementMxScaleA>();
            
            auto inputOffset = processIndex * ubMoveNum * copyInNum;
            
            int32_t rowNum = ubMoveNum;
            if (processIndex == processCount - 1) {
                rowNum = rows - processIndex * ubMoveNum;
            }
 
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            int64_t dataLen = rowNum * copyInNum;
            AscendC::DataCopy(buf, src[inputOffset], dataLen);
 
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
            auto outputOffset = processIndex * ubMoveNum * hiddenSize;
            #define U16(x) static_cast<uint16_t>(x)
            AscendC::DataCopyPad(dst[outputOffset], 
                buf, {U16(rowNum), U16(hiddenSize), U16(scaleLen / 32), 0, 0});
 
            AscendC::DataCopyPad(dstScale[processIndex * ubMoveNum * scaleLen], 
                bufScale, {U16(rowNum), U16(scaleLen), U16(hiddenSize / 32), 0, 0});
 
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        }
    }

    CATLASS_DEVICE void ResetTokenPerExpert(Params const &params)
    {
        auto &tilingData = params.tilingData;
        if (aivCoreIdx != blockNum - 1) {
            return;
        }
        int32_t num = tilingData->EP * CeilAlign(tilingData->EP * tilingData->expertPerRank, ALIGN_128);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::LocalTensor<int32_t> tmp = resource.ubBuf.template GetBufferByByte<int32_t>(0);
        AscendC::Duplicate(tmp, 0, num);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopy(tokenPerExpert, tmp, num);
    }

    CATLASS_DEVICE void InitParamsAndTensor(const Params& params)
    {
        baseOffset = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        auto &tilingData = params.tilingData;
        Get<N_VALUE>(problemShape) = tilingData->n;
        Get<K_VALUE>(problemShape) = tilingData->k;
        tokenPerExpertLayout = Layout3D(CeilAlign(tilingData->EP * tilingData->expertPerRank, ALIGN_128), tilingData->expertPerRank); // rankSize * E, E
        preSumBeforeRank.SetGlobalBuffer((__gm__ int32_t*)params.workspaceInfo.ptrSumBeforeRank);
        tokenPerExpert.SetGlobalBuffer((__gm__ int32_t*)params.peermemInfo.ptrTokenPerExpert);
        cumsumMM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.workspaceInfo.ptrcumsumMM));
        groupListGm = cumsumMM[(tilingData->EP - 1) * tilingData->expertPerRank]; // cumsumMM的最后一行
        groupFlagListGm.SetGlobalBuffer((__gm__ int32_t*)params.workspaceInfo.ptrFlagList);
        if constexpr(g_coreType == AscendC::AIV) {
            if (aivCoreIdx == 0) {
                int32_t flagNum = tilingData->expertPerRank * INT_CACHELINE * 2;
                // groupFlagList清零
                AscendC::LocalTensor<int32_t> initTensor{AscendC::TPosition::VECIN, 0, 0};
                AscendC::Duplicate(initTensor, 0, flagNum);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                AscendC::DataCopy(groupFlagListGm, initTensor, flagNum);
            }
        }
        epilogueOp.Init({params.workspaceInfo.ptrA2, params.workspaceInfo.ptrA2Scale, params.workspaceInfo.ptrFlagList,
                         nullptr, nullptr, nullptr, L1_TILE_M, L1_TILE_N});
        dispatchBlockNumPerEP = tilingData->blockNumPerEP;
    }

    CATLASS_DEVICE void GroupMatmulSwigluQuant(const Params& params)
    {
        uint32_t m = Get<M_VALUE>(problemShape);
        uint32_t n = Get<N_VALUE>(problemShape);
        uint32_t k = Get<K_VALUE>(problemShape);
        uint32_t outputN = n / SWIGLU_N_HALF;
        epilogueOp.UpdateNextProblem({m, outputN, k, 0});

        BlockMmad blockMmad(resource);

        auto scaleK = CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(m, k);
        auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(k, n);
        auto layoutMxScaleA =
            tla::MakeMxScaleLayout<ElementMxScaleA, LayoutTagA, false>(m, scaleK); // bool: is scaleB
        auto layoutMxScaleB = tla::MakeMxScaleLayout<ElementMxScaleB, LayoutTagB, true>(scaleK, n);
        auto layoutC = tla::MakeLayout<ElementMMadOut, LayoutTagC>(L1_TILE_M, L1_TILE_N);

        auto tensorA = tla::MakeTensor(aGlobal, layoutA, Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(bGlobal, layoutB, Arch::PositionGM{});
        auto tensorMxScaleA = tla::MakeTensor(aScaleGlobal, layoutMxScaleA, Arch::PositionGM{});
        auto tensorMxScaleB = tla::MakeTensor(bScaleGlobal, layoutMxScaleB, Arch::PositionGM{});
        auto tensorUbFirst = tla::MakeTensor(l0cOutUbFirst, layoutC, Arch::PositionUB{});
        auto tensorUbSecond = tla::MakeTensor(l0cOutUbSecond, layoutC, Arch::PositionUB{});

        BlockScheduler matmulBlockScheduler({m, outputN, k}, MakeCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();
        uint32_t startLoopIdx = (blockIdx < startBlockIdx ? blockIdx + blockNum : blockIdx) - startBlockIdx;
        for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += blockNum) {
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);
            uint32_t mLoc = blockCoord.m() * L1_TILE_M;
            uint32_t nLoc = blockCoord.n() * L1_TILE_N;
            uint32_t kLoc = blockCoord.k() * L1_TILE_K;

            if constexpr(g_coreType == AscendC::AIC) {
                if (loopIdx == startLoopIdx) {
                    uint32_t targetValue = params.tilingData->EP * dispatchBlockNumPerEP;
                    __gm__ int32_t* flagValueAddr = (__gm__ int32_t*)groupFlagListGm2.GetPhyAddr();
                    while(targetValue != AscendC::ReadGmByPassDCache(flagValueAddr)) {
                        int64_t st = AscendC::GetSystemCycle();
                        while (AscendC::GetSystemCycle() - st < 100) {}
                    }
                }
                if (isVecSetSyncCom) {
                    WaitForVector();
                }
                auto tensorBlockA = GetTile(
                    tensorA, tla::MakeCoord(mLoc, kLoc),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                auto tensorBlockMxScaleA = GetTile(
                    tensorMxScaleA,
                    tla::MakeCoord(mLoc, kLoc / MXFP_SCALE_GROUP_NUM),
                    tla::MakeShape(actualBlockShape.m(), CeilDiv(actualBlockShape.k(), MXFP_SCALE_GROUP_NUM)));
                auto tensorBlockUbFirst = GetTile(
                    tensorUbFirst, tla::MakeCoord(0, 0),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                auto tensorBlockUbSecond = GetTile(
                    tensorUbSecond, tla::MakeCoord(0, 0),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

                for (uint32_t weightBlock = 0; weightBlock < SWIGLU_N_HALF; ++weightBlock) {
                    auto tensorBlockB = GetTile(
                        tensorB,
                        tla::MakeCoord(kLoc, nLoc + weightBlock * outputN),
                        tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                    auto tensorBlockMxScaleB = GetTile(
                        tensorMxScaleB,
                        tla::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nLoc + weightBlock * outputN),
                        tla::MakeShape(CeilDiv(actualBlockShape.k(), MXFP_SCALE_GROUP_NUM), actualBlockShape.n()));
                    blockMmad(tensorBlockA, tensorBlockB, weightBlock == 0 ? tensorBlockUbFirst : tensorBlockUbSecond,
                              actualBlockShape, tensorBlockMxScaleA, tensorBlockMxScaleB);
                }
                NotifyVector();
            }
            isVecSetSyncCom = true;
            if constexpr(g_coreType == AscendC::AIV) {
                if (subBlockIdx == 0) {
                    AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{actualBlockShape.m(), actualBlockShape.n(),0,0};
                    AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
                        mLoc * outputN + nLoc, // yOffset
                        mLoc * CeilDiv(outputN, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
                            CeilDiv(nLoc, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, // yScaleOffset
                        0,0,0,0
                    };
                    WaitForCube();
                    epilogueOp(epilogueShape, epilogueOffset);
                    NotifyCube();
                }
            }
        }
        startBlockIdx = (startBlockIdx + coreLoops) % blockNum;
    }

    CATLASS_DEVICE void GroupMatmul2(const Params& params, uint32_t groupIdx)
    {
        uint32_t m = Get<M_VALUE>(problemShape);
        uint32_t n = Get<K_VALUE>(problemShape);
        uint32_t k = Get<N_VALUE>(problemShape) / SWIGLU_N_HALF;

        BlockMmad2 blockMmad(resource);

        auto scaleK = CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        auto layoutA = tla::MakeLayout<ElementA2, LayoutTagA2>(m, k);
        auto layoutB = tla::MakeLayout<ElementB2, LayoutTagB2>(k, n);
        auto layoutMxScaleA =
            tla::MakeMxScaleLayout<ElementMxScaleA2, LayoutTagA2, false>(m, scaleK); // bool: is scaleB
        auto layoutMxScaleB = tla::MakeMxScaleLayout<ElementMxScaleB2, LayoutTagB2, true>(scaleK, n);
        auto layoutC = tla::MakeLayout<ElementMMadOut2, LayoutTagC2>(L1_TILE_M2, L1_TILE_N2);

        auto tensorA = tla::MakeTensor(a2Global, layoutA, Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(b2Global, layoutB, Arch::PositionGM{});
        auto tensorMxScaleA = tla::MakeTensor(a2ScaleGlobal, layoutMxScaleA, Arch::PositionGM{});
        auto tensorMxScaleB = tla::MakeTensor(b2ScaleGlobal, layoutMxScaleB, Arch::PositionGM{});
        auto tensorUbFirst = tla::MakeTensor(l0cOutUbGMM2First, layoutC, Arch::PositionUB{});
        auto tensorUbSecond = tla::MakeTensor(l0cOutUbGMM2Second, layoutC, Arch::PositionUB{});

        BlockScheduler matmulBlockScheduler({m, n, k}, MakeCoord(L1_TILE_M2, L1_TILE_N2));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();
        uint32_t startLoopIdx = (blockIdx < startBlockIdx ? blockIdx + blockNum : blockIdx) - startBlockIdx;
        uint32_t currentExpertM = cumsumMM((params.tilingData->EP - 1) * params.tilingData->expertPerRank + groupIdx);
        uint32_t preSrcExpertSum = 0;

        for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += blockNum) {
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);
            uint32_t mLoc = blockCoord.m() * L1_TILE_M2;
            uint32_t nLoc = blockCoord.n() * L1_TILE_N2;
            uint32_t kLoc = blockCoord.k() * L1_TILE_K2;

            auto tensorUb = GMM2_PINGPONG_IDX == 0 ? tensorUbFirst : tensorUbSecond;
            auto l0cOutUbGMM2 = GMM2_PINGPONG_IDX == 0 ? l0cOutUbGMM2First : l0cOutUbGMM2Second;
            if constexpr(g_coreType == AscendC::AIC) {
                // wait  groupFlagListGm
                if (loopIdx == startLoopIdx) {
                    BlockScheduler gmmBlockScheduler({m, k, n}, MakeCoord(L1_TILE_M, L1_TILE_N));
                    uint32_t targetLoops = gmmBlockScheduler.GetCoreLoops();
                    __gm__ int32_t* flagValueAddr = (__gm__ int32_t*)groupFlagListGm.GetPhyAddr();
                    while(targetLoops != AscendC::ReadGmByPassDCache(flagValueAddr)) {
                        int64_t st = AscendC::GetSystemCycle();
                        while (AscendC::GetSystemCycle() - st < 100) {}
                    }
                }
                if (VecSetSyncCom2 >= GMM2_PINGPONG_COUNT) {
                    WaitForVector(GMM2_PINGPONG_IDX);
                }
                auto tensorBlockA = GetTile(
                    tensorA, tla::MakeCoord(mLoc, kLoc),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                auto tensorBlockMxScaleA = GetTile(
                    tensorMxScaleA,
                    tla::MakeCoord(mLoc, kLoc / MXFP_SCALE_GROUP_NUM),
                    tla::MakeShape(actualBlockShape.m(), CeilDiv(actualBlockShape.k(), MXFP_SCALE_GROUP_NUM)));
                auto tensorBlockUb = GetTile(
                    tensorUb, tla::MakeCoord(0, 0),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                auto tensorBlockB = GetTile(
                    tensorB,
                    tla::MakeCoord(kLoc, nLoc),
                    tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                auto tensorBlockMxScaleB = GetTile(
                    tensorMxScaleB,
                    tla::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nLoc),
                    tla::MakeShape(CeilDiv(actualBlockShape.k(), MXFP_SCALE_GROUP_NUM), actualBlockShape.n()));
                blockMmad(tensorBlockA, tensorBlockB, tensorBlockUb,
                          actualBlockShape, tensorBlockMxScaleA, tensorBlockMxScaleB);
                NotifyVector(GMM2_PINGPONG_IDX);
            }
            VecSetSyncCom2++;
            if constexpr(g_coreType == AscendC::AIV) {
                if (subBlockIdx == 0) {
                    WaitForCube(GMM2_PINGPONG_IDX);
                    /*
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
                    ub2GmParams.blockCount = actualBlockShape.m();
                    ub2GmParams.blockLen = actualBlockShape.n() * sizeof(bfloat16_t);
                    ub2GmParams.dstStride = (n - actualBlockShape.n()) * sizeof(bfloat16_t);
                    AscendC::DataCopyPad(outputGm[mLoc * n + nLoc], l0cOutUbGMM2, ub2GmParams);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
                    */
                    
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);

                    int32_t lenTile = actualBlockShape.m();
                    int32_t stTile = mLoc;
                    int32_t edTile = stTile + lenTile;
                    int32_t preSumRankInExpert = 0;
                    int32_t tileOffset = 0;
                    
                    for (int32_t dstEpIdx = 0; dstEpIdx < params.tilingData->EP; dstEpIdx++) {
                        uint32_t lenRankInExpert = tokenPerExpert(tokenPerExpertLayout(dstEpIdx, params.tilingData->rank, groupIdx));
                        uint32_t dstExpertOffset = preSumBeforeRank(dstEpIdx * params.tilingData->expertPerRank + groupIdx);
                        uint32_t stRankInExpert = preSumRankInExpert;
                        uint32_t edRankInExpert = stRankInExpert + lenRankInExpert;
                        preSumRankInExpert += lenRankInExpert;

                        if (stRankInExpert >= edTile) {
                            break;
                        } else if (edRankInExpert <= stTile) {
                            continue;
                        }

                        uint32_t stData = tla::max(stRankInExpert, stTile); 
                        uint32_t edData = tla::min(edRankInExpert, edTile);
                        uint32_t lenData = edData - stData;
                        if (lenData <= 0){
                            continue;
                        }

                        uint32_t dstOffsetInExpert = 0;
                        if (stTile > stRankInExpert) {
                            dstOffsetInExpert = stTile - stRankInExpert;
                        }
                
                        AscendC::GlobalTensor<ElementMMadOut2> gmRemoteD;
                        __gm__ void* dstPeermemPtr = params.shmem(params.peermemInfo.ptrD, dstEpIdx);
                        gmRemoteD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMMadOut2*>(dstPeermemPtr));
                        
                        // 远端偏移地址
                        uint64_t gmDstOffset = (dstOffsetInExpert + dstExpertOffset) * n + nLoc;
                        auto gmTileD = gmRemoteD[gmDstOffset];

                        AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
                        ub2GmParams.blockCount = lenData;
                        ub2GmParams.blockLen = actualBlockShape.n() * sizeof(ElementMMadOut2);
                        ub2GmParams.dstStride = (n - actualBlockShape.n()) * sizeof(ElementMMadOut2);

                        AscendC::DataCopyPad(gmTileD, l0cOutUbGMM2[tileOffset *  actualBlockShape.n()], ub2GmParams);
                        tileOffset += lenData;
                    }

                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
                    NotifyCube(GMM2_PINGPONG_IDX);
                }
            }
            GMM2_PINGPONG_IDX = 1 - GMM2_PINGPONG_IDX;
        }

        preSrcExpertSum += currentExpertM;
        startBlockIdx = (startBlockIdx + coreLoops) % blockNum;
    }

    CATLASS_DEVICE
    void GetCumsumForMMAIV(AscendC::GlobalTensor<int32_t> & tokenPerExpert, AscendC::GlobalTensor<int32_t> & result, uint32_t expertPerRank, uint32_t rankId, uint32_t EP)
    {
        int32_t expertPerRankAligned = (expertPerRank + 8 - 1) / 8 * 8;
        AscendC::LocalTensor<int32_t> tmpBuffer1 = {AscendC::TPosition::VECIN, 0, 0};
        #define U16(x) static_cast<uint16_t>(x)

        AscendC::DataCopyPad(
            tmpBuffer1,
            tokenPerExpert[rankId * expertPerRank],
            {U16(EP), U16(expertPerRank * sizeof(int32_t)), U16((AlignUp(EP * expertPerRank, 128) - expertPerRank) * sizeof(int32_t)), 0},
            {}
        );

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        for (uint32_t i = 1; i < EP; ++i) {
            AscendC::Add(tmpBuffer1[i * expertPerRankAligned], tmpBuffer1[i * expertPerRankAligned], tmpBuffer1[(i - 1) * expertPerRankAligned], expertPerRank);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        AscendC::DataCopyPad(
            result,
            tmpBuffer1,
            {U16(EP), U16((expertPerRank) * sizeof(int32_t)), 0, 0}
        );
    }

    // After InitRouting, Get tokenPerExpert from other ranks, and calculate prevsum
    CATLASS_DEVICE
    void CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2(Params const &params, GM_ADDR tokenPerExpertCurRank){
        if constexpr(g_coreType == AscendC::AIC) {
            return;
        }
        auto &tilingData = params.tilingData;
        uint32_t numPerCore = CeilAlign(tilingData->EP * tilingData->expertPerRank, ALIGN_128);
        AscendC::LocalTensor<int32_t> tmpBuffer{AscendC::TPosition::VECIN, 0, 0};
        AscendC::LocalTensor<int32_t> prevSumBuf = tmpBuffer[numPerCore];
        for(int32_t dstEpIdx = aivCoreIdx; dstEpIdx < tilingData->EP; dstEpIdx += blockNum) {
            if (dstEpIdx == tilingData->rank) {
                continue;
            }
            AscendC::GlobalTensor<int32_t> srcAddress;
            srcAddress.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tokenPerExpertCurRank));
            AscendC::GlobalTensor<int32_t> dstAddress;
            __gm__ void* dstPeermemPtr = params.shmem(tokenPerExpertCurRank, dstEpIdx);
            dstAddress.SetGlobalBuffer((__gm__ int32_t * )dstPeermemPtr);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::DataCopy(tmpBuffer, srcAddress, numPerCore);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Adds(tmpBuffer, tmpBuffer, 0x800000, numPerCore);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopy(dstAddress, tmpBuffer, numPerCore);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        for(int32_t dstEpIdx = aivCoreIdx; dstEpIdx < tilingData->EP; dstEpIdx += blockNum) {
            if (dstEpIdx != tilingData->rank) {
                int32_t intPer512 = ALIGN_512 / sizeof(int);
                for(int32_t checkIdx = 0; checkIdx < numPerCore; checkIdx += intPer512) {
                    __gm__ int32_t* sync_check = reinterpret_cast<__gm__ int32_t*>(params.peermemInfo.ptrTokenPerExpert) + tokenPerExpertLayout(dstEpIdx, 0, checkIdx);
                    gm_signal_wait_until_ne(sync_check, 0);
                }
                AscendC::DataCopy(tmpBuffer, tokenPerExpert[tokenPerExpertLayout(dstEpIdx, 0, 0)], numPerCore);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::Adds(tmpBuffer, tmpBuffer, -0x800000, numPerCore);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::DataCopy(tokenPerExpert[tokenPerExpertLayout(dstEpIdx, 0, 0)], tmpBuffer, numPerCore);
            } else {
                AscendC::DataCopy(tmpBuffer, tokenPerExpert[tokenPerExpertLayout(dstEpIdx, 0, 0)], numPerCore);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            int32_t tmpSum = 0;
            int32_t j = 0;
            for (int32_t i = 0; i < (tilingData->rank + 1) * tilingData->expertPerRank; i++) {
                if (i >= tilingData->rank * tilingData->expertPerRank) {
                    prevSumBuf(j) = tmpSum;
                    j++;
                }
                tmpSum += tmpBuffer(i);
            }
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::DataCopyPad(preSumBeforeRank[dstEpIdx * tilingData->expertPerRank], prevSumBuf,
            AscendC::DataCopyParams{1, static_cast<uint16_t>(tilingData->expertPerRank * sizeof(int32_t)), 0, 0});
        }
        AscendC::SyncAll<true>();
        if (aivCoreIdx == 0) {
            GetCumsumForMMAIV(tokenPerExpert, cumsumMM, tilingData->expertPerRank, tilingData->rank, tilingData->EP);
        }
        if (blockIdx < tilingData->EP * dispatchBlockNumPerEP) {
            prevSum = preSumBeforeRank(blockIdx / dispatchBlockNumPerEP * tilingData->expertPerRank);
        }
    }

    // 只有aiv的core1做dispatch
    CATLASS_DEVICE
    void Dispatch(Params const &params, uint32_t groupIdx) {
        int64_t widthA = Get<K_VALUE>(problemShape);
        int64_t widthAScale = CeilDiv(widthA, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        auto &tilingData = params.tilingData;
        int32_t pingpongId = 0;
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        for(int32_t idx = blockIdx; idx < tilingData->EP * dispatchBlockNumPerEP; idx += blockNum) {
            uint32_t dstEpIdx = idx / dispatchBlockNumPerEP;
            uint32_t idxInEp = idx % dispatchBlockNumPerEP;
            uint32_t rowStart = (dstEpIdx == 0 ? 0 : cumsumMM((dstEpIdx - 1) * tilingData->expertPerRank + groupIdx));
            if (rowStart < tilingData->maxOutputSize) {
                uint32_t rows = tokenPerExpert(tokenPerExpertLayout(dstEpIdx, tilingData->rank, groupIdx));
                if (rowStart + rows > tilingData->maxOutputSize) {
                    rows = tilingData->maxOutputSize - rowStart;
                }
                uint32_t rowSrc = prevSum;
                prevSum += rows;
                uint32_t rowsPerCore = CeilDiv(rows, dispatchBlockNumPerEP);
                uint32_t rowStartPerCore = rowStart + idxInEp * rowsPerCore;
                uint32_t rowSrcPerCore = rowSrc + idxInEp * rowsPerCore;
                if (rowStartPerCore + rowsPerCore > rowStart + rows) {
                    rowsPerCore = rowStart + rows - rowStartPerCore;
                }
                if (rowsPerCore > 0) {
                    AscendC::GlobalTensor <ElementA> gmRemoteA;
                    gmRemoteA.SetGlobalBuffer(
                            reinterpret_cast<__gm__ ElementA * > (params.shmem(params.peermemInfo.ptrA0, dstEpIdx)));
 
                    int64_t gmOffsetA = rowStartPerCore * widthA;
                    int64_t gmOffsetAScale = rowStartPerCore * widthAScale;
                    int64_t gmOffsetPeer = rowSrcPerCore * (widthA + widthAScale);
                    GM_ADDR aScalePtr = (GM_ADDR) aScaleGlobal[gmOffsetAScale].GetPhyAddr();
                    // Communication data
                    CopyGMToGMPerTokenV2(aGlobal[gmOffsetA],
                                         aScaleGlobal[gmOffsetAScale],
                                         gmRemoteA[gmOffsetPeer],
                                         rowsPerCore, Get<K_VALUE>(problemShape),
                                         tilingData->dispatchRows, pingpongId);
                }
            }
            AtomicAddFlag((__gm__ int32_t*)groupFlagListGm2.GetPhyAddr(), 1);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
    CATLASS_DEVICE void AivSyncAllForAic()
    {
        if constexpr(g_coreType == AscendC::AIC) {
            AscendC::CrossCoreWaitFlag<0x2>(0);
        }
        if constexpr(g_coreType == AscendC::AIV) {
            AscendC::SyncAll<true>();
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0);
        }
    }

    CATLASS_DEVICE void ArgSortExpandedRowIdx(const Params &params) {
        auto &tilingData = params.tilingData;
        int32_t coreLen, coreOffset;
        TilingByCore(tilingData->m * tilingData->topK, coreLen, coreOffset, 1);
        __gm__ int32_t *src = (__gm__ int32_t *)params.workspaceInfo.expandedRowIdx;
        __gm__ int32_t *dst = (__gm__ int32_t *)params.workspaceInfo.expandedRowIdxGather;
        Simt::VF_CALL<ArgsortSimt>(Simt::Dim3{MoeInitRoutingV3::SIMT_THREAD_NUM, 1, 1}, coreLen, coreOffset,
                                   src, dst);
    }

    CATLASS_DEVICE void Unpermute(const Params &params) {
        if constexpr(g_coreType == AscendC::AIC) {
            return;
        }
        auto &tilingData = params.tilingData;
        int32_t coreLen, coreOffset;
        TilingByCore(tilingData->m, coreLen, coreOffset, 1);
        AscendC::GlobalTensor<int32_t> expandedRowIdx;
        expandedRowIdx.SetGlobalBuffer((__gm__ int32_t*)params.workspaceInfo.expandedRowIdxGather);
        AscendC::GlobalTensor<bfloat16_t> probs;
        probs.SetGlobalBuffer((__gm__ bfloat16_t*)params.probsGmAddr);
        AscendC::GlobalTensor<bfloat16_t> expandedX;
        expandedX.SetGlobalBuffer((__gm__ bfloat16_t*)params.peermemInfo.ptrD);
        AscendC::GlobalTensor<bfloat16_t> output;
        output.SetGlobalBuffer((__gm__ bfloat16_t*)params.y2GmAddr);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        for(int32_t tokenIdx = coreOffset; tokenIdx < coreLen + coreOffset; tokenIdx++) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::LocalTensor<bfloat16_t> dataRes{AscendC::TPosition::VECIN, 0, 0};
            AscendC::LocalTensor<bfloat16_t> dataIn0 = dataRes[tilingData->k];
            AscendC::LocalTensor<bfloat16_t> dataIn1 = dataRes[tilingData->k * 2];
            for(int32_t expId = 0; expId < tilingData->topK; ++expId) {
                int32_t rowIdx = expandedRowIdx(tokenIdx * tilingData->topK + expId);
                bfloat16_t expScale = probs(tokenIdx * tilingData->topK + expId);
                auto event = (expId % 2 == 0) ? EVENT_ID0 : EVENT_ID1;
                auto dataIn = (expId % 2 == 0) ? dataIn0 : dataIn1;
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
                AscendC::DataCopy(dataIn, expandedX[rowIdx * tilingData->k], tilingData->k);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(event);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(event);
                if (expId == 0) {
                    AscendC::Muls(dataRes, dataIn, expScale, tilingData->k);
                } else {
                    AscendC::Muls(dataIn, dataIn, expScale, tilingData->k);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(dataRes, dataRes, dataIn, tilingData->k);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopy(output[tokenIdx * tilingData->k], dataRes, tilingData->k);
        }
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    }

    CATLASS_DEVICE void operator()(const Params& params)
    {
        InitParamsAndTensor(params);
        
        GM_ADDR tokenPerExpertCurRank = params.peermemInfo.ptrTokenPerExpert + tokenPerExpertLayout(params.tilingData->rank, 0, 0) * sizeof(int32_t);

        // 在moe_init_routing_v3中设置
        #if (__NPU_ARCH__ == 3510)
            int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
        #endif
        
        // InitRouting
        moe_init_routing_v3<bfloat16_t, fp8_e4m3fn_t>(params.aGmAddr, params.expertIdxGmAddr, nullptr, nullptr,
                params.peermemInfo.ptrA0, params.workspaceInfo.expandedRowIdx,
                tokenPerExpertCurRank, nullptr,
                params.workspaceInfo.ptrA, &params.tilingData->moeInitRoutingTilingData, params.tilingData->moeInitRoutingTilingData.tilingKey);

        AscendC::SyncAll<true>();
        // Get TokenPerExpert prevsum cumsum
        CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2(params, tokenPerExpertCurRank);
        AivSyncAllForAic();
        uint32_t expertPerRank = params.tilingData->expertPerRank;
        for (uint32_t groupIdx = 0; groupIdx < expertPerRank; groupIdx++) {
            if (!UpdateGroupParams(params, groupIdx)) {
                continue;
            }
            UpdateGlobalBuffer(params);
            if constexpr(g_coreType == AscendC::AIV) {
                if(subBlockIdx == 1) {
                    Dispatch(params, groupIdx);
                }
            }
            if (subBlockIdx == 0) {
                GroupMatmulSwigluQuant(params);
            }
        }
        EndSync(isVecSetSyncCom);
        // // 初始化
        preOffset = 0;
        baseOffset = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (uint32_t groupIdx = 0; groupIdx < expertPerRank; groupIdx++) {
            if (!UpdateGroupParams(params, groupIdx)) {
                continue;
            }
            UpdateGlobalBuffer(params);
            if (subBlockIdx == 0) {
                GroupMatmul2(params, groupIdx);
            }
        }

        EndGMM2Sync(VecSetSyncCom2);
        AscendC::SyncAll<true>();
        ArgSortExpandedRowIdx(params);
        if constexpr(g_coreType == AscendC::AIV) {
            ResetTokenPerExpert(params);
            params.shmem.CrossRankSync();
        }
        
        // unpermute
        Unpermute(params);

        #if (__NPU_ARCH__ == 3510)
            SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
        #endif
    }
};

#endif // KERNEL_GMM_SWIGLU_MXQUANT_H