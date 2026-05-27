/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DISPATCH_FFN_COMBINE_KERNEL_A2_HPP
#define DISPATCH_FFN_COMBINE_KERNEL_A2_HPP

#include "kernel_operator.h"

#include "utils/copy_l0c_to_gm_custom.hpp"

#include "catlass/catlass.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/detail/callback.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

#include "utils/block_mmad_w4a4.hpp"
#include "utils/block_mmad_preload_async_fixpipe_quant.hpp"
#include "utils/copy_gm_to_l1_custom.hpp"
#include "utils/block_epilogue_w4a8post_pertoken_v2.hpp"
#include "utils/block_epilogue_w4a8post_pertoken_swiglu.hpp"
#include "utils/block_epilogue_pertoken_v2.hpp"
#include "utils/block_epilogue_pertoken_swiglu.hpp"
#include "utils/block_epilogue_pertoken_row.hpp"
#include "utils/hccl_shmem.hpp"
#include "utils/const_args.hpp"
#include "utils/layout3d.hpp"

#include "moe_init_routing_quant_v2/moe_init_routing_quant_v2_tiling.h"
#include "moe_init_routing_v2/moe_init_routing_v2_tiling.h"
#include "moe_init_routing_quant_v2/moe_init_routing_quant_v2.cpp"
// Non-quant routing dispatcher (used by the BF16/FP16 weight path).
// Lives alongside the quant dispatcher in the same `moe_init_routing_quant_v2`
// directory so it can share all sort/expert-token/srcToDst helpers.
#include "moe_init_routing_v2/moe_init_routing_v2.cpp"
#include "moe_init_routing_quant_v2/moe_v2_fullload_dynamic_quant.h"
#include "unpermute/moe_token_unpermute.h"
#include "utils/get_tensor_addr.hpp"


using namespace AscendC;
// #define SYNC_MODE

namespace Catlass::Gemm::Kernel {

// SYNCFLAGC2V, SYNCFLAGV2C are defined in utils/const_args.hpp
// UB_MOVE_NUM is defined in dispatch_ffn_combine_kernel.hpp

// Atlas A2 (910B) kernel.
//
// Uses `HcclA2CombineOpParam` window context and push-based
// SendTokensV3 / RecvTokensV3 protocol for expert dispatch.
template <class BlockMmad_, class BlockScheduler_, class ElementGroupList_, class BlockEpilogue1_,
          class BlockEpilogue2_>
class DispatchFFNCombineKernelA2 {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;
    using ElementScale = uint64_t;
    using LayoutScale = typename layout::VectorLayout;
    using ElementPerTokenScale = float;
    using LayoutPerTokenScale = typename layout::VectorLayout;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue1 = BlockEpilogue1_;
    using BlockEpilogue2 = BlockEpilogue2_;
    using ElementD1 = typename BlockEpilogue1::ElementD;
    using LayoutD1 = typename BlockEpilogue1::LayoutD;
    using ElementD2 = typename BlockEpilogue2::ElementD;
    using LayoutD2 = typename BlockEpilogue2::LayoutD;
    using ElementABefore = std::conditional_t<std::is_same_v<ElementA, AscendC::int4b_t>, int8_t, ElementA>;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        __gm__ ElementABefore *ptrA;
        LayoutA layoutA;
        LayoutA layoutA2;
        __gm__ ElementB *ptrB1;
        LayoutB layoutB1;
        __gm__ ElementB *ptrB2;
        LayoutB layoutB2;
        // The unified op def exposes optional bias1/bias2 inputs; when absent
        // ptrBias1/ptrBias2 are nullptr. Per-channel scale info is fully carried
        // by `weight_scales1` / `weight_scales2` (forwarded as `ptrScale1` / `ptrScale2`).
        __gm__ ElementScale *ptrScale1;
        LayoutScale layoutScale1;
        __gm__ ElementScale *ptrScale2;
        LayoutScale layoutScale2;
        __gm__ ElementD2 *ptrOutput;
        LayoutD1 layoutD1;
        LayoutD2 layoutD2;
        GM_ADDR ptrWorkspace;
        GM_ADDR ptrExpertTokenNums;
        int32_t EP;
        int32_t listLen;
        int32_t expertPerRank;
        uint32_t maxOutputSize;
        //--------------
        GM_ADDR expertIdx;
        GM_ADDR moeInitRoutingQuantV2Scale;
        GM_ADDR moeInitRoutingQuantV2Offset;
        GM_ADDR expandedX;
        GM_ADDR expandedRowIdx;
        GM_ADDR expertTokensCountOrCumsum;
        GM_ADDR expertTokensBeforeCapacity;
        GM_ADDR dynamicQuantScale;
        GM_ADDR probs;
        GM_ADDR ptrXActiveMask;
        GM_ADDR ptrScales;
        GM_ADDR ptrBias1;
        GM_ADDR ptrBias2;
        int64_t topK;
        uint64_t initRoutingQuantTilingKey;
        uint32_t epilogueCoreNum;
        // Epilogue scheduling granularity. Default 0 keeps the legacy W4A8
        // behavior (no special throttling). INT8 path historically uses
        // `expertPerRank - 1`; BF16 path historically uses `expertPerRank - 2`.
        uint32_t epilogueGranularity{0};
        float swigluLimit;
        GM_ADDR contextGM{nullptr};
        union {
            optiling::MoeInitRoutingQuantV2TilingData moeInitRoutingQuantV2TilingData;
            optiling::MoeInitRoutingV2TilingData moeInitRoutingV2TilingData;
        };
        //--------------

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord problemShape_, uint32_t EP_, uint32_t listLen_,
               uint32_t expertPerRank_, uint32_t maxOutputSize_, int64_t topK_,
               uint64_t initRoutingQuantTilingKey_, uint32_t epilogueCoreNum_,
               GM_ADDR contextGM_, GM_ADDR ptrA_, LayoutA layoutA_, LayoutA layoutA2_, GM_ADDR ptrB1_,
               LayoutB layoutB1_, GM_ADDR ptrB2_, LayoutB layoutB2_,
               GM_ADDR ptrScale1_, LayoutScale layoutScale1_, GM_ADDR ptrScale2_, LayoutScale layoutScale2_,
               GM_ADDR ptrOutput_, LayoutD2 layoutD1_, LayoutD2 layoutD2_, GM_ADDR expertIdx_,
               GM_ADDR moeInitRoutingQuantV2Scale_, GM_ADDR moeInitRoutingQuantV2Offset_,
                GM_ADDR expertTokensBeforeCapacity_, GM_ADDR probs_, GM_ADDR ptrWorkspace_, GM_ADDR gmExpertTokenNums_,
                GM_ADDR ptrXActiveMask_, GM_ADDR ptrScales_, GM_ADDR ptrBias1_, GM_ADDR ptrBias2_,
                optiling::MoeInitRoutingQuantV2TilingData moeInitRoutingQuantV2TilingData_,
                uint32_t epilogueGranularity_ = 0,
                float swigluLimit_ = std::numeric_limits<float>::infinity())
            : problemShape(problemShape_),
              EP(EP_),
              listLen(listLen_),
              expertPerRank(expertPerRank_),
              maxOutputSize(maxOutputSize_),
              topK(topK_),
              initRoutingQuantTilingKey(initRoutingQuantTilingKey_),
              epilogueCoreNum(epilogueCoreNum_),
              epilogueGranularity(epilogueGranularity_),
              swigluLimit(swigluLimit_),
              contextGM(contextGM_),
              ptrA(reinterpret_cast<__gm__ ElementABefore *>(ptrA_)),
              layoutA(layoutA_),
              layoutA2(layoutA2_),
              ptrB1(reinterpret_cast<__gm__ ElementB *>(ptrB1_)),
              layoutB1(layoutB1_),
              ptrB2(reinterpret_cast<__gm__ ElementB *>(ptrB2_)),
              layoutB2(layoutB2_),
              ptrScale1(reinterpret_cast<__gm__ ElementScale *>(ptrScale1_)),
              layoutScale1(layoutScale1_),
              ptrScale2(reinterpret_cast<__gm__ ElementScale *>(ptrScale2_)),
              layoutScale2(layoutScale2_),
              ptrOutput(reinterpret_cast<__gm__ ElementD2 *>(ptrOutput_)),
              layoutD1(layoutD1_),
              layoutD2(layoutD2_),
              expertIdx(expertIdx_),
              moeInitRoutingQuantV2Scale(moeInitRoutingQuantV2Scale_),
              moeInitRoutingQuantV2Offset(moeInitRoutingQuantV2Offset_),
              expertTokensBeforeCapacity(expertTokensBeforeCapacity_),
              probs(probs_),
              ptrXActiveMask(ptrXActiveMask_),
              ptrScales(ptrScales_),
              ptrBias1(ptrBias1_),
              ptrBias2(ptrBias2_),
              ptrWorkspace(ptrWorkspace_),
              ptrExpertTokenNums(gmExpertTokenNums_),
              moeInitRoutingQuantV2TilingData(moeInitRoutingQuantV2TilingData_)
        {
            moeInitRoutingQuantV2TilingData.vbsComputeParamsOp = moeInitRoutingQuantV2TilingData_.vbsComputeParamsOp;
            moeInitRoutingQuantV2TilingData.vmsMiddleComputeParamsOp =
                moeInitRoutingQuantV2TilingData_.vmsMiddleComputeParamsOp;
            moeInitRoutingQuantV2TilingData.sortOutComputeParamsOp =
                moeInitRoutingQuantV2TilingData_.sortOutComputeParamsOp;
            moeInitRoutingQuantV2TilingData.srcToDstComputeParamsOp =
                moeInitRoutingQuantV2TilingData_.srcToDstComputeParamsOp;
            moeInitRoutingQuantV2TilingData.srcToDstCapacityComputeParamsOp =
                moeInitRoutingQuantV2TilingData_.srcToDstCapacityComputeParamsOp;
            moeInitRoutingQuantV2TilingData.gatherOutComputeParamsOp =
                moeInitRoutingQuantV2TilingData_.gatherOutComputeParamsOp;
        }

        CATLASS_HOST_DEVICE
        Params(GemmCoord problemShape_, uint32_t EP_, uint32_t listLen_,
               uint32_t expertPerRank_, uint32_t maxOutputSize_, int64_t topK_,
                uint64_t initRoutingQuantTilingKey_, uint32_t epilogueCoreNum_,
                GM_ADDR contextGM_, GM_ADDR ptrA_, LayoutA layoutA_, LayoutA layoutA2_, GM_ADDR ptrB1_,
                LayoutB layoutB1_, GM_ADDR ptrB2_, LayoutB layoutB2_,
                GM_ADDR ptrScale1_, LayoutScale layoutScale1_, GM_ADDR ptrScale2_, LayoutScale layoutScale2_,
                GM_ADDR ptrOutput_, LayoutD2 layoutD1_, LayoutD2 layoutD2_, GM_ADDR expertIdx_,
                GM_ADDR moeInitRoutingQuantV2Scale_, GM_ADDR moeInitRoutingQuantV2Offset_,
                GM_ADDR expertTokensBeforeCapacity_, GM_ADDR probs_, GM_ADDR ptrWorkspace_, GM_ADDR gmExpertTokenNums_,
               GM_ADDR ptrXActiveMask_, GM_ADDR ptrScales_, GM_ADDR ptrBias1_, GM_ADDR ptrBias2_,
               optiling::MoeInitRoutingV2TilingData moeInitRoutingV2TilingData_,
                uint32_t epilogueGranularity_ = 0,
                float swigluLimit_ = std::numeric_limits<float>::infinity())
            : problemShape(problemShape_),
              EP(EP_),
              listLen(listLen_),
              expertPerRank(expertPerRank_),
              maxOutputSize(maxOutputSize_),
              topK(topK_),
              initRoutingQuantTilingKey(initRoutingQuantTilingKey_),
              epilogueCoreNum(epilogueCoreNum_),
              epilogueGranularity(epilogueGranularity_),
              swigluLimit(swigluLimit_),
              contextGM(contextGM_),
              ptrA(reinterpret_cast<__gm__ ElementABefore *>(ptrA_)),
              layoutA(layoutA_),
              layoutA2(layoutA2_),
              ptrB1(reinterpret_cast<__gm__ ElementB *>(ptrB1_)),
              layoutB1(layoutB1_),
              ptrB2(reinterpret_cast<__gm__ ElementB *>(ptrB2_)),
              layoutB2(layoutB2_),
              ptrScale1(reinterpret_cast<__gm__ ElementScale *>(ptrScale1_)),
              layoutScale1(layoutScale1_),
              ptrScale2(reinterpret_cast<__gm__ ElementScale *>(ptrScale2_)),
              layoutScale2(layoutScale2_),
              ptrOutput(reinterpret_cast<__gm__ ElementD2 *>(ptrOutput_)),
              layoutD1(layoutD1_),
              layoutD2(layoutD2_),
              expertIdx(expertIdx_),
              moeInitRoutingQuantV2Scale(moeInitRoutingQuantV2Scale_),
              moeInitRoutingQuantV2Offset(moeInitRoutingQuantV2Offset_),
              expertTokensBeforeCapacity(expertTokensBeforeCapacity_),
              probs(probs_),
              ptrXActiveMask(ptrXActiveMask_),
              ptrScales(ptrScales_),
              ptrBias1(ptrBias1_),
              ptrBias2(ptrBias2_),
              ptrWorkspace(ptrWorkspace_),
              ptrExpertTokenNums(gmExpertTokenNums_),
              moeInitRoutingV2TilingData(moeInitRoutingV2TilingData_)
        {
        }
    };

    // Methods
    CATLASS_DEVICE
    DispatchFFNCombineKernelA2(Params const &params)
    {
        if ASCEND_IS_AIC {
            coreIdx = AscendC::GetBlockIdx();
            coreNum = AscendC::GetBlockNum();
        }

        if ASCEND_IS_AIV {
            coreIdx = get_block_idx() + get_subblockid() * get_block_num();
            coreNum = get_block_num() * get_subblockdim();
        }

        initBuffer(params);
    }

    CATLASS_DEVICE
    ~DispatchFFNCombineKernelA2() {
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
    {
#ifdef SYNC_MODE
        AscendC::SyncAll<false>();
#endif
        GMM1(params);
        return;
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params)
    {
        DispatchAndCombine(params);
    }

private:
    CATLASS_DEVICE
    int32_t RuntimeRank(Params const &params) const
    {
        (void)params;
        return shmem.Rank();
    }

    CATLASS_DEVICE
    int32_t RuntimeRankSize(Params const &params) const
    {
        (void)params;
        return shmem.RankSize();
    }

    CATLASS_DEVICE void initBuffer(Params const &params)
    {
        auto tmpContext = reinterpret_cast<__gm__ CommContext *>(params.contextGM);
        shmem.initShmem(tmpContext);
        // A2-only init: fetch the RDMA QP descriptor that `SendTokensV3`'s
        // cross-server path needs, and compute which physical server the
        // local rank lives on. On A3 these stay at their default values and
        // are never read.
        qp_info_ = reinterpret_cast<__gm__ HcclAiRMAInfo *>(
            reinterpret_cast<__gm__ HcclA2CombineOpParam *>(shmem.WinContext_)->aiRMAInfo);
        serverId_ = static_cast<int32_t>(RuntimeRank(params) / SERVER_RANK_SIZE_A2);

        workspaceInfo = WorkspaceInfo(params);
        peermemInfo = PeermemInfo(params, shmem);

        cumsumMM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspaceInfo.ptrcumsumMM));

        // W4A8路径在A2上不支持
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementABefore *>(shmem() + peermemInfo.offsetA));
        gmPermutedToken.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD1 *>(workspaceInfo.ptrPermutedToken));

        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(workspaceInfo.ptrC));
        gmC2.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(workspaceInfo.ptrC2));

        gmPerTokenScale1.SetGlobalBuffer(
            reinterpret_cast<__gm__ ElementPerTokenScale *>(shmem() + peermemInfo.offsetPeerPerTokenScale));
        gmPerTokenScale2.SetGlobalBuffer(
            reinterpret_cast<__gm__ ElementPerTokenScale *>(workspaceInfo.ptrPerTokenScale2));

        tokenPerExpert.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(shmem() + peermemInfo.offsetPeerTokenPerExpert));

        tokenPerExpertLayout = Layout3D(AlignUp(RuntimeRankSize(params) * params.expertPerRank, ALIGN_128), params.expertPerRank);
        preSumBeforeRank.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspaceInfo.ptrSumBeforeRank));
        gmXActiveMask.SetGlobalBuffer(reinterpret_cast<__gm__ bool*>(params.ptrXActiveMask));
    }

    template <typename T>
    CATLASS_DEVICE void CopyGMToGM(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, int32_t elemNum,
                                   int32_t ubMoveNum)
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

        using TType = Gemm::GemmType<T, layout::RowMajor>;
        using CopyGmToUb = Epilogue::Tile::CopyGm2Ub<ArchTag, TType>;
        using CopyUbToGm = Epilogue::Tile::CopyUb2Gm<ArchTag, TType>;
        CopyGmToUb copyGmToUb;
        CopyUbToGm copyUbToGm;
        constexpr int32_t BufferNum = 2;
        int tmpBufferSize = 32 * 1024 / sizeof(T);  // 32 KB
        AscendC::LocalTensor<T> tmpBuffer1 = resource.ubBuf.template GetBufferByByte<T>(0);
        tmpBuffer1.SetSize(tmpBufferSize);
        int tmpBufferOffset = 96 * 1024;  // half of UB
        AscendC::LocalTensor<T> tmpBuffer2 = resource.ubBuf.template GetBufferByByte<T>(tmpBufferOffset);
        tmpBuffer2.SetSize(tmpBufferSize);

        // [ReduceScatter] 2. Pre Interface Sync
        int pingpongId = 0;
        auto processCount = CeilDiv(elemNum, ubMoveNum);
        for (uint32_t processIndex = 0; processIndex < processCount; ++processIndex) {
            uint32_t curProcessNum = (processIndex == processCount - 1) ? elemNum - ubMoveNum * (processCount - 1) :
                                                                          ubMoveNum;
            AscendC::TEventID EVENT_ID = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
            AscendC::LocalTensor<T> buf = pingpongId == 0 ? tmpBuffer1 : tmpBuffer2;
            auto processOffset = processIndex * ubMoveNum;

            auto inputOffset = processOffset;
            auto outputOffset = processOffset;
            // [ReduceScatter] 2. Pre Interface Sync
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            // [ReduceScatter] 3. Start shmem_mte_get_mem_nbi
            copyGmToUb(buf, src[inputOffset], layout::RowMajor{1, curProcessNum}, layout::RowMajor{1, curProcessNum});
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
            copyUbToGm(dst[outputOffset], buf, layout::RowMajor{1, curProcessNum}, layout::RowMajor{1, curProcessNum});

            // [ReduceScatter] 4. Post Interface Sync
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            pingpongId = (pingpongId + 1) % BufferNum;
        }
        // [ReduceScatter] 4. Post Interface Sync

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    template <typename T>
    CATLASS_DEVICE void CopyGMToGMPerToken(
        AscendC::GlobalTensor<T> dst,
        AscendC::GlobalTensor<float> dstScale,
        AscendC::GlobalTensor<T> src,
        int32_t rows,
        int32_t hiddenSize)
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

        constexpr int32_t BufferNum = 2;
        AscendC::LocalTensor<T> tmpBuffer1 = resource.ubBuf.template GetBufferByByte<T>(0);
        constexpr int tmpBufferOffset = 96 * 1024;
        AscendC::LocalTensor<T> tmpBuffer2 = resource.ubBuf.template GetBufferByByte<T>(tmpBufferOffset);
        uint32_t copyInNum = hiddenSize + ALIGN_512;
        int pingpongId = 0;
        for (uint32_t processIndex = 0; processIndex < rows; ++processIndex) {
            AscendC::TEventID EVENT_ID = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
            AscendC::LocalTensor<T> buf = pingpongId == 0 ? tmpBuffer1 : tmpBuffer2;
            AscendC::LocalTensor<float> bufScale = buf[hiddenSize].template ReinterpretCast<float>();
            auto inputOffset = processIndex * copyInNum;
            auto outputOffset = processIndex * hiddenSize;
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            AscendC::DataCopy(buf, src[inputOffset], copyInNum);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
            AscendC::DataCopy(dst[outputOffset], buf, hiddenSize);
            AscendC::DataCopyPad(dstScale[processIndex], bufScale, {1, 4, 0, 0, 0});
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            pingpongId = (pingpongId + 1) % BufferNum;
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    CATLASS_DEVICE void FetchAndPreprocessInt8ToInt4(uint64_t gmOffsetA, AscendC::GlobalTensor<float> dstScale,
                                                     AscendC::GlobalTensor<int8_t> src, int32_t rows,
                                                     int32_t hiddenSize)
    {
        uint32_t tmpBufferOffset = 0;

        // double buffer 定义： tmpBuffer 包含int8_t token + float scale, xTensor 只包含int8_t token
        AscendC::LocalTensor<int8_t> xTensor0 = resource.ubBuf.template GetBufferByByte<int8_t>(tmpBufferOffset);
        AscendC::LocalTensor<int8_t> tmpBuffer0 = resource.ubBuf.template GetBufferByByte<int8_t>(tmpBufferOffset);
        xTensor0.SetSize(hiddenSize);
        tmpBuffer0.SetSize(hiddenSize + ALIGN_512);
        tmpBufferOffset += (hiddenSize + ALIGN_512) * sizeof(int8_t);

        AscendC::LocalTensor<int8_t> xTensor1 = resource.ubBuf.template GetBufferByByte<int8_t>(tmpBufferOffset);
        AscendC::LocalTensor<int8_t> tmpBuffer1 = resource.ubBuf.template GetBufferByByte<int8_t>(tmpBufferOffset);
        xTensor1.SetSize(hiddenSize);
        tmpBuffer1.SetSize(hiddenSize + ALIGN_512);
        tmpBufferOffset += (hiddenSize + ALIGN_512) * sizeof(int8_t);
        // double buffer 定义结束

        AscendC::LocalTensor<int4b_t> xHighI4Tensor = resource.ubBuf.template GetBufferByByte<int4b_t>(tmpBufferOffset);
        xHighI4Tensor.SetSize(hiddenSize);
        tmpBufferOffset += hiddenSize / 2;

        AscendC::LocalTensor<int4b_t> xLowI4Tensor = resource.ubBuf.template GetBufferByByte<int4b_t>(tmpBufferOffset);
        xLowI4Tensor.SetSize(hiddenSize);
        tmpBufferOffset += hiddenSize / 2;

        AscendC::LocalTensor<half> xHighHalfTensor = resource.ubBuf.template GetBufferByByte<half>(tmpBufferOffset);
        xHighHalfTensor.SetSize(hiddenSize * sizeof(half));
        tmpBufferOffset += hiddenSize * sizeof(half);

        AscendC::LocalTensor<half> xLowHalfTensor = resource.ubBuf.template GetBufferByByte<half>(tmpBufferOffset);
        xLowHalfTensor.SetSize(hiddenSize * sizeof(half));
        tmpBufferOffset += hiddenSize * sizeof(half);

        AscendC::LocalTensor<half> xLowHalfTensor2 = resource.ubBuf.template GetBufferByByte<half>(tmpBufferOffset);
        xLowHalfTensor2.SetSize(hiddenSize * sizeof(half));
        tmpBufferOffset += hiddenSize * sizeof(half);

        AscendC::LocalTensor<int16_t> xLowI16Tensor = resource.ubBuf.template GetBufferByByte<int16_t>(tmpBufferOffset);
        xLowI16Tensor.SetSize(128 * sizeof(int16_t));

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID7);

        constexpr int32_t MASK = 128;
        Duplicate(xLowI16Tensor, static_cast<int16_t>(0x0F0F), MASK);
        PipeBarrier<PIPE_V>();
        const size_t LEN_VK = (hiddenSize / 2) / 128;
        const size_t LAST_LEN_VK = (hiddenSize % 256) / 2;
        const half ONE_SIXTEENTH = static_cast<half>(0.0625f);

        uint32_t totalM = rows;

        SetFlag<HardEvent::S_MTE2>(EVENT_ID6);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID6);
        uint32_t curCoreTaskNum, curCoreStartOffset;
        CalculateTaskInfoEachCore(curCoreTaskNum, curCoreStartOffset, totalM);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID6);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID7);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID6);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID7);

        constexpr size_t LEN_128 = 128;

        constexpr int32_t BufferNum = 2;
        uint32_t copyInNum = hiddenSize + ALIGN_512;

        int pingpongId = 0;
        for (uint32_t processIndex = 0; processIndex < rows; ++processIndex) {
            uint64_t relStartAddr = processIndex * hiddenSize;
            uint64_t absStartAddr = gmOffsetA + relStartAddr;
            AscendC::TEventID EVENT_ID = pingpongId == 0 ? EVENT_ID6 : EVENT_ID7;
            AscendC::LocalTensor<int8_t> buf = pingpongId == 0 ? tmpBuffer0 : tmpBuffer1;
            AscendC::LocalTensor<int8_t> xTensor = pingpongId == 0 ? xTensor0 : xTensor1;
            AscendC::LocalTensor<float> bufScale = buf[hiddenSize].template ReinterpretCast<float>();
            auto inputOffset = processIndex * copyInNum;
            auto outputOffset = processIndex * hiddenSize;

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID);
            AscendC::DataCopy(buf, src[inputOffset], copyInNum);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
            // 远端 scale 拷贝到GM
            AscendC::DataCopyPad(dstScale[processIndex], bufScale, {1, 4, 0, 0, 0});
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID);
            // 高四位处理开始
            Cast(xHighHalfTensor, xTensor, AscendC::RoundMode::CAST_NONE, hiddenSize);
            PipeBarrier<PIPE_V>();
            Muls(xHighHalfTensor, xHighHalfTensor, ONE_SIXTEENTH, hiddenSize);
            PipeBarrier<PIPE_V>();
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID7);
            Cast(xHighI4Tensor, xHighHalfTensor, AscendC::RoundMode::CAST_FLOOR, hiddenSize);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID6);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID6);
            DataCopy(gmA1I4_I8[absStartAddr], xHighI4Tensor.ReinterpretCast<int8_t>(), hiddenSize / 2);
            // 高四位处理结束

            // 低四位处理开始
            SetFlag<HardEvent::MTE3_V>(EVENT_ID7);
            And(xLowHalfTensor.ReinterpretCast<int16_t>(), xTensor.ReinterpretCast<int16_t>(), xLowI16Tensor, LEN_128,
                LEN_VK, {1, 1, 1, 8, 8, 0});
            if (LAST_LEN_VK > 0) {
                And(xLowHalfTensor[LEN_VK * LEN_128].ReinterpretCast<int16_t>(),
                    xTensor[LEN_VK * LEN_128 * TWO].ReinterpretCast<int16_t>(), xLowI16Tensor, LAST_LEN_VK, 1,
                    {1, 1, 1, 8, 8, 0});
            }
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(EVENT_ID);
            Cast(xLowHalfTensor2.ReinterpretCast<half>(), xLowHalfTensor.ReinterpretCast<int8_t>(),
                 AscendC::RoundMode::CAST_NONE, hiddenSize);
            PipeBarrier<PIPE_V>();
            const half MINUS_EIGHT = static_cast<half>(-8);
            Adds(xHighHalfTensor, xLowHalfTensor2, MINUS_EIGHT, hiddenSize);
            PipeBarrier<PIPE_V>();
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID6);
            Cast(xLowI4Tensor, xHighHalfTensor.ReinterpretCast<half>(), AscendC::RoundMode::CAST_NONE, hiddenSize);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID7);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID7);
            DataCopy(gmA1I4_I8[absStartAddr + hiddenSize / 2], xLowI4Tensor.ReinterpretCast<int8_t>(), hiddenSize / 2);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID6);
            // 低四位处理结束
            pingpongId = (pingpongId + 1) % BufferNum;
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID7);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID6);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID7);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID6);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID7);
    }

    CATLASS_DEVICE
    void GetCumsumForMMAIV(AscendC::GlobalTensor<int32_t> &tokenPerExpert, AscendC::GlobalTensor<int32_t> &result,
                           uint32_t expertPerRank, uint32_t rankId, uint32_t EP)
    {
        int32_t expertPerRankAligned = (expertPerRank + 8 - 1) / 8 * 8;
        AscendC::LocalTensor<int32_t> tmpBuffer1 = resource.ubBuf.template GetBufferByByte<int32_t>(0);
        AscendC::LocalTensor<int32_t> tmpResult =
            resource.ubBuf.template GetBufferByByte<int32_t>(EP * expertPerRank * sizeof(int32_t));
#define U16(x) static_cast<uint16_t>(x)

        AscendC::DataCopyPad(tmpBuffer1, tokenPerExpert[rankId * expertPerRank],
                             {U16(EP), U16(expertPerRank * sizeof(int32_t)),
                              U16((AlignUp(EP * expertPerRank, 128) - expertPerRank) * sizeof(int32_t)), 0},
                             {});

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        for (uint32_t i = 1; i < EP; ++i) {
            AscendC::Add(tmpBuffer1[i * expertPerRankAligned], tmpBuffer1[i * expertPerRankAligned],
                         tmpBuffer1[(i - 1) * expertPerRankAligned], expertPerRank);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        AscendC::DataCopyPad(result, tmpBuffer1, {U16(EP), U16((expertPerRank) * sizeof(int32_t)), 0, 0});
    }

    CATLASS_DEVICE
    void GMM1(Params const &params)
    {
        icache_preload(8);
        BlockScheduler blockScheduler;
        BlockMmad blockMmad(resource);
        float aivFinishGroups = 0.0f;
        __gm__ float *aivFinishPtr = workspaceInfo.ptrSoftFlagBase + params.EP * FLAGSTRIDE;

        int64_t gmGroupOffsetA = 0;
        int64_t gmGroupOffsetB = 0;
        int64_t gmGroupOffsetC = 0;
        uint32_t startCoreIdx = 0;
        uint32_t syncGroupIdx = 0;
        int64_t preCurrentmSum = 0;
        int32_t syncLoopIdx = -1;

        uint16_t syncgmmIdx = 0;
        AscendC::CrossCoreWaitFlag<0x2>(syncgmmIdx /
                                        CROSS_CORE_FLAG_MAX_SET_COUNT);  // Wait for AIV to finish cumsum for matmul

        syncgmmIdx++;
        if constexpr (std::is_same_v<ElementA, int8_t>) {
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        AscendC::GlobalTensor<ElementB> gmB1;
        AscendC::GlobalTensor<ElementScale> gmSLocal;

        for (uint32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            uint32_t currentM = cumsumMM((params.EP - 1) * params.expertPerRank + groupIdx);
            if (preCurrentmSum >= params.maxOutputSize) {
                currentM = 0;
            } else if (preCurrentmSum + currentM >= params.maxOutputSize) {
                currentM = params.maxOutputSize - preCurrentmSum;
            }

            // W4A8路径在A2上不支持

            int32_t arrayGroupIdx = params.listLen == 1 ? 0 : groupIdx;
            gmB1.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(GetTensorAddr<ElementB>(arrayGroupIdx, reinterpret_cast<GM_ADDR>(params.ptrB1))));
            // W4A8路径在A2上不支持
            gmSLocal.SetGlobalBuffer(reinterpret_cast<__gm__ ElementScale *>(GetTensorAddr<int64_t>(arrayGroupIdx, reinterpret_cast<GM_ADDR>(params.ptrScale1))));
            /* W4A8路径在A2上不支持
            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                gmS.SetGlobalBuffer(reinterpret_cast<__gm__ ElementScale *>(GetTensorAddr<int64_t>(arrayGroupIdx, reinterpret_cast<GM_ADDR>(params.ptrScale1))));
            }
            */

            // W4A8路径在A2上不支持
            AscendC::PipeBarrier<PIPE_ALL>();

            if (currentM <= L1TileShape::M) {
                gmB1.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }
            GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};
            LayoutA layoutA = params.layoutA.GetTileLayout(inGroupProblemShape.GetCoordMK());
            if constexpr (std::is_same_v<ElementA, int8_t>) {
                layoutA = LayoutA{inGroupProblemShape.m(), static_cast<uint32_t>(inGroupProblemShape.k() + ALIGN_512)};
            }
            LayoutB layoutB1 = params.layoutB1;
            LayoutScale layoutScale = params.layoutScale1;
            LayoutC layoutC = LayoutC(inGroupProblemShape.m(), inGroupProblemShape.n());
            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();
            // Determine the starting loopIdx of the current core under the current groupIdx
            uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;
            // Loop through the matmul of each groupIdx

            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                for (; syncGroupIdx <= groupIdx; syncGroupIdx++) {
                    AscendC::CrossCoreWaitFlag<0x2>(syncgmmIdx / CROSS_CORE_FLAG_MAX_SET_COUNT);
                    syncgmmIdx++;
                }
                // Compute block location
                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);
                // Compute initial location in logical coordinates
                MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
                MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
                MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
                int64_t gmOffsetA = layoutA.GetOffset(offsetA);
                int64_t gmOffsetB = layoutB1.GetOffset(offsetB);
                int64_t gmOffsetC = layoutC.GetOffset(offsetC);
                if (currentM > 0) {
                    // W4A8路径在A2上不支持
                    if constexpr (std::is_same_v<ElementA, int8_t>) {
                        int64_t gmOffsetS = blockCoord.n() * L1TileShape::N + (params.listLen == 1 ? groupIdx * params.problemShape.n() : 0);
                        blockMmad(gmA[gmGroupOffsetA + gmOffsetA], layoutA,
                                  gmB1[gmGroupOffsetB + gmOffsetB], layoutB1,
                                  gmC[gmGroupOffsetC + gmOffsetC], layoutC,
                                  gmSLocal[gmOffsetS], layoutScale,
                                  actualBlockShape);
                    }
                }
            }

            // W4A8路径在A2上不支持
            if ((groupIdx + 1) == params.epilogueGranularity && (groupIdx < params.expertPerRank - 1)) {
                syncLoopIdx++;
                if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                    blockMmad.SynchronizeBlock();
                }
                blockMmad.Finalize(syncLoopIdx, SYNCFLAGC2V);
            }

            preCurrentmSum += currentM;
            if constexpr (std::is_same_v<ElementA, int8_t>) {
                gmGroupOffsetA += inGroupProblemShape.m() * (inGroupProblemShape.k() + ALIGN_512);
            } else {
                gmGroupOffsetA += inGroupProblemShape.m() * inGroupProblemShape.k();
            }
            if (params.listLen == 1) {
                gmGroupOffsetB += inGroupProblemShape.k() * inGroupProblemShape.n();
            }
            gmGroupOffsetC += inGroupProblemShape.m() * inGroupProblemShape.n();
            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }

        for (; syncGroupIdx < params.expertPerRank; syncGroupIdx++) {
            AscendC::CrossCoreWaitFlag<0x2>(syncgmmIdx / CROSS_CORE_FLAG_MAX_SET_COUNT);
            syncgmmIdx++;
        }

        // W4A8路径在A2上不支持
        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.SynchronizeBlock();
        }
        blockMmad.Finalize(syncLoopIdx + 1, SYNCFLAGC2V);
    }

    CATLASS_DEVICE
    void GMM2(Params const &params)
    {
        icache_preload(8);
        BlockScheduler blockScheduler;
        BlockMmad blockMmad(resource);

        uint32_t n2 = params.problemShape.k();
        uint32_t k2 = params.problemShape.n() / 2;

        int64_t gmGroupOffsetA = 0;
        int64_t gmGroupOffsetB = 0;
        int64_t gmGroupOffsetC = 0;

        uint32_t startCoreIdx = 0;

        /* W4A8路径在A2上不支持
        if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        */

        int64_t preCurrentmSum = 0;
        int32_t syncLoopIdx = -1;

        // W4A8路径在A2上不支持
        uint32_t lastDequantExpertNum = params.expertPerRank;
        if (params.epilogueGranularity < params.expertPerRank) {
            lastDequantExpertNum = params.expertPerRank - params.epilogueGranularity;
        }
        if (params.epilogueGranularity < params.expertPerRank && params.epilogueGranularity > 0) {
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        for (uint32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            uint32_t currentM = cumsumMM((params.EP - 1) * params.expertPerRank + groupIdx);
            if (preCurrentmSum >= params.maxOutputSize) {
                currentM = 0;
            } else if (preCurrentmSum + currentM > params.maxOutputSize) {
                currentM = params.maxOutputSize - preCurrentmSum;
            }

            // W4A8路径在A2上不支持

            AscendC::GlobalTensor<ElementB> gmB2;
            AscendC::GlobalTensor<ElementScale> gmS2Local;
            int32_t arrayGroupIdx = params.listLen == 1 ? 0 : groupIdx;
            gmB2.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(GetTensorAddr<ElementB>(arrayGroupIdx, reinterpret_cast<GM_ADDR>(params.ptrB2))));
            // W4A8路径在A2上不支持
            gmS2Local.SetGlobalBuffer(reinterpret_cast<__gm__ ElementScale *>(GetTensorAddr<int64_t>(arrayGroupIdx, reinterpret_cast<GM_ADDR>(params.ptrScale2))));
            if (currentM <= L1TileShape::M) {
                gmB2.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }
            GemmCoord inGroupProblemShape{currentM, n2, k2};  // M N K

            LayoutA layoutA = params.layoutA2.GetTileLayout(inGroupProblemShape.GetCoordMK());
            LayoutB layoutB2 = params.layoutB2;
            LayoutScale layoutScale = params.layoutScale2;
            LayoutC layoutC = LayoutC(inGroupProblemShape.m(), inGroupProblemShape.n());

            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();

            // Determine the starting loopIdx of the current core under the current groupIdx
            uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;

            // W4A8路径在A2上不支持
            uint32_t lastDequantExpertNum = params.expertPerRank;
            if (params.epilogueGranularity < params.expertPerRank) {
                lastDequantExpertNum = params.expertPerRank - params.epilogueGranularity;
            }
            if (params.expertPerRank > lastDequantExpertNum && groupIdx + 1 == params.expertPerRank - lastDequantExpertNum) {
                AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAGV2C);
            }

            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                if (loopIdx + coreNum >= coreLoops) {
                    syncLoopIdx = groupIdx;
                }
                // Compute block location
                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);

                // Compute initial location in logical coordinates
                MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
                MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
                MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};

                int64_t gmOffsetA = layoutA.GetOffset(offsetA);
                int64_t gmOffsetB = layoutB2.GetOffset(offsetB);
                int64_t gmOffsetC = layoutC.GetOffset(offsetC);
                if (currentM > 0) {
                    // W4A8路径在A2上不支持
                    if constexpr (std::is_same_v<ElementA, int8_t>) {
                        int64_t gmOffsetS = blockCoord.n() * L1TileShape::N + (params.listLen == 1 ? groupIdx * n2 : 0);
                        blockMmad(gmPermutedToken[gmGroupOffsetA + gmOffsetA], layoutA,
                                  gmB2[gmGroupOffsetB + gmOffsetB], layoutB2,
                                  gmC2[gmGroupOffsetC + gmOffsetC], layoutC,
                                  gmS2Local[gmOffsetS], layoutScale,
                                  actualBlockShape, syncLoopIdx, 0);
                    } else {
                        blockMmad(gmPermutedToken[gmGroupOffsetA + gmOffsetA], layoutA,
                                  gmB2[gmGroupOffsetB + gmOffsetB], layoutB2,
                                  gmC2[gmGroupOffsetC + gmOffsetC], layoutC,
                                  gmS2Local, layoutScale,
                                  actualBlockShape, syncLoopIdx, 0);
                    }
                }
            }
            preCurrentmSum += currentM;
            gmGroupOffsetA += inGroupProblemShape.m() * inGroupProblemShape.k();
            if (params.listLen == 1) {
                gmGroupOffsetB += inGroupProblemShape.k() * inGroupProblemShape.n();
            }
            gmGroupOffsetC += inGroupProblemShape.m() * inGroupProblemShape.n();

            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }
        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.SynchronizeBlock();
        }
        /* W4A8路径在A2上不支持
        if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
            blockMmad.Finalize(params.expertPerRank - 1, 0);
        }
        */
    }

    CATLASS_DEVICE
    void InitArithProgress(Params const &params)
    {
        AscendC::LocalTensor<float> tmpBuffer1 = resource.ubBuf.template GetBufferByByte<float>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::Duplicate(tmpBuffer1, 0.0f, (params.EP + 1) * FLAGSTRIDE);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        AscendC::GlobalTensor<float> flagGlobalBase;
        flagGlobalBase.SetGlobalBuffer(workspaceInfo.ptrSoftFlagBase);
        AscendC::DataCopy(flagGlobalBase, tmpBuffer1, (params.EP + 1) * FLAGSTRIDE);
    }

    CATLASS_DEVICE
    void CalculateTaskInfoEachCore(uint32_t &curCoreTaskNum_, uint32_t &curCoreStartOffset_, uint32_t totalM)
    {
        // 均分任务数
        int64_t eachCoreTaskNum = (totalM + coreNum - 1) / coreNum;  // 每个核处理的数据量
        // 尾核任务数
        int64_t taskNumPertailCore = eachCoreTaskNum - 1;
        // 实际使用核数
        int64_t usedCoreNum = totalM >= coreNum ? coreNum : totalM;
        // 尾核起始索引
        uint32_t tailCoreIdx = totalM - (eachCoreTaskNum - 1) * usedCoreNum;
        uint32_t curCoreId = GetBlockIdx();
        // 每个核处理的任务数量 = 是否为尾核 ？均分任务数 ：(均分任务数 - 1)
        curCoreTaskNum_ = curCoreId < tailCoreIdx ? eachCoreTaskNum : eachCoreTaskNum - 1;
        // 每个核处理的起始偏移地址 = 是否为尾核 ？均分任务数 * blockId : (均分任务数 - 1) * blockId + 尾核起始索引
        curCoreStartOffset_ = curCoreId < tailCoreIdx ? eachCoreTaskNum * curCoreId :
                                                        ((eachCoreTaskNum - 1) * curCoreId + tailCoreIdx);
    }

    CATLASS_DEVICE
    void CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2(Params const &params,
                                                                        int64_t localTokenPerExpertOffset)
    {
        uint32_t numPerCore = AlignUp(params.EP * params.expertPerRank, 128);
        AscendC::LocalTensor<int32_t> tmpBuffer = resource.ubBuf.template GetBufferByByte<int32_t>(0);
        AscendC::LocalTensor<int32_t> prevSumBuf = tmpBuffer[numPerCore];

        for (int32_t dstEpIdx = coreIdx; dstEpIdx < params.EP; dstEpIdx += coreNum) {
            if (dstEpIdx == RuntimeRank(params)) {
                continue;
            }
            AscendC::GlobalTensor<int32_t> srcAddress;
            srcAddress.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(shmem.windowsOutAddr() + localTokenPerExpertOffset));
            AscendC::GlobalTensor<int32_t> dstAddress;
            __gm__ void *dstPeermemPtr = shmem(localTokenPerExpertOffset, coreIdx);
            dstAddress.SetGlobalBuffer((__gm__ int32_t *)dstPeermemPtr);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            using TType = Gemm::GemmType<int32_t, layout::RowMajor>;
            using CopyGmToUb = Epilogue::Tile::CopyGm2Ub<ArchTag, TType>;
            using CopyUbToGm = Epilogue::Tile::CopyUb2Gm<ArchTag, TType>;
            CopyGmToUb copyGmToUb;
            CopyUbToGm copyUbToGm;

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);

            copyGmToUb(tmpBuffer, srcAddress[0], layout::RowMajor{1, numPerCore}, layout::RowMajor{1, numPerCore});

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Adds(tmpBuffer, tmpBuffer, 0x800000, numPerCore);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            copyUbToGm(dstAddress[0], tmpBuffer, layout::RowMajor{1, numPerCore}, layout::RowMajor{1, numPerCore});
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        for (int32_t dstEpIdx = coreIdx; dstEpIdx < params.EP; dstEpIdx += coreNum) {
            if (dstEpIdx != RuntimeRank(params)) {
                int32_t intPer512 = CACHE_LINE / sizeof(int);
                for (int32_t checkIdx = 0; checkIdx < AlignUp(params.EP * params.expertPerRank, 128);
                     checkIdx += intPer512) {
                    __gm__ int32_t *sync_check =
                        reinterpret_cast<__gm__ int32_t *>(shmem() + peermemInfo.offsetPeerTokenPerExpert) +
                        tokenPerExpertLayout(dstEpIdx, 0, checkIdx);
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
            int32_t prevSum = 0;
            int32_t j = 0;
            for (int32_t i = 0; i < (RuntimeRank(params) + 1) * params.expertPerRank; i++) {
                if (i >= RuntimeRank(params) * params.expertPerRank) {
                    prevSumBuf(j) = prevSum;
                    j++;
                }
                prevSum += tmpBuffer(i);
            }
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::DataCopyPad(
                preSumBeforeRank[dstEpIdx * params.expertPerRank], prevSumBuf,
                AscendC::DataCopyParams{1, static_cast<uint16_t>(params.expertPerRank * sizeof(int32_t)), 0, 0});
        }

        AscendC::SyncAll<true>();
    }

    CATLASS_DEVICE
    void ResetTokenPerExpert(int32_t num)
    {
        if (coreIdx != coreNum - 1) {
            return;
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::LocalTensor<int32_t> tmp = resource.ubBuf.template GetBufferByByte<int32_t>(0);
        AscendC::Duplicate(tmp, 0, num);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopy(tokenPerExpert, tmp, num);
    }

    CATLASS_DEVICE
    void UpdateAicFlags(const Params &params)
    {
        float flagBase = 1.0f * params.expertPerRank;
        __gm__ float *aicFinishPtr = workspaceInfo.ptrSoftFlagBase + params.EP * FLAGSTRIDE;
        float flag = 0.0f;
        float lastflag = -1.0f;
        AscendC::LocalTensor<float> tmpBuffer1 = resource.ubBuf.template GetBufferByByte<float>(0);
        __gm__ float *flagPtr = workspaceInfo.ptrSoftFlagBase;
        AscendC::GlobalTensor<float> flagGM;
        flagGM.SetGlobalBuffer(flagPtr);
        int32_t flagBufferSize = max(4, params.EP) * FLAGSTRIDE;
        AscendC::LocalTensor<float> dstValueBuffer = resource.ubBuf.template GetBufferByByte<float>(flagBufferSize);
        AscendC::LocalTensor<float> sharedTmpBuffer =
            resource.ubBuf.template GetBufferByByte<float>((flagBufferSize + 64));
        uint64_t mask[1] = {0};
        uint32_t repeatNum = (flagBufferSize / (4 * FLAGSTRIDE));
        for (int32_t i = 0; i < 4; i++) {
            if (i < params.EP) {
                mask[0] |= 1ull * (1ull << (i * 16));
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        while (flag < flagBase) {
            flag = flagBase;
            AscendC::DataCopy(tmpBuffer1, flagGM, params.EP * FLAGSTRIDE);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

            AscendC::ReduceMin<float>(dstValueBuffer, tmpBuffer1, sharedTmpBuffer, mask, repeatNum, 8, false);

            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            flag = min(flag, dstValueBuffer.GetValue(0));

            if (flag > lastflag) {
                *aicFinishPtr = flag;
                gm_dcci(aicFinishPtr);
                lastflag = flag;
            }
        }
    }

    CATLASS_DEVICE
    bool IsSyncTask(int32_t task_id, int32_t n_tasks) {
        int32_t offset = n_tasks - task_id;
        if (offset <= 0 || (offset & (offset - 1)) != 0) {
            return false;
        }else
        {
            return true;
        }
    }

    // ===========================================================
    // A2 RDMA send doorbell helper. Ported verbatim (parameter list
    // and field-access pattern) from
    // dispatch_ffn_combine_bf16_kernel_a2.hpp::AIVRDMAPostSend so
    // that the A2 BatchWrite flow can post WQEs to the SQ directly
    // from AIV. Only ever called from the cross-server branch of
    // `SendTokensV3`.
    // ===========================================================
    CATLASS_DEVICE
    void AIVRDMAPostSend(GM_ADDR srcDmaAddr, GM_ADDR destDmaAddr, uint64_t destRankId, uint64_t messageLen,
                         __gm__ HcclAiRMAInfo *QpInfo,
                         AscendC::LocalTensor<uint64_t> &ubLocal,
                         AscendC::LocalTensor<uint32_t> &ubLocalHead)
    {
        auto qpNum = QpInfo->qpNum;
        auto qp_ctx_entry = reinterpret_cast<__gm__ HcclAiRMAWQ *>(
            QpInfo->sqPtr + destRankId * qpNum * static_cast<uint64_t>(QpInfo->sizeOfAiRMAWQ));
        auto mem_info_table = QpInfo->memPtr;
        auto sizeof_memdetail = QpInfo->sizeOfAiRMAMem;
        auto sqBaseAddr = qp_ctx_entry->bufAddr;
        auto wqeSize = qp_ctx_entry->wqeSize;
        auto curHardwareHead = qp_ctx_entry->headAddr;
        cacheWriteThrough(reinterpret_cast<__gm__ uint8_t *>(curHardwareHead), 8);
        uint64_t curHead = *reinterpret_cast<__gm__ uint32_t *>(curHardwareHead);
        auto curHardwareTailAddr = qp_ctx_entry->tailAddr;
        constexpr uint64_t shift = 15U;
        auto QP_DEPTH = qp_ctx_entry->depth;

        PipeBarrier<PIPE_ALL>();

        // Spin until there is room in the SQ. Host watchdog will kill the
        // kernel if this never makes progress (link loss, etc.).
        while (true) {
            cacheWriteThrough(reinterpret_cast<__gm__ uint8_t *>(curHardwareTailAddr), 8);
            if ((curHead - *reinterpret_cast<__gm__ uint32_t *>(curHardwareTailAddr)) < QP_DEPTH - 1) {
                break;
            }
            (void)AscendC::GetSystemCycle();
        }

        __gm__ uint8_t *wqeAddr = reinterpret_cast<__gm__ uint8_t *>(sqBaseAddr + wqeSize * (curHead % QP_DEPTH));

        uint64_t ownBit = (curHead >> shift) & 1U;
        uint32_t byte_4 = 3U;                      // opcode = RDMA_WRITE
        byte_4 |= ((~ownBit) << 7U) & (1U << 7U);  // owner bit
        byte_4 |= 1U << 8U;                        // IBV_SEND_SIGNALED

        *reinterpret_cast<__gm__ uint32_t *>(wqeAddr) = byte_4;
        *reinterpret_cast<__gm__ uint32_t *>(wqeAddr + 4) = messageLen;
        *reinterpret_cast<__gm__ uint32_t *>(wqeAddr + 8) = 0;
        *reinterpret_cast<__gm__ uint32_t *>(wqeAddr + 12) = 1U << 24U;  // num_sge = 1
        *reinterpret_cast<__gm__ uint32_t *>(wqeAddr + 16) = 0;          // start_sge_idx = 0
        __gm__ HcclAiRMAMemInfo *memDetail = reinterpret_cast<__gm__ HcclAiRMAMemInfo *>(
            mem_info_table + sizeof_memdetail * destRankId);
        *reinterpret_cast<__gm__ uint32_t *>(wqeAddr + 20) = reinterpret_cast<__gm__ MemDetails *>(
            memDetail->memDetailPtr +
            memDetail->sizeOfMemDetails * static_cast<uint32_t>(HcclAiRMAMemType::REMOTE_INPUT))->key;
        *reinterpret_cast<__gm__ uint64_t *>(wqeAddr + 24) = reinterpret_cast<uint64_t>(destDmaAddr);

        __gm__ uint8_t *sgeAddr = wqeAddr + sizeof(struct hns_roce_rc_sq_wqe);
        *reinterpret_cast<__gm__ uint32_t *>(sgeAddr) = messageLen;
        memDetail = reinterpret_cast<__gm__ HcclAiRMAMemInfo *>(mem_info_table + sizeof_memdetail * destRankId);
        *reinterpret_cast<__gm__ uint32_t *>(sgeAddr + sizeof(uint32_t)) = reinterpret_cast<__gm__ MemDetails *>(
            memDetail->memDetailPtr +
            memDetail->sizeOfMemDetails * static_cast<uint32_t>(HcclAiRMAMemType::LOCAL_OUTPUT))->key;
        *reinterpret_cast<__gm__ uint64_t *>(sgeAddr + 2 * sizeof(uint32_t)) = reinterpret_cast<uint64_t>(srcDmaAddr);

        cacheWriteThrough(wqeAddr, sizeof(struct hns_roce_rc_sq_wqe) + sizeof(struct hns_roce_lite_wqe_data_seg));
        PipeBarrier<PIPE_ALL>();
        curHead++;

        uint64_t doorBellInfo = 0;
        doorBellInfo |= qp_ctx_entry->wqn;                      // DB_TAG (qp_num)
        doorBellInfo |= 0UL << 24UL;                            // DB_CMD = SQ_DB
        doorBellInfo |= (curHead % 65536UL) << 32UL;            // DB_PI
        doorBellInfo |= static_cast<uint64_t>(qp_ctx_entry->sl) << 48UL;  // DB_SL

        __gm__ uint64_t *doorBellAddr = reinterpret_cast<__gm__ uint64_t *>(qp_ctx_entry->dbAddr);
        PipeBarrier<PIPE_ALL>();

        ubLocal.SetValue(0, doorBellInfo);
        AscendC::GlobalTensor<uint64_t> DBGlobalTensor;
        DBGlobalTensor.SetGlobalBuffer(doorBellAddr);
        AscendC::DataCopyExtParams copyParams{1, 1 * sizeof(uint64_t), 0, 0, 0};
        PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPad(DBGlobalTensor, ubLocal, copyParams);
        PipeBarrier<PIPE_ALL>();

        ubLocalHead.SetValue(0, static_cast<uint32_t>(curHead));
        AscendC::GlobalTensor<uint32_t> HeadGlobalTensor;
        HeadGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(curHardwareHead));
        AscendC::DataCopyExtParams copyParamsHead{1, 1 * sizeof(uint32_t), 0, 0, 0};
        PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPad(HeadGlobalTensor, ubLocalHead, copyParamsHead);
        PipeBarrier<PIPE_ALL>();
    }

    // ===========================================================
    // A2 token sender. Three paths:
    //   - self-rank      : windowsOut -> UB -> local peer mem, no flag
    //   - same-server    : IPC direct write to remote peer mem, then IPC
    //                       DataCopyPad of epoch flag
    //   - cross-server   : batched RDMA write of the token slab, then a
    //                       second RDMA write of the epoch flag (same QP,
    //                       so order is preserved on the receiver)
    // Epoch is encoded as `groupIdx + 1`, RecvTokensV3 spins on
    // `gm_signal_wait_until_ne(flagAddr, groupIdx)`.
    // ===========================================================
    template <typename T>
    CATLASS_DEVICE void SendTokensV3(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src,
                                     int32_t rows, uint32_t hiddenSize, int32_t dstEpIdx,
                                     int32_t groupIdx, int32_t rowStart, Params const &params)
    {
        if (rows == 0) {
            return;
        }
        constexpr bool kIsInt8Path = std::is_same_v<ElementA, int8_t>;
        uint32_t copyInNum = kIsInt8Path ? (hiddenSize + ALIGN_512) : hiddenSize;

        // UB layout:
        //   [0, copyInNum*sizeof(T))             : bufT     (token slab)
        //   [ubRdmaOffset, +UB_ALIGN)            : rdmaUbLocal     (doorbell)
        //   [ubRdmaOffset+UB_ALIGN, +UB_ALIGN)   : rdmaUbLocalHead (head ptr)
        //   [ubFlagOffset, +UB_ALIGN)            : flagUb          (epoch flag)
        uint64_t ubRdmaOffset = 2 * copyInNum * sizeof(uint32_t);
        AscendC::LocalTensor<uint64_t> rdmaUbLocal =
            resource.ubBuf.template GetBufferByByte<uint64_t>(ubRdmaOffset);
        AscendC::LocalTensor<uint32_t> rdmaUbLocalHead =
            resource.ubBuf.template GetBufferByByte<uint32_t>(ubRdmaOffset + UB_ALIGN);
        uint64_t ubFlagOffset = ubRdmaOffset + 2 * UB_ALIGN;
        AscendC::LocalTensor<int32_t> flagUb =
            resource.ubBuf.template GetBufferByByte<int32_t>(ubFlagOffset);
        AscendC::LocalTensor<T> bufT = resource.ubBuf.template GetBufferByByte<T>(0);
        AscendC::LocalTensor<float> bufScale = bufT[hiddenSize].template ReinterpretCast<float>();

        int32_t dstServerId = dstEpIdx / static_cast<int32_t>(SERVER_RANK_SIZE_A2);

        using TType = Gemm::GemmType<T, layout::RowMajor>;
        using CopyGmToUb = Epilogue::Tile::CopyGm2Ub<ArchTag, TType>;
        using CopyUbToGm = Epilogue::Tile::CopyUb2Gm<ArchTag, TType>;
        CopyGmToUb copyGmToUb;
        CopyUbToGm copyUbToGm;

        // -- Path A: self-rank --
        if (static_cast<int32_t>(RuntimeRank(params)) == dstEpIdx) {
            AscendC::GlobalTensor<ElementPerTokenScale> gmScale;
            gmScale.SetGlobalBuffer(reinterpret_cast<__gm__ ElementPerTokenScale *>(
                shmem() + peermemInfo.offsetPeerPerTokenScale));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            for (int32_t i = 0; i < rows; ++i) {
                auto offset = i * copyInNum;
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                copyGmToUb(bufT, src[offset], layout::RowMajor{1, copyInNum},
                           layout::RowMajor{1, copyInNum});
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                copyUbToGm(dst[offset], bufT, layout::RowMajor{1, hiddenSize},
                           layout::RowMajor{1, hiddenSize});
                if constexpr (kIsInt8Path) {
                    AscendC::DataCopyPad(gmScale[rowStart + i], bufScale, {1, 4, 0, 0, 0});
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            return;
        }

        // -- Path B: same server (IPC direct write) --
        if (dstServerId == serverId_) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            for (int32_t i = 0; i < rows; ++i) {
                auto offset = i * copyInNum;
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                copyGmToUb(bufT, src[offset], layout::RowMajor{1, copyInNum},
                           layout::RowMajor{1, copyInNum});
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                copyUbToGm(dst[offset], bufT, layout::RowMajor{1, copyInNum},
                           layout::RowMajor{1, copyInNum});
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);

            // Epoch flag (IPC). Receiver waits for value != groupIdx.
            AscendC::GlobalTensor<int32_t> remoteFlag;
            remoteFlag.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
                shmem(0, dstEpIdx) + peermemInfo.offsetFlag + RuntimeRank(params) * sizeof(uint64_t)));
            flagUb.SetValue(0, groupIdx + 1);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyPad(remoteFlag, flagUb,
                                 AscendC::DataCopyParams{1, sizeof(int32_t), 0, 0});
            AscendC::PipeBarrier<PIPE_ALL>();
            return;
        }

        // -- Path C: cross-server (RDMA data then RDMA flag) --
        AscendC::GlobalTensor<int32_t> localFlagGm;
        localFlagGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
            shmem.windowsOutAddr() + peermemInfo.offsetFlag + RuntimeRank(params) * sizeof(uint64_t)));
        flagUb.SetValue(0, groupIdx + 1);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPad(localFlagGm, flagUb,
                             AscendC::DataCopyParams{1, sizeof(int32_t), 0, 0});
        AscendC::PipeBarrier<PIPE_ALL>();

        uint64_t totalBytes = static_cast<uint64_t>(rows) * copyInNum * sizeof(T);
        AIVRDMAPostSend(const_cast<GM_ADDR>(reinterpret_cast<const GM_ADDR>(src.GetPhyAddr())),
                        const_cast<GM_ADDR>(reinterpret_cast<const GM_ADDR>(dst.GetPhyAddr())),
                        static_cast<uint64_t>(dstEpIdx), totalBytes,
                        qp_info_, rdmaUbLocal, rdmaUbLocalHead);
        AscendC::PipeBarrier<PIPE_ALL>();

        GM_ADDR remoteFlagAddr = reinterpret_cast<GM_ADDR>(
            reinterpret_cast<uintptr_t>(shmem(0, dstEpIdx)) +
            static_cast<uintptr_t>(peermemInfo.offsetFlag) +
            static_cast<uintptr_t>(RuntimeRank(params)) * sizeof(uint64_t));
        AIVRDMAPostSend(const_cast<GM_ADDR>(reinterpret_cast<const GM_ADDR>(localFlagGm.GetPhyAddr())), remoteFlagAddr,
                        static_cast<uint64_t>(dstEpIdx), sizeof(int32_t),
                        qp_info_, rdmaUbLocal, rdmaUbLocalHead);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // ===========================================================
    // A2 token receiver. The actual data has already been pushed to
    // local peer mem by the sender(s); this routine just spins on the
    // matching epoch flag so the AIC GMM1 won't read partial data.
    //   - srcEpIdx == rank : self-rank handled by SendTokensV3 path A
    //   - rows2 == 0       : skip, no flag will arrive
    // For INT8 path: after flag arrives, copy data and separate scale.
    // ===========================================================
    template <typename T>
    CATLASS_DEVICE void RecvTokensV3(int32_t rows2, int32_t rowStart2, uint32_t hiddenSize,
                                     int32_t srcEpIdx, int32_t groupIdx, Params const &params)
    {
        if (srcEpIdx == static_cast<int32_t>(RuntimeRank(params))) {
            return;
        }
        if (rows2 == 0) {
            return;
        }
        __gm__ int32_t *flagAddr = reinterpret_cast<__gm__ int32_t *>(
            shmem() + peermemInfo.offsetFlag + srcEpIdx * sizeof(uint64_t));
        gm_signal_wait_until_ne(flagAddr, groupIdx);

        constexpr bool kIsInt8Path = std::is_same_v<ElementA, int8_t>;
        if constexpr (kIsInt8Path) {
            uint32_t copyInNum = hiddenSize + ALIGN_512;
            AscendC::LocalTensor<T> bufT = resource.ubBuf.template GetBufferByByte<T>(0);
            AscendC::LocalTensor<float> bufScale = bufT[hiddenSize].template ReinterpretCast<float>();

            using TType = Gemm::GemmType<T, layout::RowMajor>;
            using CopyGmToUb = Epilogue::Tile::CopyGm2Ub<ArchTag, TType>;
            using CopyUbToGm = Epilogue::Tile::CopyUb2Gm<ArchTag, TType>;
            CopyGmToUb copyGmToUb;
            CopyUbToGm copyUbToGm;

            AscendC::GlobalTensor<T> gmDstA;
            gmDstA.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(shmem() + peermemInfo.offsetA));
            AscendC::GlobalTensor<ElementPerTokenScale> gmScale;
            gmScale.SetGlobalBuffer(reinterpret_cast<__gm__ ElementPerTokenScale *>(
                shmem() + peermemInfo.offsetPeerPerTokenScale));

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            for (int32_t i = 0; i < rows2; ++i) {
                auto offset = (rowStart2 + i) * copyInNum;
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                copyGmToUb(bufT, gmDstA[offset],
                           layout::RowMajor{1, copyInNum}, layout::RowMajor{1, copyInNum});
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                copyUbToGm(gmDstA[offset], bufT,
                           layout::RowMajor{1, hiddenSize}, layout::RowMajor{1, hiddenSize});
                AscendC::DataCopyPad(gmScale[rowStart2 + i], bufScale, {1, 4, 0, 0, 0});
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }

    CATLASS_DEVICE
    void CombineSetFlag()
    {
        if constexpr (std::is_same_v<ElementB, int8_t>) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        } else { // W4A8路径在A2上不支持
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
    }

    CATLASS_DEVICE
    void ApplyXActiveMask(Params const &params) {
        if (params.ptrXActiveMask == nullptr) {
            return;
        }
        int32_t m = params.problemShape.m();
        int32_t topK = params.topK;
        int32_t expertNum = params.expertPerRank * params.EP;
        AscendC::GlobalTensor<int32_t> expertIdxGm;
        expertIdxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.expertIdx));

        int32_t totalElements = m * topK;
        int32_t base = totalElements / coreNum;
        int32_t rem = totalElements % coreNum;

        int32_t startIdx = coreIdx * base + (coreIdx < rem ? coreIdx : rem);
        int32_t endIdx = (coreIdx + 1) * base + (coreIdx + 1 < rem ? coreIdx + 1 : rem);

        AscendC::LocalTensor<int32_t> tmpExpertIdx = resource.ubBuf.template GetBufferByByte<int32_t>(0);
        int32_t copySize = endIdx - startIdx;

        AscendC::DataCopyPad(tmpExpertIdx[0], expertIdxGm[startIdx], 
                    {1, static_cast<uint16_t>(copySize * sizeof(int32_t)), 0, 0}, {}
        );

        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

        for (int32_t i = 0; i < copySize; ++i) {
            int32_t tokenIdx = (startIdx + i) / topK;
            bool isActive = gmXActiveMask(tokenIdx);
            if (!isActive) {
                tmpExpertIdx.SetValue(i, expertNum);
            }
        }

        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::DataCopyPad(expertIdxGm[startIdx], tmpExpertIdx[0], {1, static_cast<uint16_t>(copySize * sizeof(int32_t)), 0, 0, 0});
        AscendC::SyncAll<true>();
    }

    CATLASS_DEVICE
    void DispatchAndCombine(Params const &params)
    {
        icache_preload(8);
        int64_t localTokenPerExpertOffset =
            peermemInfo.offsetPeerTokenPerExpert + tokenPerExpertLayout(RuntimeRank(params), 0, 0) * sizeof(int32_t);
        GM_ADDR localTokenPerExpert =
            shmem.windowsOutAddr() + localTokenPerExpertOffset;  // Place the entire communication matrix in peermem
        uint32_t expandedRowIdxOffset = AlignUp(params.problemShape.m(), 256) * params.topK * sizeof(int32_t);

        ApplyXActiveMask(params);

        // ----------------------------------------------------------------
        // [MoeInitRoutingV2 quant vs non-quant dispatch]
        //
        // The routing variant is decided purely from the weight element type
        // (`ElementB`), matching the host tiling's `info.isQuantRouting`:
        //   - ElementB ∈ {int8_t} -> quant routing (W4A8路径在A2上不支持)
        //         * tilingKey 20000 / 11000-series / 10000-series with quantMode=1
        //         * emits INT8 tokens + per-token quant scale
        //   - ElementB ∈ {fp16, bf16}      -> non-quant routing
        //         * tilingKey 20000 / 10000-series with quantMode=0
        //         * forwards X directly without quantization
        //
        // The two dispatchers (`moe_init_routing_quant_v2` / `moe_init_routing_v2`)
        // share all underlying sort / expert-token / srcToDst helpers since
        // both are defined under `namespace MoeInitRoutingQuantV2`.
        //
        // Note on `peermemInfo.offsetPeerPerTokenScale`: this output slot only
        // matters for the quant path (it carries the dynamic-quant scale).
        // The non-quant path doesn't write it, so we never need to set it up
        // differently here.
        // ----------------------------------------------------------------
        //---initRouting------
        constexpr bool kRoutingIsQuant = std::is_same_v<ElementB, int8_t>; // W4A8路径在A2上不支持
        if constexpr (kRoutingIsQuant) {
            moe_init_routing_quant_v2<ElementD2>(
                reinterpret_cast<GM_ADDR>(params.ptrA), params.expertIdx, params.moeInitRoutingQuantV2Scale,
                params.moeInitRoutingQuantV2Offset, shmem.windowsOutAddr() + peermemInfo.offsetA, workspaceInfo.expandedRowIdx,
                localTokenPerExpert, params.expertTokensBeforeCapacity, shmem.windowsOutAddr() + peermemInfo.offsetPeerPerTokenScale,
                params.ptrWorkspace + expandedRowIdxOffset, &params.moeInitRoutingQuantV2TilingData,
                params.initRoutingQuantTilingKey);
        } else {
            // BF16 / FP16 path: no scale/offset/dynamicQuantScale inputs and
            // outputs, tiling data is the Inner (non-quant) variant.
            moe_init_routing_v2<ElementABefore>(
                reinterpret_cast<GM_ADDR>(params.ptrA), params.expertIdx,
                shmem.windowsOutAddr() + peermemInfo.offsetA, workspaceInfo.expandedRowIdx,
                localTokenPerExpert, params.expertTokensBeforeCapacity,
                params.ptrWorkspace + expandedRowIdxOffset, &params.moeInitRoutingV2TilingData,
                params.initRoutingQuantTilingKey);
        }
        
        AscendC::SyncAll<true>();

        CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2(params, localTokenPerExpertOffset);

        if (coreIdx == 0) {
            GetCumsumForMMAIV(tokenPerExpert, cumsumMM, params.expertPerRank, RuntimeRank(params), params.EP);
        }

        uint32_t curGroupOffset = 0;
        int32_t prevSum = 0;
        if (coreIdx < params.EP) {
            prevSum = preSumBeforeRank(coreIdx * params.expertPerRank);
        }
        AscendC::SyncAll<true>();

        AscendC::GlobalTensor<int32_t> ExpertTokenNums;
        ExpertTokenNums.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.ptrExpertTokenNums));
        if (coreIdx == 0) {
            CopyGMToGM(ExpertTokenNums, cumsumMM[(params.EP - 1) * params.expertPerRank], params.expertPerRank,
                       UB_MOVE_NUM);
        }
        uint16_t syncgmm1Idx = 0;
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(syncgmm1Idx / CROSS_CORE_FLAG_MAX_SET_COUNT);
        syncgmm1Idx++;

        uint32_t prevGroupSum1 = 0;
        uint32_t prevGroupSum2 = 0;
        nSyncSwiglu = 0;
        dequantSum[0] = 0;
        uint32_t dequantSumTemp = 0;
        uint32_t dequantSum1 = 0;
        uint32_t dequantSum2 = 0;
        icache_preload(8);
        for (int32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            // ----------------------------------------------------------
            // A3 path (default): each core pulls its `dstEpIdx`'s tokens
            // out of the remote rank's local peer-mem and runs the W4A8
            // pre-process (int8 -> int4) in the same step.
            //
            // A2 path: each core PUSHES its tokens to the remote rank
            // using SendTokensV3 (self-rank fast path / IPC / RDMA), then
            // every core PULLS waits on the epoch flag via RecvTokensV3.
            // The W4A8 pre-process step does NOT apply here because A2 is
            // only enabled for the BF16/INT8 weight paths (W4A8 op def
            // only registers `ascend910_93`).
            // ----------------------------------------------------------
            uint32_t currentM;
            currentM = static_cast<uint32_t>(cumsumMM(tokenPerExpertLayout(params.EP - 1, RuntimeRank(params), groupIdx)));
            // SEND phase: parallel-over-cores send to each dstEpIdx.
            // prevGroupSum1: dstEpIdx peer mem offset base (SEND phase)
            // prevGroupSum2: rank peer mem offset base (RECV phase)
            uint32_t currentMSend = 0;
            constexpr bool kIsInt8Path = std::is_same_v<ElementA, int8_t>;
            uint32_t copyInNum = kIsInt8Path ? (params.problemShape.k() + ALIGN_512) : params.problemShape.k();
            for (int32_t dstEpIdx = static_cast<int32_t>(coreIdx);
                 dstEpIdx < static_cast<int32_t>(params.EP);
                 dstEpIdx += static_cast<int32_t>(coreNum)) {
                uint32_t rowStart = prevGroupSum1 +
                    ((RuntimeRank(params) == 0) ? 0u
                        : static_cast<uint32_t>(cumsumMM(
                            tokenPerExpertLayout(RuntimeRank(params) - 1, dstEpIdx, groupIdx))));
                currentMSend = static_cast<uint32_t>(
                    cumsumMM(tokenPerExpertLayout(params.EP - 1, dstEpIdx, groupIdx)));
                if (rowStart < params.maxOutputSize) {
                    uint32_t rows = tokenPerExpert(tokenPerExpertLayout(RuntimeRank(params), dstEpIdx, groupIdx));
                    if (rowStart + rows > params.maxOutputSize) {
                        rows = params.maxOutputSize - rowStart;
                    }
                    uint32_t rowSrc = prevSum;
                    prevSum += rows;
                    AscendC::GlobalTensor<ElementABefore> gmSrcA;
                    gmSrcA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementABefore *>(
                        shmem.windowsOutAddr() + peermemInfo.offsetA));
                    int64_t gmSrcOffset = static_cast<int64_t>(rowSrc) * copyInNum;

                    AscendC::GlobalTensor<ElementABefore> gmRemoteDstA;
                    gmRemoteDstA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementABefore *>(
                        shmem(0, dstEpIdx) + peermemInfo.offsetA));
                    int64_t gmDstOffset = static_cast<int64_t>(rowStart) * copyInNum;

                    SendTokensV3<ElementABefore>(
                        gmRemoteDstA[gmDstOffset], gmSrcA[gmSrcOffset],
                        static_cast<int32_t>(rows), params.problemShape.k(),
                        dstEpIdx, groupIdx, static_cast<int32_t>(rowStart), params);
                }
            }

            AscendC::SyncAll<true>();

            // RECV phase: parallel-over-cores wait on epoch flags.
            for (int32_t srcEpIdx = static_cast<int32_t>(coreIdx);
                 srcEpIdx < static_cast<int32_t>(params.EP);
                 srcEpIdx += static_cast<int32_t>(coreNum)) {
                uint32_t rowStart2 = prevGroupSum2 +
                    ((srcEpIdx == 0) ? 0u
                        : static_cast<uint32_t>(cumsumMM(
                            tokenPerExpertLayout(srcEpIdx - 1, RuntimeRank(params), groupIdx))));
                if (rowStart2 < params.maxOutputSize) {
                    uint32_t rows2 = static_cast<uint32_t>(
                        tokenPerExpert(tokenPerExpertLayout(srcEpIdx, RuntimeRank(params), groupIdx)));
                    if (rows2 + rowStart2 > params.maxOutputSize) {
                        rows2 = params.maxOutputSize - rowStart2;
                    }
                    RecvTokensV3<ElementABefore>(
                        static_cast<int32_t>(rows2), static_cast<int32_t>(rowStart2),
                        params.problemShape.k(), srcEpIdx, groupIdx, params);
                }
            }
            AscendC::SyncAll<true>();

            prevGroupSum1 += currentMSend;
            prevGroupSum2 += currentM;
            AscendC::SyncAll<true>();
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(syncgmm1Idx / CROSS_CORE_FLAG_MAX_SET_COUNT);
            syncgmm1Idx++;

            // W4A8路径在A2上不支持
            /* if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) { ... } else */ {
                uint32_t tokenCount = currentM;

                if (groupIdx + 1 <= params.epilogueGranularity) {
                    if (dequantSum1 + tokenCount <= params.maxOutputSize) {
                        dequantSum1 += tokenCount;
                    } else if (dequantSum1 < params.maxOutputSize) {
                        dequantSum1 = params.maxOutputSize;
                    }
                }

                if (groupIdx + 1 > params.epilogueGranularity && dequantSum1 < params.maxOutputSize) {
                    if (dequantSum1 + dequantSum2 + tokenCount <= params.maxOutputSize) {
                        dequantSum2 += tokenCount;
                    } else if (dequantSum1 + dequantSum2 < params.maxOutputSize) {
                        dequantSum2 += params.maxOutputSize - dequantSum1 - dequantSum2;
                    }
                }
            }
        }
#ifdef SYNC_MODE
        AscendC::SyncAll<false>();
#endif

        uint32_t n2 = params.problemShape.k();

        typename BlockEpilogue2::Params epilogueParams{
            static_cast<int32_t>(params.EP),
            static_cast<int32_t>(params.expertPerRank),
            static_cast<int32_t>(RuntimeRank(params)),
            reinterpret_cast<__gm__ int32_t *>(shmem() + peermemInfo.offsetPeerTokenPerExpert),
            params.layoutD2,
            static_cast<int32_t>(n2),
            static_cast<int32_t>(L1TileShape::N),
            shmem,
            static_cast<int32_t>(peermemInfo.offsetD)};

        uint32_t n = params.problemShape.n();
        BlockEpilogue2 blockEpilogue2(resource, epilogueParams);
        BlockEpilogue1 blockEpilogue1(resource, n);
#ifdef SYNC_MODE
        AscendC::SyncAll<false>();
#endif
        // W4A8路径在A2上不支持
        /* if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) { ... } else */ {
            AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAGC2V);
            AscendC::SyncAll<true>();
            if (dequantSum1 > 0) {
                uint32_t rowStartThisCore = 0;
                MatrixCoord offsetC{0U, 0};
                MatrixCoord shapeC{dequantSum1, params.problemShape.n()};
                LayoutC layoutC{dequantSum1, params.problemShape.n()};
                int64_t gmOffsetC = layoutC.GetOffset(offsetC);
                int64_t gmOffsetD = params.layoutD1.GetOffset(offsetC);
                if constexpr (std::is_same_v<ElementB, int8_t>) {
                    blockEpilogue1(gmC[gmOffsetC], shapeC, gmPerTokenScale1[rowStartThisCore],
                                    gmPermutedToken[gmOffsetD], gmPerTokenScale2[rowStartThisCore],
                                    params.epilogueCoreNum, params.swigluLimit);
                } else if constexpr (!std::is_same_v<ElementB, AscendC::int4b_t>) {
                    blockEpilogue1(gmC[gmOffsetC], shapeC, gmPermutedToken[gmOffsetD],
                                    params.epilogueCoreNum, params.swigluLimit);
                }
            }
            AscendC::SyncAll<true>();
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNCFLAGV2C);

            if ((params.epilogueGranularity < params.expertPerRank && params.epilogueGranularity > 0)) {
                AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAGC2V);
                AscendC::SyncAll<true>();
                if (dequantSum2 > 0) {
                    uint32_t rowStartThisCore = dequantSum1;
                    MatrixCoord offsetC{rowStartThisCore, 0};
                    uint32_t dequantLen = dequantSum2;
                    MatrixCoord shapeC{dequantLen, params.problemShape.n()};
                    LayoutC layoutC{dequantLen, params.problemShape.n()};
                    int64_t gmOffsetC = layoutC.GetOffset(offsetC);
                    int64_t gmOffsetD = params.layoutD1.GetOffset(offsetC);
                    if constexpr (std::is_same_v<ElementB, int8_t>) {
                        blockEpilogue1(gmC[gmOffsetC], shapeC, gmPerTokenScale1[rowStartThisCore],
                                        gmPermutedToken[gmOffsetD], gmPerTokenScale2[rowStartThisCore],
                                        coreNum, params.swigluLimit);
                    } else if constexpr (!std::is_same_v<ElementB, AscendC::int4b_t>) {
                        blockEpilogue1(gmC[gmOffsetC], shapeC, gmPermutedToken[gmOffsetD],
                                        coreNum, params.swigluLimit);
                    }
                }
                AscendC::SyncAll<true>();
                AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNCFLAGV2C);
            }
        }
        blockEpilogue1.Finalize();
#ifdef SYNC_MODE
        AscendC::SyncAll<false>();
        AscendC::SyncAll<false>();
#endif
        // W4A8路径在A2上不支持
        CombineSetFlag();

        CombineV2(params, blockEpilogue2);
        AscendC::SyncAll<true>();
        ResetTokenPerExpert(params.EP * AlignUp(params.EP * params.expertPerRank, 128));
        shmem.InitStatusTargetSum();
        if (get_subblockid() == 0) {
            AscendC::LocalTensor<int32_t> ctrBuffer = resource.ubBuf.template GetBufferByByte<int32_t>(0);
            shmem.CrossRankSyncV2Set(ctrBuffer);
        } else {
            uint32_t uboffset = 0;
            uint32_t aicCoreNum = coreNum / 2;
            uint32_t aicCoreIdx = get_block_idx();
            uint32_t sendRankNum_ = params.EP / aicCoreNum;
            uint32_t remainderRankNum = params.EP % aicCoreNum;
            if (aicCoreIdx < remainderRankNum) {
                sendRankNum_++;
            }
            AscendC::LocalTensor<float> statusTensor = resource.ubBuf.template GetBufferByByte<float>(uboffset);
            uboffset += sendRankNum_ * UB_ALIGN;
            AscendC::LocalTensor<float> gatherMaskOutTensor = resource.ubBuf.template GetBufferByByte<float>(uboffset);
            uboffset += AlignUp(params.EP * sizeof(float), 32);
            AscendC::LocalTensor<uint32_t> gatherTmpTensor =
                resource.ubBuf.template GetBufferByByte<uint32_t>(uboffset);
            uboffset += AlignUp(sizeof(uint32_t), 32);
            AscendC::LocalTensor<float> statusSumOutTensor = resource.ubBuf.template GetBufferByByte<float>(uboffset);
            uboffset += AlignUp(sizeof(float), 32);
            shmem.CrossRankSyncV2Wait(statusTensor, gatherMaskOutTensor, gatherTmpTensor, statusSumOutTensor);
            MoeTokenUnpermuteTilingData tilingData;
            MoeTokenUnpermuteTiling(params.problemShape.m() * params.topK, n2, params.topK, tilingData, coreNum / 2);
            KernelMoeTokenUnpermute<ElementD2, int32_t, float, true> kernelMoeTokenUnpermuteOp;
            kernelMoeTokenUnpermuteOp.Init(shmem() + peermemInfo.offsetD, workspaceInfo.expandedRowIdx, params.probs,
                                           reinterpret_cast<GM_ADDR>(params.ptrOutput), &tilingData);
            kernelMoeTokenUnpermuteOp.Process();
        }
    }

    CATLASS_DEVICE
    void CombineV2(Params const &params, BlockEpilogue2 &blockEpilogue)
    {
        blockEpilogue.InitFlag();
        BlockScheduler blockScheduler;
        int32_t syncLoopIdx = 0;
        uint32_t startCoreIdx = 0;
        uint32_t aicCoreNum = coreNum / 2;
        uint32_t aicCoreIdx = get_block_idx();
        uint32_t aivSubCoreIdx = get_subblockid();
        uint32_t preSrcExpertSum = 0;
        uint32_t n2 = params.problemShape.k();
        uint32_t k2 = params.problemShape.n() / 2;

        int64_t gmGroupOffsetC = 0;
        uint32_t aivCoreNum = coreNum;
        uint32_t aivCoreIdx = coreIdx;

        int32_t m0 = 32;
        if constexpr (std::is_same_v<ElementB, int8_t>) {
            m0 = 16;
        }

        icache_preload(8);
        for (uint32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            uint32_t currentExpertM = cumsumMM((params.EP - 1) * params.expertPerRank + groupIdx);
            if (preSrcExpertSum >= params.maxOutputSize) {
                currentExpertM = 0;
            } else if (preSrcExpertSum + currentExpertM > params.maxOutputSize) {
                currentExpertM = params.maxOutputSize - preSrcExpertSum;
            }

            // W4A8路径在A2上不支持

            GemmCoord inGroupProblemShape{currentExpertM, n2, k2};  // M N K
            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();
            uint32_t startLoopIdx =
                ((aicCoreIdx < startCoreIdx) ? (aicCoreIdx + aicCoreNum) : aicCoreIdx) - startCoreIdx;

            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += aicCoreNum) {
                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);

                int32_t m_rows = (actualBlockShape.m() + m0 - 1) / m0;
                int32_t aiv_m_rows = m_rows / 2;
                if (aivSubCoreIdx == 1 && aiv_m_rows * 2 < m_rows) {
                    aiv_m_rows += 1;
                }
                uint32_t m_offset = blockCoord.m() * L1TileShape::M;  // blockOffset
                if (aivSubCoreIdx == 1) {
                    m_offset += (m_rows / 2) * m0;
                }

                for (; syncLoopIdx <= groupIdx; syncLoopIdx++) {
                    int32_t flag_id = syncLoopIdx / CROSS_CORE_FLAG_MAX_SET_COUNT;
                    AscendC::CrossCoreWaitFlag<0x2>(flag_id);
                }

                for (int32_t cur_row = 0; cur_row < aiv_m_rows; cur_row++) {
                    GemmCoord realTileCoord{m_offset, blockCoord.n() * L1TileShape::N, 1};
                    uint32_t actualm = m0;
                    if (aivSubCoreIdx == 1 && cur_row == aiv_m_rows - 1) {
                        actualm = actualBlockShape.m() - (m_rows / 2) * m0 - cur_row * m0;
                    }
                    GemmCoord realTileShape{actualm, actualBlockShape.n(), 1};
                    // W4A8路径在A2上不支持
                    if constexpr (std::is_same_v<ElementB, int8_t>) {
                        blockEpilogue(gmC2, gmPerTokenScale2, realTileCoord, realTileShape,
                                      groupIdx, preSrcExpertSum, preSumBeforeRank);
                    } else if constexpr (!std::is_same_v<ElementB, AscendC::int4b_t>) {
                        blockEpilogue(gmC2, realTileCoord, realTileShape,
                                      groupIdx, preSrcExpertSum, preSumBeforeRank);
                    }
                    m_offset += m0;
                }
            }

            preSrcExpertSum += currentExpertM;
            startCoreIdx = (startCoreIdx + coreLoops) % aicCoreNum;
        }

        /* W4A8路径在A2上不支持
        if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
            for (; syncLoopIdx < params.expertPerRank; syncLoopIdx++) {
                int32_t flag_id = syncLoopIdx / CROSS_CORE_FLAG_MAX_SET_COUNT;
                AscendC::CrossCoreWaitFlag<0x2>(flag_id);
            }
        }
        */
        blockEpilogue.Finalize();
    }

private:
    struct WorkspaceInfo {
        GM_ADDR ptrA;
        GM_ADDR ptrPerTokenScale;
        GM_ADDR ptrcumsumMM;
        GM_ADDR ptrC;
        GM_ADDR ptrC2;
        GM_ADDR ptrPermutedToken;
        GM_ADDR ptrPerTokenScale2;
        GM_ADDR expandedRowIdx;
        GM_ADDR ptrTokenPerExpert;
        GM_ADDR ptrA1Int4;
        GM_ADDR ptrA2Int4;
        GM_ADDR ptrCGMM1;
        GM_ADDR ptrCGMM2;
        GM_ADDR ptrSumBeforeRank;
#ifdef ENABLE_TIMER
        GM_ADDR ptrTimer;
#endif
        __gm__ float *ptrSoftFlagBase;

        CATLASS_DEVICE
        WorkspaceInfo() {}

        CATLASS_DEVICE
        WorkspaceInfo(const Params &params)
        {
            uint32_t k2 = params.problemShape.n() / 2;
            uint32_t n2 = params.problemShape.k();
            int64_t workspaceOffset = 0;
            expandedRowIdx = params.ptrWorkspace;

            workspaceOffset += AlignUp(params.problemShape.m(), 256) * params.topK * sizeof(int32_t);
            ptrcumsumMM = params.ptrWorkspace + workspaceOffset;

            workspaceOffset += (params.EP * params.EP * params.expertPerRank) * sizeof(int32_t);

            workspaceOffset += (params.EP * params.EP * params.expertPerRank) * sizeof(int32_t);
            ptrPerTokenScale = params.ptrWorkspace + workspaceOffset;

            workspaceOffset += params.maxOutputSize * sizeof(ElementPerTokenScale);
            ptrPerTokenScale2 = params.ptrWorkspace + workspaceOffset;

            workspaceOffset += params.maxOutputSize * sizeof(ElementPerTokenScale);
            ptrTokenPerExpert = params.ptrWorkspace + workspaceOffset;

            workspaceOffset += (params.EP * params.EP * params.expertPerRank) * sizeof(int32_t);
            ptrC = params.ptrWorkspace + workspaceOffset;  // 7

            // W4A8路径在A2上不支持
            workspaceOffset += params.maxOutputSize * params.problemShape.n() * sizeof(ElementC);
            ptrC2 = params.ptrWorkspace + workspaceOffset;  // 8

            // W4A8路径在A2上不支持
            workspaceOffset += params.maxOutputSize * n2 * sizeof(ElementC);
            // W4A8路径在A2上不支持
            ptrA = params.ptrWorkspace + workspaceOffset;
            workspaceOffset += params.maxOutputSize * params.problemShape.k() * sizeof(ElementA);
            ptrPermutedToken = params.ptrWorkspace + workspaceOffset;
            workspaceOffset += params.maxOutputSize * k2 * sizeof(ElementA);
            ptrSumBeforeRank = params.ptrWorkspace + workspaceOffset;
            // W4A8路径在A2上不支持
            workspaceOffset += params.EP * sizeof(int32_t) * FLAGSTRIDE;
            ptrSoftFlagBase = reinterpret_cast<__gm__ float *>(params.ptrWorkspace + workspaceOffset);

#ifdef ENABLE_TIMER
            ptrTimer = params.ptrWorkspace + workspaceOffset;
            workspaceOffset += AscendTimer::N_TIMING_COUNTER * sizeof(int64_t);
#endif
        }
    };

    struct PeermemInfo {
        int64_t offsetA;
        int64_t offsetPeerPerTokenScale;
        int64_t offsetPeerTokenPerExpert;
        int64_t offsetD;
        // Only used by the A2 BatchWrite path: location inside peer-memory
        // where one int32 epoch-flag per source rank is written by SendTokensV3
        // and polled by RecvTokensV3. On A3 we don't push flags through
        // peer-mem at all, so this stays 0.
        int64_t offsetFlag{0};

        CATLASS_DEVICE
        PeermemInfo() {}

        CATLASS_DEVICE
        PeermemInfo(const Params &params, const HcclShmem<true> &shmem)
        {
            offsetA = 0;                                                                // 占用1/3的BUFFSIZE
            offsetPeerPerTokenScale = offsetA + AlignUp(shmem.SegmentSize() / 3, 512);  // 占用1MB
            offsetD = offsetPeerPerTokenScale + MB_SIZE;                                // 占用剩下的
            offsetPeerTokenPerExpert = shmem.SegmentSize() - 2 * MB_SIZE;               // 占用最后2MB
            // Layout one int32 flag per source-rank, packed into the 32 B
            // slot starting at offsetPeerTokenPerExpert - one cacheline.
            // The 1 MB region just before the tokenPerExpert slab is
            // repurposed for the A2 epoch flags (SendTokensV3 writes at
            // `offsetFlag + srcRank * sizeof(uint64_t)`).
            offsetFlag = offsetPeerTokenPerExpert - MB_SIZE;
        }
    };

    Arch::Resource<ArchTag> resource;

    uint32_t coreIdx;
    uint32_t coreNum;
    uint32_t nSyncSwiglu;

    Params params;
    WorkspaceInfo workspaceInfo;
    PeermemInfo peermemInfo;

    uint32_t dequantSum[16] = {0};

    AscendC::GlobalTensor<ElementABefore> gmA;
    AscendC::GlobalTensor<ElementC> gmC;
    AscendC::GlobalTensor<ElementScale> gmS;

    AscendC::GlobalTensor<int4b_t> gmA1I4;
    AscendC::GlobalTensor<int8_t> gmA1I4_I8;
    AscendC::GlobalTensor<int8_t> gmB1_I8;
    AscendC::GlobalTensor<int4b_t> gmA2I4;
    AscendC::GlobalTensor<int8_t> gmA2I4_I8;
    AscendC::GlobalTensor<float> gmCGMM1;
    AscendC::GlobalTensor<float> gmCGMM2;

    AscendC::GlobalTensor<ElementD1> gmPermutedToken;
    AscendC::GlobalTensor<ElementScale> gmS2;
    AscendC::GlobalTensor<ElementC> gmC2;

    AscendC::GlobalTensor<ElementPerTokenScale> gmPerTokenScale1;
    AscendC::GlobalTensor<ElementPerTokenScale> gmPerTokenScale2;

    AscendC::GlobalTensor<bool> gmXActiveMask;

    AscendC::GlobalTensor<int32_t> tokenPerExpert;
    AscendC::GlobalTensor<int32_t> cumsumMM;
    AscendC::GlobalTensor<int32_t> preSumBeforeRank;
    Layout3D tokenPerExpertLayout;
    HcclShmem<true> shmem;

    // A2 state.
    __gm__ HcclAiRMAInfo *qp_info_{nullptr};
    int32_t serverId_{0};
};

}  // namespace Catlass::Gemm::Kernel

#endif // DISPATCH_FFN_COMBINE_KERNEL_A2_HPP