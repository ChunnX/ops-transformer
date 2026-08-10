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
 * \file mega_moe_kernel_a3.hpp
 * \brief
 */

#ifndef MEGA_MOE_KERNEL_A3_HPP
#define MEGA_MOE_KERNEL_A3_HPP

// Guard against 3rdparty copy_l0c_to_gm.hpp so the custom PER_CHANNEL
// specialization in copy_l0c_to_gm_custom.hpp takes precedence.
#define CATLASS_GEMM_TILE_COPY_L0C_TO_GM_HPP

#include "kernel_operator.h"

#include "utils/copy_l0c_to_gm_custom.hpp"

#include "template_linear_algebra_v2/catlass.hpp"
#include "template_linear_algebra_v2/arch/cross_core_sync.hpp"
#include "template_linear_algebra_v2/arch/resource.hpp"
#include "template_linear_algebra_v2/coord.hpp"
#include "template_linear_algebra_v2/detail/callback.hpp"
#include "template_linear_algebra_v2/gemm_coord.hpp"
#include "template_linear_algebra_v2/matrix_coord.hpp"
#include "template_linear_algebra_v2/epilogue/tile/tile_copy.hpp"

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
#include "moe_init_routing_quant_v2/moe_init_routing_quant_v2.hpp"
// Non-quant routing dispatcher (used by the BF16/FP16 weight path).
// Lives alongside the quant dispatcher in the same `moe_init_routing_quant_v2`
// directory so it can share all sort/expert-token/srcToDst helpers.
#include "moe_init_routing_v2/moe_init_routing_v2.hpp"
#include "moe_init_routing_quant_v2/moe_v2_fullload_dynamic_quant.h"
#include "unpermute/moe_token_unpermute.h"
#include "utils/get_tensor_addr.hpp"
#include "mega_moe_exception_dump_policy.h"

namespace Catlass::Gemm::Kernel {

using namespace AscendC;
using namespace Mc2Tiling;

constexpr uint16_t SYNCFLAGC2V = 9;
constexpr uint16_t SYNCFLAGV2C = 10;
constexpr int32_t UB_MOVE_NUM = 16 * 1024;
// dispatch/combine 二维并行分块参数：每个 (rank, 专家组) 的 token 行按该行数切块，
// 让 EP < AIV 核数（如 EP=8 < 40）时空转的 AIV 也能参与搬运。
//
// 【为什么要有下限，且下限不能只夹块大小】
// 块太小会把大 burst 搬运碎片化、拉低 MTE2 效率。实测教训（ws16 大 batch）：
//   - 固定 8 行/块：kimi_k2_ws16 mte2 0.928→0.855、Duration +26%
//   - 只夹块大小到 8 行：kimi_k2_ws16 每 rank 仅 10.7 行 → 切成 8 + 2.7 行，
//     第二块是纯碎片，且 epr=24 把这个代价放大 24 倍 → 仍 +17%
// 因此判据改为「切完之后每块仍要 ≥ MIN_ROWS_PER_BLOCK 行」，即
// 只有 rowsPerRank >= 2×MIN_ROWS_PER_BLOCK 才值得切第一刀。
constexpr uint32_t MIN_ROWS_PER_BLOCK = 8;
// dispatch/CombineV1 每 dstEpIdx 块数前缀和数组的上限，
// 与 host 侧 tiling 校验的 VALID_EP_WORLD_SIZE 最大值（128）对齐。
static constexpr uint32_t MAX_DISPATCH_RANKS = 128;
// cumsumMM 列切片 UB 缓存的起始偏移与容量。
//
// 【为什么放 UB 尾部，而不是此前的 64KB】
// 原实现放在 64KB 并注释「CombineV1 期间 epilogue 从 0 起分配，均不触及本区间」——该假设是错的。
// BlockEpilogue2（EpilogueAtlasA2PerTokenDequant<ubStages=2>，见 utils/block_epilogue_pertoken_row.hpp）
// 从 UB[0] 起按 stage 连续分配，每 stage 占 n2×(sizeof(ElementC)+sizeof(ElementD)+sizeof(float))
// = hidden×10B，两个 stage 合计：
//     hidden=4096 → 80KB   hidden=6144 → 120KB   hidden=7168 → 140KB   hidden=8192 → 160KB
// 即 64KB 必落在 epilogue 的占用区内 → 每个 group 调用 epilogue 都会冲掉 cumsumCache，
// 后续 group 的 BuildRankBlockPrefix / 任务反查读到 epilogue 中间数据，块映射错乱，
// 表现为 cos 0.002~0.02、Duration 涨 5 倍、偶发 CRASH（rowsPerRank∈[21,32] 时稳定复现）。
//
// 现改为放在 UB 尾部：起点 = UB_SIZE - CACHE_BYTES。epilogue 最坏占用 160KB（hidden=8192），
// 尾部 8KB 区间 [184KB,192KB) 不被触及；dispatch 的搬运 ping-pong 占 [0,32KB)/[96KB,128KB)、
// int4 预处理占 [0,~30KB)，同样不触及。
// 容量 8KB = 2048 个 int32，覆盖 EP×expertPerRank 的上界（EP≤128、epr≤128 时按 host 校验
// 组合不会同时取满；实际最大用例 EP16×epr24=384 个 int32 = 1.5KB）。
static constexpr uint32_t DISPATCH_CACHE_BYTES = 8 * 1024;
static constexpr uint32_t DISPATCH_CACHE_UB_OFFSET = Arch::AtlasA2::UB_SIZE - DISPATCH_CACHE_BYTES;

// 计算每块行数。返回值 == rowsPerRank 表示不分块（退化为原「一 AIV 一 rank」语义）。
// 分块条件（三个都要满足）：
//   1) EP < coreNum          —— 否则原实现已无空转核，分块只有碎片化代价
//   2) coreNum/EP >= 2       —— 每 rank 至少能分到 2 块才有并行意义
//   3) rowsPerRank >= 2×MIN  —— 保证切完每块仍是有意义的大 burst，不产生碎片尾块
__aicore__ inline uint32_t CalcRowsPerBlock(uint32_t rowsPerRank, uint32_t ep, uint32_t coreNum)
{
    if (rowsPerRank == 0) {
        return 1U;
    }
    if (ep == 0 || ep >= coreNum || rowsPerRank < 2U * MIN_ROWS_PER_BLOCK) {
        return rowsPerRank;  // 不分块
    }
    uint32_t wantBlocks = coreNum / ep;  // 每 rank 期望切几块才填满核
    if (wantBlocks <= 1U) {
        return rowsPerRank;
    }
    // 块数上限：保证每块 >= MIN_ROWS_PER_BLOCK，避免尾块碎片
    uint32_t maxBlocks = rowsPerRank / MIN_ROWS_PER_BLOCK;
    if (wantBlocks > maxBlocks) {
        wantBlocks = maxBlocks;
    }
    return CeilDiv(rowsPerRank, wantBlocks);
}

// Atlas A3 (910_93) kernel.
template <class BlockMmad_, class BlockScheduler_, class ElementGroupList_, class BlockEpilogue1_,
          class BlockEpilogue2_, class BlockEpilogue3_>
class MegaMoeKernel {
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
    using BlockEpilogue3 = BlockEpilogue3_;
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
        __gm__ float *ptrBias1;
        __gm__ ElementB *ptrB2;
        LayoutB layoutB2;
        __gm__ float *ptrBias2;
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
        uint64_t maxOutputSize;
        // for reuse workspace, gmm1 out preRow Stride use max(n,k)
        // single axis must lower than uint32, layoutC need uint32
        uint32_t gmmOutPreRowStride;
        // GMM1 input reuses the SwiGLU output workspace. Inputs are K-dense
        // inside each expert, while every expert reserves currentM * max(K, K2)
        // elements. The tail gap prevents an earlier expert's K2-wide SwiGLU
        // output from overwriting a later expert's unconsumed GMM1 input.
        uint32_t gmm1InputExpertSlotWidth;
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
        int64_t topK;
        uint64_t initRoutingQuantTilingKey;
        uint32_t epilogueCoreNum;
        // Epilogue scheduling granularity. Default 0 keeps the legacy W4A8
        // behavior (no special throttling). INT8 path historically uses
        // `expertPerRank - 1`; BF16 path historically uses `expertPerRank - 2`.
        uint32_t epilogueGranularity{0};
        float swigluLimit;
        GM_ADDR contextGM{nullptr};
        // 算子tiling数据地址（GM），ADump启动时由0核dump到Tiling段
        GM_ADDR tilingGM{nullptr};
        union {
            MoeInitRoutingQuantV2TilingData moeInitRoutingQuantV2TilingData;
            MoeInitRoutingV2TilingData moeInitRoutingV2TilingData;
        };
        //--------------

        // Methods
        CATLASS_HOST_DEVICE
        Params()
        {
        }

        CATLASS_HOST_DEVICE
        Params(GemmCoord problemShape_, uint32_t EP_, uint32_t listLen_, uint32_t expertPerRank_,
               uint64_t maxOutputSize_, int64_t topK_, uint64_t initRoutingQuantTilingKey_, uint32_t epilogueCoreNum_,
               GM_ADDR contextGM_, GM_ADDR ptrA_, LayoutA layoutA_, LayoutA layoutA2_, GM_ADDR ptrB1_,
               LayoutB layoutB1_, GM_ADDR ptrBias1_, GM_ADDR ptrB2_, LayoutB layoutB2_, GM_ADDR ptrBias2_,
               GM_ADDR ptrScale1_, LayoutScale layoutScale1_, GM_ADDR ptrScale2_, LayoutScale layoutScale2_,
               GM_ADDR ptrOutput_, LayoutD1 layoutD1_, LayoutD2 layoutD2_, GM_ADDR expertIdx_,
               GM_ADDR moeInitRoutingQuantV2Scale_, GM_ADDR moeInitRoutingQuantV2Offset_,
               GM_ADDR expertTokensBeforeCapacity_, GM_ADDR probs_, GM_ADDR ptrWorkspace_, GM_ADDR gmExpertTokenNums_,
               GM_ADDR ptrXActiveMask_, GM_ADDR ptrScales_,
               MoeInitRoutingQuantV2TilingData moeInitRoutingQuantV2TilingData_, uint32_t epilogueGranularity_ = 0,
               float swigluLimit_ = std::numeric_limits<float>::infinity(), GM_ADDR tilingGM_ = nullptr)
            : problemShape(problemShape_), EP(EP_), listLen(listLen_), expertPerRank(expertPerRank_),
              maxOutputSize(maxOutputSize_), topK(topK_), initRoutingQuantTilingKey(initRoutingQuantTilingKey_),
              epilogueCoreNum(epilogueCoreNum_), epilogueGranularity(epilogueGranularity_), swigluLimit(swigluLimit_),
              contextGM(contextGM_), tilingGM(tilingGM_), ptrA(reinterpret_cast<__gm__ ElementABefore *>(ptrA_)),
              layoutA(layoutA_), layoutA2(layoutA2_), ptrB1(reinterpret_cast<__gm__ ElementB *>(ptrB1_)),
              layoutB1(layoutB1_), ptrBias1(reinterpret_cast<__gm__ float *>(ptrBias1_)),
              ptrB2(reinterpret_cast<__gm__ ElementB *>(ptrB2_)), layoutB2(layoutB2_),
              ptrBias2(reinterpret_cast<__gm__ float *>(ptrBias2_)),
              ptrScale1(reinterpret_cast<__gm__ ElementScale *>(ptrScale1_)), layoutScale1(layoutScale1_),
              ptrScale2(reinterpret_cast<__gm__ ElementScale *>(ptrScale2_)), layoutScale2(layoutScale2_),
              ptrOutput(reinterpret_cast<__gm__ ElementD2 *>(ptrOutput_)), layoutD1(layoutD1_), layoutD2(layoutD2_),
              expertIdx(expertIdx_), moeInitRoutingQuantV2Scale(moeInitRoutingQuantV2Scale_),
              moeInitRoutingQuantV2Offset(moeInitRoutingQuantV2Offset_),
              expertTokensBeforeCapacity(expertTokensBeforeCapacity_), probs(probs_), ptrXActiveMask(ptrXActiveMask_),
              ptrScales(ptrScales_), ptrWorkspace(ptrWorkspace_), ptrExpertTokenNums(gmExpertTokenNums_),
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
            gmmOutPreRowStride = problemShape.n() > problemShape.k() ? problemShape.n() : problemShape.k();
            uint32_t k2 = problemShape.n() / 2;
            gmm1InputExpertSlotWidth = problemShape.k() > k2 ? problemShape.k() : k2;
        }

        CATLASS_HOST_DEVICE
        Params(GemmCoord problemShape_, uint32_t EP_, uint32_t listLen_, uint32_t expertPerRank_,
               uint64_t maxOutputSize_, int64_t topK_, uint64_t initRoutingQuantTilingKey_, uint32_t epilogueCoreNum_,
               GM_ADDR contextGM_, GM_ADDR ptrA_, LayoutA layoutA_, LayoutA layoutA2_, GM_ADDR ptrB1_,
               LayoutB layoutB1_, GM_ADDR ptrBias1_, GM_ADDR ptrB2_, LayoutB layoutB2_, GM_ADDR ptrBias2_,
               GM_ADDR ptrScale1_, LayoutScale layoutScale1_, GM_ADDR ptrScale2_, LayoutScale layoutScale2_,
               GM_ADDR ptrOutput_, LayoutD1 layoutD1_, LayoutD2 layoutD2_, GM_ADDR expertIdx_,
               GM_ADDR moeInitRoutingQuantV2Scale_, GM_ADDR moeInitRoutingQuantV2Offset_,
               GM_ADDR expertTokensBeforeCapacity_, GM_ADDR probs_, GM_ADDR ptrWorkspace_, GM_ADDR gmExpertTokenNums_,
               GM_ADDR ptrXActiveMask_, GM_ADDR ptrScales_, MoeInitRoutingV2TilingData moeInitRoutingV2TilingData_,
               uint32_t epilogueGranularity_ = 0, float swigluLimit_ = std::numeric_limits<float>::infinity(),
               GM_ADDR tilingGM_ = nullptr)
            : problemShape(problemShape_), EP(EP_), listLen(listLen_), expertPerRank(expertPerRank_),
              maxOutputSize(maxOutputSize_), topK(topK_), initRoutingQuantTilingKey(initRoutingQuantTilingKey_),
              epilogueCoreNum(epilogueCoreNum_), epilogueGranularity(epilogueGranularity_), swigluLimit(swigluLimit_),
              contextGM(contextGM_), tilingGM(tilingGM_), ptrA(reinterpret_cast<__gm__ ElementABefore *>(ptrA_)),
              layoutA(layoutA_), layoutA2(layoutA2_), ptrB1(reinterpret_cast<__gm__ ElementB *>(ptrB1_)),
              layoutB1(layoutB1_), ptrBias1(reinterpret_cast<__gm__ float *>(ptrBias1_)),
              ptrB2(reinterpret_cast<__gm__ ElementB *>(ptrB2_)), layoutB2(layoutB2_),
              ptrBias2(reinterpret_cast<__gm__ float *>(ptrBias2_)),
              ptrScale1(reinterpret_cast<__gm__ ElementScale *>(ptrScale1_)), layoutScale1(layoutScale1_),
              ptrScale2(reinterpret_cast<__gm__ ElementScale *>(ptrScale2_)), layoutScale2(layoutScale2_),
              ptrOutput(reinterpret_cast<__gm__ ElementD2 *>(ptrOutput_)), layoutD1(layoutD1_), layoutD2(layoutD2_),
              expertIdx(expertIdx_), moeInitRoutingQuantV2Scale(moeInitRoutingQuantV2Scale_),
              moeInitRoutingQuantV2Offset(moeInitRoutingQuantV2Offset_),
              expertTokensBeforeCapacity(expertTokensBeforeCapacity_), probs(probs_), ptrXActiveMask(ptrXActiveMask_),
              ptrScales(ptrScales_), ptrWorkspace(ptrWorkspace_), ptrExpertTokenNums(gmExpertTokenNums_),
              moeInitRoutingV2TilingData(moeInitRoutingV2TilingData_)
        {
            gmmOutPreRowStride = problemShape.n() > problemShape.k() ? problemShape.n() : problemShape.k();
            uint32_t k2 = problemShape.n() / 2;
            gmm1InputExpertSlotWidth = problemShape.k() > k2 ? problemShape.k() : k2;
        }
    };

    // Methods
    CATLASS_DEVICE
    MegaMoeKernel(Params const &params)
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
    ~MegaMoeKernel()
    {
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
    {
        if constexpr (!std::is_same_v<ElementB, AscendC::int4b_t>) {
            GMM1(params);
            AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAGV2C);
            GMM2(params);
        } else {
            GMM1(params);
            GMM2(params);
        }
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

    CATLASS_DEVICE void initBuffer(Params const &params)
    {
        auto tmpContext = reinterpret_cast<__gm__ Mc2Aclnn::Mc2MoeContext *>(params.contextGM);
        shmem.initShmem(tmpContext);
        if ASCEND_IS_AIV {
            GM_ADDR dumpBase = shmem();
            exceptionDump_.Init(dumpBase, params.tilingGM);
        }
        workspaceInfo = WorkspaceInfo(params);
        peermemInfo = PeermemInfo(params, shmem);

        cumsumMM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspaceInfo.ptrcumsumMM));

        if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
            gmS.SetGlobalBuffer(params.ptrScale1);
            gmA1I4.SetGlobalBuffer(reinterpret_cast<__gm__ int4b_t *>(workspaceInfo.ptrA1Int4));
            gmA1I4_I8.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(workspaceInfo.ptrA1Int4));
            gmA2I4.SetGlobalBuffer(reinterpret_cast<__gm__ int4b_t *>(workspaceInfo.ptrA2Int4));
            gmA2I4_I8.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(workspaceInfo.ptrA2Int4));
            gmS2.SetGlobalBuffer(params.ptrScale2);
        } else {
            gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementABefore *>(workspaceInfo.ptrA));
            gmPermutedToken.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD1 *>(workspaceInfo.ptrPermutedToken));
        }

        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(workspaceInfo.ptrC));
        gmC2.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(workspaceInfo.ptrC2));

        gmPerTokenScale1.SetGlobalBuffer(
            reinterpret_cast<__gm__ ElementPerTokenScale *>(workspaceInfo.ptrPerTokenScale));
        gmPerTokenScale2.SetGlobalBuffer(
            reinterpret_cast<__gm__ ElementPerTokenScale *>(workspaceInfo.ptrPerTokenScale2));

        tokenPerExpert.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(shmem() + peermemInfo.offsetPeerTokenPerExpert));

        paddedExpertNumAligned = AlignUp(params.EP * params.expertPerRank + 1, ALIGN_128);
        tokenPerExpertLayout = Layout3D(paddedExpertNumAligned, params.expertPerRank);
        preSumBeforeRank.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspaceInfo.ptrSumBeforeRank));
        gmXActiveMask.SetGlobalBuffer(reinterpret_cast<__gm__ bool *>(params.ptrXActiveMask));

        isCombineV1 = false;
        if constexpr (std::is_same_v<ElementB, int8_t>) {
            isCombineV1 = true;
            if (params.problemShape.m() * params.topK <= 4096) {
                isCombineV1 = false;
            }
        }
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
        int tmpBufferSize = 32 * 1024 / sizeof(T); // 32 KB
        AscendC::LocalTensor<T> tmpBuffer1 = resource.ubBuf.template GetBufferByByte<T>(0);
        tmpBuffer1.SetSize(tmpBufferSize);
        int tmpBufferOffset = 96 * 1024; // half of UB
        AscendC::LocalTensor<T> tmpBuffer2 = resource.ubBuf.template GetBufferByByte<T>(tmpBufferOffset);
        tmpBuffer2.SetSize(tmpBufferSize);

        int pingpongId = 0;
        auto processCount = CeilDiv(elemNum, ubMoveNum);
        for (uint32_t processIndex = 0; processIndex < processCount; ++processIndex) {
            uint32_t curProcessNum =
                (processIndex == processCount - 1) ? elemNum - ubMoveNum * (processCount - 1) : ubMoveNum;
            AscendC::TEventID EVENT_ID = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
            AscendC::LocalTensor<T> buf = pingpongId == 0 ? tmpBuffer1 : tmpBuffer2;
            auto processOffset = processIndex * ubMoveNum;

            auto inputOffset = processOffset;
            auto outputOffset = processOffset;
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            copyGmToUb(buf, src[inputOffset], layout::RowMajor{1, curProcessNum}, layout::RowMajor{1, curProcessNum});
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
            copyUbToGm(dst[outputOffset], buf, layout::RowMajor{1, curProcessNum}, layout::RowMajor{1, curProcessNum});

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            pingpongId = (pingpongId + 1) % BufferNum;
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    template <typename T>
    CATLASS_DEVICE void CopyGMToGMPerToken(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<float> dstScale,
                                           AscendC::GlobalTensor<T> src, int32_t rows, int32_t hiddenSize,
                                           int32_t ubMoveNum, int32_t &pingpongId)
    {
        constexpr int32_t BufferNum = 2;
        AscendC::LocalTensor<T> tmpBuffer1 = resource.ubBuf.template GetBufferByByte<T>(0);
        constexpr int tmpBufferOffset = 96 * 1024;
        AscendC::LocalTensor<T> tmpBuffer2 = resource.ubBuf.template GetBufferByByte<T>(tmpBufferOffset);
        uint32_t copyInNum = hiddenSize + UB_ALIGN;
        auto processCount = CeilDiv(rows, ubMoveNum);
        for (uint32_t processIndex = 0; processIndex < processCount; ++processIndex) {
            pingpongId = (pingpongId + 1) % BufferNum;
            AscendC::TEventID EVENT_ID = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
            AscendC::LocalTensor<T> buf = pingpongId == 0 ? tmpBuffer1 : tmpBuffer2;
            AscendC::LocalTensor<float> bufScale = buf[hiddenSize].template ReinterpretCast<float>();
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
            AscendC::DataCopyPad(dst[outputOffset], buf, {U16(rowNum), U16(hiddenSize), 1, 0, 0});
            AscendC::DataCopyPad(dstScale[processIndex * ubMoveNum], bufScale,
                                 {U16(rowNum), U16(sizeof(float)), static_cast<uint32_t>(hiddenSize / 32), 0, 0});
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        }
    }

    CATLASS_DEVICE void FetchAndPreprocessInt8ToInt4(uint64_t gmOffsetA, AscendC::GlobalTensor<float> dstScale,
                                                     AscendC::GlobalTensor<int8_t> src, int32_t rows,
                                                     int32_t hiddenSize)
    {
        uint32_t tmpBufferOffset = 0;

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
        const half MINUS_EIGHT = static_cast<half>(-8);

        uint32_t totalM = rows;

        SetFlag<HardEvent::S_MTE2>(EVENT_ID6);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID6);
        uint32_t curCoreTaskNum;
        uint32_t curCoreStartOffset;
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

    // 单条向量指令的 repeat 上限为 255，int32 每 repeat 处理 64 个元素，
    // 因此「按数据量」的向量接口一次最多覆盖 255*64 = 16320 个 int32。
    static constexpr uint32_t kVecI32MaxCalCount = 255U * 64U;

    // 沿 rank 维（EP 维）对 tokenPerExpert 做 inclusive 前缀和，结果密排写入 result(cumsumMM)。
    //
    // 语义：tokenPerExpert(srcRank, dstRank, e) = srcRank 发给 dstRank 的第 e 号本地专家的 token 数。
    //       本函数只取 dstRank == rankId 的那一列切片，得到 EP × expertPerRank 的矩阵
    //       cnt[i][e]，然后逐列做前缀和：result[i][e] = Σ_{j<=i} cnt[j][e]。
    //       消费方：result[(EP-1)*epr + g] = 本 rank 第 g 号专家的总行数；
    //               result[(i-1)*epr + g]  = 第 i 个源 rank 在该专家内的起始行号。
    //
    // 性能：原实现是 EP-1 次「单行 Add + PipeBarrier<PIPE_V>」，依赖链长度 O(EP)
    //       （EP=16 时 15 次全 V 流水排空，只为累加 16 个 int32，是全 kernel 性价比最低的一段）。
    //       这里改成 Hillis-Steele 倍增扫描，依赖链降到 ceil(log2(EP))（EP=16 → 4 步），
    //       且每步用「整段多行」的向量指令一次覆盖 (EP-d) 行，指令条数 15 → 2*4。
    //       用 bufA/bufB ping-pong 避免 dst 与 src 部分重叠（AscendC 向量接口只允许完全重叠）。
    CATLASS_DEVICE
    void GetCumsumForMMAIV(AscendC::GlobalTensor<int32_t> &tokenPerExpert, AscendC::GlobalTensor<int32_t> &result,
                           AscendC::GlobalTensor<int32_t> &expertTokenNums, uint32_t expertPerRank, uint32_t rankId,
                           uint32_t EP)
    {
        // DataCopyPad(GM->UB) 的 dstStride=0 会把每个 block 落到下一个 32B 对齐位置，
        // 因此第 i 个 rank 的行首恰好落在 i * expertPerRankAligned（int32 元素）处。
        const uint32_t expertPerRankAligned = (expertPerRank + 8 - 1) / 8 * 8;
        const uint32_t rowsTotal = EP * expertPerRankAligned; // ping-pong 单块元素数（32B 对齐）

        AscendC::LocalTensor<int32_t> bufA = resource.ubBuf.template GetBufferByByte<int32_t>(0);
        AscendC::LocalTensor<int32_t> bufB =
            resource.ubBuf.template GetBufferByByte<int32_t>(rowsTotal * sizeof(int32_t));
#define U16(x) static_cast<uint16_t>(x)

        AscendC::DataCopyPad(bufA, tokenPerExpert[rankId * expertPerRank],
                             {U16(EP), U16(expertPerRank * sizeof(int32_t)),
                              U16((paddedExpertNumAligned - expertPerRank) * sizeof(int32_t)), 0},
                             {});

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        // 倍增扫描：step d 满足 dst[i] = src[i] + src[i-d] (i>=d)，dst[i] = src[i] (i<d)。
        // 走完 d = 1,2,4,...(<EP) 共 ceil(log2(EP)) 步后即为 inclusive 前缀和。
        // 行尾 padding lane（expertPerRank..expertPerRankAligned）也会被一并相加，但
        // 最终 DataCopyPad(UB->GM) 只取每行前 expertPerRank 个元素，padding 不会写出去。
        AscendC::LocalTensor<int32_t> src = bufA;
        AscendC::LocalTensor<int32_t> dst = bufB;
        for (uint32_t d = 1; d < EP; d <<= 1) {
            const uint32_t head = d * expertPerRankAligned;
            // 前 d 行本步不变，原样带到 ping-pong 的另一块（Adds 与 Add 同在 PIPE_V，共用一次 barrier）
            for (uint32_t off = 0; off < head; off += kVecI32MaxCalCount) {
                uint32_t n = head - off;
                n = (n > kVecI32MaxCalCount) ? kVecI32MaxCalCount : n;
                AscendC::Adds(dst[off], src[off], 0, static_cast<int32_t>(n));
            }
            // 其余行做 dst[i] = src[i] + src[i-d]；src/dst 分属两块 UB，无地址重叠
            for (uint32_t off = head; off < rowsTotal; off += kVecI32MaxCalCount) {
                uint32_t n = rowsTotal - off;
                n = (n > kVecI32MaxCalCount) ? kVecI32MaxCalCount : n;
                AscendC::Add(dst[off], src[off], src[off - head], static_cast<int32_t>(n));
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<int32_t> swapTmp = src;
            src = dst;
            dst = swapTmp;
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        AscendC::DataCopyPad(result, src, {U16(EP), U16((expertPerRank) * sizeof(int32_t)), 0, 0});
        // ExpertTokenNums 就是最后一行（全 rank 汇总），直接从 UB 写出，
        // 省掉原先「SyncAll 之后再 CopyGMToGM(cumsumMM -> ExpertTokenNums)」的 GM->UB->GM 往返。
        AscendC::DataCopyPad(expertTokenNums, src[(EP - 1) * expertPerRankAligned],
                             {1, U16(expertPerRank * sizeof(int32_t)), 0, 0});
    }

    CATLASS_DEVICE
    void GMM1(Params const &params)
    {
        icache_preload(8);
        BlockScheduler blockScheduler;
        BlockMmad blockMmad(resource);

        int64_t gmGroupOffsetA = 0;
        int64_t gmGroupOffsetB = 0;
        int64_t gmGroupOffsetC = 0;
        uint32_t startCoreIdx = 0;
        uint32_t syncGroupIdx = 0;
        int64_t preCurrentmSum = 0;
        int32_t syncLoopIdx = -1;

        uint16_t syncgmmIdx = 0;
        AscendC::CrossCoreWaitFlag<0x2>(syncgmmIdx /
                                        CROSS_CORE_FLAG_MAX_SET_COUNT); // Wait for AIV to finish cumsum for matmul

        syncgmmIdx++;
        AscendC::GlobalTensor<ElementB> gmB1;

        for (uint32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            uint32_t currentM = cumsumMM((params.EP - 1) * params.expertPerRank + groupIdx);
            if (preCurrentmSum >= params.maxOutputSize) {
                currentM = 0;
            } else if (preCurrentmSum + currentM >= params.maxOutputSize) {
                currentM = params.maxOutputSize - preCurrentmSum;
            }

            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                currentM = currentM * 2;
            }

            int32_t arrayGroupIdx = params.listLen == 1 ? 0 : groupIdx;
            gmB1.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(
                GetTensorAddr<ElementB>(arrayGroupIdx, reinterpret_cast<GM_ADDR>(params.ptrB1))));
            gmS.SetGlobalBuffer(reinterpret_cast<__gm__ ElementScale *>(
                GetTensorAddr<int64_t>(arrayGroupIdx, reinterpret_cast<GM_ADDR>(params.ptrScale1))));

            if constexpr (!std::is_same_v<ElementB, AscendC::int4b_t>) {
                AscendC::PipeBarrier<PIPE_ALL>();
            }

            if (currentM <= L1TileShape::M) {
                gmB1.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }
            GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};
            LayoutA layoutA = params.layoutA.GetTileLayout(inGroupProblemShape.GetCoordMK());
            LayoutB layoutB1 = params.layoutB1;
            LayoutScale layoutScale = params.layoutScale1;
            LayoutC layoutC;
            layoutC = LayoutC(inGroupProblemShape.m(), inGroupProblemShape.n(), params.gmmOutPreRowStride);
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
                    if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                        int64_t gmOffsetS = blockCoord.n() * L1TileShape::N +
                                            (params.listLen == 1 ? groupIdx * params.problemShape.n() : 0);
                        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                            blockMmad(gmS[gmOffsetS], layoutScale, gmA1I4[gmGroupOffsetA + gmOffsetA], layoutA,
                                      gmB1[gmGroupOffsetB + gmOffsetB], layoutB1, gmC[gmGroupOffsetC + gmOffsetC],
                                      layoutC, actualBlockShape, Callback{}, Callback{});
                        } else {
                            blockMmad(gmS[gmOffsetS], layoutScale, gmA1I4[gmGroupOffsetA + gmOffsetA], layoutA,
                                      gmB1[gmGroupOffsetB + gmOffsetB], layoutB1, gmC[gmGroupOffsetC + gmOffsetC],
                                      layoutC, actualBlockShape);
                        }
                    } else if constexpr (std::is_same_v<ElementA, int8_t>) {
                        int64_t gmOffsetS = blockCoord.n() * L1TileShape::N +
                                            (params.listLen == 1 ? groupIdx * params.problemShape.n() : 0);
                        blockMmad(gmA[gmGroupOffsetA + gmOffsetA], layoutA, gmB1[gmGroupOffsetB + gmOffsetB], layoutB1,
                                  gmC[gmGroupOffsetC + gmOffsetC], layoutC, gmS[gmOffsetS], layoutScale,
                                  actualBlockShape);
                    } else {
                        blockMmad(gmA[gmGroupOffsetA + gmOffsetA], layoutA, gmB1[gmGroupOffsetB + gmOffsetB], layoutB1,
                                  gmC[gmGroupOffsetC + gmOffsetC], layoutC, gmS, layoutScale, actualBlockShape);
                    }
                }
            }

            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                if (IsSyncTask(groupIdx, params.expertPerRank)) {
                    syncLoopIdx++;
                    if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                        blockMmad.SynchronizeBlock();
                    }
                    blockMmad.Finalize(syncLoopIdx, SYNCFLAGC2V);
                }
            } else {
                if ((groupIdx + 1) == params.epilogueGranularity && (groupIdx < params.expertPerRank - 1)) {
                    syncLoopIdx++;
                    if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                        blockMmad.SynchronizeBlock();
                    }
                    blockMmad.Finalize(syncLoopIdx, SYNCFLAGC2V);
                }
            }

            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                preCurrentmSum += currentM / 2;
            } else {
                preCurrentmSum += currentM;
            }
            // 每个专家之间在k小于k2的时候不是密排，所以这里不取k
            gmGroupOffsetA += inGroupProblemShape.m() * params.gmm1InputExpertSlotWidth;
            if (params.listLen == 1) {
                gmGroupOffsetB += inGroupProblemShape.k() * inGroupProblemShape.n();
            }
            gmGroupOffsetC += inGroupProblemShape.m() * params.gmmOutPreRowStride;
            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }

        for (; syncGroupIdx < params.expertPerRank; syncGroupIdx++) {
            AscendC::CrossCoreWaitFlag<0x2>(syncgmmIdx / CROSS_CORE_FLAG_MAX_SET_COUNT);
            syncgmmIdx++;
        }

        if constexpr (!std::is_same_v<ElementB, AscendC::int4b_t>) {
            if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                blockMmad.SynchronizeBlock();
            }
            blockMmad.Finalize(syncLoopIdx + 1, SYNCFLAGC2V);
        }
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

        if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        int64_t preCurrentmSum = 0;
        int32_t syncLoopIdx = -1;
        uint32_t lastDequantExpertNum = params.expertPerRank;
        if constexpr (!std::is_same_v<ElementB, AscendC::int4b_t>) {
            if (params.epilogueGranularity < params.expertPerRank) {
                lastDequantExpertNum = params.expertPerRank - params.epilogueGranularity;
            }
        }

        for (uint32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            uint32_t currentM = cumsumMM((params.EP - 1) * params.expertPerRank + groupIdx);
            if (preCurrentmSum >= params.maxOutputSize) {
                currentM = 0;
            } else if (preCurrentmSum + currentM > params.maxOutputSize) {
                currentM = params.maxOutputSize - preCurrentmSum;
            }

            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                currentM = currentM * 2;
            }

            AscendC::GlobalTensor<ElementB> gmB2;
            int32_t arrayGroupIdx = params.listLen == 1 ? 0 : groupIdx;
            gmB2.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(
                GetTensorAddr<ElementB>(arrayGroupIdx, reinterpret_cast<GM_ADDR>(params.ptrB2))));
            gmS2.SetGlobalBuffer(reinterpret_cast<__gm__ ElementScale *>(
                GetTensorAddr<int64_t>(arrayGroupIdx, reinterpret_cast<GM_ADDR>(params.ptrScale2))));
            if (currentM <= L1TileShape::M) {
                gmB2.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }
            GemmCoord inGroupProblemShape{currentM, n2, k2}; // M N K

            LayoutA layoutA = params.layoutA2.GetTileLayout(inGroupProblemShape.GetCoordMK());
            LayoutB layoutB2 = params.layoutB2;
            LayoutScale layoutScale = params.layoutScale2;
            LayoutC layoutC = LayoutC(inGroupProblemShape.m(), inGroupProblemShape.n());

            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();

            // Determine the starting loopIdx of the current core under the current groupIdx
            uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;

            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                if (groupIdx == 0 || IsSyncTask(groupIdx - 1, params.expertPerRank)) {
                    AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAGV2C);
                }
            } else {
                if (params.expertPerRank > lastDequantExpertNum &&
                    groupIdx + 1 == params.expertPerRank - lastDequantExpertNum) {
                    AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAGV2C);
                }
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
                    if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                        int64_t gmOffsetS = blockCoord.n() * L1TileShape::N + (params.listLen == 1 ? groupIdx * n2 : 0);
                        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                            blockMmad(gmS2[gmOffsetS], layoutScale, gmA2I4[gmGroupOffsetA + gmOffsetA], layoutA,
                                      gmB2[gmGroupOffsetB + gmOffsetB], layoutB2, gmC2[gmGroupOffsetC + gmOffsetC],
                                      layoutC, actualBlockShape, Callback{}, Callback{}, syncLoopIdx, 0);
                        }
                    } else if constexpr (std::is_same_v<ElementA, int8_t>) {
                        int64_t gmOffsetS = blockCoord.n() * L1TileShape::N + (params.listLen == 1 ? groupIdx * n2 : 0);
                        blockMmad(gmPermutedToken[gmGroupOffsetA + gmOffsetA], layoutA,
                                  gmB2[gmGroupOffsetB + gmOffsetB], layoutB2, gmC2[gmGroupOffsetC + gmOffsetC], layoutC,
                                  gmS2[gmOffsetS], layoutScale, actualBlockShape, syncLoopIdx, 0);
                    } else {
                        blockMmad(gmPermutedToken[gmGroupOffsetA + gmOffsetA], layoutA,
                                  gmB2[gmGroupOffsetB + gmOffsetB], layoutB2, gmC2[gmGroupOffsetC + gmOffsetC], layoutC,
                                  gmS2, layoutScale, actualBlockShape, syncLoopIdx, 0);
                    }
                }
            }
            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                preCurrentmSum += currentM / 2;
            } else {
                preCurrentmSum += currentM;
            }
            gmGroupOffsetA += inGroupProblemShape.m() * inGroupProblemShape.k();
            if (params.listLen == 1) {
                gmGroupOffsetB += inGroupProblemShape.k() * inGroupProblemShape.n();
            }
            gmGroupOffsetC += inGroupProblemShape.m() * inGroupProblemShape.n();

            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }
        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.SynchronizeBlock();
            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                blockMmad.Finalize(params.expertPerRank - 1, 0);
            }
        }
        if constexpr (std::is_same_v<ElementB, int8_t>) {
            if (isCombineV1) {
                blockMmad.Finalize(params.expertPerRank - 1, 0);
            }
        }
    }

    CATLASS_DEVICE
    void CalculateTaskInfoEachCore(uint32_t &curCoreTaskNum_, uint32_t &curCoreStartOffset_, uint32_t totalM)
    {
        // 均分任务数
        int64_t eachCoreTaskNum = (totalM + coreNum - 1) / coreNum; // 每个核处理的数据量
        // 实际使用核数
        int64_t usedCoreNum = totalM >= coreNum ? coreNum : totalM;
        // 尾核起始索引
        uint32_t tailCoreIdx = totalM - (eachCoreTaskNum - 1) * usedCoreNum;
        uint32_t curCoreId = GetBlockIdx();
        // 每个核处理的任务数量 = 是否为尾核 ？均分任务数 ：(均分任务数 - 1)
        curCoreTaskNum_ = curCoreId < tailCoreIdx ? eachCoreTaskNum : eachCoreTaskNum - 1;
        // 每个核处理的起始偏移地址 = 是否为尾核 ？均分任务数 * blockId : (均分任务数 - 1) * blockId + 尾核起始索引
        curCoreStartOffset_ =
            curCoreId < tailCoreIdx ? eachCoreTaskNum * curCoreId : ((eachCoreTaskNum - 1) * curCoreId + tailCoreIdx);
    }

    CATLASS_DEVICE
    void CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2(Params const &params,
                                                                        int64_t localTokenPerExpertOffset)
    {
        uint32_t numPerCore = paddedExpertNumAligned;
        AscendC::LocalTensor<int32_t> tmpBuffer = resource.ubBuf.template GetBufferByByte<int32_t>(0);
        AscendC::LocalTensor<int32_t> prevSumBuf = tmpBuffer[numPerCore];

        int32_t runtimeRank = RuntimeRank(params);
        for (int32_t dstEpIdx = coreIdx; dstEpIdx < params.EP; dstEpIdx += coreNum) {
            if (dstEpIdx == runtimeRank) {
                continue;
            }
            AscendC::GlobalTensor<int32_t> srcAddress;
            srcAddress.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(shmem() + localTokenPerExpertOffset));
            AscendC::GlobalTensor<int32_t> dstAddress;
            __gm__ void *dstPeermemPtr = shmem(localTokenPerExpertOffset, dstEpIdx);
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
            if (dstEpIdx != runtimeRank) {
                int32_t intPer512 = CACHE_LINE / sizeof(int);
                for (int32_t checkIdx = 0; checkIdx < paddedExpertNumAligned; checkIdx += intPer512) {
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
            for (int32_t i = 0; i < (runtimeRank + 1) * params.expertPerRank; i++) {
                if (i >= runtimeRank * params.expertPerRank) {
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
    bool IsSyncTask(int32_t task_id, int32_t n_tasks)
    {
        int32_t offset = n_tasks - task_id;
        if (offset <= 0 || (offset & (offset - 1)) != 0) {
            return false;
        } else {
            return true;
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
        } else if constexpr (!std::is_same_v<ElementB, AscendC::int4b_t>) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
    }

    CATLASS_DEVICE
    void CombineWaitFlag()
    {
        if constexpr (std::is_same_v<ElementB, int8_t>) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        } else if constexpr (!std::is_same_v<ElementB, AscendC::int4b_t>) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
    }

    CATLASS_DEVICE
    void ApplyXActiveMask(Params const &params)
    {
        if (params.ptrXActiveMask == nullptr) {
            return;
        }
        int32_t m = params.problemShape.m();
        int32_t topK = params.topK;
        int32_t expertNum = params.expertPerRank * params.EP;
        AscendC::GlobalTensor<int32_t> expertIdxGm;
        expertIdxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.expertIdx));

        int32_t totalElements = m * topK;
        int32_t base = totalElements / coreNum;
        int32_t rem = totalElements % coreNum;

        int32_t startIdx = coreIdx * base + (coreIdx < rem ? coreIdx : rem);
        int32_t endIdx = (coreIdx + 1) * base + (coreIdx + 1 < rem ? coreIdx + 1 : rem);

        AscendC::LocalTensor<int32_t> tmpExpertIdx = resource.ubBuf.template GetBufferByByte<int32_t>(0);
        int32_t copySize = endIdx - startIdx;

        AscendC::DataCopyPad(tmpExpertIdx[0], expertIdxGm[startIdx],
                             {1, static_cast<uint16_t>(copySize * sizeof(int32_t)), 0, 0}, {});

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
        AscendC::DataCopyPad(expertIdxGm[startIdx], tmpExpertIdx[0],
                             {1, static_cast<uint16_t>(copySize * sizeof(int32_t)), 0, 0, 0});
        AscendC::SyncAll<true>();
    }

    CATLASS_DEVICE
    void DispatchAndCombine(Params const &params)
    {
        icache_preload(8);
        exceptionDump_.Dump(shmem() + peermemInfo.offsetPeerTokenPerExpert,
                            static_cast<size_t>(paddedExpertNumAligned) * params.expertPerRank *
                                static_cast<uint32_t>(shmem.RankSize()) * sizeof(int32_t));
        shmem.DumpSyncRegions(exceptionDump_);
        int32_t runtimeRank = RuntimeRank(params);
        int64_t localTokenPerExpertOffset =
            peermemInfo.offsetPeerTokenPerExpert + tokenPerExpertLayout(runtimeRank, 0, 0) * sizeof(int32_t);
        GM_ADDR localTokenPerExpert =
            shmem() + localTokenPerExpertOffset; // Place the entire communication matrix in peermem
        uint32_t expandedRowIdxOffset = AlignUp(params.problemShape.m(), 256) * params.topK * sizeof(int32_t);

        exceptionDump_.UpdateStage(MC2MegaMoeAdump::Stage::APPLY_XACTIVE_MASK);
        ApplyXActiveMask(params);

        constexpr int64_t colsAlign =
            std::is_same_v<ElementB, AscendC::int4b_t> ? int64_t{512} : static_cast<int64_t>(UB_ALIGN);
        exceptionDump_.UpdateStage(MC2MegaMoeAdump::Stage::MOE_INIT_ROUTING);
        if constexpr (kRoutingIsQuant) {
            moe_init_routing_quant_v2<ElementD2>(
                reinterpret_cast<GM_ADDR>(params.ptrA), params.expertIdx, params.moeInitRoutingQuantV2Scale,
                params.moeInitRoutingQuantV2Offset, shmem() + peermemInfo.offsetA, workspaceInfo.expandedRowIdx,
                localTokenPerExpert, params.expertTokensBeforeCapacity, shmem() + peermemInfo.offsetPeerPerTokenScale,
                params.ptrWorkspace + expandedRowIdxOffset, &params.moeInitRoutingQuantV2TilingData,
                params.initRoutingQuantTilingKey, colsAlign);
        } else {
            // BF16 / FP16 path: no scale/offset/dynamicQuantScale inputs and
            // outputs, tiling data is the Inner (non-quant) variant.
            moe_init_routing_v2<ElementABefore>(reinterpret_cast<GM_ADDR>(params.ptrA), params.expertIdx,
                                                shmem() + peermemInfo.offsetA, workspaceInfo.expandedRowIdx,
                                                localTokenPerExpert, params.expertTokensBeforeCapacity,
                                                params.ptrWorkspace + expandedRowIdxOffset,
                                                &params.moeInitRoutingV2TilingData, params.initRoutingQuantTilingKey);
        }

        AscendC::SyncAll<true>();

        exceptionDump_.UpdateStage(MC2MegaMoeAdump::Stage::ALLGATHER_TOKEN_PER_EXPERT);
        CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2(params, localTokenPerExpertOffset);

        if (coreIdx == 0) {
            exceptionDump_.UpdateStage(MC2MegaMoeAdump::Stage::CUMSUM_TOKEN_PER_EXPERT);
            // ExpertTokenNums（cumsum 最后一行）由 GetCumsumForMMAIV 直接从 UB 写出，
            // 不再走「SyncAll 之后 CopyGMToGM(cumsumMM -> ExpertTokenNums)」的 GM->UB->GM 往返。
            AscendC::GlobalTensor<int32_t> expertTokenNums;
            expertTokenNums.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.ptrExpertTokenNums));
            GetCumsumForMMAIV(tokenPerExpert, cumsumMM, expertTokenNums, params.expertPerRank, runtimeRank, params.EP);
        }

        AscendC::SyncAll<true>();

        uint16_t syncgmm1Idx = 0;
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(syncgmm1Idx / CROSS_CORE_FLAG_MAX_SET_COUNT);
        syncgmm1Idx++;

        uint32_t prevGroupSum1 = 0;
        uint32_t prevGroupSum2 = 0;
        // P2 修复：每 dstEpIdx 块数前缀和（EP ≤ 128），每个 group 重建一次。
        uint32_t rankBlockPrefix[MAX_DISPATCH_RANKS + 1];
        nSyncSwiglu = 0;
        dequantSum[0] = 0;
        uint32_t dequantSumTemp = 0;
        uint32_t dequantSum1 = 0;
        uint32_t dequantSum2 = 0;
        int32_t pingpongIdx = 0;
        icache_preload(8);
        if constexpr (std::is_same_v<ElementB, int8_t>) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
        exceptionDump_.UpdateStage(MC2MegaMoeAdump::Stage::DISPATCH);
        // P2 修复：cumsumMM 列切片一次性搬进 UB 缓存，group 循环内每 rank 行数 /
        // 组内起始行全部从 UB 读（避免每 group 每核 O(EP) 次 GM 标量读）。
        AscendC::LocalTensor<int32_t> cumsumCache = PrefetchCumsumColumn(params);
        for (int32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            // ----------------------------------------------------------
            // Each core pulls its `dstEpIdx`'s tokens out of the remote
            // rank's local peer-mem and runs the pre-process step.
            // ----------------------------------------------------------
            uint32_t currentM = cumsumMM((params.EP - 1) * params.expertPerRank + groupIdx);
            // P2 优化：dispatch 从「一 AIV 一 rank」改为「rank × token 行块」二维并行，
            // 让 EP < 40 时空转的 AIV 也参与搬运。块大小由 CalcRowsPerBlock 自适应：
            // EP 已达核数时退化为不分块，避免把大 burst 搬运碎片化（ws16 大 batch 实测教训）。
            //
            // ⚠️ P2 bug 修复：任务块数必须按【每 dstEpIdx 的实际行数】算，不能按全局平均。
            // 旧实现 tokenBlocks = CeilDiv(rowsPerRankEst, rowsPerBlock)（rowsPerRankEst 是
            // currentM/EP 的平均行数）且任务网格是 EP × tokenBlocks 均匀网格；稀疏路由下
            // 某 dstEpIdx 的实际行数远超平均时，其超出 tokenBlocks×rowsPerBlock 的尾段行
            // 没有任何任务块覆盖 → 行丢失。现在用 BuildRankBlockPrefix 把任务空间压成
            // 「每 rank 块数 Σ 的连续序列」，taskIdx 经前缀和反查 (dstEpIdx, blockIdx)，
            // 每 rank 的全部行都被恰好一个块覆盖一次。
            // rowStart / rowSrc / gmOffsetA 偏移公式不变（分块仍是行区间的有序切分，
            // 每块绝对地址不变），GMM1 侧读 ptrA 的布局不受影响。
            uint32_t rowsPerRankEst = CeilDiv(currentM, static_cast<uint32_t>(params.EP));
            uint32_t rowsPerBlock =
                CalcRowsPerBlock(rowsPerRankEst, static_cast<uint32_t>(params.EP), coreNum);
            uint32_t totalTasks = BuildRankBlockPrefix(params, groupIdx, rowsPerBlock, prevGroupSum1,
                                                       params.maxOutputSize, cumsumCache, rankBlockPrefix);
            for (uint32_t taskIdx = static_cast<uint32_t>(coreIdx); taskIdx < totalTasks;
                 taskIdx += coreNum) {
                uint32_t dstEpIdx = 0;
                while (dstEpIdx + 1 < static_cast<uint32_t>(params.EP) &&
                       rankBlockPrefix[dstEpIdx + 1] <= taskIdx) {
                    ++dstEpIdx;
                }
                uint32_t blockIdx = taskIdx - rankBlockPrefix[dstEpIdx];
                uint32_t prevCumsumDst = dstEpIdx == 0 ? 0 : cumsumCache((dstEpIdx - 1) * params.expertPerRank + groupIdx);
                uint32_t rowStartInGroup = prevCumsumDst;
                uint32_t rowStart = rowStartInGroup + prevGroupSum1 + blockIdx * rowsPerBlock;
                if (rowStart < params.maxOutputSize) {
                    uint32_t rowsTotal =
                        cumsumCache(dstEpIdx * params.expertPerRank + groupIdx) - prevCumsumDst;
                    // 该源 rank 的行数可能不足（rowsTotal==0 或 < blockIdx*rowsPerBlock），
                    // 直接跳过该块，避免 rowsTotal - blockIdx*rowsPerBlock 下溢成巨数。
                    if (blockIdx * rowsPerBlock >= rowsTotal) {
                        continue;
                    }
                    uint32_t rows = rowsTotal - blockIdx * rowsPerBlock;
                    if (rows > rowsPerBlock) {
                        rows = rowsPerBlock;
                    }
                    if (rowStart + rows > params.maxOutputSize) {
                        rows = params.maxOutputSize - rowStart;
                    }
                    uint32_t rowSrc = preSumBeforeRank(dstEpIdx * params.expertPerRank + groupIdx) +
                                      blockIdx * rowsPerBlock;
                    GM_ADDR otherRankPtr = shmem(0, dstEpIdx);
                    AscendC::GlobalTensor<ElementABefore> gmRemoteA;
                    gmRemoteA.SetGlobalBuffer(
                        reinterpret_cast<__gm__ ElementABefore *>(otherRankPtr + peermemInfo.offsetA));

                    MatrixCoord offsetPeer{rowSrc, 0};
                    // 在做dispatch的时候，因为复用了gmm1的A矩阵与swiglu的输出
                    // 所以dispatch的时候每个专家之间间隔从密排改为max（k， k2）
                    // 但是专家之内仍然是密排的
                    int64_t gmOffsetA = static_cast<int64_t>(prevGroupSum1) * params.gmm1InputExpertSlotWidth +
                                        static_cast<int64_t>(rowStartInGroup + blockIdx * rowsPerBlock) *
                                            params.problemShape.k();
                    if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                        int64_t gmOffsetPeer = rowSrc * (params.problemShape.k() + ALIGN_512);
                        FetchAndPreprocessInt8ToInt4(static_cast<uint64_t>(gmOffsetA), gmPerTokenScale1[rowStart],
                                                     gmRemoteA[gmOffsetPeer], rows, params.problemShape.k());
                    } else if constexpr (std::is_same_v<ElementB, int8_t>) {
                        int64_t gmOffsetPeer = rowSrc * (params.problemShape.k() + UB_ALIGN);
                        int32_t ubMoveNum = 2;
                        CopyGMToGMPerToken(gmA[gmOffsetA], gmPerTokenScale1[rowStart], gmRemoteA[gmOffsetPeer], rows,
                                           params.problemShape.k(), ubMoveNum, pingpongIdx);
                    } else {
                        int64_t gmOffsetPeer = params.layoutA.GetOffset(offsetPeer);
                        CopyGMToGM(gmA[gmOffsetA], gmRemoteA[gmOffsetPeer], rows * params.problemShape.k(),
                                   UB_MOVE_NUM);
                    }
                }
            }
            AscendC::SyncAll<true>();
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(syncgmm1Idx / CROSS_CORE_FLAG_MAX_SET_COUNT);
            syncgmm1Idx++;

            prevGroupSum1 += currentM;

            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                if (dequantSumTemp + currentM <= params.maxOutputSize) {
                    dequantSumTemp += currentM;
                } else if (dequantSumTemp < params.maxOutputSize) {
                    dequantSumTemp = params.maxOutputSize;
                }

                if (IsSyncTask(groupIdx, params.expertPerRank)) {
                    nSyncSwiglu++;
                    dequantSum[nSyncSwiglu] = dequantSumTemp;
                }
            } else {
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
        if constexpr (std::is_same_v<ElementB, int8_t>) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        }

        uint32_t n2 = params.problemShape.k();

        typename BlockEpilogue2::Params epilogueParams2{
            static_cast<int32_t>(params.EP), static_cast<int32_t>(params.expertPerRank),
            reinterpret_cast<__gm__ int32_t *>(shmem() + peermemInfo.offsetPeerTokenPerExpert),
            static_cast<int32_t>(n2)};

        typename BlockEpilogue3::Params epilogueParams3{
            static_cast<int32_t>(params.EP),
            static_cast<int32_t>(params.expertPerRank),
            runtimeRank,
            reinterpret_cast<__gm__ int32_t *>(shmem() + peermemInfo.offsetPeerTokenPerExpert),
            params.layoutD2,
            static_cast<int32_t>(n2),
            static_cast<int32_t>(L1TileShape::N),
            shmem,
            peermemInfo.offsetD,
            tokenPerExpertLayout};

        uint32_t n = params.problemShape.n();
        BlockEpilogue2 blockEpilogue2(resource, epilogueParams2);
        BlockEpilogue3 blockEpilogue3(resource, epilogueParams3);
        exceptionDump_.UpdateStage(MC2MegaMoeAdump::Stage::SWIGLU);
        BlockEpilogue1 blockEpilogue1(resource, n);
        if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
            for (int32_t syncIdx = 0; syncIdx < nSyncSwiglu; syncIdx++) {
                AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAGC2V);
                AscendC::SyncAll<true>();
                uint32_t curRowNum = dequantSum[syncIdx + 1] - dequantSum[syncIdx];
                if (curRowNum > 0) {
                    uint32_t rowStartThisCore = dequantSum[syncIdx];
                    MatrixCoord offsetC{rowStartThisCore, 0};
                    MatrixCoord shapeC{curRowNum, params.problemShape.n()};
                    LayoutC layoutC{curRowNum, params.gmmOutPreRowStride};
                    int64_t gmOffsetC = layoutC.GetOffset(offsetC);
                    int64_t gmOffsetD = params.layoutD1.GetOffset(offsetC);
                    blockEpilogue1(gmC[gmOffsetC * 2], shapeC, gmPerTokenScale1[rowStartThisCore],
                                   reinterpret_cast<__gm__ float *>(params.ptrBias1), gmA2I4_I8[gmOffsetD], cumsumMM,
                                   rowStartThisCore, gmPerTokenScale2[rowStartThisCore], params.expertPerRank,
                                   params.EP, runtimeRank, params.listLen, resource, params.epilogueCoreNum,
                                   params.swigluLimit, params.gmmOutPreRowStride);
                }
                AscendC::SyncAll<true>();
                AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNCFLAGV2C);
            }
        } else {
            AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAGC2V);
            AscendC::SyncAll<true>();
            if (dequantSum1 > 0) {
                uint32_t rowStartThisCore = 0;
                MatrixCoord offsetC{0U, 0};
                MatrixCoord shapeC{dequantSum1, params.problemShape.n()};
                LayoutC layoutC;
                layoutC = LayoutC{dequantSum1, params.gmmOutPreRowStride};
                int64_t gmOffsetC = layoutC.GetOffset(offsetC);
                int64_t gmOffsetD = params.layoutD1.GetOffset(offsetC);
                if constexpr (std::is_same_v<ElementB, int8_t>) {
                    blockEpilogue1(gmC[gmOffsetC], shapeC, gmPerTokenScale1[rowStartThisCore],
                                   gmPermutedToken[gmOffsetD], gmPerTokenScale2[rowStartThisCore], resource,
                                   params.epilogueCoreNum, params.swigluLimit, params.gmmOutPreRowStride);
                } else {
                    blockEpilogue1(gmC[gmOffsetC], shapeC, gmPermutedToken[gmOffsetD], resource, params.epilogueCoreNum,
                                   params.swigluLimit, params.gmmOutPreRowStride);
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
                    LayoutC layoutC;
                    layoutC = LayoutC{dequantLen, params.gmmOutPreRowStride};
                    int64_t gmOffsetC = layoutC.GetOffset(offsetC);
                    int64_t gmOffsetD = params.layoutD1.GetOffset(offsetC);
                    if constexpr (std::is_same_v<ElementB, int8_t>) {
                        blockEpilogue1(gmC[gmOffsetC], shapeC, gmPerTokenScale1[rowStartThisCore],
                                       gmPermutedToken[gmOffsetD], gmPerTokenScale2[rowStartThisCore], resource,
                                       coreNum, params.swigluLimit, params.gmmOutPreRowStride);
                    } else {
                        blockEpilogue1(gmC[gmOffsetC], shapeC, gmPermutedToken[gmOffsetD], resource, coreNum,
                                       params.swigluLimit, params.gmmOutPreRowStride);
                    }
                }
                AscendC::SyncAll<true>();
                AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNCFLAGV2C);
            }
        }
        blockEpilogue1.Finalize();
        exceptionDump_.UpdateStage(MC2MegaMoeAdump::Stage::COMBINE);
        if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
            blockEpilogue3.InitFlag();
            CombineV2(params, blockEpilogue3);
        } else if (isCombineV1) {
            blockEpilogue2.SetFlag();
            CombineV1(params, blockEpilogue2);
        } else {
            CombineSetFlag();
            CombineV2(params, blockEpilogue3);
        }

        AscendC::SyncAll<true>();
        exceptionDump_.UpdateStage(MC2MegaMoeAdump::Stage::RESET_TOKEN_PER_EXPERT);
        ResetTokenPerExpert(params.EP * paddedExpertNumAligned);

        exceptionDump_.UpdateStage(MC2MegaMoeAdump::Stage::CROSS_RANK_SYNC);
        if constexpr (!std::is_same_v<ElementB, int8_t>) {
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
                AscendC::LocalTensor<float> gatherMaskOutTensor =
                    resource.ubBuf.template GetBufferByByte<float>(uboffset);
                uboffset += AlignUp(params.EP * sizeof(float), 32);
                AscendC::LocalTensor<uint32_t> gatherTmpTensor =
                    resource.ubBuf.template GetBufferByByte<uint32_t>(uboffset);
                uboffset += AlignUp(sizeof(uint32_t), 32);
                AscendC::LocalTensor<float> statusSumOutTensor =
                    resource.ubBuf.template GetBufferByByte<float>(uboffset);
                uboffset += AlignUp(sizeof(float), 32);
                shmem.CrossRankSyncV2Wait(statusTensor, gatherMaskOutTensor, gatherTmpTensor, statusSumOutTensor);

                exceptionDump_.UpdateStage(MC2MegaMoeAdump::Stage::UNPERMUTE);
                MoeTokenUnpermuteTilingData tilingData;
                MoeTokenUnpermuteTiling(params.problemShape.m() * params.topK, n2, params.topK, tilingData,
                                        coreNum / 2);
                KernelMoeTokenUnpermute<ElementD2, int32_t, float, true> kernelMoeTokenUnpermuteOp;
                kernelMoeTokenUnpermuteOp.Init(shmem() + peermemInfo.offsetD, workspaceInfo.expandedRowIdx,
                                               params.probs, reinterpret_cast<GM_ADDR>(params.ptrOutput), &tilingData);
                kernelMoeTokenUnpermuteOp.Process();
            }
        } else {
            shmem.CrossRankSync();

            exceptionDump_.UpdateStage(MC2MegaMoeAdump::Stage::UNPERMUTE);
            MoeTokenUnpermuteTilingData tilingData;
            MoeTokenUnpermuteTiling(params.problemShape.m() * params.topK, n2, params.topK, tilingData, coreNum);
            KernelMoeTokenUnpermute<ElementD2, int32_t, float, true> kernelMoeTokenUnpermuteOp;
            kernelMoeTokenUnpermuteOp.Init(shmem() + peermemInfo.offsetD, workspaceInfo.expandedRowIdx, params.probs,
                                           reinterpret_cast<GM_ADDR>(params.ptrOutput), &tilingData, true);
            kernelMoeTokenUnpermuteOp.Process();
        }
    }

    // dispatch / CombineV1 共用：构建「每 dstEpIdx 块数」的前缀和映射（P2 bug 修复）。
    //
    // 旧实现用【全局平均行数】rowsPerRankEst 推 tokenBlocks，任务网格是 EP × tokenBlocks
    // 的均匀网格；稀疏路由下某 dstEpIdx 的实际行数远超平均时，其尾段行（超出
    // tokenBlocks×rowsPerBlock 的部分）没有任何任务块覆盖，数据整段丢失——这是
    // 4 个 FAIL case 的共同根因（etn_ok=True、cos 0.82~0.98、偶发 NaN、FAIL rank 漂移）。
    //
    // 本函数按【每个 dstEpIdx 的实际行数】计算块数 = CeilDiv(rowsTotal, rowsPerBlock)，
    // 并把任务空间压成跨 rank 连续的序列：rankBlockPrefix[r] = Σ_{i<r} blocks(i)，
    // 返回总任务数 = rankBlockPrefix[EP]。调用方用 taskIdx 经前缀和线性反查
    // (dstEpIdx, blockIdx)，保证每 rank 的全部行被恰好一个块覆盖一次；分块仍是行区间
    // 的有序切分，GM 偏移公式（rowStart / rowSrc / gmOffsetA）因此保持不变。
    //
    // rowsTotal 按 maxOutputSize 剩余容量收敛：rank 的组内起始行 groupBase + prevCumsum
    // >= maxOutputSize 时块数为 0，与旧实现的行级截断守卫语义一致，避免容量溢出时
    // 生成纯跳过的空块。
    //
    // 【性能】每 rank 行数不再单独读 tokenPerExpert，而是取 cumsumMM 列切片的相邻差：
    //   rowsTotal(dstEpIdx) = cumsumCache[dstEpIdx*epr+g] - cumsumCache[(dstEpIdx-1)*epr+g]
    // （cumsumMM 就是 tokenPerExpert 的 rank 维 inclusive 前缀和，相邻差严格等于原始值）。
    // cumsumCache 由 PrefetchCumsumColumn 在 group 循环前一次性向量搬进 UB，
    // 避免每 group 每核在 GM 上做 O(EP) 次标量读（AIC 标量 GM 读延迟高且几乎无流水，
    // 实测 qwen3_235b_ws8 因此回退 +260us）。
    CATLASS_DEVICE
    uint32_t BuildRankBlockPrefix(Params const &params, uint32_t groupIdx, uint32_t rowsPerBlock,
                                  uint64_t groupBase, uint64_t maxOutputSize,
                                  AscendC::LocalTensor<int32_t> const &cumsumCache, uint32_t *rankBlockPrefix)
    {
        const uint32_t ep = static_cast<uint32_t>(params.EP);
        const uint32_t epr = static_cast<uint32_t>(params.expertPerRank);
        uint32_t totalTasks = 0;
        uint32_t prevCumsum = 0;
        rankBlockPrefix[0] = 0;
        for (uint32_t dstEpIdx = 0; dstEpIdx < ep; ++dstEpIdx) {
            uint32_t curCumsum = cumsumCache(dstEpIdx * epr + groupIdx);
            uint32_t rowsTotal = curCumsum - prevCumsum;  // 该 rank 实际行数（cumsum 相邻差）
            uint64_t base = groupBase + prevCumsum;       // 该 rank 组内起始行（= rowStartInGroup）
            if (base >= maxOutputSize) {
                rowsTotal = 0;
            } else {
                uint64_t cap = maxOutputSize - base;
                if (cap < rowsTotal) {
                    rowsTotal = static_cast<uint32_t>(cap);
                }
            }
            totalTasks += CeilDiv(rowsTotal, rowsPerBlock);
            rankBlockPrefix[dstEpIdx + 1] = totalTasks;
            prevCumsum = curCumsum;
        }
        return totalTasks;
    }

    // 把 cumsumMM 列切片（EP×epr 个 int32，本 rank workspace 内连续）向量搬进 UB 缓存，
    // 供 BuildRankBlockPrefix 与 dispatch/CombineV1 的 task 循环从 UB 读。
    // 返回的 LocalTensor 在 DISPATCH_CACHE_UB_OFFSET 起（≤8KB）。
    CATLASS_DEVICE
    AscendC::LocalTensor<int32_t> PrefetchCumsumColumn(Params const &params)
    {
        const uint32_t elems = static_cast<uint32_t>(params.EP) * static_cast<uint32_t>(params.expertPerRank);
        AscendC::LocalTensor<int32_t> cache =
            resource.ubBuf.template GetBufferByByte<int32_t>(DISPATCH_CACHE_UB_OFFSET);
        const uint32_t aligned = AlignUp(elems, 8U);  // DataCopy 要求 32B 对齐（8 个 int32）
        cache.SetSize(static_cast<int64_t>(aligned));
        AscendC::DataCopy(cache, cumsumMM[0], aligned);
        AscendC::PipeBarrier<PIPE_ALL>();
        return cache;
    }

    CATLASS_DEVICE
    // CombineV1: selected when ElementB == int8_t && m * topK > 4096.
    void CombineV1(Params const &params, BlockEpilogue2 &blockEpilogue)
    {
        uint32_t n2 = params.problemShape.k();
        int32_t prevGroupSum2 = 0;
        // P2 修复：每 dstEpIdx 块数前缀和（EP ≤ 128），每个 group 重建一次；
        // cumsumMM 列切片搬进 UB 缓存后逐 group 从 UB 读。
        uint32_t rankBlockPrefix[MAX_DISPATCH_RANKS + 1];
        AscendC::LocalTensor<int32_t> cumsumCache = PrefetchCumsumColumn(params);

        icache_preload(8);
        for (uint32_t t_groupIdx = 0; t_groupIdx < params.expertPerRank; ++t_groupIdx) {
            int32_t flagId = t_groupIdx / CROSS_CORE_FLAG_MAX_SET_COUNT;
            AscendC::CrossCoreWaitFlag<0x2>(flagId);
            AscendC::SyncAll<true>();

            uint32_t groupIdx = t_groupIdx;
            uint32_t groupTokenTotal = cumsumMM((params.EP - 1) * params.expertPerRank + groupIdx);
            if (prevGroupSum2 >= static_cast<int32_t>(params.maxOutputSize)) {
                groupTokenTotal = 0;
            } else if (prevGroupSum2 + groupTokenTotal > params.maxOutputSize) {
                groupTokenTotal = params.maxOutputSize - prevGroupSum2;
            }

            // P2 优化：CombineV1 从「一 AIV 一 rank」改为「rank × token 行块」二维并行，
            // 与 dispatch 同理，让空转 AIV 也参与 combine 搬运；块大小同样自适应，
            // EP 达到核数时退化为不分块，避免碎片化拖低 MTE2。
            // ⚠️ P2 bug 修复：任务块数按【每 dstEpIdx 的实际行数】算（groupTokenTotal
            // 只是跨全部 rank 的总行数），映射方式与 dispatch 一致（BuildRankBlockPrefix
            // + 前缀和反查），保证每 rank 全部行被恰好一个块覆盖一次。
            uint32_t rowsPerRankEst = CeilDiv(groupTokenTotal, static_cast<uint32_t>(params.EP));
            uint32_t rowsPerBlock =
                CalcRowsPerBlock(rowsPerRankEst, static_cast<uint32_t>(params.EP), coreNum);
            uint32_t totalTasks = BuildRankBlockPrefix(params, groupIdx, rowsPerBlock, prevGroupSum2,
                                                       params.maxOutputSize, cumsumCache, rankBlockPrefix);
            for (uint32_t taskIdx = static_cast<uint32_t>(coreIdx); taskIdx < totalTasks;
                 taskIdx += coreNum) {
                uint32_t dstEpIdx = 0;
                while (dstEpIdx + 1 < static_cast<uint32_t>(params.EP) &&
                       rankBlockPrefix[dstEpIdx + 1] <= taskIdx) {
                    ++dstEpIdx;
                }
                uint32_t blockIdx = taskIdx - rankBlockPrefix[dstEpIdx];
                uint32_t prevCumsumDst =
                    dstEpIdx == 0 ? 0 : cumsumCache((dstEpIdx - 1) * params.expertPerRank + groupIdx);
                uint32_t rowStartInGroup = prevCumsumDst + prevGroupSum2;
                uint32_t srcRowOffset = rowStartInGroup + blockIdx * rowsPerBlock;
                if (srcRowOffset < params.maxOutputSize) {
                    uint32_t rowsTotal =
                        cumsumCache(dstEpIdx * params.expertPerRank + groupIdx) - prevCumsumDst;
                    // 该源 rank 的行数可能不足（rowsTotal==0 或 < blockIdx*rowsPerBlock），
                    // 直接跳过该块，避免 rowsTotal - blockIdx*rowsPerBlock 下溢成巨数。
                    if (blockIdx * rowsPerBlock >= rowsTotal) {
                        continue;
                    }
                    uint32_t dataRows = rowsTotal - blockIdx * rowsPerBlock;
                    if (dataRows > rowsPerBlock) {
                        dataRows = rowsPerBlock;
                    }
                    if (srcRowOffset + dataRows > params.maxOutputSize) {
                        dataRows = params.maxOutputSize - srcRowOffset;
                    }
                    __gm__ void *dstPeermemPtr = shmem(peermemInfo.offsetD, dstEpIdx);
                    AscendC::GlobalTensor<ElementD2> gmRemotePeer;
                    gmRemotePeer.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD2 *>(dstPeermemPtr));
                    uint32_t dstRowOffset = preSumBeforeRank(dstEpIdx * params.expertPerRank + groupIdx) +
                                            blockIdx * rowsPerBlock;
                    MatrixCoord offsetC{srcRowOffset, 0};
                    MatrixCoord offsetPeer{dstRowOffset, 0};
                    MatrixCoord shapeC{dataRows, n2};
                    int64_t gmOffsetC = params.layoutD2.GetOffset(offsetC);
                    int64_t gmOffsetPeer = params.layoutD2.GetOffset(offsetPeer);
                    if constexpr (std::is_same_v<ElementA, int8_t>) {
                        blockEpilogue(gmC2[gmOffsetC], shapeC, gmPerTokenScale2[srcRowOffset],
                                      gmRemotePeer[gmOffsetPeer]);
                    }
                }
            }
            prevGroupSum2 += groupTokenTotal;
        }
        blockEpilogue.Finalize();
    }

    CATLASS_DEVICE
    // CombineV2: selected when W4A8 (int4b_t), or ElementB == int8_t && m * topK <= 4096,
    //   or BF16/FP16 (all non-INT8 paths).
    void CombineV2(Params const &params, BlockEpilogue3 &blockEpilogue)
    {
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

            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                currentExpertM = currentExpertM * 2;
            }
            GemmCoord inGroupProblemShape{currentExpertM, n2, k2}; // M N K
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
                uint32_t m_offset = blockCoord.m() * L1TileShape::M;
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
                    if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                        blockEpilogue(gmC2, gmPerTokenScale2, reinterpret_cast<__gm__ float *>(params.ptrBias2),
                                      realTileCoord, realTileShape, groupIdx, preSrcExpertSum * 2, preSumBeforeRank,
                                      params.listLen);
                    } else if constexpr (std::is_same_v<ElementB, int8_t>) {
                        blockEpilogue(gmC2, gmPerTokenScale2, realTileCoord, realTileShape, groupIdx, preSrcExpertSum,
                                      preSumBeforeRank);
                    } else {
                        blockEpilogue(gmC2, realTileCoord, realTileShape, groupIdx, preSrcExpertSum, preSumBeforeRank);
                    }
                    m_offset += m0;
                }
            }

            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                preSrcExpertSum += currentExpertM / 2;
            } else {
                preSrcExpertSum += currentExpertM;
            }
            startCoreIdx = (startCoreIdx + coreLoops) % aicCoreNum;
        }

        if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
            for (; syncLoopIdx < params.expertPerRank; syncLoopIdx++) {
                int32_t flag_id = syncLoopIdx / CROSS_CORE_FLAG_MAX_SET_COUNT;
                AscendC::CrossCoreWaitFlag<0x2>(flag_id);
            }
        }
        blockEpilogue.Finalize();
        CombineWaitFlag();
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
        GM_ADDR ptrA1Int4;
        GM_ADDR ptrA2Int4;
        GM_ADDR ptrSumBeforeRank;
        __gm__ float *ptrSoftFlagBase;

        CATLASS_DEVICE
        WorkspaceInfo()
        {
        }

        CATLASS_DEVICE
        WorkspaceInfo(const Params &params)
        {
            uint64_t workspaceOffset = 0;

            expandedRowIdx = params.ptrWorkspace;
            workspaceOffset += AlignUp(params.problemShape.m(), 256) * params.topK * sizeof(int32_t);

            ptrcumsumMM = params.ptrWorkspace + workspaceOffset;
            workspaceOffset += (params.EP * params.EP * params.expertPerRank) * sizeof(int32_t);

            ptrPerTokenScale = params.ptrWorkspace + workspaceOffset;
            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t> || std::is_same_v<ElementB, int8_t>) {
                workspaceOffset += params.maxOutputSize * sizeof(ElementPerTokenScale);
            }

            ptrPerTokenScale2 = params.ptrWorkspace + workspaceOffset;
            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t> || std::is_same_v<ElementB, int8_t>) {
                workspaceOffset += params.maxOutputSize * sizeof(ElementPerTokenScale);
            }

            // gmm out resues workspace
            ptrC = params.ptrWorkspace + workspaceOffset;
            ptrC2 = params.ptrWorkspace + workspaceOffset;
            workspaceOffset += params.maxOutputSize * params.gmmOutPreRowStride * sizeof(ElementC);

            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                // when W4A8 M
                workspaceOffset += params.maxOutputSize * params.gmmOutPreRowStride * sizeof(ElementC);
            }

            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                // swiglu out and gmm2 prechannel dequant res reuse workspace
                ptrA1Int4 = params.ptrWorkspace + workspaceOffset;
                ptrA2Int4 = params.ptrWorkspace + workspaceOffset;
                // sizeof(ElementA) is 1
                workspaceOffset += params.maxOutputSize * params.gmm1InputExpertSlotWidth;
            } else {
                ptrA = params.ptrWorkspace + workspaceOffset;
                ptrPermutedToken = params.ptrWorkspace + workspaceOffset;
                workspaceOffset += params.maxOutputSize * params.gmm1InputExpertSlotWidth * sizeof(ElementA);
            }

            ptrSumBeforeRank = params.ptrWorkspace + workspaceOffset;
            workspaceOffset += params.EP * AlignUp(params.expertPerRank, 16) * sizeof(int32_t);

            // W4A8 not use
            if constexpr (!std::is_same_v<ElementB, AscendC::int4b_t>) {
                ptrSoftFlagBase = reinterpret_cast<__gm__ float *>(params.ptrWorkspace + workspaceOffset);
                workspaceOffset += params.EP * sizeof(int32_t) * FLAGSTRIDE;
            }
        }
    };

    struct PeermemInfo {
        int64_t offsetA;
        int64_t offsetPeerPerTokenScale;
        int64_t offsetPeerTokenPerExpert;
        int64_t offsetD;

        CATLASS_DEVICE
        PeermemInfo()
        {
        }

        CATLASS_DEVICE
        PeermemInfo(const Params &params, const HcclShmem<false> &shmem)
        {
            // 布局：10MB reserved → A → scale → D → cumsum → flag
            const int64_t segSize = static_cast<int64_t>(shmem.SegmentSize());
            const int64_t EP = params.EP;
            const int64_t E = params.expertPerRank;
            const int64_t M = params.maxOutputSize;
            const int64_t h = params.problemShape.k();
            const int64_t bs = params.problemShape.m();
            const int64_t topK = params.topK;

            constexpr bool RoutingIsQuant =
                std::is_same_v<ElementB, AscendC::int4b_t> || std::is_same_v<ElementB, int8_t>;

            // 尾部：CrossRankSync + tokenPerExpert
            int64_t tailSyncSize = static_cast<int64_t>(shmem.TailReservedSize());
            int64_t tokenPerExpertSize =
                EP * AlignUp(EP * MAX_EXPERTS_PER_RANK, ALIGN_128) * static_cast<int64_t>(sizeof(int32_t));

            // A: dispatch 数据区（量化时含行内 scale）
            int64_t offsetASize = bs * topK * (RoutingIsQuant ? (h + ALIGN_512) : h * sizeof(int16_t));

            // scale: per-token 量化 scale 区（仅量化路径）
            int64_t perTokenScaleSize = RoutingIsQuant ? (bs * topK * static_cast<int64_t>(sizeof(float))) : 0;

            // D: 输出区
            int64_t DSize = bs * topK * h * sizeof(int16_t);

            offsetA = RESERVED_SPACE_SIZE;
            if constexpr (RoutingIsQuant) {
                offsetPeerPerTokenScale = offsetA + offsetASize;
                offsetD = offsetPeerPerTokenScale + perTokenScaleSize;
            } else {
                offsetPeerPerTokenScale = 0;
                offsetD = offsetA + offsetASize;
            }
            // 两个offset倒排并固定大小防止清零的位置与下一个case需要的不相同
            // 这个flag在shmem中计算并使用
            const int64_t offsetFlag = shmem.SegmentSize() - shmem.TailReservedSize();
            offsetPeerTokenPerExpert = offsetFlag - tokenPerExpertSize;
        }
    };

    Arch::Resource<ArchTag> resource;

    uint32_t coreIdx;
    uint32_t coreNum;
    uint32_t nSyncSwiglu;

    Params params;
    WorkspaceInfo workspaceInfo;
    PeermemInfo peermemInfo;

    uint32_t dequantSum[kMaxDequantSyncGroups] = {0};

    // ========== Common tensors (all types) ==========
    AscendC::GlobalTensor<ElementC> gmC;
    AscendC::GlobalTensor<ElementC> gmC2;
    AscendC::GlobalTensor<ElementPerTokenScale> gmPerTokenScale1;
    AscendC::GlobalTensor<ElementPerTokenScale> gmPerTokenScale2;
    AscendC::GlobalTensor<bool> gmXActiveMask;
    AscendC::GlobalTensor<int32_t> tokenPerExpert;
    AscendC::GlobalTensor<int32_t> cumsumMM;
    AscendC::GlobalTensor<int32_t> preSumBeforeRank;
    Layout3D tokenPerExpertLayout;
    HcclShmem<false> shmem;
    int32_t paddedExpertNumAligned;
    bool isCombineV1;
    // ExceptionDump引擎：记录执行阶段时间戳，并提供Dump接口由host侧dump指定GM地址内容。
    // 基址取通信域首地址（shmem()()），根据kRoutingIsQuant选择对应tiling结构体的Policy，
    // ArchTag传入Policy供架构差异扩展。
    static constexpr bool kRoutingIsQuant =
        std::is_same_v<ElementB, AscendC::int4b_t> || std::is_same_v<ElementB, int8_t>;
    MC2MegaMoeAdump::ExceptionDumpEngine<kRoutingIsQuant, ArchTag> exceptionDump_;

    AscendC::GlobalTensor<ElementScale> gmS;
    AscendC::GlobalTensor<ElementScale> gmS2;
    AscendC::GlobalTensor<int4b_t> gmA1I4;
    AscendC::GlobalTensor<int8_t> gmA1I4_I8;
    AscendC::GlobalTensor<int4b_t> gmA2I4;
    AscendC::GlobalTensor<int8_t> gmA2I4_I8;

    AscendC::GlobalTensor<ElementABefore> gmA;
    AscendC::GlobalTensor<ElementD1> gmPermutedToken;
};
} // namespace Catlass::Gemm::Kernel
#endif // MEGA_MOE_KERNEL_HPP
