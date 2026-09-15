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
 * \file quant_matmul_reduce_scatter_urma_impl.h
 * \brief quantMatmulReduceScatter — PUT 模式实现 + FragmentTensor 消除 R 循环
 *
 * 参考 allgather 的 FragmentTensor 设计思想:
 *   - A 数据全在本卡 GM 连续 [m,k], 各 rank 段打包为 FragmentTensor
 *   - BlockMmadFragC 在 tile 跨越 rank 边界时自动切换 fragment, 保持 per-fragment L1 隔离
 *   - 一次 matmul 调用覆盖所有 rank 段, 无 R 循环
 *   - per-tile 流水: AIC 逐 tile SetFlag, AIV 逐 tile WaitFlag + AllToAll + reduceSum
 *
 * 架构:
 *   AIC: for tileIdx = 0..commTurn-1:
 *          构建 FragmentTensor (rankSize 个 fragment, 每个 curTileM 行, 间距 mPerRank 行)
 *          matmul(rankSize * curTileM) → workspace
 *          remote→local 边界 SetFlag(FLAG_ID_REMOTE_DONE) 提前通知 AIV 启动 AllToAll (与本地段计算重叠)
 *          本 tile 全部完成后 SetFlag(FLAG_ID_ALL_DONE)
 *   AIV: for tileIdx = 0..commTurn-1:
 *          WaitFlag(FLAG_ID_REMOTE_DONE) → SyncAll → 通信核 AllToAll(tileIdx) ‖ 计算核 ReduceSum(tileIdx-1)
 *          → WaitFlag(FLAG_ID_ALL_DONE) → SyncAll → CrossDevice → SyncAll; 末 tile ReduceSum 循环后补做
 *
 * 关键设计:
 *   - 无 R 循环: FragmentTensor 打包 rankSize 个 fragment, 一次调用覆盖一个 tile 的全部 rank 段
 *   - AIC 侧无 if/else: 统一 for t 循环, T=1 自然退化 (AIV 侧有 commTurn<=1 单 tile 快速路径)
 *   - L1 局部性: BlockMmadFragC per-fragment 独立加载 L1 (与 R 循环一致)
 *   - flag 计数式: 每 tile 一次 Set(1)/Set(0), AIV 按序对应 Wait(1)/Wait(0)
 */

#pragma once

#include "kernel_operator.h"
#include "include/tensor_api/tensor.h"

#include "adv_api/hcomm/hcomm.h"
#include "blaze/gemm/utils/common_utils.h"

#include "apace/core/aiv_comm/collective_comm_api.h"
#include "apace/core/aiv_comm/collective_comm_context.h"
#include "apace/tiling/comm_tiling_data.h"
#include "apace/core/aiv_comm/barrier/barrier_ubmem.h"

#include "quant_matmul_reduce_scatter_urma_tiling_data.h"
#include "apace/kernel/matmul/quant_batch_matmul/reduce_scatter_qbmm_mx_kernel.h"
#include "reduce_sum_ref.h"

namespace Apace {

using namespace AscendC;
using namespace Blaze::Gemm;
using AscendC::Te::Get;
using namespace Apace::AivComm;

/**
 * @brief quantMatmulReduceScatter 核心实现类 (PUT 模式)
 *
 * 计算在前(AIC) + PUT 通信(AIV) + reduceSum(AIV)
 */
template <typename AType, typename BType, typename CType, bool TransA, bool TransB>
class QuantMatmulReduceScatterImpl {
public:
    __aicore__ inline QuantMatmulReduceScatterImpl(){};
    __aicore__ inline ~QuantMatmulReduceScatterImpl() {}

    __aicore__ inline void Init(__gm__ CommContext *hcommCtx, GM_ADDR aGM, GM_ADDR scaleAGM, GM_ADDR bGM,
                                GM_ADDR scaleBGM, GM_ADDR workspaceGM, GM_ADDR cGM,
                                const QuantMatmulReduceScatterTilingData *tilingData);

    __aicore__ inline void Run();

    using TypeA = AType;
    using TypeB = BType;
    using TypeC = CType;
    using TypeScaleA = ::fp8_e8m0_t;
    using TypeScaleB = ::fp8_e8m0_t;
    static constexpr int32_t SCALE_C0 = 2;

    // Layout 定义
    using LayoutA =
        typename AscendC::Std::conditional_t<TransA, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
    using LayoutB =
        typename AscendC::Std::conditional_t<TransB, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutScaleA = typename AscendC::Te::FrameLayoutFormat<
        AscendC::Std::conditional_t<TransA, AscendC::Te::ScaleADNLayoutPtn, AscendC::Te::ScaleANDLayoutPtn>,
        AscendC::Std::Int<SCALE_C0>>;
    using LayoutScaleB = typename AscendC::Te::FrameLayoutFormat<
        AscendC::Std::conditional_t<TransB, AscendC::Te::ScaleBDNLayoutPtn, AscendC::Te::ScaleBNDLayoutPtn>,
        AscendC::Std::Int<SCALE_C0>>;

    // 组件定义 — 使用 FragmentTensor 版 kernel (参考 allgather QmmMxKernelAgUdma)
    using FragKernelImpl = ReduceScatterFragImpl::QmmMxKernelRsFrag<AType, BType, CType>;
    using FragKernelParams = typename FragKernelImpl::Params;
    using FragFragmentParams = typename FragKernelImpl::FragmentParams;

    FragKernelImpl fragKernelImpl_;

private:
    // ---------------- 通信相关组件 ----------------
    __gm__ CommUbmemContext *syncBuffer_{nullptr};
    __gm__ CommUdmaContext *udmaCtx_{nullptr};
    // PUT 模式: AIC 写 C_i 到 workspaceGM, AIV PUT 推送到其他 rank win 区
    CollectiveComm<CommCollectiveOp::AllToAll, CommMode::PUT, CType, TeamBarrier> allToAllC_;
    TeamBarrier teamBarrier_;

    struct BaseParams {
        GM_ADDR selfWinAddr{nullptr}; // 本卡通信窗口地址 (AIC 写入 C_i 的目标)
        GM_ADDR aGm{nullptr};
        GM_ADDR scaleAGm{nullptr};
        GM_ADDR bGm{nullptr};
        GM_ADDR scaleBGm{nullptr};
        GM_ADDR workspaceGm{nullptr}; // workspace GM (AIC 写入 C_i, AIV PUT 源数据)
        GM_ADDR cGm{nullptr};         // 最终输出 GM

        uint32_t rankId{0};
        int32_t commTurn{0}; // 总通信轮次
        uint32_t rankSize{0};
        uint64_t axisN{0};          // N 轴大小
        uint64_t axisK{0};          // K 轴大小
        uint64_t headMSize{0};      // 通信头块 M 大小 (= chunkM for single tile)
        uint64_t tailMSize{0};      // 通信尾块 M 大小 (0 if no tail)
        uint32_t headTileCnt{0};    // 头块数量
        uint32_t chunkM{0};         // 每个 rank 的 M 分片大小 (m / rankSize)
        uint64_t scaleARowBytes{0}; // ScaleA 每行字节数 = CeilDiv(axisK, 64) * 2 (MXFP8)
    } baseParams_;

    // ---------------- reduceSum 组件 ----------------
    AiVReduceSumImpl::ReduceSumForAlltoAll<CType, float> reduceSum_;

    const QuantMatmulReduceScatterTilingData *tilingData_{nullptr};

    // ---------------- AIC→AIV CrossCore flag 常量 ----------------
    // modeId 与 reduce_scatter_qbmm_mx_kernel.h 的 CrossCoreSetFlag<0x2, PIPE_FIX> 保持一致
    constexpr static uint8_t AIC_NOTIFY_MODE = 0x2;
    // AIC 在 remote→local 边界 Set(1) (配对 reduce_scatter_qbmm_mx_kernel.h 内 SetFlag(1))
    constexpr static uint16_t FLAG_ID_REMOTE_DONE = 1;
    // AIC 本 tile matmul 全部完成 Set(0) (配对 RunMatmul 尾部 SetFlag(0))
    constexpr static uint16_t FLAG_ID_ALL_DONE = 0;
    // RunOneTileTurn 哨兵: 本轮计算核不做 reduceSum (tile 0 预热 / 单 tile 路径)
    constexpr static int32_t NO_REDUCE_TILE = -1;

    // AIV 流水上下文: 角色分类 + reduceSum 形状派生 (RunAllToAllAndReduceSum 使用)
    struct AivRunContext {
        uint32_t commTurn;   // 通信轮次 (= tile 总数)
        uint32_t rsCoreNum;  // 计算核数 (前 rsCoreNum 个 block)
        uint32_t nSize;      // N 轴元素数
        uint32_t rowBytes;   // 每行字节数
        uint64_t rowStride;  // 相邻 rank 部分和行间距 (元素数)
        bool isCommBlock;    // 后 rankSize 核: AllToAll 通信
        bool isComputeBlock; // 前 rsCoreNum 核: reduceSum
    };

    // ---------------- 私有方法 ----------------
    __aicore__ inline void InitBaseParams(const QuantMatmulReduceScatterTilingData *tilingData);
    __aicore__ inline void RunMatmul();               // AIC: per-tile matmul → workspaceGM
    __aicore__ inline void RunAllToAllAndReduceSum(); // AIV: per-tile WaitFlag + AllToAll PUT + reduceSum
    __aicore__ inline void InitAivRunContext(AivRunContext &ctx); // AIV: 角色判定 + 形状派生
    __aicore__ inline void RunOneTileTurn(const AivRunContext &ctx,
                                          int32_t reduceTileIdx); // AIV: 一轮通算重叠步
    __aicore__ inline void ReduceSumTile(const AivRunContext &ctx, uint32_t tileIdx); // AIV: 单 tile reduceSum
    __aicore__ inline uint32_t GetTileM(uint32_t tileIdx);       // 获取 tile 的 M 大小
    __aicore__ inline uint64_t GetTileMOffset(uint32_t tileIdx); // 获取 tile 的 M 偏移
    __aicore__ inline void DoReduceSum(uint32_t tileM, uint64_t tileMOffset, uint32_t nSize, uint32_t rowBytes,
                                       uint64_t stride, uint32_t rsCoreNum);
};

// =================================================================================
// 公有方法实现
// =================================================================================

template <typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void QuantMatmulReduceScatterImpl<AType, BType, CType, TransA, TransB>::Init(
    __gm__ CommContext *hcommCtx, GM_ADDR aGM, GM_ADDR scaleAGM, GM_ADDR bGM, GM_ADDR scaleBGM, GM_ADDR workspaceGM,
    GM_ADDR cGM, const QuantMatmulReduceScatterTilingData *tilingData)
{
    tilingData_ = tilingData;
    baseParams_.aGm = aGM;
    baseParams_.scaleAGm = scaleAGM;
    baseParams_.bGm = bGM;
    baseParams_.scaleBGm = scaleBGM;
    baseParams_.workspaceGm = workspaceGM;
    baseParams_.cGm = cGM;

    syncBuffer_ = &(hcommCtx->ubmemCtx);
    udmaCtx_ = &(hcommCtx->udmaCtx);
    baseParams_.rankId = udmaCtx_->rankId;
    baseParams_.rankSize = udmaCtx_->rankSize;

    InitBaseParams(tilingData);

    // selfWinAddr: 本 rank 的 win 区数据起始地址 (跳过 barrier counter 区域)
    // Barrier 布局 (totalJobs=1): localFlag(32) + teamSyncCounter(32) + crossCoreFlag(32) = 96B
    // 对齐到 128B, PUT 数据从 offset 128 开始写入, 不覆盖 counter
    constexpr uint64_t WIN_DATA_OFFSET = 128;
    baseParams_.selfWinAddr =
        reinterpret_cast<GM_ADDR>(udmaCtx_->commBufferAddrs[baseParams_.rankId]) + WIN_DATA_OFFSET;

    // 多核并行通信: 后 rankSize 个核做 AllToAll PUT
    // teamBarrier: totalJobs=1 → jobIndex=0 (最后一个 block) 的核执行 CrossDevice, step=1 轮询所有 remote
    // AllToAll: totalJobs=rankSize → 后 R 个核各负责 1 个 target, 并行 PUT
    // jobIndex = BlockNum - 1 - blockIdx → 最后一个 block jobIndex=0, 倒数第二个 jobIndex=1
    uint32_t rankSize = baseParams_.rankSize;
    uint32_t jobIndex = static_cast<uint32_t>(GetBlockNum() - 1 - GetBlockIdx());

    // UB 分配: commBuf + barrierBuf
    uint32_t ubOffset = 0;
    auto commBuf = AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, uint8_t>(ubOffset);
    ubOffset += COMM_WORKSPACE_SIZE;
    auto barrierBuf = AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, uint8_t>(ubOffset);
    // teamBarrier totalJobs=1: 仅 jobIndex=0 的核执行 CrossDevice (step=1, 轮询所有 remote)
    teamBarrier_.Init(barrierBuf.Get(), syncBuffer_, 1, jobIndex);

    allToAllC_.template Init<BARRIER_NONE>(udmaCtx_, teamBarrier_, tilingData->commTilingData, baseParams_.workspaceGm,
                                           commBuf.Get(), rankSize, jobIndex, WIN_DATA_OFFSET);
}

template <typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void QuantMatmulReduceScatterImpl<AType, BType, CType, TransA, TransB>::Run()
{
    if ASCEND_IS_AIC {
        // AIC 侧: 逐 tile matmul → workspace + SetFlag
        RunMatmul();
    }

    if ASCEND_IS_AIV {
        // AIV 侧: per-tile WaitFlag + AllToAll PUT + reduceSum
        RunAllToAllAndReduceSum();
    }
}

// =================================================================================
// 参数初始化
// =================================================================================

template <typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void QuantMatmulReduceScatterImpl<AType, BType, CType, TransA, TransB>::InitBaseParams(
    const QuantMatmulReduceScatterTilingData *tilingData)
{
    baseParams_.headMSize = tilingData->commTilingData.splitAxisTileSize;
    baseParams_.tailMSize = tilingData->commTilingData.splitAxisTailSize;
    baseParams_.headTileCnt = static_cast<uint32_t>(tilingData->commTilingData.splitAxisTileCnt);
    baseParams_.commTurn = tilingData->commTilingData.splitAxisTileCnt + tilingData->commTilingData.splitAxisTailCnt;

    baseParams_.axisN = tilingData->mmTilingData.n;
    baseParams_.axisK = tilingData->mmTilingData.k;

    // scaleARowBytes = CeilDiv(axisK, 64) * 2 (MXFP8 scale: 每组 64 元素, 2 字节)
    baseParams_.scaleARowBytes = (baseParams_.axisK + 63U) / 64U * 2U;

    // chunkSize = 每个 rank 分到的 M 行数 = m / rankSize
    uint64_t chunkSize = tilingData->commTilingData.splitAxisTileSize * tilingData->commTilingData.splitAxisTileCnt +
                         tilingData->commTilingData.splitAxisTailSize * tilingData->commTilingData.splitAxisTailCnt;
    baseParams_.chunkM = static_cast<uint32_t>(chunkSize);
}

// =================================================================================
// AIC 侧: FragmentTensor matmul → workspaceGM (统一路径, 无 if/else, 无 R 循环)
// =================================================================================

template <typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void QuantMatmulReduceScatterImpl<AType, BType, CType, TransA, TransB>::RunMatmul()
{
    // FragmentTensor 方案: 每 tile 一次 matmul 调用覆盖该 tile 的所有 rank 段
    //   totalLogicalM = rankSize * curTileM (headRows 由本循环按 tile 设置)
    //   FragmentTensor 打包各 rank 段地址, BlockMmadFragC per-fragment L1 隔离
    //   无 rank 循环, 无 if/else — 统一 for tileIdx 循环, T=1 自然退化
    const auto &mmT = tilingData_->mmTilingData;
    uint32_t commTurn = static_cast<uint32_t>(baseParams_.commTurn);
    uint32_t rankSize = baseParams_.rankSize;

    for (uint32_t tileIdx = 0; tileIdx < commTurn; ++tileIdx) {
        uint32_t curTileM = GetTileM(tileIdx);
        uint64_t tileMOffset = GetTileMOffset(tileIdx);

        const auto &subMmT = tilingData_->subMmTilingData;
        const auto &tailMmT = tilingData_->tailMmTilingData;
        bool isHeadTile = (tileIdx < baseParams_.headTileCnt);
        const auto &tileT = (commTurn == 1) ? mmT : (isHeadTile ? subMmT : tailMmT);

        // localMatmul 后置: FragmentTensor localLast=true, fragment 顺序 [remote..., local]
        // Process 在 remote→local 边界自动 SetFlag(1), 此处 SetFlag(0) 为 all done
        FragFragmentParams fragParams;
        fragParams.tileCnt = 0;
        fragParams.tileM = curTileM;
        fragParams.tailCnt = 0;
        fragParams.tailM = 0;
        fragParams.paddedTailM = 0;
        fragParams.commTurn = 1;
        fragParams.headRows = rankSize * curTileM;
        fragParams.rankId = baseParams_.rankId;
        fragParams.rankSize = rankSize;
        fragParams.mPerRank = baseParams_.chunkM;
        fragParams.k = baseParams_.axisK;
        fragParams.n = baseParams_.axisN;
        fragParams.scaleKLen = (baseParams_.axisK + 63U) / 64U * 2U;
        fragParams.localLast = true;
        fragParams.fragCount = 0;

        FragKernelParams params;
        params.mmTile = &tileT;
        params.qbmmParams = {tileT.baseM, tileT.baseN, tileT.baseK, tileT.dbL0c};
        params.fragParams = fragParams;
        params.aGM = baseParams_.aGm + tileMOffset * baseParams_.axisK;
        params.aScaleGM = baseParams_.scaleAGm + tileMOffset * baseParams_.scaleARowBytes;
        params.bGM = baseParams_.bGm;
        params.bScaleGM = baseParams_.scaleBGm;
        params.cGM = baseParams_.workspaceGm + tileMOffset * baseParams_.axisN * sizeof(CType);
        params.dataBytesPerMRow = baseParams_.axisK * sizeof(AType);
        params.scaleBytesPerMRow = baseParams_.scaleARowBytes;
        params.cBytesPerM = baseParams_.axisN * sizeof(CType);

        fragKernelImpl_(params);

        CrossCoreSetFlag<AIC_NOTIFY_MODE, PIPE_FIX>(FLAG_ID_ALL_DONE); // all done (remote+local)
    }
}

// =================================================================================
// AIV 侧: per-tile WaitFlag + AllToAll PUT + reduceSum (流水掩盖)
// =================================================================================

template <typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void QuantMatmulReduceScatterImpl<AType, BType, CType, TransA, TransB>::RunAllToAllAndReduceSum()
{
    // 严格分离 — 后 rankSize 核仅通信, 前 rsCoreNum 核仅计算, 其余核仅参与 SyncAll:
    //   通信核: AllToAll(tileIdx)
    //   计算核: ReduceSum(tileIdx-1) ‖ 通信核 AllToAll(tileIdx); 末 tile 循环后补做
    //   通信核不补做 reduceSum
    AivRunContext ctx;
    InitAivRunContext(ctx);

    if (ctx.commTurn <= 1) {
        // 单 tile: 无流水可掩盖, 本轮通信完成后计算核再做 reduceSum
        RunOneTileTurn(ctx, NO_REDUCE_TILE);
        if (ctx.isComputeBlock) {
            ReduceSumTile(ctx, 0);
        }
        if (ctx.isCommBlock) {
            allToAllC_.template Finalize<BARRIER_NONE>();
        }
        return;
    }

    // 多 tile 流水: tile 0 预热 (仅通信), 之后每轮计算核 reduceSum(tileIdx-1) ‖ 通信核 AllToAll(tileIdx)
    RunOneTileTurn(ctx, NO_REDUCE_TILE);
    for (uint32_t tileIdx = 1; tileIdx < ctx.commTurn; ++tileIdx) {
        RunOneTileTurn(ctx, static_cast<int32_t>(tileIdx - 1));
    }

    // 末 tile: AllToAll 全部完成后, 仅计算核补做 reduceSum
    SyncAll<true>();
    if (ctx.isComputeBlock) {
        ReduceSumTile(ctx, ctx.commTurn - 1);
    }
    if (ctx.isCommBlock) {
        allToAllC_.template Finalize<BARRIER_NONE>();
    }
}

// =================================================================================
// AIV 侧辅助: InitAivRunContext / RunOneTileTurn / ReduceSumTile
// =================================================================================

// AIV 流水上下文初始化: 角色判定 (通信核/计算核/多余核) + reduceSum 形状派生
template <typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void QuantMatmulReduceScatterImpl<AType, BType, CType, TransA, TransB>::InitAivRunContext(
    AivRunContext &ctx)
{
    ctx.commTurn = static_cast<uint32_t>(baseParams_.commTurn);
    ctx.nSize = static_cast<uint32_t>(baseParams_.axisN);
    ctx.rowBytes = ctx.nSize * sizeof(CType);
    ctx.rowStride = static_cast<uint64_t>(baseParams_.chunkM) * ctx.nSize;
    ctx.rsCoreNum = tilingData_->mmTilingData.usedCoreNum - baseParams_.rankSize; // 计算核数 (前 rsCoreNum 个)

    uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
    uint32_t jobIndex = static_cast<uint32_t>(GetBlockNum() - 1 - blockIdx);
    ctx.isCommBlock = (jobIndex < baseParams_.rankSize); // 后 rankSize 核: AllToAll 通信
    ctx.isComputeBlock = (blockIdx < ctx.rsCoreNum);     // 前 rsCoreNum 核: reduceSum
}

// 一轮通算重叠步 (单 tile / 预热 / 流水轮次的公共序列):
//   WaitFlag(REMOTE_DONE) → SyncAll#1 → 通信核 AllToAll+WaitFlag(ALL_DONE) ‖ 计算核
//   WaitFlag(ALL_DONE)+ReduceSum(reduceTileIdx) → SyncAll#2 → CrossDevice → SyncAll#3
//   reduceTileIdx == NO_REDUCE_TILE 时本轮计算核不做 reduceSum (tile 0 预热 / 单 tile 路径)
template <typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void QuantMatmulReduceScatterImpl<AType, BType, CType, TransA, TransB>::RunOneTileTurn(
    const AivRunContext &ctx, int32_t reduceTileIdx)
{
    // localMatmul 后置: AIC 在 remote→local 边界 SetFlag(FLAG_ID_REMOTE_DONE), 本 tile 全部完成后
    // SetFlag(FLAG_ID_ALL_DONE); AIV 双角色核按序对应 Wait
    if (ctx.isCommBlock) {
        CrossCoreWaitFlag<AIC_NOTIFY_MODE, PIPE_MTE2>(FLAG_ID_REMOTE_DONE); // remote done on this core
    }
    if (ctx.isComputeBlock) {
        CrossCoreWaitFlag<AIC_NOTIFY_MODE, PIPE_MTE2>(FLAG_ID_REMOTE_DONE); // consume remote flag
    }
    SyncAll<true>(); // #1: all cores' remote done

    if (ctx.isCommBlock) {
        allToAllC_.template Commit<BARRIER_NONE>();
        allToAllC_.template Wait<BARRIER_NONE>();
        CrossCoreWaitFlag<AIC_NOTIFY_MODE, PIPE_MTE2>(FLAG_ID_ALL_DONE); // consume local flag
    }
    if (ctx.isComputeBlock) {
        CrossCoreWaitFlag<AIC_NOTIFY_MODE, PIPE_MTE2>(FLAG_ID_ALL_DONE); // all done (local included)
        if (reduceTileIdx != NO_REDUCE_TILE) {
            ReduceSumTile(ctx, static_cast<uint32_t>(reduceTileIdx)); // 与通信核 AllToAll 重叠执行
        }
    }
    SyncAll<true>(); // #2: AllToAll done / all matmul done
    teamBarrier_.CrossDevice();
    SyncAll<true>(); // #3: remote data visible
}

// 单 tile reduceSum: 计算核将 tileIdx 行区间的 rankSize 份部分和归约写入输出区
template <typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void QuantMatmulReduceScatterImpl<AType, BType, CType, TransA, TransB>::ReduceSumTile(
    const AivRunContext &ctx, uint32_t tileIdx)
{
    uint32_t tileM = GetTileM(tileIdx);
    if (tileM == 0) {
        return;
    }
    uint64_t tileMOffset = GetTileMOffset(tileIdx);
    DoReduceSum(tileM, tileMOffset, ctx.nSize, ctx.rowBytes, ctx.rowStride, ctx.rsCoreNum);
}

// =================================================================================
// 辅助方法: GetTileM / GetTileMOffset / DoReduceSum
// =================================================================================

template <typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline uint32_t QuantMatmulReduceScatterImpl<AType, BType, CType, TransA, TransB>::GetTileM(uint32_t tileIdx)
{
    if (tileIdx < baseParams_.headTileCnt) {
        return static_cast<uint32_t>(baseParams_.headMSize);
    }
    return static_cast<uint32_t>(baseParams_.tailMSize);
}

template <typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline uint64_t QuantMatmulReduceScatterImpl<AType, BType, CType, TransA, TransB>::GetTileMOffset(
    uint32_t tileIdx)
{
    if (tileIdx < baseParams_.headTileCnt) {
        return static_cast<uint64_t>(tileIdx) * baseParams_.headMSize;
    }
    return static_cast<uint64_t>(baseParams_.headTileCnt) * baseParams_.headMSize;
}

template <typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void QuantMatmulReduceScatterImpl<AType, BType, CType, TransA, TransB>::DoReduceSum(
    uint32_t tileM, uint64_t tileMOffset, uint32_t nSize, uint32_t rowBytes, uint64_t stride, uint32_t rsCoreNum)
{
    using ReduceSumType = AiVReduceSumImpl::ReduceSumForAlltoAll<CType, float>;
    typename ReduceSumType::RsParams rsParams;
    rsParams.recvWindowAddr = baseParams_.selfWinAddr + tileMOffset * rowBytes;  // 远端 rank 部分和窗口
    rsParams.selfPartialAddr = baseParams_.workspaceGm + tileMOffset * rowBytes; // 本 rank 局部 matmul 输出
    rsParams.outputAddr = baseParams_.cGm + tileMOffset * rowBytes;
    rsParams.rankDim = baseParams_.rankSize;
    rsParams.rankId = baseParams_.rankId;
    rsParams.aivNum = rsCoreNum;
    rsParams.rowStride = stride;
    rsParams.nSize = nSize;
    reduceSum_.Init(rsParams);
    reduceSum_.ExecuteReduceSum(tileM);
}

} // namespace Apace
