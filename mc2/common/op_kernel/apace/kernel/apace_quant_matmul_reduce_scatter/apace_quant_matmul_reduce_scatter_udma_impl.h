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
 * \file quant_matmul_reduce_scatter_impl.h
 * \brief quantMatmulReduceScatter — PUT 模式实现 (计算→通信) + 方案 D per-tile 流水重叠
 *        + Round 4 子问题独立 tiling 优化
 *
 * 架构（PUT 模式 = 计算→通信，方案 D per-tile 流水，Round 4 子问题 tiling）：
 *   AIC: RunMatmul()  per-tile 循环: T×R 次子调用, 每次算一个 tile 的 M 范围
 *                        每轮 T 完成后 CrossCoreSetFlag<0x2, PIPE_FIX>(0) 通知 AIV
 *   AIV: RunAllToAllAndReduceSum()  per-tile 循环:
 *                        CrossCoreWaitFlag<0x2, PIPE_MTE2>(0) 等待 AIC tile t
 *                        AllToAll PUT: 推送 tile t 到所有目标 rank
 *                        reduceSum: 对 tile t 的 R 个源求和 → output
 *
 * 流水重叠:
 *   AIC: [tile0 R段mm] SetFlag [tile1 R段mm] SetFlag [tile2 R段mm] SetFlag ...
 *   AIV:                WaitFlag [comm0+rs0]  WaitFlag [comm1+rs1]  WaitFlag [comm2+rs2]
 *
 * 关键设计:
 *   - T=1 退化路径: 保持当前行为 (单次全量 mm + 单次 SetFlag + 单次 WaitFlag)
 *   - T>1 per-tile 流水路径: T×R 次子调用, flag 计数式 (T 次 Set ↔ T 次 Wait)
 *   - CrossCore flag flagId=0 计数式: T 次 Set(0) 对应 T 次 Wait(0), 峰值 T ≤ 15
 *   - 零 tile 核: SetFlag 在 R-loop 外, 所有 AIC 核都执行 (flag 计数平衡)
 *   - PUT 模式: AIC 写 workspace, AIV 从 workspace 推送到其他 rank win 区
 *   - reduceSum: self 从 workspace 读, received 从 win 读, per-tile 分块处理
 *   - 所有 AIC 核参与 matmul (BlockScheduler 分配), 仅 block 0 的 AIV 参与通信
 *   - AIC 不参与 SyncAll (SyncAll<true> isAIVOnly=true)
 */

#pragma once

#include "basic_api/kernel_basic_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "include/tensor_api/tensor.h"

#include "adv_api/hcomm/hcomm.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "apace/kernel/all_to_all_quant_matmul/quant_matmul_mx_kernel.h"

#include "apace/block/aiv_comm/collective_comm_api.h"
#include "apace/block/aiv_comm/collective_comm_context.h"
#include "apace/tiling/comm_tiling_data.h"
#include "apace/block/aiv_comm/barrier/barrier_ubmem.h"

#include "apace_quant_matmul_reduce_scatter_tiling_data.h"
#include "reduce_sum.h"

namespace Apace {

using namespace AscendC;
using namespace Blaze::Gemm;
using AscendC::Te::Get;
using namespace Apace::AivComm;

using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

/**
 * @brief quantMatmulReduceScatter 核心实现类 (GET 模式)
 *
 * 计算在前(AIC) + GET 通信(AIV) + reduceSum(AIV)
 */
template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
class ApaceQuantMatmulReduceScatterUdmaImpl {
public:
    __aicore__ inline ApaceQuantMatmulReduceScatterUdmaImpl() {};
    __aicore__ inline ~ApaceQuantMatmulReduceScatterUdmaImpl() {}

    __aicore__ inline void Init(__gm__ CommContext *hcommCtx,
                     GM_ADDR aGM, GM_ADDR scaleAGM,
                     GM_ADDR bGM, GM_ADDR scaleBGM,
                     GM_ADDR workspaceGM,
                     GM_ADDR cGM,
                     const ApaceQuantMatmulReduceScatterTilingData *tilingData);

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
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;
    using BiasType = float;
    using LayoutScaleA = typename AscendC::Te::FrameLayoutFormat<
        AscendC::Std::conditional_t<TransA, AscendC::Te::ScaleADNLayoutPtn, AscendC::Te::ScaleANDLayoutPtn>,
        AscendC::Std::Int<SCALE_C0>>;
    using LayoutScaleB = typename AscendC::Te::FrameLayoutFormat<
        AscendC::Std::conditional_t<TransB, AscendC::Te::ScaleBDNLayoutPtn, AscendC::Te::ScaleBNDLayoutPtn>,
        AscendC::Std::Int<SCALE_C0>>;

    // 组件定义 — 与 all_to_all_quant_matmul 保持一致
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape,
        NONE_FULL_LOAD_MODE, LayoutA, LayoutB, TypeA>;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<NONE_FULL_LOAD_MODE, false>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, TypeA, LayoutA, TypeB, LayoutB, TypeC, LayoutC, BiasType, LayoutBias>;
    // CommPolicy: LOCAL 模式不调用 WaitTile, 用 no-op policy
    struct NoOpCommWaitPolicy {
        __aicore__ inline void WaitTile(uint32_t) {}
    };
    using QuantMatmulKernelImpl = Kernel::QuantMatmulMxKernel<ProblemShape, BlockMmad, BlockScheduler,
        NoOpCommWaitPolicy>;

    // 参数类型
    using Params = typename QuantMatmulKernelImpl::Params;
    using BlockMmadParams = typename QuantMatmulKernelImpl::BlockMmadParams;
    using L1Params = typename QuantMatmulKernelImpl::L1Params;
    using LocalParams = typename QuantMatmulKernelImpl::LocalParams;
    using BlockSchedulerParams = typename QuantMatmulKernelImpl::BlockSchedulerParams;
    using QBMMTiling = typename QuantMatmulKernelImpl::QBMMTiling;
    using MatmulMode = typename QuantMatmulKernelImpl::MatmulMode;

    QuantMatmulKernelImpl quantMatmulKernelImpl_;

private:
    // ---------------- 通信相关组件 ----------------
    __gm__ CommUbmemContext* syncBuffer_{nullptr};
    __gm__ CommUdmaContext* udmaCtx_{nullptr};
    // PUT 模式: AIC 写 C_i 到 workspaceGM, AIV PUT 推送到其他 rank win 区
    // (GET 模式在 4+ rank 有数据可见性问题, 回退到 PUT)
    CollectiveComm<CommCollectiveOp::AllToAll, CommMode::PUT, CType, TeamBarrier> allToAllC_;
    TeamBarrier teamBarrier_;

    struct BaseParams {
        GM_ADDR selfWinAddr{nullptr};   // 本卡通信窗口地址 (AIC 写入 C_i 的目标)
        GM_ADDR aGm{nullptr};
        GM_ADDR scaleAGm{nullptr};
        GM_ADDR bGm{nullptr};
        GM_ADDR scaleBGm{nullptr};
        GM_ADDR workspaceGm{nullptr};   // workspace GM (AllToAll GET 接收缓冲区)
        GM_ADDR cGm{nullptr};           // 最终输出 GM

        uint32_t rankId{0};
        int32_t  commTurn{0};          // 总通信轮次
        uint32_t rankSize{0};
        uint64_t axisM{0};             // M 轴总大小
        uint64_t axisN{0};             // N 轴大小
        uint64_t axisK{0};             // K 轴大小
        uint64_t headMSize{0};         // 通信头块 M 大小 (= chunkM for single tile)
        uint64_t tailMSize{0};         // 通信尾块 M 大小 (0 if no tail)
        uint32_t headTileCnt{0};       // 头块数量
        uint32_t tailTileCnt{0};       // 尾块数量
        uint64_t chunkBytes{0};        // 每个 rank 的 C 数据块字节数
        uint32_t chunkM{0};            // 每个 rank 的 M 分片大小 (m / rankSize)
        uint64_t scaleARowBytes{0};    // ScaleA 每行字节数 = CeilDiv(axisK, 64) * 2 (MXFP8)
    } baseParams_;

    const ApaceQuantMatmulReduceScatterTilingData* tilingData_{nullptr};

    // ---------------- 私有方法 ----------------
    __aicore__ inline void InitBaseParams(const ApaceQuantMatmulReduceScatterTilingData *tilingData);
    __aicore__ inline void SetupMatmulParams(const QuantMatmulTilingData* mmTile, Params& out);
    __aicore__ inline void RunMatmul();                    // AIC: 量化 MatMul 计算 → workspaceGM
    __aicore__ inline void RunAllToAllAndReduceSum();      // AIV: per-tile WaitFlag + AllToAll PUT + reduceSum
    __aicore__ inline void RunReduceSumTile(uint32_t t);   // AIV: 单 tile reduceSum → output
    __aicore__ inline void RunReduceSum();                 // AIV: 全量 reduceSum (T=1 退化路径)
    __aicore__ inline void RunAllToAll();                  // AIV: AllToAll PUT 通信 (T=1 退化路径)
};

// =================================================================================
// 公有方法实现
// =================================================================================

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void ApaceQuantMatmulReduceScatterUdmaImpl<AType, BType, CType, TransA, TransB>::Init(
    __gm__ CommContext *hcommCtx, GM_ADDR aGM, GM_ADDR scaleAGM, GM_ADDR bGM, GM_ADDR scaleBGM,
    GM_ADDR workspaceGM, GM_ADDR cGM,
    const ApaceQuantMatmulReduceScatterTilingData *tilingData)
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

    // selfWinAddr: 本 rank 的 win 区基址, AIC 将 C_i 写入此处
    baseParams_.selfWinAddr = reinterpret_cast<GM_ADDR>(udmaCtx_->commBufferAddrs[baseParams_.rankId]);

    // UB 分配: commBuf + barrierBuf
    uint32_t ubOffset = 0;
    auto commBuf = AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, uint8_t>(ubOffset);
    ubOffset += COMM_WORKSPACE_SIZE;
    auto barrierBuf = AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, uint8_t>(ubOffset);
    // teamBarrier_ totalJobs=1: 仅 block 0 参与 barrier (与 AllToAll totalJobs=1 一致)
    // CrossDevice step = min(1, nranks) = 1, core 0 轮询所有其他 rank
    teamBarrier_.Init(barrierBuf.Get(), syncBuffer_, 1,
                      static_cast<uint32_t>(GetBlockIdx()));

    // 初始化 AllToAll PUT 通信链
    // localAddr = workspaceGM (PUT 模式: 源数据, AIC 写入的 C_i)
    // totalJobs=1: 仅 block 0 处理所有 target ranks
    allToAllC_.template Init<BARRIER_NONE>(udmaCtx_, teamBarrier_, tilingData->commTilingData,
        baseParams_.workspaceGm, commBuf.Get(), 1,
        static_cast<uint32_t>(GetBlockIdx()));
}

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void ApaceQuantMatmulReduceScatterUdmaImpl<AType, BType, CType, TransA, TransB>::Run()
{
    if ASCEND_IS_AIC {
        // AIC 侧: per-tile 子调用 + SetFlag (方案 D)
        // T=1 退化路径: 单次全量 mm + 单次 SetFlag
        // T>1 流水路径: T×R 次子调用, 每 tile 后 SetFlag 通知 AIV
        RunMatmul();
        // AIC 不参与 SyncAll: SyncAll<true> isAIVOnly=true, AIC 调用为 no-op
    }

    if ASCEND_IS_AIV {
        // AIV 侧: per-tile WaitFlag + AllToAll PUT + reduceSum (方案 D)
        // WaitFlag 已在 per-tile 循环内, 不需要单独的 CrossCoreWaitFlag
        RunAllToAllAndReduceSum();
    }
}

// =================================================================================
// 参数初始化
// =================================================================================

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void ApaceQuantMatmulReduceScatterUdmaImpl<AType, BType, CType, TransA, TransB>::InitBaseParams(
    const ApaceQuantMatmulReduceScatterTilingData *tilingData)
{
    baseParams_.headMSize = tilingData->commTilingData.splitAxisTileSize;
    baseParams_.tailMSize = tilingData->commTilingData.splitAxisTailSize;
    baseParams_.headTileCnt = static_cast<uint32_t>(tilingData->commTilingData.splitAxisTileCnt);
    baseParams_.tailTileCnt = static_cast<uint32_t>(tilingData->commTilingData.splitAxisTailCnt);
    baseParams_.commTurn = tilingData->commTilingData.splitAxisTileCnt
                        + tilingData->commTilingData.splitAxisTailCnt;

    baseParams_.axisM = tilingData->mmTilingData.m;
    baseParams_.axisN = tilingData->mmTilingData.n;
    baseParams_.axisK = tilingData->mmTilingData.k;

    // scaleARowBytes = CeilDiv(axisK, 64) * 2 (MXFP8 scale: 每组 64 元素, 2 字节)
    baseParams_.scaleARowBytes = (baseParams_.axisK + 63U) / 64U * 2U;

    // chunkSize = 每个 rank 分到的 M 行数 = m / rankSize
    // CommTilingData: chunkSize = splitAxisTileSize * splitAxisTileCnt + splitAxisTailSize * splitAxisTailCnt
    uint64_t chunkSize = tilingData->commTilingData.splitAxisTileSize * tilingData->commTilingData.splitAxisTileCnt
                       + tilingData->commTilingData.splitAxisTailSize * tilingData->commTilingData.splitAxisTailCnt;
    baseParams_.chunkM = static_cast<uint32_t>(chunkSize);
    // chunkBytes = chunkSize * n * sizeof(CType)
    baseParams_.chunkBytes = chunkSize * tilingData->commTilingData.nonSplitAxisSize * sizeof(CType);
}

// =================================================================================
// AIC 侧: 量化 MatMul 计算 → selfWinAddr
// =================================================================================

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void ApaceQuantMatmulReduceScatterUdmaImpl<AType, BType, CType, TransA, TransB>::SetupMatmulParams(
    const QuantMatmulTilingData* mmTile, Params& out)
{
    // 本算子每卡独立计算完整 C_i = A_i × B
    // K 不切分, 无 TP 切分, 无 L0C 原子累加
    // rankId=0, rankSize=1: matmul kernel 按单卡模式计算
    // localMatmul=1: localFirst=true, 走 LOCAL 路径 (从 localAGmAddr 读 A, 不等通信 flag)
    ProblemShape problemShape{mmTile->m, mmTile->n, mmTile->k, 1UL};
    BlockMmadParams mmadParams;
    LocalParams localParams{0, 1, baseParams_.axisM, baseParams_.aGm,
        baseParams_.scaleAGm, 1, 1UL, MatmulMode::LOCAL, static_cast<uint32_t>(baseParams_.headMSize)};
    L1Params l1Params{static_cast<uint64_t>(mmTile->stepK) * mmTile->baseK, mmTile->scaleKL1,
        mmTile->nBufferNum};

    // PUT 模式: AIC 输出写入 workspaceGM (PUT 从此处推送到其他 rank win 区)
    // C_i [m, n] 连续存储, PUT 按 targetRankId * chunkBytes 偏移读取各 chunk
    mmadParams.cGmAddr = baseParams_.workspaceGm;
    mmadParams.bGmAddr = baseParams_.bGm;
    mmadParams.scaleBGmAddr = baseParams_.scaleBGm;

    BlockSchedulerParams schedulerParams{
        mmTile->baseM, mmTile->baseN, mmTile->mTailTile, mmTile->nTailTile,
        mmTile->mBaseTailSplitCnt, mmTile->nBaseTailSplitCnt, mmTile->mTailMain, mmTile->nTailMain};
    QBMMTiling qbmmParams{mmTile->baseM, mmTile->baseN, mmTile->baseK, mmTile->dbL0c, false};

    out = {problemShape, mmadParams, l1Params, schedulerParams, qbmmParams, localParams};
}

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void ApaceQuantMatmulReduceScatterUdmaImpl<AType, BType, CType, TransA, TransB>::RunMatmul()
{
    const auto& mmT = tilingData_->mmTilingData;
    const auto& commTd = tilingData_->commTilingData;
    uint32_t commTurn = static_cast<uint32_t>(commTd.splitAxisTileCnt + commTd.splitAxisTailCnt);

    if (commTurn <= 1) {
        // T=1 退化路径: 当前行为, 单次全量 mm + 单次 SetFlag
        Params params;
        SetupMatmulParams(&tilingData_->mmTilingData, params);
        // 无 SetAtomicAdd (无 L0C 原子累加, 每卡独立完整计算)
        quantMatmulKernelImpl_(params);
        CrossCoreSetFlag<0x2, PIPE_FIX>(0);
    } else {
        // T>1 per-tile 流水路径 (方案 D, Round 4 优化: 子问题独立 tiling)
        // 将 AIC 的单次全量 matmul 改为 T×R 次子调用
        // 每轮 T 完成后 SetFlag 通知 AIV, AIV WaitFlag 后执行该 tile 的 AllToAll + reduceSum
        //
        // Round 4 优化: head tile 使用子问题 tiling (subMmTilingData, m=headMSize)
        //   - SWAT 为小 M 生成独立 baseM/baseN/baseK/stepK 等, 可能产生更多 tiles → 更好的核间负载均衡
        //   - tail tile 仍使用全量 tiling (mmTilingData), 因 tail M 大小不固定
        //   - blockDim 固定使用 mmTilingData.usedCoreNum (kernel launch 时确定)
        uint64_t tileM = commTd.splitAxisTileSize;  // headMSize
        uint32_t R = baseParams_.rankSize;
        uint64_t chunkM = baseParams_.chunkM;
        const auto& subMmT = tilingData_->subMmTilingData;  // 子问题 tiling (head tile)

        for (uint32_t t = 0; t < commTurn; ++t) {
            // 当前 tile 的 M 大小 (head tile 或 tail tile)
            bool isHeadTile = (t < commTd.splitAxisTileCnt);
            uint32_t curTileM = isHeadTile
                              ? static_cast<uint32_t>(commTd.splitAxisTileSize)
                              : static_cast<uint32_t>(commTd.splitAxisTailSize);

            // 选择 tiling: head tile 用子问题 tiling, tail tile 用全量 tiling
            const auto& tileT = isHeadTile ? subMmT : mmT;

            for (uint32_t r = 0; r < R; ++r) {
                // rowOffset = r * chunkM + t * tileM
                // 与 AllToAll PUT srcAddr 公式一致: localAddr + r * chunkBytes + t * tileMaxByteSize
                uint64_t rowOffset = r * chunkM + t * tileM;

                Params params;
                // 子问题: {curTileM, N, K, 1}
                params.problemShape = ProblemShape{
                    static_cast<int64_t>(curTileM),
                    static_cast<int64_t>(baseParams_.axisN),
                    static_cast<int64_t>(baseParams_.axisK), 1LL};

                // A/ScaleA 带行偏移 (fp8 1 byte, 指针加法 = 元素偏移)
                // LOCAL 模式: matmul kernel 从 localAGmAddr 读 A, actualMPos = rankId(0) * oriM + mPos
                params.localParams = LocalParams{
                    0U, 1U, static_cast<uint64_t>(curTileM),
                    baseParams_.aGm + rowOffset * baseParams_.axisK,
                    baseParams_.scaleAGm + rowOffset * baseParams_.scaleARowBytes,
                    1U, 1U, MatmulMode::LOCAL, curTileM};

                // C 输出到 workspace 对应行区间
                // CType (bfloat16_t) 2 bytes, 偏移 = rowOffset * axisN * sizeof(CType)
                params.mmadParams.cGmAddr = baseParams_.workspaceGm
                    + rowOffset * baseParams_.axisN * sizeof(CType);
                params.mmadParams.bGmAddr = baseParams_.bGm;
                params.mmadParams.scaleBGmAddr = baseParams_.scaleBGm;

                // L1 参数使用子问题 tiling (stepK * baseK, scaleKL1, nBufferNum)
                params.l1Params = L1Params{
                    static_cast<uint64_t>(tileT.stepK) * tileT.baseK,
                    tileT.scaleKL1, tileT.nBufferNum};

                // M 尾块重置: tileM 是 baseM 的整数倍 → 无 M 尾块
                // mBaseTailSplitCnt=1 时 mBaseTailMain_ = mMergeSize (忽略 mTailMain=0)
                // N 侧参数使用子问题 tiling
                params.schParams = BlockSchedulerParams{
                    tileT.baseM, tileT.baseN,
                    1U, tileT.nTailTile, 1U, tileT.nBaseTailSplitCnt, 0U, tileT.nTailMain};

                // QBMM 参数使用子问题 tiling
                params.qbmmParams = QBMMTiling{
                    tileT.baseM, tileT.baseN, tileT.baseK, tileT.dbL0c, false};

                quantMatmulKernelImpl_(params);
            }
            // 每轮 R 个子调用完成 → SetFlag 通知 AIV
            // 全核无条件执行 (零 tile 核也执行, 保证 flag 计数平衡)
            CrossCoreSetFlag<0x2, PIPE_FIX>(0);
        }
    }
}

// =================================================================================
// AIV 侧: AllToAll PUT 通信 (T=1 退化路径, 被 RunAllToAllAndReduceSum 替代)
// =================================================================================

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void ApaceQuantMatmulReduceScatterUdmaImpl<AType, BType, CType, TransA, TransB>::RunAllToAll()
{
    // AllToAll PUT: 从 workspaceGM 推送 C_i chunks 到其他 rank win 区
    // totalJobs=1: 仅 block 0 执行所有通信
    // PUT 模式: 先 CrossDevice 确保所有 rank 就绪, 再 Commit+Wait+SyncAll
    if (GetBlockIdx() == 0) {
        teamBarrier_.CrossDevice();
    }
    SyncAll<true>();

    for (uint32_t tid = 0; tid < static_cast<uint32_t>(baseParams_.commTurn); ++tid) {
        if (GetBlockIdx() == 0) {
            allToAllC_.template Commit<BARRIER_NONE>();
            allToAllC_.template Wait<BARRIER_DEVICE>();
        }
        AscendC::SyncAll<true>();
    }
    if (GetBlockIdx() == 0) {
        allToAllC_.template Finalize<BARRIER_NONE>();
    }
}

// =================================================================================
// AIV 侧: reduceSum 求和 → output (T=1 退化路径, 全量处理所有 tile)
// =================================================================================

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void ApaceQuantMatmulReduceScatterUdmaImpl<AType, BType, CType, TransA, TransB>::RunReduceSum()
{
    // reduceSum: output = Σ_{r=0}^{R-1} C_r[rankId * chunkM : (rankId+1) * chunkM, :]
    //
    // PUT 模式数据来源:
    //   - self (r=rankId): workspaceGM[rankId * chunkBytes]  (AIC 直接写入 workspace)
    //   - received (r != rankId): selfWinAddr[r * chunkBytes]  (PUT 推送到 win 区)
    //
    // 输出: cGm[0 : chunkM * n]  (最终求和结果, shape [chunkM, n])
    //
    // 多 tile: 按 comm tile 分块处理，每 tile 处理 headMSize/tailMSize 行
    //   - 地址偏移用 chunkM（跨 rank 步进）
    //   - 处理大小用 tileM（当前 tile 行数）
    //   - tile 偏移 = tileMOffset（在 chunk 内的行偏移）
    // 注意: 此方法处理所有 tile, 用于 T=1 退化路径 (通信完成后一次性 reduceSum)

    ReduceSumImpl<CType> reduceSum;
    uint32_t nSize = static_cast<uint32_t>(baseParams_.axisN);
    uint32_t chunkM = baseParams_.chunkM;
    uint32_t tileMOffset = 0;

    // head tiles
    for (uint32_t tid = 0; tid < baseParams_.headTileCnt; ++tid) {
        uint32_t tileM = static_cast<uint32_t>(baseParams_.headMSize);
        reduceSum.RunTile(
            baseParams_.workspaceGm,    // self data (AIC wrote to workspace)
            baseParams_.selfWinAddr,    // received data (PUT pushed to win)
            baseParams_.cGm,
            baseParams_.rankId,
            baseParams_.rankSize,
            chunkM,
            nSize,
            tileM,
            tileMOffset);
        tileMOffset += tileM;
    }

    // tail tiles
    for (uint32_t tid = 0; tid < baseParams_.tailTileCnt; ++tid) {
        uint32_t tileM = static_cast<uint32_t>(baseParams_.tailMSize);
        reduceSum.RunTile(
            baseParams_.workspaceGm,    // self data (AIC wrote to workspace)
            baseParams_.selfWinAddr,    // received data (PUT pushed to win)
            baseParams_.cGm,
            baseParams_.rankId,
            baseParams_.rankSize,
            chunkM,
            nSize,
            tileM,
            tileMOffset);
        tileMOffset += tileM;
    }
}

// =================================================================================
// AIV 侧: 单 tile reduceSum (方案 D per-tile 流水路径)
// =================================================================================

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void ApaceQuantMatmulReduceScatterUdmaImpl<AType, BType, CType, TransA, TransB>::RunReduceSumTile(
    uint32_t t)
{
    // 对 tile t 执行 reduceSum:
    //   - self data: workspaceGM[rankId * chunkBytes + t * tileM * nBytes]
    //   - received data: selfWinAddr[r * chunkBytes + t * tileM * nBytes] (PUT 推送)
    //   - output: cGm[t * tileM * nBytes]
    //
    // tileMOffset = t * headMSize (head 和 tail tile 的行偏移均按 headMSize 步进)
    //   - head tile (t < headTileCnt): tileMOffset = t * headMSize
    //   - tail tile (t == headTileCnt): tileMOffset = headTileCnt * headMSize
    // 这与 AllToAll PUT 的 tileByteOffset_ 累加规则一致

    ReduceSumImpl<CType> reduceSum;
    uint32_t nSize = static_cast<uint32_t>(baseParams_.axisN);
    uint32_t chunkM = baseParams_.chunkM;

    uint32_t tileM;
    uint32_t tileMOffset;

    if (t < baseParams_.headTileCnt) {
        tileM = static_cast<uint32_t>(baseParams_.headMSize);
        tileMOffset = t * tileM;
    } else {
        tileM = static_cast<uint32_t>(baseParams_.tailMSize);
        tileMOffset = baseParams_.headTileCnt * static_cast<uint32_t>(baseParams_.headMSize);
    }

    reduceSum.RunTile(
        baseParams_.workspaceGm,    // self data (AIC wrote to workspace for this tile)
        baseParams_.selfWinAddr,    // received data (PUT pushed to win for this tile)
        baseParams_.cGm,
        baseParams_.rankId,
        baseParams_.rankSize,
        chunkM,
        nSize,
        tileM,
        tileMOffset);
}

// =================================================================================
// AIV 侧: per-tile WaitFlag + AllToAll PUT + reduceSum (方案 D)
// =================================================================================

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void ApaceQuantMatmulReduceScatterUdmaImpl<AType, BType, CType, TransA, TransB>::RunAllToAllAndReduceSum()
{
    // 方案 D per-tile 流水路径:
    //   AIC: [tile0 R段mm] SetFlag [tile1 R段mm] SetFlag [tile2 R段mm] SetFlag ...
    //   AIV:                WaitFlag [comm0+rs0]  WaitFlag [comm1+rs1]  WaitFlag [comm2+rs2]
    //
    // per-tile WaitFlag(0): CrossCore flag 是计数式, T 次 Set 对应 T 次 Wait
    // SyncAll 在 WaitFlag 后、Commit 前 (确保所有 AIV 核就绪)
    // reduceSum 按 tile 分块处理 (与 AllToAll PUT 数据布局一致)
    //
    // T=1 退化: 循环执行 1 次, 等效于 WaitFlag → SyncAll → Commit → Wait → SyncAll → reduceSum

    uint32_t commTurn = static_cast<uint32_t>(baseParams_.commTurn);

    for (uint32_t t = 0; t < commTurn; ++t) {
        // 1. 等待 AIC tile t 完成 (per-tile WaitFlag, 计数式 flag)
        //    flagId=0: T 次 SetFlag(0) 对应 T 次 WaitFlag(0), 峰值 T ≤ 15
        CrossCoreWaitFlag<0x2, PIPE_MTE2>(0);
        SyncAll<true>();

        // 2. AllToAll PUT: 推送 tile t 到所有目标 rank
        //    Commit 使用 currentTileIdx_ 自动递增, 每次推送一个 tile
        //    Wait<BARRIER_DEVICE> 包含 CrossDevice, 确保所有 rank PUT 完成
        if (GetBlockIdx() == 0) {
            allToAllC_.template Commit<BARRIER_NONE>();
            allToAllC_.template Wait<BARRIER_DEVICE>();
        }
        SyncAll<true>();

        // 3. reduceSum: 对 tile t 的 R 个源求和 → output
        //    仅 block 0 执行 (避免多核并发写同一 output)
        if (GetBlockIdx() == 0) {
            RunReduceSumTile(t);
        }
    }

    // 4. Finalize (PUT 模式 DoFinalize 为空, 但保持调用一致性)
    if (GetBlockIdx() == 0) {
        allToAllC_.template Finalize<BARRIER_NONE>();
    }
}

} // namespace Apace
