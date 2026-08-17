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
 * \brief quantMatmulReduceScatter — PUT 模式实现 (计算→通信) + per-tile 通信流水
 *
 * 架构（PUT 模式 = 计算→通信，per-tile 通信流水）：
 *   AIC: RunMatmul()  一次 matmul 调用覆盖完整 [m, n] (M_total = m = rankSize * chunkM)
 *                        完成后 CrossCoreSetFlag<0x2, PIPE_FIX>(0) 通知 AIV
 *   AIV: RunAllToAllAndReduceSum()  per-tile 循环:
 *                        CrossCoreWaitFlag<0x2, PIPE_MTE2>(0) 等待 AIC 完成
 *                        AllToAll PUT: 逐 tile 推送 C_i 到所有目标 rank
 *                        reduceSum: 逐 tile 对 R 个源求和 → output
 *
 * 关键设计:
 *   - 消除 R 循环: M_total = m, A 数据连续 [m,k], C 输出到 workspace 连续 [m,n]
 *     一次 matmul 调用覆盖全部 rank 段, PUT 自动按 rank 偏移读取
 *   - 统一路径: 无 if/else, AIC 一次算完后发 1 次 SetFlag
 *   - per-tile 通信: AIV 侧按 commTurn 分 tile 通信+reduceSum (流水在通信侧)
 *   - CrossCore flag flagId=0: 1 次 Set(0) 对应 1 次 Wait(0)
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
    struct NoOpCommWaitPolicy { __aicore__ inline void WaitTile(uint32_t) {} };
    using QuantMatmulKernelImpl = Kernel::QuantMatmulMxKernel<ProblemShape, BlockMmad, BlockScheduler, NoOpCommWaitPolicy>;

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
    __aicore__ inline void RunMatmul();                    // AIC: per-tile matmul → workspaceGM
    __aicore__ inline void RunAllToAllAndReduceSum();      // AIV: per-tile WaitFlag + AllToAll PUT + reduceSum
    __aicore__ inline void RunReduceSumTile(uint32_t t);   // AIV: 单 tile reduceSum → output
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
        // AIC 侧: 一次 matmul 覆盖完整 [m, n] + SetFlag
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
// AIC 侧: per-tile matmul → workspaceGM (统一路径, 无 if/else, 无 R 循环)
// =================================================================================

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void ApaceQuantMatmulReduceScatterUdmaImpl<AType, BType, CType, TransA, TransB>::RunMatmul()
{
    // 一次 matmul 调用覆盖完整 [m, n] (M_total = m = rankSize * chunkM)
    // A 数据连续 [m, k], C 输出到 workspace 连续 [m, n]
    // PUT 自动按 targetRankId * chunkBytes 偏移从 workspace 读取各 rank 段
    // 无 R 循环, 无 per-tile 切分 — 消除 FragmentTensor 需求
    const auto& mmT = tilingData_->mmTilingData;

    ProblemShape problemShape{
        static_cast<int64_t>(mmT.m),
        static_cast<int64_t>(mmT.n),
        static_cast<int64_t>(mmT.k), 1LL};

    BlockMmadParams mmadParams;
    mmadParams.cGmAddr = baseParams_.workspaceGm;
    mmadParams.bGmAddr = baseParams_.bGm;
    mmadParams.scaleBGmAddr = baseParams_.scaleBGm;

    // rankId=0, rankSize=1: 单卡模式, A 从本地完整读取, 不等通信 flag
    LocalParams localParams{
        0U, 1U, static_cast<uint64_t>(mmT.m),
        baseParams_.aGm, baseParams_.scaleAGm,
        1U, 1UL, MatmulMode::LOCAL, static_cast<uint32_t>(baseParams_.headMSize)};

    L1Params l1Params{
        static_cast<uint64_t>(mmT.stepK) * mmT.baseK,
        mmT.scaleKL1, mmT.nBufferNum};

    BlockSchedulerParams schedulerParams{
        mmT.baseM, mmT.baseN, mmT.mTailTile, mmT.nTailTile,
        mmT.mBaseTailSplitCnt, mmT.nBaseTailSplitCnt, mmT.mTailMain, mmT.nTailMain};

    QBMMTiling qbmmParams{
        mmT.baseM, mmT.baseN, mmT.baseK, mmT.dbL0c, false};

    Params params = {problemShape, mmadParams, l1Params, schedulerParams, qbmmParams, localParams};

    // 无 SetAtomicAdd (无 L0C 原子累加, 每卡独立完整计算)
    quantMatmulKernelImpl_(params);

    // 通知 AIV: matmul 全部完成
    CrossCoreSetFlag<0x2, PIPE_FIX>(0);
}

// =================================================================================
// AIV 侧: 单 tile reduceSum → output
// =================================================================================

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void ApaceQuantMatmulReduceScatterUdmaImpl<AType, BType, CType, TransA, TransB>::RunReduceSumTile(
    uint32_t t)
{
    // 对 tile t 执行 reduceSum:
    //   - self data: workspaceGM[rankId * chunkBytes + tileMOffset * nBytes]
    //   - received data: selfWinAddr[r * chunkBytes + tileMOffset * nBytes] (PUT 推送)
    //   - output: cGm[tileMOffset * nBytes]

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
        baseParams_.workspaceGm,
        baseParams_.selfWinAddr,
        baseParams_.cGm,
        baseParams_.rankId,
        baseParams_.rankSize,
        chunkM,
        nSize,
        tileM,
        tileMOffset);
}

// =================================================================================
// AIV 侧: per-tile WaitFlag + AllToAll PUT + reduceSum
// =================================================================================

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void ApaceQuantMatmulReduceScatterUdmaImpl<AType, BType, CType, TransA, TransB>::RunAllToAllAndReduceSum()
{
    // AIC 一次 matmul 算完 [m, n], 发 1 次 SetFlag
    // AIV WaitFlag 1 次, 然后按 commTurn 分 tile 通信 + reduceSum
    //
    // 通信侧 per-tile 流水:
    //   AIV: WaitFlag → [tile0 comm+rs0] → [tile1 comm+rs1] → ... → Finalize

    // 1. 等待 AIC matmul 完成
    CrossCoreWaitFlag<0x2, PIPE_MTE2>(0);
    SyncAll<true>();

    // 2. CrossDevice 确保所有 rank 的 AIV 就绪
    if (GetBlockIdx() == 0) {
        teamBarrier_.CrossDevice();
    }
    SyncAll<true>();

    // 2. 逐 tile 通信 + reduceSum
    uint32_t commTurn = static_cast<uint32_t>(baseParams_.commTurn);

    for (uint32_t t = 0; t < commTurn; ++t) {
        // AllToAll PUT: 推送 tile t 到所有目标 rank
        if (GetBlockIdx() == 0) {
            allToAllC_.template Commit<BARRIER_NONE>();
            allToAllC_.template Wait<BARRIER_DEVICE>();
        }
        SyncAll<true>();

        // reduceSum: 对 tile t 的 R 个源求和 → output
        if (GetBlockIdx() == 0) {
            RunReduceSumTile(t);
        }
    }

    // 3. Finalize
    if (GetBlockIdx() == 0) {
        allToAllC_.template Finalize<BARRIER_NONE>();
    }
}

} // namespace Apace
