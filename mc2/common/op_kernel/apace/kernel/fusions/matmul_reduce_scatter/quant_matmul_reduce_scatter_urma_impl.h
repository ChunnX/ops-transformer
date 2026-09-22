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
 * 架构 (v3: AIC 三段式 + tid flag + 并行跨卡 fence + 收尾统一归约):
 *   AIC: RunMatmul()      段1: head tiles 的 REMOTE 片段, 每 tile SetFlag(tid)
 *        RunTailMatmul()  段2: tail tile 的 REMOTE 片段 (存在时), 一次 SetFlag(tid); tiling 切换前排空 L1
 *        RunLocalMatmul() 段3: 全部 tiles 的 LOCAL 片段 (无 per-tile SetFlag), 与段1/2 之间及段内
 *                          head→tail 切换 tiling 前排空 L1, 尾部一次性 SetFlag(FLAG_ID_LOCAL_DONE)
 *   AIV: for t = 0..commTurn-1: RunOneTileTurn(t)
 *          = 全核 WaitFlag(tid) → SyncAll → 通信核 Commit/Wait (PUT tile t)
 *        收尾四步: WaitFlag(LOCAL_DONE) → SyncAll → 通信核并行 CrossDevice (R 核各 fence 1 个远端)
 *          → SyncAll → 计算核逐 tile ReduceSum(0..T-1)
 *
 * 关键设计:
 *   - 无 R 循环: FragmentTensor 打包 rankSize 个 fragment, 一次调用覆盖一个 tile 的全部 rank 段
 *   - tid flag (三轮评审意见1): remote tile 通知直接用 tid 作 flagid, 每 id 严格一次 Set ↔ 一次 Wait,
 *     无轮转复用积压; flagid 空间: tiles {0..T-1} (T ≤ QMRS_MAX_COMM_TILES=14), 14 保留给
 *     SyncAll<true> (SYNC_AIV_ONLY_ALL), 15 = LOCAL_DONE
 *   - tiling 切换 L1 排空 (三轮评审意见3): 段间/段内 tiling 切换前 SetFlag/WaitFlag(MTE1_MTE2)
 *     排空 L1 (同 mx_quant_matmul_all_to_all_urma_impl.h), 防止新 tiling 的 MTE2 写覆盖未消费的 L1
 *   - 并行跨卡 fence (三轮评审意见2): teamBarrier totalJobs=rankSize, R 个通信核各轮询 1 个远端 win,
 *     替代单核串行轮询全部远端; win 数据区起点随 barrier 区 (2*R*32) 后移对齐 128B
 *   - local 片段就绪: 收尾 LOCAL_DONE(15) + SyncAll 聚合 (per-tile local 通知已随旧 ALL_DONE 删除)
 *   - 归约统一后移: per-turn reduceSum 读 win 存在跨设备读-写竞态, 移至 kernel 末尾由一次性 fence 排序
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

    // ---------------- AIC→AIV CrossCore flag (v3: tid 直通 + 一次性收尾 flag) ----------------
    // modeId 与既有形态一致 (kernel 内 Set 已移出, 形态仅在 impl 层使用)
    constexpr static uint8_t AIC_NOTIFY_MODE = 0x2;
    // REMOTE tile 通知: flagid 直接使用 tid (三轮评审意见1), 每 id 严格一次 Set ↔ 一次 Wait,
    // 无轮转复用积压 (旧 t%11 轮转在大 case 下 AIC 领先 ≥11 tile 时存在卡死风险)。
    // flagid 空间 (dav_3510 为 4bit): tiles {0..T-1}, T ≤ QMRS_MAX_COMM_TILES(14);
    // 14 保留给 SyncAll<true> (SYNC_AIV_ONLY_ALL), 15 = LOCAL_DONE, 二者与 tiles 不相交。
    // LOCAL_DONE: 收尾一次性完成信号 (AIC 三段全部完成); 专用 id 15
    constexpr static uint16_t FLAG_ID_LOCAL_DONE = 15;

    // ---------------- tiling 切换 L1 排空 (三轮评审意见3, 同 mx_quant_matmul_all_to_all) ----------------
    // 不同段/不同 tile 的 matmul tiling 切分不同 (L1Params 变化), 切换前须排空 L1:
    // SetFlag(MTE1_MTE2) 等 MTE1 (L1→L0) 此前的读全部完成, WaitFlag 让切换后新 tiling 的
    // MTE2 (GM→L1) 写等待其结束, 避免覆盖未消费的 L1 数据。event id 6 与 mmad 流水内部
    // 使用的 {0..5} (INPUT_BUFFER_0..3 / SCALE_BUFFER_0..1) 不相交。
    static constexpr uint16_t MTE1_MTE2_EVENT_FLAG = 6;

    static __aicore__ inline void DrainL1BeforeTilingSwitch()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_FLAG);
    }

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
    __aicore__ inline void RunMatmul();     // AIC 段1: head tiles 的 REMOTE 片段, 每 tile 一次 SetFlag
    __aicore__ inline void RunTailMatmul(); // AIC 段2: tail tile 的 REMOTE 片段 (存在时), 一次 SetFlag
    __aicore__ inline void RunLocalMatmul(); // AIC 段3: 全部 tiles 的 LOCAL 片段, 尾部一次性 SetFlag(LOCAL_DONE)
    __aicore__ inline void RunAllToAllAndReduceSum();             // AIV: 统一 turn 循环 + 收尾归约
    __aicore__ inline void InitAivRunContext(AivRunContext &ctx); // AIV: 角色判定 + 形状派生
    __aicore__ inline void RunOneTileTurn(const AivRunContext &ctx, uint32_t tileIdx); // AIV: 一轮通信步
    __aicore__ inline void ReduceSumTile(const AivRunContext &ctx, uint32_t tileIdx);  // AIV: 单 tile reduceSum
    __aicore__ inline uint32_t GetTileM(uint32_t tileIdx);                             // 获取 tile 的 M 大小
    __aicore__ inline uint64_t GetTileMOffset(uint32_t tileIdx);                       // 获取 tile 的 M 偏移
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

    // selfWinAddr: 本 rank 的 win 区数据起始地址 (跳过 barrier 区域)
    // Barrier 布局 (totalJobs=rankSize): localFlag rank 槽位区 (R*32) + teamSyncCounter 任务槽位区
    // (R*32) = 2*R*32, 向上对齐 128B (R=2 → 128, R=4 → 256, R=8 → 512);
    // PUT 数据从 WIN_DATA_OFFSET 开始写入, 不覆盖 barrier 区域
    uint32_t rankSize = baseParams_.rankSize;
    constexpr uint64_t WIN_DATA_ALIGN = 128;
    uint64_t winDataOffset =
        ((2UL * rankSize * BARRIER_FLAG_SIZE + WIN_DATA_ALIGN - 1) / WIN_DATA_ALIGN) * WIN_DATA_ALIGN;
    baseParams_.selfWinAddr = reinterpret_cast<GM_ADDR>(udmaCtx_->commBufferAddrs[baseParams_.rankId]) + winDataOffset;

    // 多核并行通信: 后 rankSize 个核做 AllToAll PUT
    // teamBarrier: totalJobs=rankSize → 后 R 个核各执行 1 个 job, CrossDevice 时 step=R,
    // 每核只轮询 1 个远端 win (三轮评审意见2: 替代 totalJobs=1 单核串行轮询全部远端)
    // AllToAll: totalJobs=rankSize → 后 R 个核各负责 1 个 target, 并行 PUT
    // jobIndex = BlockNum - 1 - blockIdx → 最后一个 block jobIndex=0, 倒数第二个 jobIndex=1
    uint32_t jobIndex = static_cast<uint32_t>(GetBlockNum() - 1 - GetBlockIdx());

    // UB 分配: commBuf + barrierBuf
    uint32_t ubOffset = 0;
    auto commBuf = AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, uint8_t>(ubOffset);
    ubOffset += COMM_WORKSPACE_SIZE;
    auto barrierBuf = AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, uint8_t>(ubOffset);
    // teamBarrier totalJobs=rankSize: R 个通信核并行执行 CrossDevice, 每核 fence 1 个远端
    teamBarrier_.Init(barrierBuf.Get(), syncBuffer_, rankSize, jobIndex);

    allToAllC_.template Init<BARRIER_NONE>(udmaCtx_, teamBarrier_, tilingData->commTilingData, baseParams_.workspaceGm,
                                           commBuf.Get(), rankSize, jobIndex, winDataOffset);
}

template <typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void QuantMatmulReduceScatterImpl<AType, BType, CType, TransA, TransB>::Run()
{
    if ASCEND_IS_AIC {
        // AIC 侧三段式: 逐 tile matmul → workspace + SetFlag
        RunMatmul();      // 段1: head tiles 的 REMOTE 片段, 每 tile 一次 SetFlag
        RunTailMatmul();  // 段2: tail tile 的 REMOTE 片段 (存在时), 一次 SetFlag
        RunLocalMatmul(); // 段3: 全部 tiles 的 LOCAL 片段, 尾部一次性 SetFlag(LOCAL_DONE)
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
    // 段1: head tiles 的 REMOTE 片段 (仅 remote rank 段, fragScope 剔除 self), 每 tile 一次 SetFlag。
    // headTileCnt ≥ 1 恒成立 (host 侧 headMSize 钳位到 chunkM 后 headTileCnt = chunkM/headMSize ≥ 1);
    // 单 tile 退化时 commTurn == 1, 本段即全覆盖。tiling 用 remoteHead (m=(R-1)*headMSize,
    // T=1 时 headMSize=chunkM) — 段级 tiling 的 m 必须等于本段实际逻辑 M (REMOTE=(R-1)*tileM)。
    const auto &tileT = tilingData_->remoteHeadMmTilingData;
    uint32_t rankSize = baseParams_.rankSize;

    for (uint32_t tileIdx = 0; tileIdx < baseParams_.headTileCnt; ++tileIdx) {
        uint32_t curTileM = GetTileM(tileIdx);
        uint64_t tileMOffset = GetTileMOffset(tileIdx);

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
        fragParams.fragScope = ReduceScatterFragImpl::FRAG_SCOPE_REMOTE;
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

        CrossCoreSetFlag<AIC_NOTIFY_MODE, PIPE_FIX>(tileIdx); // remote frag done (flagid = tid)
    }
}

template <typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void QuantMatmulReduceScatterImpl<AType, BType, CType, TransA, TransB>::RunTailMatmul()
{
    // 段2: tail tile 的 REMOTE 片段。tailTileCnt ∈ {0,1}, commTurn > headTileCnt ⇔ tailTileCnt == 1
    // (所有 AIC 核对同一 tilingData 求值, 判定一致; 单 tile 时 headTileCnt==1、tailTileCnt==0, 本段跳过)
    if (static_cast<uint32_t>(baseParams_.commTurn) > baseParams_.headTileCnt) {
        const uint32_t tileIdx = static_cast<uint32_t>(baseParams_.commTurn) - 1; // = headTileCnt
        uint32_t curTileM = GetTileM(tileIdx);
        uint64_t tileMOffset = GetTileMOffset(tileIdx);

        // tiling 用 remoteTail (m=(R-1)*tailMSize) — 与本段实际逻辑 M 一致
        const auto &tailMmT = tilingData_->remoteTailMmTilingData;

        // 三轮评审意见3: head 段 (remoteHead) → tail 段 (remoteTail) tiling 切换, 前排空 L1
        DrainL1BeforeTilingSwitch();

        FragFragmentParams fragParams;
        fragParams.tileCnt = 0;
        fragParams.tileM = curTileM;
        fragParams.tailCnt = 0;
        fragParams.tailM = 0;
        fragParams.paddedTailM = 0;
        fragParams.commTurn = 1;
        fragParams.headRows = baseParams_.rankSize * curTileM;
        fragParams.rankId = baseParams_.rankId;
        fragParams.rankSize = baseParams_.rankSize;
        fragParams.mPerRank = baseParams_.chunkM;
        fragParams.k = baseParams_.axisK;
        fragParams.n = baseParams_.axisN;
        fragParams.scaleKLen = (baseParams_.axisK + 63U) / 64U * 2U;
        fragParams.fragScope = ReduceScatterFragImpl::FRAG_SCOPE_REMOTE;
        fragParams.fragCount = 0;

        FragKernelParams params;
        params.mmTile = &tailMmT;
        params.qbmmParams = {tailMmT.baseM, tailMmT.baseN, tailMmT.baseK, tailMmT.dbL0c};
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

        CrossCoreSetFlag<AIC_NOTIFY_MODE, PIPE_FIX>(tileIdx); // tail tile remote frag done (flagid = tid)
    }
}

template <typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void QuantMatmulReduceScatterImpl<AType, BType, CType, TransA, TransB>::RunLocalMatmul()
{
    // 段3: 全部 tiles 的 LOCAL 片段 (仅本 rank 段, 输出落 workspace 本 rank 段对应 tile 行区),
    // 无任何 per-tile SetFlag (意见2)。tiling 按段选择 (LOCAL 段实际逻辑 M = tileM):
    // head → localHead (m=headMSize), tail → localTail (m=tailMSize)。
    // 三轮评审意见3: remote 段 (段1/2) → local 段 (段3) 之间 tiling/scope 切换, 前排空 L1
    // (同 mx_quant_matmul_all_to_all 的 RunLocalMatmul 前置排空)。
    DrainL1BeforeTilingSwitch();

    uint32_t commTurn = static_cast<uint32_t>(baseParams_.commTurn);
    uint32_t rankSize = baseParams_.rankSize;

    bool switchFlag = false; // head (localHead) → tail (localTail) tiling 切换标志
    for (uint32_t tileIdx = 0; tileIdx < commTurn; ++tileIdx) {
        uint32_t curTileM = GetTileM(tileIdx);
        uint64_t tileMOffset = GetTileMOffset(tileIdx);

        const auto &localHeadT = tilingData_->localHeadMmTilingData;
        const auto &localTailT = tilingData_->localTailMmTilingData;
        bool isHeadTile = (tileIdx < baseParams_.headTileCnt);
        if (!isHeadTile && !switchFlag) {
            // 三轮评审意见3: 段内 head→tail tiling 切换前排空 L1
            DrainL1BeforeTilingSwitch();
            switchFlag = true;
        }
        const auto &tileT = isHeadTile ? localHeadT : localTailT;

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
        fragParams.fragScope = ReduceScatterFragImpl::FRAG_SCOPE_LOCAL;
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
    }

    // 尾部一次性完成信号: AIC 三段全部完成 (PIPE_FIX 保证 Set 排在段3 全部 fixpipe 写出之后入队生效)。
    // RunLocalMatmul 由全部 AIC 核进入, Set 在 scheduler 分派之外 — 逐核 Set/Wait 计数平衡。
    CrossCoreSetFlag<AIC_NOTIFY_MODE, PIPE_FIX>(FLAG_ID_LOCAL_DONE);
}

// =================================================================================
// AIV 侧: 统一 turn 循环 + 收尾统一归约 (v2)
// =================================================================================

template <typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void QuantMatmulReduceScatterImpl<AType, BType, CType, TransA, TransB>::RunAllToAllAndReduceSum()
{
    // 角色分离 — 后 rankSize 核仅通信, 前 rsCoreNum 核仅计算, 其余核仅参与 SyncAll。
    // 统一 turn 循环: 单 tile / 多 tile 无分支 (T=1 自然退化为 1 轮, 无单 tile 快速路径)。
    AivRunContext ctx;
    InitAivRunContext(ctx);

    for (uint32_t tileIdx = 0; tileIdx < ctx.commTurn; ++tileIdx) {
        RunOneTileTurn(ctx, tileIdx);
    }

    // ---- kernel 收尾四步: 结构上即旧 per-turn 序列「Wait → SyncAll → CrossDevice → SyncAll」的一次性收拢 ----

    // F4: 全核 Wait 一次性 LOCAL_DONE — 本 block AIC 三段全部完成 (local 片段已写出 workspace)。
    //     单核 Wait 只覆盖本 block (modeId 0x2 为 block 内配对), 跨 AIC 聚合由下一步 F1' 完成。
    CrossCoreWaitFlag<AIC_NOTIFY_MODE, PIPE_S>(FLAG_ID_LOCAL_DONE);

    // F1': 全核 SyncAll — 双重聚合: (i) 本 rank 全部通信核的末次 Drain 完成 (全部 PUT 已发出且完成);
    //      (ii) 全部 AIV 核的 F4 已通过 ⇒ 全部 AIC 的段 3 完成 (reduceSum 读 workspace 的全 AIC 聚合语义)。
    SyncAll<true>();

    // F2: 一次性跨卡 fence — R 个通信核并行执行 (三轮评审意见2: 替代旧 totalJobs=1 单核串行
    //     轮询全部远端); rendezvous 语义: 本 rank 通信核 job j 轮询远端 rank j 的 ubmem win
    //     本 rank 槽位计数, 对端到达 ⇔ 对端全核已过其 F1' ⇔ 对端全部 PUT 完成。位置在
    //     RunOneTileTurn 之外 (kernel 收尾段), 每 launch 仅 1 次。
    teamBarrier_.CrossDevice();

    // F3: 全核 SyncAll — 将 F2 的跨卡可见性结论传播给全部核。
    SyncAll<true>();

    if (ctx.isComputeBlock) {
        for (uint32_t tileIdx = 0; tileIdx < ctx.commTurn; ++tileIdx) {
            ReduceSumTile(ctx, tileIdx); // 末尾统一消费; ReduceSumTile/DoReduceSum 零改动
        }
    }
    if (ctx.isCommBlock) {
        allToAllC_.template Finalize<BARRIER_NONE>(); // 框架四段式契约 (空实现, 保留调用)
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

// 一轮通信步 (意见1/4 落地): 全核 WaitFlag → 每 turn 唯一一次 SyncAll → 通信核 Commit/Wait。
// 计算核无 per-turn 任务 (win 数据未证到达、local 片段在段3 末尾才就绪), 归约整体后移至收尾统一执行。
template <typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void QuantMatmulReduceScatterImpl<AType, BType, CType, TransA, TransB>::RunOneTileTurn(
    const AivRunContext &ctx, uint32_t tileIdx)
{
    // 全核 Wait (通信核/计算核/多余核同序同次, 计数平衡); flagid = tid (三轮评审意见1), 无 remote/local 之分。
    // WaitFlag 形态 PIPE_S: 阻塞标量流 (兄弟先例 mx_quant_matmul_all_to_all_urma_impl.h 同形态)。
    CrossCoreWaitFlag<AIC_NOTIFY_MODE, PIPE_S>(tileIdx);

    // 意见1: 每 turn 唯一一次全卡同核同步。
    // 位置固定在 Wait 之后、Commit 之前 (不可后移): PUT(t) 读 workspace remote 片段 (全 AIC 分片写出),
    // 单核 Wait 只证本 block AIC 完成, SyncAll 汇集全部 AIV 核的 Wait 通过后才证明全 AIC 的 tile t
    // remote 片段完成。
    SyncAll<true>();

    if (ctx.isCommBlock) {
        allToAllC_.template Commit<BARRIER_NONE>(); // PUT tile t (WriteNbi, 跳过 self target)
        allToAllC_.template Wait<BARRIER_NONE>();   // Drain (本核 outbound PUT 完成)
    }
    // 计算核: 无 per-turn 任务, 直接进入下一 turn 的 WaitFlag。
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
