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
 * FragmentTensor 设计:
 *   - A 数据全在本卡 GM 连续 [m,k], 各 rank 段打包为 FragmentTensor
 *   - BlockMmadFragC 在 tile 跨越 rank 边界时自动切换 fragment, 保持 per-fragment L1 隔离
 *   - 一次 matmul 调用覆盖所有 rank 段, 无 R 循环
 *
 * 架构 (AIC/AIV 分工 + 逐 tile 交错 + 通算重叠流水):
 *   AIC: RunInterleavedMatmul(): for t = 0..T-1:
 *          scope①(t) → [L1 布局变化时排空] → scope②(t) → SetFlag(tid)
 *          (scope 顺序由 tilingData.localScopeLast 控制: 0=LOCAL 先(默认), 1=LOCAL 后置)
 *        flag 语义 = 本 block tile t 全部片段完成 (local+remote)
 *   AIV: for t = 0..T-1: RunOneTileTurn(t)
 *          = 全核 WaitFlag(tid) → SyncAll → 通信核 Commit(t)+fence(t-1)+Drain(t)
 *            ‖ 计算核 [本地轮询 fence(t-1)] + ReduceSum(t-1)
 *        尾声: 通信核 fence(T-1); 计算核 [本地轮询] + ReduceSum(T-1)
 *
 * 关键设计:
 *   - 无 R 循环: FragmentTensor 打包 rankSize 个 fragment, 一次调用覆盖一个 tile 的全部 rank 段
 *   - tid flag: tile 通知直接用 tid 作 flagid, 每 id 严格一次 Set ↔ 一次 Wait,
 *     无轮转复用积压; flagid 空间: tiles {0..T-1} (T ≤ QMRS_MAX_COMM_TILES=14), 14 保留给
 *     SyncAll<true> (SYNC_AIV_ONLY_ALL), 15 保留未用
 *   - tiling 切换 L1 排空: 仅当前一 scope 与当前 scope 的 L1 布局字段不同时
 *     SetFlag/WaitFlag(MTE1_MTE2) 排空 L1, 防止新布局的 MTE2 写覆盖未消费的 L1
 *     (rank2 下 local/remote tiling 相同, 稳定轮转不排空)
 *   - 并行跨卡 fence: teamBarrier totalJobs=rankSize, R 个通信核各轮询 1 个远端 win;
 *     win 数据区起点随 barrier 区 (2*R*32) 后移对齐 128B
 *   - 通算重叠: reduceSum(t-1) 与 PUT(t) 逐 turn 重叠, 用通信时间掩盖 AIV 归约访存
 *   - fence 重叠化: fence(t-1) 移至 Commit(t) 之后 (与 PUT(t) 传输重叠) 且 Drain(t) 之前;
 *     计算核本地轮询 R 个通信核的 fence 计数槽 (teamSyncCounter, 纯本地 GM 读) —
 *     fence(t-1) 完成即启动 RS(t-1), lag 保持 1 turn
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
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
    static constexpr int32_t SCALE_C0 = 2;         // MX 每组 scale 字节数 (E8M0)
    static constexpr uint32_t MX_GROUP_SIZE = 64U; // MX 量化组大小 (每组 64 个元素共用 1 个 scale)

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
        GM_ADDR selfWinAddr{nullptr}; // 本卡通信窗口数据区地址 (远端 rank PUT 落点, reduceSum 读取)
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

    // 计算核本地轮询 fence 完成计数用的 UB scratch (32B, 复用 barrierBuf;
    // 计算核不执行 CrossDevice, 该 buffer 在计算核上空闲; RS 会覆写 UB 但轮询每次
    // 从 GM 重读, 内容失效无影响)
    __ubuf__ uint8_t *fencePollBuf_{nullptr};

    // ---------------- AIC→AIV CrossCore flag (tid 直通) ----------------
    // modeId 与既有形态一致 (kernel 内 Set 已移出, 形态仅在 impl 层使用)
    constexpr static uint8_t AIC_NOTIFY_MODE = 0x2;
    // tile 通知: flagid 直接使用 tid, 每 id 严格一次 Set ↔ 一次 Wait,
    // 不做轮转复用 (同 id 连续两次 Set 在 AIC 领先时有丢失唤醒卡死风险)。
    // flag 语义 = 本 block tile t 的 LOCAL+REMOTE 片段全部完成。
    // flagid 空间 (dav_3510 为 4bit): tiles {0..T-1}, T ≤ QMRS_MAX_COMM_TILES(14);
    // 14 保留给 SyncAll<true> (SYNC_AIV_ONLY_ALL), 15 保留未用, 二者与 tiles 不相交。

    // ---------------- tiling 切换 L1 排空 ----------------
    // 仅当前一 scope 与当前 scope 的 L1 布局字段不同 (IsL1LayoutChanged) 时调用:
    // SetFlag(MTE1_MTE2) 等 MTE1 (L1→L0) 此前的读全部完成, WaitFlag 让新布局的
    // MTE2 (GM→L1) 写等待其结束, 避免覆盖未消费的 L1 数据。
    // event id 6 说明: mmad 流水内部 (qmm_mx_block_mmad_fragment.h) 占用 MTE1_MTE2 的
    // {0..5} (INPUT_BUFFER_0..3 / SCALE_BUFFER_0..1), 本排空用 6 与之不相交; TPipe 队列
    // 场景官方文档保留 eventID 6/7, 本实现为静态 LocalTensor (无 TPipe 队列), 且兄弟实现
    // mx_quant_matmul_all_to_all_urma_impl.h 同样使用 id 6 (仓内既有先例)。
    static constexpr uint16_t MTE1_MTE2_EVENT_FLAG = 6;

    static __aicore__ inline void DrainL1BeforeTilingSwitch()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_FLAG);
    }

    // L1 布局相关字段判定: 仅 baseM/baseN/baseK/scaleKL1/stepK/nBufferNum/dbL0c 这 7 个
    // 字段经 l0TileShape={baseM,baseN,baseK} 与 L1Params{kL1=stepK*baseK, scaleKL1,
    // l1BufNum=nBufferNum} (+dbL0c 控制 L0C 乒乓) 进入 mmad, 决定 L1 缓冲几何; 其余字段
    // (m/尾块调度/usedCoreNum) 只影响 scope 内循环次数与 L0 寻址, 不改变 L1 布局。
    // rank2 下 (R-1)*tileM == tileM, local/remote tiling 逐字节相同 → 稳定轮转不排空。
    static __aicore__ inline bool IsL1LayoutChanged(const QuantMatmulTilingData &a,
                                                    const QuantMatmulTilingData &b)
    {
        return a.baseM != b.baseM || a.baseN != b.baseN || a.baseK != b.baseK ||
               a.scaleKL1 != b.scaleKL1 || a.stepK != b.stepK || a.nBufferNum != b.nBufferNum ||
               a.dbL0c != b.dbL0c;
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
    // 逐 tile 交错流水: AIC 每 tile 先 LOCAL 后 REMOTE 片段, 完成后 SetFlag(tid);
    // flag 语义 = 本 block 的 tile t 全部片段 (local+remote) 完成
    __aicore__ inline void RunInterleavedMatmul();
    __aicore__ inline void RunOneFragScope(uint32_t tileIdx, bool isLocal,
                                           const QuantMatmulTilingData &tileT); // AIC: 单 tile 单 scope 片段
    __aicore__ inline void RunAllToAllAndReduceSum();             // AIV: 逐 turn 通信 + 重叠归约流水
    __aicore__ inline void InitAivRunContext(AivRunContext &ctx); // AIV: 角色判定 + 形状派生
    __aicore__ inline void RunOneTileTurn(const AivRunContext &ctx, uint32_t tileIdx); // AIV: 一轮通算重叠步
    __aicore__ inline void ReduceSumTile(const AivRunContext &ctx, uint32_t tileIdx,
                                         uint32_t fullBatch);           // AIV: 单 tile reduceSum
    __aicore__ inline void WaitRemoteFenceDone(uint32_t expectedCount); // AIV: 本地轮询 fence 完成计数
    __aicore__ inline uint32_t GetTileM(uint32_t tileIdx);              // 获取 tile 的 M 大小
    __aicore__ inline uint64_t GetTileMOffset(uint32_t tileIdx);        // 获取 tile 的 M 偏移
    __aicore__ inline void DoReduceSum(uint32_t tileM, uint64_t tileMOffset, uint32_t nSize, uint32_t rowBytes,
                                       uint64_t stride, uint32_t rsCoreNum, uint32_t fullBatch);
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
    // PUT 数据从 winDataOffset 开始写入, 不覆盖 barrier 区域
    uint32_t rankSize = baseParams_.rankSize;
    constexpr uint64_t WIN_DATA_ALIGN = 128;
    uint64_t winDataOffset =
        ((2UL * rankSize * BARRIER_FLAG_SIZE + WIN_DATA_ALIGN - 1) / WIN_DATA_ALIGN) * WIN_DATA_ALIGN;
    baseParams_.selfWinAddr = reinterpret_cast<GM_ADDR>(udmaCtx_->commBufferAddrs[baseParams_.rankId]) + winDataOffset;

    // 多核并行通信: 后 rankSize 个核做 AllToAll PUT
    // teamBarrier: totalJobs=rankSize → 后 R 个核各执行 1 个 job, CrossDevice 时 step=R,
    // 每核只轮询 1 个远端 win
    // AllToAll: totalJobs=rankSize → 后 R 个核各负责 1 个 target, 并行 PUT
    // jobIndex = BlockNum - 1 - blockIdx → 最后一个 block jobIndex=0, 倒数第二个 jobIndex=1
    uint32_t jobIndex = static_cast<uint32_t>(GetBlockNum() - 1 - GetBlockIdx());

    // UB 分配: commBuf + barrierBuf
    uint32_t ubOffset = 0;
    auto commBuf = AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, uint8_t>(ubOffset);
    ubOffset += COMM_WORKSPACE_SIZE;
    auto barrierBuf = AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, uint8_t>(ubOffset);
    fencePollBuf_ = barrierBuf.Get(); // 计算核 fence 轮询 scratch (见成员注释)
    // teamBarrier totalJobs=rankSize: R 个通信核并行执行 CrossDevice, 每核 fence 1 个远端
    teamBarrier_.Init(barrierBuf.Get(), syncBuffer_, rankSize, jobIndex);

    allToAllC_.template Init<BARRIER_NONE>(udmaCtx_, teamBarrier_, tilingData->commTilingData, baseParams_.workspaceGm,
                                           commBuf.Get(), rankSize, jobIndex, winDataOffset);
}

template <typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void QuantMatmulReduceScatterImpl<AType, BType, CType, TransA, TransB>::Run()
{
    if ASCEND_IS_AIC {
        // AIC 侧: 逐 tile 交错 — 每 tile LOCAL 片段 → REMOTE 片段 → SetFlag(tid)
        RunInterleavedMatmul();
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

    // scaleARowBytes = CeilDiv(axisK, MX_GROUP_SIZE) * SCALE_C0 (MXFP8: 每组 64 元素, 2 字节)
    baseParams_.scaleARowBytes = (baseParams_.axisK + MX_GROUP_SIZE - 1U) / MX_GROUP_SIZE * SCALE_C0;

    // chunkSize = 每个 rank 分到的 M 行数 = m / rankSize
    uint64_t chunkSize = tilingData->commTilingData.splitAxisTileSize * tilingData->commTilingData.splitAxisTileCnt +
                         tilingData->commTilingData.splitAxisTailSize * tilingData->commTilingData.splitAxisTailCnt;
    baseParams_.chunkM = static_cast<uint32_t>(chunkSize);
}

// =================================================================================
// AIC 侧: 逐 tile 交错流水 (LOCAL 片段 → REMOTE 片段 → SetFlag(tid))
// =================================================================================

// 单 tile 单 scope 的 FragmentTensor matmul 调用 (统一参数装配, 无 R 循环)
// isLocal=true: 仅本 rank 片段 (fragScope=LOCAL, 1 个 fragment), tiling 用 localHead/localTail (m=tileM);
// isLocal=false: 仅 remote rank 片段 (fragScope=REMOTE, R-1 个 fragment), tiling 用 remoteHead/remoteTail
// (m=(R-1)*tileM)。scope 级 tiling 的 m 必须等于该 scope 实际逻辑 M (见 tiling_data.h 头注)。
template <typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void QuantMatmulReduceScatterImpl<AType, BType, CType, TransA, TransB>::RunOneFragScope(
    uint32_t tileIdx, bool isLocal, const QuantMatmulTilingData &tileT)
{
    uint32_t curTileM = GetTileM(tileIdx);
    uint64_t tileMOffset = GetTileMOffset(tileIdx);
    uint32_t rankSize = baseParams_.rankSize;

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
    fragParams.scaleKLen = (baseParams_.axisK + MX_GROUP_SIZE - 1U) / MX_GROUP_SIZE * SCALE_C0;
    fragParams.fragScope = isLocal ? ReduceScatterFragImpl::FRAG_SCOPE_LOCAL : ReduceScatterFragImpl::FRAG_SCOPE_REMOTE;
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

// 逐 tile 交错: 每 tile 依次执行 LOCAL/REMOTE 两个片段 (顺序由 tilingData.localScopeLast
// 控制: 0=LOCAL 先(默认), 1=LOCAL 后置), 全部完成后 SetFlag(tid)。
// 单 flag 语义下两种顺序时序等价 (flag = 双 scope 全部完成, 触发时间 = max(两 scope));
// 稳态流水中 PUT(t) 已与 AIC 整个 tile t+1 重叠, tile 内顺序对 AIV 不可见 — 开关为策略实验保留。
// flag 语义 = 本 block 的 tile t 两个片段全部完成 (PIPE_FIX 保证 fixpipe 写出后生效);
// AIV 侧 Wait(tid)+SyncAll 聚合后即证全 AIC 核 tile t 完成 — PUT(t) 源数据与 reduceSum 的
// selfPartial 依赖同步就绪, 无需独立完成信号。
// tiling 切换排空: 仅当前一 scope 与当前 scope 的 L1 布局字段不同 (IsL1LayoutChanged) 时
// SetFlag/WaitFlag(MTE1_MTE2) 排空 L1, 防止新布局的 MTE2 写覆盖未消费的 L1 —
// rank2 下 local/remote tiling 逐字节相同, 稳定轮转不触发; head→tail 切换由统一的
// prev scope 比较覆盖 (无独立特判, 亦无重复排空)。
// (与 mmad ctor/dtor flag 纪律冗余的防御性排空, 开销为空转 flag 对)。
template <typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void QuantMatmulReduceScatterImpl<AType, BType, CType, TransA, TransB>::RunInterleavedMatmul()
{
    const auto &localHeadT = tilingData_->localHeadMmTilingData;
    const auto &localTailT = tilingData_->localTailMmTilingData;
    const auto &remoteHeadT = tilingData_->remoteHeadMmTilingData;
    const auto &remoteTailT = tilingData_->remoteTailMmTilingData;
    uint32_t commTurn = static_cast<uint32_t>(baseParams_.commTurn);

    // scope 顺序开关: 0 = LOCAL 先 (默认), 1 = LOCAL 后置
    const bool localLast = (tilingData_->localScopeLast != 0U);

    // 前一 scope 的 tiling (首个 scope 无前排空); head→tail 切换由统一 prev 比较覆盖
    const QuantMatmulTilingData *prevScopeT = nullptr;
    for (uint32_t tileIdx = 0; tileIdx < commTurn; ++tileIdx) {
        bool isHeadTile = (tileIdx < baseParams_.headTileCnt);
        const QuantMatmulTilingData &localT = isHeadTile ? localHeadT : localTailT;
        const QuantMatmulTilingData &remoteT = isHeadTile ? remoteHeadT : remoteTailT;
        const QuantMatmulTilingData &firstT = localLast ? remoteT : localT;
        const QuantMatmulTilingData &secondT = localLast ? localT : remoteT;

        // 第一片段 (LOCAL: 本 rank 段 → workspace 本 rank 槽位, tiling m = tileM;
        //          REMOTE: remote rank 段 → workspace 各 remote rank 槽位, tiling m = (R-1)*tileM)
        if (prevScopeT != nullptr && IsL1LayoutChanged(*prevScopeT, firstT)) {
            DrainL1BeforeTilingSwitch(); // 第二片段(t-1) → 第一片段(t) L1 布局变化 (含 head→tail)
        }
        RunOneFragScope(tileIdx, !localLast, firstT);

        // 第二片段 (与第一片段互补的 scope)
        if (IsL1LayoutChanged(firstT, secondT)) {
            DrainL1BeforeTilingSwitch(); // tile 内 scope 切换 L1 布局变化
        }
        RunOneFragScope(tileIdx, localLast, secondT);
        prevScopeT = &secondT;

        // tile t 全部片段完成 (flagid = tid)
        CrossCoreSetFlag<AIC_NOTIFY_MODE, PIPE_FIX>(tileIdx);
    }
}

// =================================================================================
// AIV 侧: 逐 turn 通信 + 重叠归约流水 (fence 与 PUT 传输重叠, 无终局 barrier)
// =================================================================================

template <typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void QuantMatmulReduceScatterImpl<AType, BType, CType, TransA, TransB>::RunAllToAllAndReduceSum()
{
    // 角色分离 — 后 rankSize 核仅通信, 前 rsCoreNum 核仅计算, 其余核仅参与 SyncAll。
    // 流水: turn t 内通信核 Commit(t)+fence(t-1)+Drain(t) 与计算核
    // [轮询 fence(t-1)]+RS(t-1) 并行; fence 与 PUT(t) 传输重叠, 不占用通道空闲窗口。
    AivRunContext ctx;
    InitAivRunContext(ctx);

    for (uint32_t tileIdx = 0; tileIdx < ctx.commTurn; ++tileIdx) {
        RunOneTileTurn(ctx, tileIdx);
    }

    // 尾声 (无终局 SyncAll, 每核 SyncAll 计数 = T 保持平衡):
    //  - 通信核: fence(T-1) — Drain(T-1) 已在 turn T-1 末完成, 此处仅需握手
    //  - 计算核: 本地轮询 R 个计数槽 >= T (证 fence(T-1) 全部完成) 后补做 RS(T-1)
    if (ctx.isCommBlock) {
        teamBarrier_.CrossDevice();
        allToAllC_.template Finalize<BARRIER_NONE>(); // 框架四段式契约 (空实现, 保留调用)
    }
    if (ctx.isComputeBlock) {
        WaitRemoteFenceDone(ctx.commTurn);
        ReduceSumTile(ctx, ctx.commTurn - 1, 1U); // 尾声: 无并发 PUT, 全量批量加载
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
    // rsCoreNum 无下溢保护说明: host 侧 (ST main.cpp) 已强校验 usedCoreNum > rankNum,
    // 本减法恒为正; 若脱离 ST 直调接入, 调用方须保证该不变量 (uint32 下溢回绕会破坏角色模型)

    uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
    uint32_t jobIndex = static_cast<uint32_t>(GetBlockNum() - 1 - blockIdx);
    ctx.isCommBlock = (jobIndex < baseParams_.rankSize); // 后 rankSize 核: AllToAll 通信
    ctx.isComputeBlock = (blockIdx < ctx.rsCoreNum);     // 前 rsCoreNum 核: reduceSum
}

// 一轮通算重叠步: Wait(tid) → SyncAll → 通信核 Commit(t)+fence(t-1)+Drain(t)
// ‖ 计算核 [轮询 fence(t-1)]+RS(t-1)。
//  - Wait(tid) + SyncAll: 本 block 及 (聚合后) 全部 AIC 核的 tile t LOCAL+REMOTE 片段完成
//    ⇒ PUT(t) 源 (workspace remote 片段) 就绪;
//  - 通信核: Commit(t) 紧跟 SyncAll 立即 post (通道尽早开始传输);
//    fence(t-1) 位于 Commit(t) 与 Drain(t) 之间 — 与 PUT(t) 传输重叠 (fence(t-1) 所需的
//    对端 Drain(t-1) 在上一轮已完成, 握手仅需 1~2 次轮询即可通过);
//    Drain(t) 收尾等待 PUT(t) 传输完成;
//  - 计算核: RS(t-1) 的 win 依赖 (PUT(t-1) 到达) 改为本地轮询 R 个通信核 fence 计数槽
//    (WaitRemoteFenceDone, 纯本地 GM 读) — fence(t-1) 完成即启动 RS(t-1), lag 保持
//    1 turn; selfPartial 依赖由 WaitFlag(t-1) 传递证明;
//  - 同步开销: 每核每 turn 恰 1 次 SyncAll, 无终局 barrier (尾声明见 RunAllToAllAndReduceSum)。
//
// 正确性论证:
//  1. fence(t-1) 于 turn t 执行的前提 Drain(t-1) 已完成: 通信核 turn t-1 以 Drain(t-1)
//     收尾 (程序序), 跨 turn 无重排;
//  2. fence 结论本地可观测: 通信核 CrossDevice 返回前将递增后的 count 写回本 rank win
//     barrier 区的 teamSyncCounter 槽 (MTE3_S 同步保证 GM 可见), 计算核 MTE2 读同区
//     可见;
//  3. counter[j] >= k ⟺ 通信核 j 已完成 k 次 CrossDevice ⟺ fence(0..k-1) 全部完成
//     (计数单调, 跨轮 skew 下提前观察到更大值仍满足判定, 无丢失唤醒);
//  4. RS(t-1) 依赖齐备: counter[j] >= t (全部 j) ⇒ fence(t-1) 完成 ⇒ 各远端 PUT(t-1)
//     已到达本 rank win; selfPartial(t-1) 由 turn t-1 的 WaitFlag 证明;
//  5. SyncAll 计数平衡: 通信核/计算核/多余核每核 T 次, 无终局, 全核一致;
//  6. win/workspace 每 tile 区域互不相交; T=1 退化 (turn 0 仅 Commit+Drain, 尾声
//     fence(0)+轮询+RS(0)) 同样成立。
template <typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void QuantMatmulReduceScatterImpl<AType, BType, CType, TransA, TransB>::RunOneTileTurn(
    const AivRunContext &ctx, uint32_t tileIdx)
{
    // 全核 Wait (通信核/计算核/多余核同序同次, 计数平衡); flagid = tid
    CrossCoreWaitFlag<AIC_NOTIFY_MODE, PIPE_S>(tileIdx);

    // SyncAll: 聚合全部 AIV 核的 Wait 通过 ⇒ 全部 AIC 核 tile t 的 LOCAL+REMOTE 片段完成
    SyncAll<true>();

    if (ctx.isCommBlock) {
        // 立即 post PUT tile t (WriteNbi, 跳过 self target) — 通道尽早开始传输
        allToAllC_.template Commit<BARRIER_NONE>();
        if (tileIdx > 0) {
            // fence(t-1): R 个通信核并行轮询 (每核 1 个远端), 证 PUT(t-1) 已到达本 rank win;
            // 位于 PUT(t) 传输窗口内, 不占用通道空闲时间
            teamBarrier_.CrossDevice();
        }
        // Drain: 等待 PUT(t) 发送侧完成 (含此前 post 的全部写), turn 收尾
        allToAllC_.template Wait<BARRIER_NONE>();
    }
    if (ctx.isComputeBlock && tileIdx > 0) {
        // 本地轮询全部通信核 fence 计数 >= tileIdx ⇒ fence(tileIdx-1) 完成 ⇒ win[t-1] 数据齐
        WaitRemoteFenceDone(tileIdx);
        ReduceSumTile(ctx, tileIdx - 1, 0U); // 在途: 与 PUT(t) 并发, 配对批量 (防 GM 突发争抢)
    }
}

// fence 完成本地轮询: 自旋读取本 rank win barrier 区 R 个 teamSyncCounter 槽,
// 直到全部 >= expectedCount。计数槽布局与 TeamBarrier::CrossDevice 一致:
//   commBufferAddrs[rankId] + rankSize*BARRIER_FLAG_SIZE + j*BARRIER_FLAG_SIZE
// 每槽 32B, int64 count 位于槽首。仅计算核调用 (通信核自身执行 fence, 多余核不消费)。
// S_MTE2/MTE2_S flag 对 (id 1) 与 RS 组件的 MTE2_V/V_MTE2 等 flag 对不同 pipe  pair,
// 且轮询与 RS 在同核程序序串行, 无并发冲突; RS 覆写 UB 不影响轮询 (每次从 GM 重读)。
template <typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void QuantMatmulReduceScatterImpl<AType, BType, CType, TransA, TransB>::WaitRemoteFenceDone(
    uint32_t expectedCount)
{
    __gm__ int32_t *counterBase =
        reinterpret_cast<__gm__ int32_t *>(reinterpret_cast<GM_ADDR>(syncBuffer_->commBufferAddrs[baseParams_.rankId]) +
                                           baseParams_.rankSize * BARRIER_FLAG_SIZE);

    auto copyGM2UB = Te::MakeCopy(Te::CopyGM2UB{});
    auto ubTmp = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::UB, int32_t>(reinterpret_cast<uint64_t>(fencePollBuf_)),
        Te::FrameLayoutFormat<Te::NDExtLayoutPtn, Te::LayoutTraitDefault<int32_t>>{}(1, BARRIER_FLAG_ELEMS));

    do {
        bool allDone = true;
        for (uint32_t j = 0; j < baseParams_.rankSize; ++j) {
            auto gmSlot = Te::MakeTensor(
                Te::MakeMemPtr<Te::Location::GM>(counterBase + j * BARRIER_FLAG_ELEMS),
                Te::FrameLayoutFormat<Te::NDExtLayoutPtn, Te::LayoutTraitDefault<int32_t>>{}(1, BARRIER_FLAG_ELEMS));
            SetFlag<HardEvent::S_MTE2>(1);
            WaitFlag<HardEvent::S_MTE2>(1);
            Te::Copy(copyGM2UB, ubTmp, gmSlot);
            SetFlag<HardEvent::MTE2_S>(1);
            WaitFlag<HardEvent::MTE2_S>(1);
            if (*reinterpret_cast<__ubuf__ int64_t *>(fencePollBuf_) < static_cast<int64_t>(expectedCount)) {
                allDone = false;
                break;
            }
        }
        if (allDone) {
            return;
        }
    } while (true);
}

// 单 tile reduceSum: 计算核将 tileIdx 行区间的 rankSize 份部分和归约写入输出区
template <typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void QuantMatmulReduceScatterImpl<AType, BType, CType, TransA, TransB>::ReduceSumTile(
    const AivRunContext &ctx, uint32_t tileIdx, uint32_t fullBatch)
{
    uint32_t tileM = GetTileM(tileIdx);
    if (tileM == 0) {
        return;
    }
    uint64_t tileMOffset = GetTileMOffset(tileIdx);
    DoReduceSum(tileM, tileMOffset, ctx.nSize, ctx.rowBytes, ctx.rowStride, ctx.rsCoreNum, fullBatch);
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
    uint32_t tileM, uint64_t tileMOffset, uint32_t nSize, uint32_t rowBytes, uint64_t stride, uint32_t rsCoreNum,
    uint32_t fullBatch)
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
    // UB 低地址保留区: commBuf (COMM_WORKSPACE_SIZE) + barrierBuf/fence 轮询 buffer
    // (BARRIER_FLAG_SIZE), RS slot 布局整体上移避开, 防轮询 MTE2 写与 RS V 读竞态
    rsParams.ubReservedBytes = COMM_WORKSPACE_SIZE + BARRIER_FLAG_SIZE;
    rsParams.fullBatch = fullBatch;
    reduceSum_.Init(rsParams);
    reduceSum_.ExecuteReduceSum(tileM);
}

} // namespace Apace
