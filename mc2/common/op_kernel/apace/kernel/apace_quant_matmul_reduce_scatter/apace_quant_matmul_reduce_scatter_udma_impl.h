/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file apace_quant_matmul_reduce_scatter_udma_impl.h
 * \brief v2: 3 级流水 ReduceScatter (AllToAll PUT + workspace 轮转)
 *
 * v2 架构改造（相比 v1）：
 *   1. 3 级流水：AIC mm → AIV aivComm → AIV reduceSum，PIPELINE_NUM=4 多 buffer workspace 轮转
 *   2. workspace 入参：mm 输出写 workspace，reduceSum 从 workspace 读取
 *   3. 无 AtomicAdd：L0C 跨 rank FP32 累加后单次 fixpipe 写 workspace，reduceSum 仅 DataCopyPad 搬运
 *   4. AIV 时分复用：先 AllToAll PUT 通信，后 reduceSum
 *   5. 3 通道 CrossCore flag 编排（通信就绪 + mm 完成 + 回压）
 *   6. localMatmul 默认 0（REMOTE-only），可选 1（DEFERRED_SYNC）
 *
 * Run():
 *   AIV: RunAllToAll() → RunReduceSum()  (时分复用：先通信后 reduceSum)
 *   AIC: RunMatmul()   (mm → L0C 跨 rank 累加 → fixpipe → workspace)
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
#include "quant_matmul_mx_kernel.h"
#include "apace_quant_matmul_reduce_scatter_tiling_data.h"

#include "apace/block/aiv_comm/collective_comm_api.h"
#include "apace/block/aiv_comm/collective_comm_context.h"
#include "apace/tiling/comm_tiling_data.h"
#include "apace/block/aiv_comm/barrier/barrier_ubmem.h"

namespace Apace {

using namespace AscendC;
using namespace Blaze::Gemm;
using AscendC::Te::Get;
using namespace Apace::AivComm;

// 通信就绪 flagId 映射（DESIGN.md §2.6）：避开 3（回压）、5（mm 完成）、[11-14]（SyncAll 硬同步占用）
// 可用集 {0,1,2,4,6,7,8,9,10,15} 共 10 个，host 侧已约束 tileCnt ≤ 10。
// 注：DESIGN 表格公式 "(tid<3)?tid:(tid<5)?tid+1:tid+2" 在 tid=4 处会映射到 5（与 mm 完成通道冲突），
// 此处按可用集修正：tid 0,1,2→0,1,2；tid=3→4；tid=4..8→tid+2；tid=9→15。
__aicore__ inline uint64_t CommFlagIdMapped(uint32_t tid)
{
    uint64_t fid = (tid < 3) ? tid : ((tid == 3) ? 4 : (tid + 2));
    if (fid >= 11 && fid <= 14) {
        fid = 15;
    }
    return fid & 0xF;
}

// UDMA 通信等待策略：AIC 等 AIV 通信完成的 CrossCore flag
struct UdmaCommWaitPolicy {
    __aicore__ inline void WaitTile(uint32_t tileIdx)
    {
        AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(CommFlagIdMapped(tileIdx));
    }
};

// 定义问题形状：[M, N, K, Batch]
using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

/**
 * @brief v2 ReduceScatter (AllToAll PUT + workspace 流水线) MX 量化 Matmul 核心实现类
 *
 * 3 级流水架构：
 * 1. AIC: Blaze QuantMatmul → L0C FP32 跨 rank 累加 → fixpipe → workspace[BF16]
 * 2. AIV Phase 1: AllToAll PUT 通信（发送 A 数据到各 rank Win 区）
 * 3. AIV Phase 2: reduceSum（DataCopyPad workspace→UB→yGm，无 AtomicAdd）
 *
 * CrossCore flag 三通道编排（950PR 模板参数生效，modeId=2）：
 *   通道 1（通信就绪）：AIV SetFlag<0x2, PIPE_MTE3>(tid) → AIC WaitFlag<0x2, PIPE_MTE2>(tid)
 *   通道 2（mm 完成）：AIC SetFlag<SYNC_MODE2, PIPE_FIX>(5) → AIV WaitFlag<SYNC_MODE2, PIPE_MTE2>(5)
 *   通道 3（回压）：AIV SetFlag<SYNC_MODE2, PIPE_MTE3>(3) → AIC WaitFlag<SYNC_MODE2, PIPE_M>(3)
 */
template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
class ApaceQuantMatmulReduceScatterUdmaImpl {
public:
    __aicore__ inline ApaceQuantMatmulReduceScatterUdmaImpl() {};
    __aicore__ inline ~ApaceQuantMatmulReduceScatterUdmaImpl() {}

    /**
     * @brief 初始化算子状态和参数（v2 新增 workspaceGM）
     */
    __aicore__ inline void Init(__gm__ CommContext *hcommCtx,
                    GM_ADDR aGM, GM_ADDR scaleAGM,
                    GM_ADDR bGM, GM_ADDR scaleBGM,
                    GM_ADDR cGM,
                    GM_ADDR workspaceGM,
                    const ApaceQuantMatmulReduceScatterTilingData *tilingData);
    /**
     * @brief 执行算子逻辑（AIV 时分复用：先通信后 reduceSum；AIC mm 写 workspace）
     */
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

    // 组件定义
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape,
        NONE_FULL_LOAD_MODE, LayoutA, LayoutB, TypeA>;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<NONE_FULL_LOAD_MODE, false>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, TypeA, LayoutA, TypeB, LayoutB, TypeC, LayoutC, BiasType, LayoutBias>;
    using QuantMatmulKernelImpl = Kernel::QuantMatmulMxKernel<ProblemShape, BlockMmad, BlockScheduler,
        UdmaCommWaitPolicy>;

    // 参数类型
    using Params = typename QuantMatmulKernelImpl::Params;
    using BlockMmadParams = typename QuantMatmulKernelImpl::BlockMmadParams;
    using L1Params = typename QuantMatmulKernelImpl::L1Params;
    using LocalParams = typename QuantMatmulKernelImpl::LocalParams;
    using BlockSchedulerParams = typename QuantMatmulKernelImpl::BlockSchedulerParams;
    using QBMMTiling = typename QuantMatmulKernelImpl::QBMMTiling;
    using MatmulMode = typename QuantMatmulKernelImpl::MatmulMode;

    // v2: BlockScheduler 调度相关类型（从 QuantMatmulKernelImpl 透传）
    using BlockShape = typename QuantMatmulKernelImpl::BlockShape;
    using BlockCoord = typename QuantMatmulKernelImpl::BlockCoord;
    static constexpr bool weightNz = QuantMatmulKernelImpl::weightNz;

    QuantMatmulKernelImpl quantMatmulKernelImpl_;

private:
    // ---------------- 通信相关组件与参数 ----------------
    __gm__ CommUbmemContext* syncBuffer_{nullptr};
    __gm__ CommUdmaContext* udmaCtx_{nullptr};
    CollectiveComm<CommCollectiveOp::AllToAll, CommMode::PUT, AType, TeamBarrier> allToAllA_;
    CollectiveComm<
        CommCollectiveOp::AllToAll, CommMode::PUT, TypeScaleA, TeamBarrier> allToAllScaleA_;
    TeamBarrier teamBarrier_;

    // v2 reduceSum UB buffer（DataCopyPad 暂存）
    TPipe pipe_;  // v2: AIV 侧 UB 管理
    TBuf<TPosition::VECCALC> reduceSumBuf_;
    // v2 reduceSum 流水同步事件（MTE2→MTE3 RAW、MTE3→MTE2 WAR）
    AscendC::TEventID evtMte2ToMte3_;
    AscendC::TEventID evtMte3ToMte2_;
    __gm__ CType* workspaceGm_{nullptr};  // v2 workspace 基址
    GlobalTensor<CType> yGm_;             // v2 输出 yGm（reduceSum 写入目标）
    GlobalTensor<CType> workspaceGmGlobal_;  // v2 workspace GlobalTensor（reduceSum 读取源）

    struct BaseParams {
        GM_ADDR selfWinAddr{nullptr}; // 通信窗口地址
        GM_ADDR aGm{nullptr};
        GM_ADDR scaleAGm{nullptr};
        GM_ADDR bGm{nullptr};
        GM_ADDR scaleBGm{nullptr};
        GM_ADDR cGm{nullptr};
        GM_ADDR workspaceGm{nullptr};  // v2 新增：workspace 基址

        uint32_t rankId{0};
        int32_t  commTurn{0};     // 总流水步数
        uint32_t rankSize{0};    // 卡数
        uint64_t axisM{0};     // M 轴总大小 (= M/rankSize)
        uint64_t axisN{0};     // N 轴大小
        uint64_t axisKa{0};    // K 轴大小 (per-rank K)
        uint64_t headMSize{0};   // 长块 M 大小
        uint64_t scaleKaSize{0};   // Scale 的 K 轴字节长度
        uint64_t rankDataBytes{0};
        // v2 新增字段
        uint32_t pipelineNum{PIPELINE_NUM};  // PIPELINE_NUM=4
        uint32_t coreNum{0};                    // AIC 核数
        // totalMmTiles 已废除（DESIGN.md §2.16 Issue 2）：AIV 循环由 bsAiv.GetTileIdx() 隐式终止
        uint32_t baseM{0};                      // mm tile baseM
        uint32_t baseN{0};                      // mm tile baseN
    } baseParams_;

    const ApaceQuantMatmulReduceScatterTilingData* tilingData_{nullptr};

    // ---------------- 私有方法 ----------------
    __aicore__ inline void InitBaseParams(const ApaceQuantMatmulReduceScatterTilingData *tilingData);
    __aicore__ inline void SetupParams(const QuantMatmulTilingData* mmTile,
                        Params& out, MatmulMode matmulMode);

    __aicore__ inline void RunAllToAll();    // AIV Phase 1: AllToAll PUT 通信
    __aicore__ inline void RunReduceSum();   // AIV Phase 2: reduceSum (workspace→yGm)
    __aicore__ inline void RunMatmul();      // AIC: mm → workspace
};

// =================================================================================
// 公有方法实现
// =================================================================================

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void ApaceQuantMatmulReduceScatterUdmaImpl<AType, BType, CType, TransA, TransB>::Init(
    __gm__ CommContext *hcommCtx, GM_ADDR aGM, GM_ADDR scaleAGM, GM_ADDR bGM, GM_ADDR scaleBGM, GM_ADDR cGM,
    GM_ADDR workspaceGM, const ApaceQuantMatmulReduceScatterTilingData *tilingData)
{
    tilingData_ = tilingData;
    baseParams_.aGm = aGM;
    baseParams_.scaleAGm = scaleAGM;
    baseParams_.bGm = bGM;
    baseParams_.scaleBGm = scaleBGM;
    baseParams_.cGm = cGM;
    baseParams_.workspaceGm = workspaceGM;  // v2 新增
    workspaceGm_ = reinterpret_cast<__gm__ CType*>(workspaceGM);
    yGm_.SetGlobalBuffer((__gm__ CType*)cGM);  // v2 输出 yGm
    workspaceGmGlobal_.SetGlobalBuffer(workspaceGm_);  // v2 workspace GlobalTensor

    syncBuffer_ = &(hcommCtx->ubmemCtx);
    udmaCtx_ = &(hcommCtx->udmaCtx);
    baseParams_.rankId = udmaCtx_->rankId;
    baseParams_.rankSize = udmaCtx_->rankSize;

    InitBaseParams(tilingData);

    baseParams_.selfWinAddr = reinterpret_cast<GM_ADDR>(udmaCtx_->commBufferAddrs[baseParams_.rankId]);
    uint32_t ubOffset = 0;
    auto commBuf = AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, uint8_t>(ubOffset); // 512B
    ubOffset += COMM_WORKSPACE_SIZE;
    auto commScaleBuf = AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, uint8_t>(ubOffset); // 512B
    ubOffset += COMM_WORKSPACE_SIZE;
    auto barrierBuf = AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, uint8_t>(ubOffset);
    teamBarrier_.Init(barrierBuf.Get(), syncBuffer_, baseParams_.rankSize, static_cast<uint32_t>(GetBlockIdx()));

    allToAllA_.template Init<BARRIER_NONE>(udmaCtx_, teamBarrier_, tilingData->commTilingData, baseParams_.aGm,
        commBuf.Get(), baseParams_.rankSize, static_cast<uint32_t>(GetBlockIdx()));

    allToAllScaleA_.Init(udmaCtx_, teamBarrier_, tilingData->scaleCommTilingData, baseParams_.scaleAGm,
        commScaleBuf.Get(), baseParams_.rankSize, static_cast<uint32_t>(GetBlockIdx()),
        baseParams_.rankSize * baseParams_.rankDataBytes);

    // v2: AIV 侧初始化 reduceSum UB buffer
    // UB 行距按 64B（32 元素）对齐取安全上界（GM→UB 2D 搬运 dstStride 公式为 64B 对齐，
    // 见 DESIGN.md §2.16 DataCopyPad 参数修正；baseN 32 对齐时与原 baseM×baseN×2B 相同）
    if ASCEND_IS_AIV {
        uint32_t alignBaseN = (baseParams_.baseN + 31) & ~31U;
        pipe_.InitBuffer(reduceSumBuf_, baseParams_.baseM * alignBaseN * sizeof(CType));
        // reduceSum 流水同步事件：GM→UB(MTE2) 与 UB→GM(MTE3) 之间的 RAW/WAR 依赖
        evtMte2ToMte3_ = pipe_.FetchEventID(HardEvent::MTE2_MTE3);
        evtMte3ToMte2_ = pipe_.FetchEventID(HardEvent::MTE3_MTE2);
    }
}

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void ApaceQuantMatmulReduceScatterUdmaImpl<AType, BType, CType, TransA, TransB>::Run()
{
    if ASCEND_IS_AIV {
        RunAllToAll();     // Phase 1: AllToAll PUT 通信
        RunReduceSum();    // Phase 2: reduceSum (workspace→yGm)
    }

    if ASCEND_IS_AIC {
        RunMatmul();
    }
}

// =================================================================================
// AIV Phase 1: 通信实现 (与 v1 相同)
// =================================================================================

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void ApaceQuantMatmulReduceScatterUdmaImpl<AType, BType, CType, TransA, TransB>::RunAllToAll()
{
    for (uint32_t tid = 0; tid < baseParams_.commTurn; ++tid) {
        // 必选保证 baseParams_.rankSize <= BlockNum
        if (AscendC::GetBlockIdx() < baseParams_.rankSize) {
            allToAllScaleA_.Commit();
            allToAllA_.Commit();
            allToAllA_.template Wait<BARRIER_DEVICE>();
        }

        AscendC::SyncAll<true>();
        // 通道 1：通知 AIC 通信完成（per-tile flagId = CommFlagIdMapped(tid)，避开 3/5/[11-14]）
        CrossCoreSetFlag<0x2, PIPE_MTE3>(CommFlagIdMapped(tid));
    }

    allToAllScaleA_.Finalize();
    allToAllA_.Finalize();
}

// =================================================================================
// AIV Phase 2: reduceSum 实现 (v2 新增)
// =================================================================================

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void ApaceQuantMatmulReduceScatterUdmaImpl<AType, BType, CType, TransA, TransB>::RunReduceSum()
{
    // v2 reduceSum：AIV 复用 BlockScheduler（DESIGN.md §2.16 方案 b）
    // 与 AIC ProcessSingleBatch 完全相同的 problemShape/schParams 构造调度器、完全相同的调用序列
    // 枚举 tile，mPos/nPos/curBaseM/curBaseN 由调度器推导，循环由 bsAiv.GetTileIdx() 隐式终止
    // （与 AIC SetFlag(5) 次数构造一致，不需要 totalMmTiles）
    Params params;
    MatmulMode mode = (tilingData_->localMatmul == 1) ? MatmulMode::DEFERRED_SYNC : MatmulMode::REMOTE;
    SetupParams(&tilingData_->tileQbmmTilingData, params, mode);

    // 复刻项 6：coreIdx = GetBlockIdx()（MIX_AIC_1_1 下 AIV 与配对 AIC 一致）
    uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());
    uint32_t coreNum = baseParams_.coreNum;
    uint32_t pipelineNum = baseParams_.pipelineNum;
    uint32_t baseM = baseParams_.baseM;
    uint32_t baseN = baseParams_.baseN;
    uint64_t N = baseParams_.axisN;

    // 复刻项 1：同一 problemShape{m,n,k,1} + 同一 schParams 构造调度器（复用同一 SetupParams 产物）
    BlockScheduler bsAiv(params.problemShape, params.schParams);
    // 复刻项 2：UpdateTailTile 判定逐字复刻（restBatch=0, isTailRound=true）
    auto& mTailTile = params.schParams.mTailTile;
    auto& nTailTile = params.schParams.nTailTile;
    if (((bsAiv.GetEndBlockIdx() + 1)) * mTailTile * nTailTile <= AscendC::GetBlockNum()) {
        bsAiv.UpdateTailTile(mTailTile, nTailTile);
    }

    LocalTensor<CType> reduceSumLocal = reduceSumBuf_.Get<CType>();
    uint32_t cubeNum = 0;  // 复刻项 5：仅在实际处理的 tile 上自增（SetFlag(3) 之后）
    BlockCoord blockIdx;
    int64_t mPos = 0L;
    int64_t nPos = 0L;
    // 复刻项 3：循环内调用顺序 GetTileIdx → GetBlockShape（同模板参数）→ 零形状检查 → GetTileCoord
    while (bsAiv.GetTileIdx(blockIdx)) {
        BlockShape singleShape =
            bsAiv.template GetBlockShape<QuantMode::MX_PERGROUP_MODE, QuantMode::MX_PERGROUP_MODE, weightNz>(
                blockIdx);
        if ((Te::Get<Blaze::Gemm::IDX_M_TILEIDX>(singleShape) <= 0) ||
            (Te::Get<Blaze::Gemm::IDX_N_TILEIDX>(singleShape) <= 0)) {
            break;  // 复刻项 4：零形状 break（对齐 AIC 的 return 早退，终止本 AIV 后续所有 WaitFlag(5)）
        }
        bsAiv.GetTileCoord(blockIdx, mPos, nPos);
        uint32_t curBaseM = static_cast<uint32_t>(Te::Get<Blaze::Gemm::IDX_M_TILEIDX>(singleShape));
        uint32_t curBaseN = static_cast<uint32_t>(Te::Get<Blaze::Gemm::IDX_N_TILEIDX>(singleShape));

        // 通道 2：等 AIC mm 完成（flagId=5，阻塞 MTE2 流水）
        CrossCoreWaitFlag<SYNC_MODE2, PIPE_MTE2>(SYNC_MM_DONE_FLAG);

        // workSpaceOffset 与 AIC 侧一致（coreIdx 两侧相同，cubeNum 序列构造一致）
        uint64_t workSpaceOffset = static_cast<uint64_t>(baseM) * baseN *
            (coreIdx + (cubeNum % pipelineNum) * coreNum);

        // GM→UB 2D 搬运（DESIGN.md §2.16 DataCopyPad 参数修正）：
        // slot 内数据为 fixpipe 按 MakeLayoutC(curBaseM, curBaseN) 写入的紧密排列（行距=curBaseN），srcStride=0；
        // UB 内行距按 64B 对齐 pitch = rowBytes + dstStride*32，dstStride 公式同 ops-nn DataCopyPad2D（单位 32B 块）
        uint32_t rowBytes = curBaseN * sizeof(CType);
        DataCopyExtParams cpIn;
        cpIn.blockCount = curBaseM;
        cpIn.blockLen = rowBytes;
        cpIn.srcStride = 0;
        cpIn.dstStride = (rowBytes + 63) / 64 * 2 - (rowBytes + 31) / 32;
        DataCopyPadExtParams<CType> padIn{false, 0, 0, 0};
        // WAR：上一轮 UB→GM(MTE3) 读取完成后才能覆写 UB
        if (cubeNum > 0) {
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(evtMte3ToMte2_);
        }
        DataCopyPad(reduceSumLocal, workspaceGmGlobal_[workSpaceOffset], cpIn, padIn);
        // RAW：GM→UB(MTE2) 完成后 UB→GM(MTE3) 才能读取 UB
        AscendC::SetFlag<HardEvent::MTE2_MTE3>(evtMte2ToMte3_);
        AscendC::WaitFlag<HardEvent::MTE2_MTE3>(evtMte2ToMte3_);

        // UB→yGm（2D 搬运，无 AtomicAdd）：outOffset = mPos * N + nPos（yGm 行主序，行距 N）
        // 注意（对 DESIGN.md §2.16 公式的实现级修正）：UB 读行距必须等于 GM→UB 建立的写行距
        // （pitch = rowBytes + dstStride*32，64B 对齐）。DESIGN 原文 srcStride=(AlignUp(curBaseN,16)-curBaseN)*2/32
        // 对应 32B 对齐 pitch，与写侧 64B pitch 在 rowBytes%64!=0 时不一致（如 baseN=48 → rowBytes=96），
        // 会导致除每 tile 首行外全部错位。故 srcStride 取与 cpIn.dstStride 相同值。
        DataCopyExtParams cpOut;
        cpOut.blockCount = curBaseM;
        cpOut.blockLen = rowBytes;
        cpOut.srcStride = cpIn.dstStride;
        cpOut.dstStride = (N - curBaseN) * sizeof(CType);
        uint64_t outOffset = static_cast<uint64_t>(mPos) * N + static_cast<uint64_t>(nPos);
        DataCopyPad(yGm_[outOffset], reduceSumLocal, cpOut);
        // WAR 标记：供下一轮 GM→UB 等待
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(evtMte3ToMte2_);

        // 通道 3：通知 AIC workspace 可复用（flagId=3，每核每 tile 一次，与 AIC 回压 Wait 配对）
        CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
        cubeNum++;
    }
}

// =================================================================================
// AIC: mm 计算实现 (v2 改造：mm 输出写 workspace)
// =================================================================================

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void ApaceQuantMatmulReduceScatterUdmaImpl<AType, BType, CType, TransA, TransB>::InitBaseParams(
    const ApaceQuantMatmulReduceScatterTilingData *tilingData)
{
    baseParams_.headMSize = tilingData->commTilingData.splitAxisTileSize;
    baseParams_.commTurn = tilingData->commTilingData.splitAxisTileCnt
                        + tilingData->commTilingData.splitAxisTailCnt;

    baseParams_.axisM = tilingData->tileQbmmTilingData.m;
    baseParams_.axisN = tilingData->tileQbmmTilingData.n;
    baseParams_.axisKa = tilingData->tileQbmmTilingData.k;
    // v2 新增字段
    baseParams_.pipelineNum = tilingData->pipelineNum;
    baseParams_.coreNum = tilingData->coreNum;
    // totalMmTiles 已废除（DESIGN.md §2.16 Issue 2）：kernel 不消费该字段
    baseParams_.baseM = tilingData->tileQbmmTilingData.baseM;
    baseParams_.baseN = tilingData->tileQbmmTilingData.baseN;

    // 计算 MXFP8 格式下 Scale 每一行的字节数
    baseParams_.scaleKaSize =
        CeilDiv(baseParams_.axisKa, Blaze::Gemm::MXFP_DIVISOR_SIZE) * Blaze::Gemm::MXFP_MULTI_BASE_SIZE;

    // 计算地址偏移所需的参数
    baseParams_.rankDataBytes = baseParams_.axisM * baseParams_.axisKa * sizeof(AType);
}

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void ApaceQuantMatmulReduceScatterUdmaImpl<AType, BType, CType, TransA, TransB>::SetupParams(
    const QuantMatmulTilingData* mmTile, Params& out, MatmulMode matmulMode)
{
    ProblemShape problemShape{mmTile->m, mmTile->n, mmTile->k, 1UL};
    BlockMmadParams mmadParams;
    // v2: localParams 新增 workspaceGmAddr/pipelineNum/coreNum
    LocalParams localParams;
    localParams.rankId = baseParams_.rankId;
    localParams.rankSize = baseParams_.rankSize;
    localParams.originalM = baseParams_.axisM;
    localParams.localAGmAddr = baseParams_.aGm;
    localParams.localScaleAGmAddr = baseParams_.scaleAGm;
    localParams.localMatmul = tilingData_->localMatmul;
    localParams.matmulMode = matmulMode;
    localParams.headTileSize = static_cast<uint32_t>(baseParams_.headMSize);
    // v2 新增
    localParams.workspaceGmAddr = baseParams_.workspaceGm;
    localParams.pipelineNum = baseParams_.pipelineNum;
    localParams.coreNum = baseParams_.coreNum;

    L1Params l1Params{static_cast<uint64_t>(mmTile->stepK) * mmTile->baseK, mmTile->scaleKL1,
        mmTile->nBufferNum};

    // v2: mm 输出到 workspace（不再到 yGm）
    // localMatmul=0（默认）: REMOTE-only，splitKNum=rankSize
    // localMatmul=1（可选）: DEFERRED_SYNC，splitKNum=rankSize
    localParams.splitKNum = baseParams_.rankSize;

    mmadParams.bGmAddr = baseParams_.bGm;
    mmadParams.scaleBGmAddr = baseParams_.scaleBGm;
    // cGmAddr 在 RunMatmul 中设置为 workspace

    // 调度器参数
    BlockSchedulerParams schedulerParams{
        mmTile->baseM, mmTile->baseN, mmTile->mTailTile, mmTile->nTailTile,
        mmTile->mBaseTailSplitCnt, mmTile->nBaseTailSplitCnt, mmTile->mTailMain, mmTile->nTailMain};
    // 基础 Tiling 参数
    QBMMTiling qbmmParams{mmTile->baseM, mmTile->baseN, mmTile->baseK, mmTile->dbL0c, false};

    out = {problemShape, mmadParams, l1Params, schedulerParams, qbmmParams, localParams};
}

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void ApaceQuantMatmulReduceScatterUdmaImpl<AType, BType, CType, TransA, TransB>::RunMatmul()
{
    Params params;
    // v2: localMatmul=0 → REMOTE-only；localMatmul=1 → DEFERRED_SYNC
    MatmulMode mode = (tilingData_->localMatmul == 1) ? MatmulMode::DEFERRED_SYNC : MatmulMode::REMOTE;
    SetupParams(&tilingData_->tileQbmmTilingData, params, mode);
    // A 从 Win 区读取（与 v1 相同）
    params.mmadParams.aGmAddr = baseParams_.selfWinAddr;
    params.mmadParams.scaleAGmAddr = baseParams_.selfWinAddr + baseParams_.rankSize * baseParams_.rankDataBytes;
    // v2 改造：mm 输出到 workspace（不再到 yGm）
    params.mmadParams.cGmAddr = baseParams_.workspaceGm;
    params.localParams.localAGmAddr = baseParams_.aGm;
    params.localParams.localScaleAGmAddr = baseParams_.scaleAGm;
    quantMatmulKernelImpl_(params);
}

} // namespace Apace
