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
 *   AIC: for t = 0..commTurn-1:
 *          构建 FragmentTensor (rankSize 个 fragment, 每个 curTileM 行, 间距 mPerRank 行)
 *          matmul(rankSize * curTileM) → workspace
 *          remote→local 边界 SetFlag(1) 提前通知 AIV 启动 AllToAll (与本地段计算重叠)
 *          本 tile 全部完成后 SetFlag(0)
 *   AIV: for t = 0..commTurn-1:
 *          WaitFlag(1) → SyncAll → 通信核 AllToAll(t) ‖ 计算核 ReduceSum(t-1)
 *          → WaitFlag(0) → SyncAll → CrossDevice → SyncAll; 末 tile ReduceSum 循环后补做
 *
 * 关键设计:
 *   - 无 R 循环: FragmentTensor 打包 rankSize 个 fragment, 一次调用覆盖一个 tile 的全部 rank 段
 *   - AIC 侧无 if/else: 统一 for t 循环, T=1 自然退化 (AIV 侧有 commTurn<=1 单 tile 快速路径)
 *   - L1 局部性: BlockMmadFragC per-fragment 独立加载 L1 (与 R 循环一致)
 *   - flag 计数式: 每 tile 一次 Set(1)/Set(0), AIV 按序对应 Wait(1)/Wait(0)
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

using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

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
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;
    using BiasType = float;
    using LayoutScaleA = typename AscendC::Te::FrameLayoutFormat<
        AscendC::Std::conditional_t<TransA, AscendC::Te::ScaleADNLayoutPtn, AscendC::Te::ScaleANDLayoutPtn>,
        AscendC::Std::Int<SCALE_C0>>;
    using LayoutScaleB = typename AscendC::Te::FrameLayoutFormat<
        AscendC::Std::conditional_t<TransB, AscendC::Te::ScaleBDNLayoutPtn, AscendC::Te::ScaleBNDLayoutPtn>,
        AscendC::Std::Int<SCALE_C0>>;

    // 组件定义 — 使用 FragmentTensor 版 kernel (参考 allgather QmmMxKernelAgUdma)
    using FragKernelImpl = ReduceScatterFragImpl::QmmMxKernelRsFrag<AType, BType, CType>;
    using FragKernelParams = typename FragKernelImpl::Params;
    using FragQBMMTiling = typename FragKernelImpl::QBMMTiling;
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
        uint64_t axisM{0};          // M 轴总大小
        uint64_t axisN{0};          // N 轴大小
        uint64_t axisK{0};          // K 轴大小
        uint64_t headMSize{0};      // 通信头块 M 大小 (= chunkM for single tile)
        uint64_t tailMSize{0};      // 通信尾块 M 大小 (0 if no tail)
        uint32_t headTileCnt{0};    // 头块数量
        uint32_t tailTileCnt{0};    // 尾块数量
        uint64_t chunkBytes{0};     // 每个 rank 的 C 数据块字节数
        uint32_t chunkM{0};         // 每个 rank 的 M 分片大小 (m / rankSize)
        uint64_t scaleARowBytes{0}; // ScaleA 每行字节数 = CeilDiv(axisK, 64) * 2 (MXFP8)
    } baseParams_;

    // ---------------- reduceSum 组件 ----------------
    AiVReduceSumImpl::ReduceSumForAlltoAll<CType> reduceSum_;

    const QuantMatmulReduceScatterTilingData *tilingData_{nullptr};

    // ---------------- 私有方法 ----------------
    __aicore__ inline void InitBaseParams(const QuantMatmulReduceScatterTilingData *tilingData);
    __aicore__ inline void RunMatmul();                    // AIC: per-tile matmul → workspaceGM
    __aicore__ inline void RunAllToAllAndReduceSum();      // AIV: per-tile WaitFlag + AllToAll PUT + reduceSum
    __aicore__ inline uint32_t GetTileM(uint32_t t);       // 获取 tile t 的 M 大小
    __aicore__ inline uint64_t GetTileMOffset(uint32_t t); // 获取 tile t 的 M 偏移
    __aicore__ inline void DoReduceSum(uint32_t tileIdx, uint32_t tileM, uint64_t tileMOffset, uint32_t nSize,
                                       uint32_t rowBytes, uint64_t stride, uint32_t usedCoreNum, uint32_t blockIdx);
    // 单 tile (commTurn<=1) 路径专用; 批量接口存在时序竞态, 已回退为逐行 ExecuteReduceSum
    __aicore__ inline void DoReduceSumBatch(uint32_t tileIdx, uint32_t tileM, uint64_t tileMOffset, uint32_t nSize,
                                            uint32_t rowBytes, uint64_t stride, uint32_t usedCoreNum,
                                            uint32_t blockIdx);
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
    baseParams_.tailTileCnt = static_cast<uint32_t>(tilingData->commTilingData.splitAxisTailCnt);
    baseParams_.commTurn = tilingData->commTilingData.splitAxisTileCnt + tilingData->commTilingData.splitAxisTailCnt;

    baseParams_.axisM = tilingData->mmTilingData.m;
    baseParams_.axisN = tilingData->mmTilingData.n;
    baseParams_.axisK = tilingData->mmTilingData.k;

    // scaleARowBytes = CeilDiv(axisK, 64) * 2 (MXFP8 scale: 每组 64 元素, 2 字节)
    baseParams_.scaleARowBytes = (baseParams_.axisK + 63U) / 64U * 2U;

    // chunkSize = 每个 rank 分到的 M 行数 = m / rankSize
    // CommTilingData: chunkSize = splitAxisTileSize * splitAxisTileCnt + splitAxisTailSize * splitAxisTailCnt
    uint64_t chunkSize = tilingData->commTilingData.splitAxisTileSize * tilingData->commTilingData.splitAxisTileCnt +
                         tilingData->commTilingData.splitAxisTailSize * tilingData->commTilingData.splitAxisTailCnt;
    baseParams_.chunkM = static_cast<uint32_t>(chunkSize);
    // chunkBytes = chunkSize * n * sizeof(CType)
    baseParams_.chunkBytes = chunkSize * tilingData->commTilingData.nonSplitAxisSize * sizeof(CType);
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
    //   无 R 循环, 无 if/else — 统一 for t 循环, T=1 自然退化
    const auto &mmT = tilingData_->mmTilingData;
    const auto &commTd = tilingData_->commTilingData;
    uint32_t commTurn = static_cast<uint32_t>(baseParams_.commTurn);
    uint32_t R = baseParams_.rankSize;

    for (uint32_t t = 0; t < commTurn; ++t) {
        uint32_t curTileM = GetTileM(t);
        uint64_t tileMOffset = GetTileMOffset(t);

        const auto &subMmT = tilingData_->subMmTilingData;
        const auto &tailMmT = tilingData_->tailMmTilingData;
        bool isHeadTile = (t < baseParams_.headTileCnt);
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
        fragParams.headRows = R * curTileM;
        fragParams.rankId = baseParams_.rankId;
        fragParams.rankSize = R;
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

        CrossCoreSetFlag<0x2, PIPE_FIX>(0); // all done (remote+local)
    }
}

// =================================================================================
// AIV 侧: per-tile WaitFlag + AllToAll PUT + reduceSum (流水掩盖)
// =================================================================================

template <typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void QuantMatmulReduceScatterImpl<AType, BType, CType, TransA, TransB>::RunAllToAllAndReduceSum()
{
    // 严格分离 — 后 R 核仅通信, 前 (usedCoreNum-R) 核仅计算
    //   通信核: AllToAll(t)
    //   计算核: ReduceSum(t-1) ‖ 通信核 AllToAll(t)
    //   通信核不补做 reduceSum

    uint32_t commTurn = static_cast<uint32_t>(baseParams_.commTurn);
    uint32_t nSize = static_cast<uint32_t>(baseParams_.axisN);
    uint32_t rowBytes = nSize * sizeof(CType);
    uint32_t usedCoreNum = tilingData_->mmTilingData.usedCoreNum;
    uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
    uint64_t stride = static_cast<uint64_t>(baseParams_.chunkM) * nSize;
    uint32_t rankSize = baseParams_.rankSize;
    uint32_t jobIndex = static_cast<uint32_t>(GetBlockNum() - 1 - blockIdx);
    bool isCommBlock = (jobIndex < rankSize);
    uint32_t rsCoreNum = usedCoreNum - rankSize; // 计算核数 (前 usedCoreNum-R 个)
    bool isComputeBlock = (blockIdx < rsCoreNum);

    if (commTurn <= 1) {
        // localMatmul 后置: AIC 在 remote→local 边界 SetFlag(1), 全部完成后 SetFlag(0)
        // comm: WaitFlag(1) → SyncAll(跨core同步, 确保所有 AIC remote 完成) → AllToAll → WaitFlag(0)
        // comp: WaitFlag(1) → SyncAll → WaitFlag(0) → ReduceSum
        if (isCommBlock) {
            CrossCoreWaitFlag<0x2, PIPE_MTE2>(1); // remote done on this core
        }
        if (isComputeBlock) {
            CrossCoreWaitFlag<0x2, PIPE_MTE2>(1); // consume remote flag
        }
        SyncAll<true>(); // #1: all cores' remote done

        if (isCommBlock) {
            allToAllC_.template Commit<BARRIER_NONE>();
            allToAllC_.template Wait<BARRIER_NONE>();
            CrossCoreWaitFlag<0x2, PIPE_MTE2>(0); // consume local flag
        }
        if (isComputeBlock) {
            CrossCoreWaitFlag<0x2, PIPE_MTE2>(0); // all done (local included)
        }
        SyncAll<true>(); // #2: AllToAll done / all matmul done
        teamBarrier_.CrossDevice();
        SyncAll<true>(); // #3: remote data visible
        if (isComputeBlock) {
            uint32_t tileM = GetTileM(0);
            uint64_t tileMOffset = GetTileMOffset(0);
            if (tileM > 0) {
                DoReduceSumBatch(0, tileM, tileMOffset, nSize, rowBytes, stride, rsCoreNum, blockIdx);
            }
        }
        if (isCommBlock) {
            allToAllC_.template Finalize<BARRIER_NONE>();
        }
        return;
    }

    // 多 tile
    if (isCommBlock) {
        CrossCoreWaitFlag<0x2, PIPE_MTE2>(1);
    }
    if (isComputeBlock) {
        CrossCoreWaitFlag<0x2, PIPE_MTE2>(1);
    }
    SyncAll<true>();
    if (isCommBlock) {
        allToAllC_.template Commit<BARRIER_NONE>();
        allToAllC_.template Wait<BARRIER_NONE>();
        CrossCoreWaitFlag<0x2, PIPE_MTE2>(0);
    }
    if (isComputeBlock) {
        CrossCoreWaitFlag<0x2, PIPE_MTE2>(0);
    }
    SyncAll<true>();
    teamBarrier_.CrossDevice();
    SyncAll<true>();

    for (uint32_t t = 1; t < commTurn; ++t) {
        if (isCommBlock) {
            CrossCoreWaitFlag<0x2, PIPE_MTE2>(1);
        }
        if (isComputeBlock) {
            CrossCoreWaitFlag<0x2, PIPE_MTE2>(1);
        }
        SyncAll<true>();

        if (isCommBlock) {
            allToAllC_.template Commit<BARRIER_NONE>();
            allToAllC_.template Wait<BARRIER_NONE>();
            CrossCoreWaitFlag<0x2, PIPE_MTE2>(0);
        } else if (isComputeBlock) {
            CrossCoreWaitFlag<0x2, PIPE_MTE2>(0);
            uint32_t rsT = t - 1;
            uint32_t rsTileM = GetTileM(rsT);
            uint64_t rsTileMOffset = GetTileMOffset(rsT);
            if (rsTileM > 0) {
                DoReduceSum(rsT, rsTileM, rsTileMOffset, nSize, rowBytes, stride, rsCoreNum, blockIdx);
            }
        }
        SyncAll<true>();
        teamBarrier_.CrossDevice();
        SyncAll<true>();
    }

    // Final: 仅计算核做 ReduceSum(last tile)
    SyncAll<true>();
    if (isComputeBlock) {
        uint32_t lastT = commTurn - 1;
        uint32_t lastTileM = GetTileM(lastT);
        uint64_t lastTileMOffset = GetTileMOffset(lastT);
        if (lastTileM > 0) {
            DoReduceSum(lastT, lastTileM, lastTileMOffset, nSize, rowBytes, stride, rsCoreNum, blockIdx);
        }
    }

    if (isCommBlock) {
        allToAllC_.template Finalize<BARRIER_NONE>();
    }
}

// =================================================================================
// 辅助方法: GetTileM / GetTileMOffset / DoReduceSum
// =================================================================================

template <typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline uint32_t QuantMatmulReduceScatterImpl<AType, BType, CType, TransA, TransB>::GetTileM(uint32_t t)
{
    if (t < baseParams_.headTileCnt) {
        return static_cast<uint32_t>(baseParams_.headMSize);
    }
    return static_cast<uint32_t>(baseParams_.tailMSize);
}

template <typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline uint64_t QuantMatmulReduceScatterImpl<AType, BType, CType, TransA, TransB>::GetTileMOffset(uint32_t t)
{
    if (t < baseParams_.headTileCnt) {
        return static_cast<uint64_t>(t) * baseParams_.headMSize;
    }
    return static_cast<uint64_t>(baseParams_.headTileCnt) * baseParams_.headMSize;
}

template <typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void QuantMatmulReduceScatterImpl<AType, BType, CType, TransA, TransB>::DoReduceSum(
    uint32_t tileIdx, uint32_t tileM, uint64_t tileMOffset, uint32_t nSize, uint32_t rowBytes, uint64_t stride,
    uint32_t usedCoreNum, uint32_t blockIdx)
{
    uint64_t outputSize = static_cast<uint64_t>(tileM) * nSize;
    GM_ADDR srcAddr = baseParams_.selfWinAddr + tileMOffset * rowBytes;
    GM_ADDR selfAddr = baseParams_.workspaceGm + tileMOffset * rowBytes;
    GM_ADDR dstAddr = baseParams_.cGm + tileMOffset * rowBytes;
    reduceSum_.Init(outputSize, stride, baseParams_.rankSize, baseParams_.rankId, usedCoreNum, srcAddr, selfAddr,
                    dstAddr);
    reduceSum_.ExecuteReduceSum(tileM, nSize);
}

// 单 tile (commTurn<=1) 路径专用; 批量接口存在时序竞态, 已回退为逐行 ExecuteReduceSum
template <typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void QuantMatmulReduceScatterImpl<AType, BType, CType, TransA, TransB>::DoReduceSumBatch(
    uint32_t tileIdx, uint32_t tileM, uint64_t tileMOffset, uint32_t nSize, uint32_t rowBytes, uint64_t stride,
    uint32_t usedCoreNum, uint32_t blockIdx)
{
    uint64_t outputSize = static_cast<uint64_t>(tileM) * nSize;
    GM_ADDR srcAddr = baseParams_.selfWinAddr + tileMOffset * rowBytes;
    GM_ADDR selfAddr = baseParams_.workspaceGm + tileMOffset * rowBytes;
    GM_ADDR dstAddr = baseParams_.cGm + tileMOffset * rowBytes;
    reduceSum_.Init(outputSize, stride, baseParams_.rankSize, baseParams_.rankId, usedCoreNum, srcAddr, selfAddr,
                    dstAddr);
    // ExecuteReduceSumBatch 的多行批量 + 单次 flag 模式存在时序竞态（尾段数据错乱），
    // 改用逐行 flag 的 ExecuteReduceSum（bit-exact 验证通过）；tileCnt=1 仅小 shape 命中
    reduceSum_.ExecuteReduceSum(tileM, nSize);
}

} // namespace Apace
