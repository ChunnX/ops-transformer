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
 * \file quant_matmul_mx_kernel.h
 * \brief mxfp8 场景 quantMatmul 实现。
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "blaze/gemm/utils/common_utils.h"
#include "include/tensor_api/tensor.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"

// v2 流水线常量（与 tiling_data.h 中定义一致，此处前向定义避免 include 循环）
#ifndef SYNC_MODE2_DEFINED
#define SYNC_MODE2_DEFINED
constexpr uint64_t SYNC_MODE2 = 2;
constexpr uint64_t SYNC_AIV_AIC_FLAG = 3;
constexpr uint64_t SYNC_MM_DONE_FLAG = 5;
constexpr uint32_t PIPELINE_NUM = 4;
#endif

namespace Blaze {
namespace Gemm {
namespace Kernel {

#define QBMM_MX_KERNEL_CLASS_TEM_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockScheduler, class CommPolicy>
#define QBMM_MX_KERNEL_FUNC_TEM_PARAMS ProblemShape, BlockMmad, BlockScheduler, CommPolicy

using namespace AscendC;
using AscendC::Te::Get;

/**
 * @brief SWAT MX 量化矩阵乘内核实现
 * 该类负责具体的矩阵乘块调度和计算，支持本地(LOCAL)和远程(REMOTE)两种切片模式。
 * 通信等待逻辑经 CommPolicy 策略类注入（组合模式）：基类持 commPolicy_ 成员对象，
 * 调用点直接 commPolicy_.WaitTile(tileIdx)，编译期由模板参数绑定具体策略，
 * 无需继承与 static_cast。
 */

template <class ProblemShape, class BlockMmad, class BlockScheduler,
          class CommPolicy>
class QuantMatmulMxKernel {
public:
    __aicore__ inline QuantMatmulMxKernel()
    {}
    __aicore__ inline ~QuantMatmulMxKernel()
    {}

    static constexpr bool weightNz = BlockMmad::WEIGHT_NZ;
    static constexpr bool transA = BlockMmad::TRANS_A;
    static constexpr bool transB = BlockMmad::TRANS_B;

    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    static constexpr int64_t C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;
    static constexpr int64_t kCacheLineAlignMask = IsFp4<AType>() ? 0xff : 0x7f;
    static constexpr int32_t SCALE_C0 = 2;

    using BlockShape = Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = Te::Coord<int64_t, int64_t, int64_t, int64_t>;

    using BlockSchedulerParams = typename BlockScheduler::Params;
    using MakeLayoutA = Te::FrameLayoutFormat<LayoutA, Std::Int<C0_SIZE>>;
    using MakeLayoutB = Te::FrameLayoutFormat<LayoutB, Std::Int<C0_SIZE>>;
    using MakeLayoutC = Te::FrameLayoutFormat<LayoutC, Std::Int<C0_SIZE>>;
    using MakeLayoutScaleA = Std::conditional_t<
        transA, Te::FrameLayoutFormat<Te::ScaleADNLayoutPtn, Std::Int<SCALE_C0>>,
        Te::FrameLayoutFormat<Te::ScaleANDLayoutPtn, Std::Int<SCALE_C0>>>;
    using MakeLayoutScaleB = Std::conditional_t<
        transB, Te::FrameLayoutFormat<Te::ScaleBDNLayoutPtn, Std::Int<SCALE_C0>>,
        Te::FrameLayoutFormat<Te::ScaleBNDLayoutPtn, Std::Int<SCALE_C0>>>;
    /**
     * @brief 算子模式：NORMAL (常规), LOCAL (仅本地计算), REMOTE (仅远程同步数据后的计算),
     *                 DEFERRED_SYNC (per-tile 本地先算驻留 L0C → wait_flag → 远程累加 → 单次 fixpipe)
     */
    enum class MatmulMode : uint32_t {
        REMOTE = 0,
        LOCAL = 1,
        DEFERRED_SYNC = 2
    };
    /**
     * @brief Tiling 配置
     */
    struct QBMMTiling {
        enum BiasMode : uint32_t {
            BIAS_DISABLED = 0,
            BIAS_ENABLED = 1
        };
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t dbL0C;
        uint32_t isBias;
    };

    /**
     * @brief 本地 Rank 相关参数
     */
    struct LocalParams {
        uint32_t rankId;
        uint32_t rankSize;
        uint64_t originalM;    // 单卡负责的总 M 行数
        GM_ADDR  localAGmAddr;
        GM_ADDR  localScaleAGmAddr;
        uint32_t localMatmul; // 0:REMOTE-only; 1:DEFERRED_SYNC
        uint32_t splitKNum;
        MatmulMode matmulMode; // REMOTE / LOCAL / DEFERRED_SYNC
        uint32_t headTileSize;
        // v2 新增：workspace 流水线参数
        GM_ADDR  workspaceGmAddr;  // workspace 基址（mm 输出目标）
        uint32_t pipelineNum;       // PIPELINE_NUM=4
        uint32_t coreNum;           // AIC 核数
    };

    /**
     * @brief 顶层参数结构
     */
    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        L1Params l1Params;
        BlockSchedulerParams schParams;
        QBMMTiling qbmmParams;
        LocalParams localParams;
    };

public:
    __aicore__ inline void Init(const Params &params);
    __aicore__ inline void Run(const Params &params);
    __aicore__ inline void operator()(const Params &params)
    {
        Run(params);
    }

    __aicore__ inline CommPolicy& GetCommPolicy() { return commPolicy_; }

private:
    __aicore__ inline void ResetGmAddr(const Params &params);
    __aicore__ inline void ProcessSingleBatch(const Params &params, BlockScheduler& bs,
                            uint64_t restBatch, bool isTailRound);
    __aicore__ inline void PostCompute();  // v2 新增：drain 剩余回压 flag
    __aicore__ inline int32_t CalcDependTileIdx(int64_t mPos, uint32_t headTileSize, uint32_t totalTiles) const;

    template <typename TensorB, typename TensorScaleB, typename TensorC>
    __aicore__ inline void SetL2Cache(
        const ProblemShape &problemShape, uint64_t curBaseM, uint64_t baseN, uint64_t scaleKL1, TensorB& gmB,
        TensorScaleB &gmScaleB, TensorC& gmC);

    template <typename TensorScaleB>
    __aicore__ inline void SetScaleL2Cache(
        const ProblemShape &problemShape, uint64_t baseN, uint64_t scaleKL1, TensorScaleB &gmScaleB);

private:
    BlockMmad mmadOp_;
    CommPolicy commPolicy_;

    __gm__ AType *aGmAddr_;               // 远程数据基址（通信缓冲区）
    __gm__ AType *localAGmAddr_;          // 本地数据基址
    __gm__ BType *bGmAddr_;
    __gm__ CType *cGmAddr_;               // 输出基址（v2: = workspaceGmAddr_）
    __gm__ BiasType *biasGmAddr_ = nullptr;
    __gm__ ::fp8_e8m0_t *scaleAGmAddr_;     // 远程 Scale 基址
    __gm__ ::fp8_e8m0_t *localScaleAGmAddr_;
    __gm__ ::fp8_e8m0_t *scaleBGmAddr_;

    // v2 新增：workspace 流水线参数
    __gm__ CType *workspaceGmAddr_{nullptr};  // v2 workspace 基址
    uint32_t pipelineNum_{4};                   // PIPELINE_NUM=4
    uint32_t coreNum_{0};                       // AIC 核数
    uint32_t cubeNum_{0};                       // mm tile 完成计数

    bool isAtomicAdd_{false}; // v2: 恒为 false（无 AtomicAdd，mm 输出到 workspace）
    bool isBias_{false};
    bool needUpdateTail_{false};
};

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernel<QBMM_MX_KERNEL_FUNC_TEM_PARAMS>::Run(const Params &params)
{
    Init(params);

    // v2: 无 AtomicAdd（mm 输出到 workspace，reduceSum 仅 DataCopyPad 搬运）
    BlockScheduler bs(params.problemShape, params.schParams);

    BlockShape l0TileShape{params.qbmmParams.baseM, params.qbmmParams.baseN, params.qbmmParams.baseK, 0};
    mmadOp_.Init(params.problemShape, l0TileShape, params.l1Params, isBias_, params.qbmmParams.dbL0C > 1,
        params.localParams.splitKNum);

    ProcessSingleBatch(params, bs, 0, true);

    // v2: drain 剩余回压 flag（等待 AIV reduceSum 消费完所有 workspace slot）
    PostCompute();
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
template <typename TensorScaleB>
__aicore__ inline void QuantMatmulMxKernel<QBMM_MX_KERNEL_FUNC_TEM_PARAMS>::SetScaleL2Cache(
    const ProblemShape &problemShape, uint64_t baseN, uint64_t scaleKL1, TensorScaleB &gmScaleB)
{
    if (Te::Get<MNK_B>(problemShape) != 1) {
        return;
    }
    if constexpr (transB) {
        const int64_t scaleKRowBytes =
            Blaze::Gemm::CeilDiv(Te::Get<MNK_K>(problemShape), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
            MXFP_MULTI_BASE_SIZE;
        const int64_t scaleKL1RowBytes = Blaze::Gemm::CeilDiv(scaleKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        // 0x7f: 128B cache line alignment for mx scale GM streaming
        const bool scaleAlignForL2Stream = (scaleKRowBytes & kCacheLineAlignMask) == 0 &&
            (scaleKL1RowBytes & kCacheLineAlignMask) == 0;
        gmScaleB.SetL2CacheHint(
            scaleAlignForL2Stream ? Te::CacheMode::CACHE_MODE_DISABLE : Te::CacheMode::CACHE_MODE_NORMAL);
    } else {
        const int64_t scaleNStrideBytes = Te::Get<MNK_N>(problemShape) * MXFP_MULTI_BASE_SIZE;
        const int64_t scaleBaseNStrideBytes = baseN * MXFP_MULTI_BASE_SIZE;
        // 0x7f: 128B cache line alignment for mx scale GM streaming
        const bool scaleAlignForL2Stream = (scaleNStrideBytes & kCacheLineAlignMask) == 0 &&
            (scaleBaseNStrideBytes & kCacheLineAlignMask) == 0;
        gmScaleB.SetL2CacheHint(
            scaleAlignForL2Stream ? Te::CacheMode::CACHE_MODE_DISABLE : Te::CacheMode::CACHE_MODE_NORMAL);
    }
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
template <typename TensorB, typename TensorScaleB, typename TensorC>
__aicore__ inline void QuantMatmulMxKernel<QBMM_MX_KERNEL_FUNC_TEM_PARAMS>::SetL2Cache(
    const ProblemShape &problemShape, uint64_t curBaseM, uint64_t baseN, uint64_t scaleKL1, TensorB& gmB,
    TensorScaleB &gmScaleB, TensorC& gmC)
{
    if (isAtomicAdd_) {
        gmC.SetL2CacheHint(Te::CacheMode::CACHE_MODE_DISABLE);
    }

    const bool fullMTile = curBaseM >= Te::Get<MNK_M>(problemShape);
    if (!fullMTile) {
        return;
    }

    SetScaleL2Cache(problemShape, baseN, scaleKL1, gmScaleB);

    if constexpr (weightNz) {
        gmB.SetL2CacheHint(Te::CacheMode::CACHE_MODE_DISABLE);
    } else {
        if constexpr (transB) {
            bool bAlignForL2Stream = (Te::Get<MNK_K>(problemShape) & kCacheLineAlignMask) == 0;
            gmB.SetL2CacheHint(
                bAlignForL2Stream ? Te::CacheMode::CACHE_MODE_DISABLE : Te::CacheMode::CACHE_MODE_NORMAL);
        } else {
            bool bAlignForL2Stream =
                (Te::Get<MNK_N>(problemShape) & kCacheLineAlignMask) == 0 && (baseN & kCacheLineAlignMask) == 0;
            gmB.SetL2CacheHint(
                bAlignForL2Stream ? Te::CacheMode::CACHE_MODE_DISABLE : Te::CacheMode::CACHE_MODE_NORMAL);
        }
    }
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernel<QBMM_MX_KERNEL_FUNC_TEM_PARAMS>::Init(const Params &params)
{
    if ASCEND_IS_AIV {
        return;
    }
    if (params.qbmmParams.isBias == QBMMTiling::BIAS_ENABLED) {
        isBias_ = true;
    }

    // v2: isAtomicAdd_ 恒为 false（mm 输出到 workspace，reduceSum 仅 DataCopyPad 无 AtomicAdd）
    isAtomicAdd_ = false;
    needUpdateTail_ = false;
    cubeNum_ = 0;
    // v2 新增参数初始化
    pipelineNum_ = params.localParams.pipelineNum;
    coreNum_ = params.localParams.coreNum;
    ResetGmAddr(params);
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernel<QBMM_MX_KERNEL_FUNC_TEM_PARAMS>::ResetGmAddr(const Params &params)
{
    if ASCEND_IS_AIV {
        return;
    }
    aGmAddr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
    bGmAddr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
    cGmAddr_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
    localAGmAddr_ = reinterpret_cast<__gm__ AType*>(params.localParams.localAGmAddr);
    localScaleAGmAddr_ = reinterpret_cast<__gm__ ::fp8_e8m0_t*>(params.localParams.localScaleAGmAddr);
    scaleAGmAddr_ = reinterpret_cast<__gm__ ::fp8_e8m0_t*>(params.mmadParams.scaleAGmAddr);
    scaleBGmAddr_ = reinterpret_cast<__gm__ ::fp8_e8m0_t*>(params.mmadParams.scaleBGmAddr);
    // v2: 绑定 workspace 基址（mm 输出目标）
    workspaceGmAddr_ = reinterpret_cast<__gm__ CType*>(params.localParams.workspaceGmAddr);
    if (isBias_) {
        biasGmAddr_ = reinterpret_cast<__gm__ BiasType*>(params.mmadParams.biasGmAddr);
    }
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline int32_t QuantMatmulMxKernel<QBMM_MX_KERNEL_FUNC_TEM_PARAMS>::CalcDependTileIdx(
    int64_t mPos, uint32_t headTileSize, uint32_t totalTiles) const
{
    int32_t tileIdx = static_cast<int32_t>(mPos / headTileSize);
    if (tileIdx >= totalTiles) {
        tileIdx = totalTiles - 1;
    }
    return tileIdx;
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernel<QBMM_MX_KERNEL_FUNC_TEM_PARAMS>::ProcessSingleBatch(
    const Params &params, BlockScheduler& bs, uint64_t restBatch, bool isTailRound)
{
    auto rankId = params.localParams.rankId;
    auto rankSize = params.localParams.rankSize;
    auto oriM = params.localParams.originalM;
    // v2: localFirst 不再使用（v2 不支持 LOCAL 模式，localMatmul=1 映射 DEFERRED_SYNC）
    // DEFERRED_SYNC: per-tile 内 local 先算驻留 L0C → wait_flag → remote 累加 → 单次 fixpipe。
    bool deferredSync = (params.localParams.matmulMode == MatmulMode::DEFERRED_SYNC);
    auto scaleKLen =
        Blaze::Gemm::CeilDiv(Te::Get<MNK_K>(params.problemShape), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
        MXFP_MULTI_BASE_SIZE;

    // 构建各 Tensor 的全局布局
    auto layoutA = MakeLayoutA{}(rankSize * params.localParams.originalM, Te::Get<MNK_K>(params.problemShape));
    auto layoutALocal = MakeLayoutA{}(rankSize * oriM, Te::Get<MNK_K>(params.problemShape));
    auto layoutScaleA = MakeLayoutScaleA{}(rankSize * oriM, scaleKLen);

    auto layoutB = MakeLayoutB{}(rankSize * Te::Get<MNK_K>(params.problemShape), Te::Get<MNK_N>(params.problemShape));
    auto layoutScaleB = MakeLayoutScaleB{}(rankSize * scaleKLen, Te::Get<MNK_N>(params.problemShape));
    auto layoutBias = Te::MakeFrameLayout<Te::NDExtLayoutPtn>(1L, Te::Get<MNK_N>(params.problemShape));
    // v2: gmC 指向 workspace（cGmAddr_ 已在 RunMatmul 中设置为 workspaceGm）
    auto layoutC = MakeLayoutC{}(Te::Get<MNK_M>(params.problemShape), Te::Get<MNK_N>(params.problemShape));

    // 创建 Tensor 句柄
    auto gmA = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(aGmAddr_), layoutA);
    auto gmALocal = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(localAGmAddr_), layoutALocal); // local输入
    auto gmScaleA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(scaleAGmAddr_), layoutScaleA);
    auto gmScaleALocal = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(localScaleAGmAddr_), layoutScaleA);
    auto gmB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(bGmAddr_), layoutB);
    auto gmScaleB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(scaleBGmAddr_), layoutScaleB);
    auto gmBias = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(biasGmAddr_), layoutBias);
    // v2: gmC 指向 workspace 基址（用于 SetL2Cache，实际 mm 输出用 per-tile gmCWorkspace）
    auto gmC = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(cGmAddr_), layoutC);

    // 尾块更新逻辑
    auto& mTailTile = params.schParams.mTailTile;
    auto& nTailTile = params.schParams.nTailTile;
    if (needUpdateTail_ ||
        (isTailRound && ((bs.GetEndBlockIdx() + 1) + (restBatch * bs.GetTotalCnt())) * mTailTile * nTailTile <=
                AscendC::GetBlockNum())) {
        needUpdateTail_ = true;
        bs.UpdateTailTile(mTailTile, nTailTile);
    }
    SetL2Cache(
        params.problemShape, params.qbmmParams.baseM, params.qbmmParams.baseN, params.l1Params.scaleKL1, gmB, gmScaleB,
        gmC);

    BlockCoord blockIdx;
    int64_t mPos = 0L;
    int64_t nPos = 0L;
    constexpr int64_t kPos = 0L;
    uint32_t totalTiles = (oriM + params.localParams.headTileSize - 1) / params.localParams.headTileSize;
    int32_t readyTileIdx = -1;
    uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());

    // 遍历当前块的调度任务
    while (bs.GetTileIdx(blockIdx)) {
        BlockShape singleShape =
            bs.template GetBlockShape<QuantMode::MX_PERGROUP_MODE, QuantMode::MX_PERGROUP_MODE, weightNz>(blockIdx);
        if ((Te::Get<IDX_M_TILEIDX>(singleShape) <= 0) || (Te::Get<IDX_N_TILEIDX>(singleShape) <= 0)) {
            return;
        }

        bs.GetTileCoord(blockIdx, mPos, nPos);

        // v2 通道 3：回压——当 cubeNum >= PIPELINE_NUM 时，等 AIV reduceSum 释放 workspace slot
        if (cubeNum_ >= pipelineNum_) {
            CrossCoreWaitFlag<SYNC_MODE2, PIPE_M>(SYNC_AIV_AIC_FLAG);
        }

        // 通信等待逻辑（与 v1 相同：等 AIV 通信完成，去重）
        int64_t blockM = Te::Get<IDX_M_TILEIDX>(singleShape);
        int32_t dependTileIdx = CalcDependTileIdx(mPos + blockM - 1, params.localParams.headTileSize, totalTiles);
        while (readyTileIdx < dependTileIdx) {
            readyTileIdx++;
            commPolicy_.WaitTile(readyTileIdx);
        }

        // ★ v2 改造点：per-tile 创建 workspace Tensor（不再用循环外的 gmC.Slice）
        // workspace slot = baseM×baseN 的整块，workSpaceOffset 以元素个数为单位
        uint64_t workSpaceOffset = static_cast<uint64_t>(params.qbmmParams.baseM) * params.qbmmParams.baseN *
            (coreIdx + (cubeNum_ % pipelineNum_) * coreNum_);
        auto layoutCTile = MakeLayoutC{}(Get<MNK_M>(singleShape), Get<MNK_N>(singleShape));
        auto gmCWorkspace = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::GM>(workspaceGmAddr_ + workSpaceOffset),
            layoutCTile);

        auto gmBlockBias =
            gmBias.Slice(Te::MakeCoord(0L, nPos), Te::MakeShape(1L, Te::Get<IDX_N_TILEIDX>(singleShape)));

        if (deferredSync) {
            // DEFERRED_SYNC 模式：per-tile local→L0C→wait→remote→累加→单次 fixpipe→workspace
            // Phase 1: 本 rank 的 local A × 本 rank 的 B 段 → L0C（reset，remoteRankCnt=0）
            auto selfMPos = rankId * oriM + mPos;
            auto gmBlockA_self =
                gmALocal.Slice(AscendC::Te::MakeCoord(selfMPos, kPos),
                    AscendC::Te::MakeShape(Get<MNK_M>(singleShape), Get<MNK_K>(params.problemShape)));
            auto gmBlockScaleA_self = gmScaleALocal.Slice(AscendC::Te::MakeCoord(selfMPos, kPos),
                    AscendC::Te::MakeShape(Get<MNK_M>(singleShape), scaleKLen));
            auto gmBlockB_self = gmB.Slice(AscendC::Te::MakeCoord(rankId * Get<MNK_K>(params.problemShape), nPos),
                AscendC::Te::MakeShape(Get<MNK_K>(params.problemShape), Get<MNK_N>(singleShape)));
            auto gmBlockScaleB_self = gmScaleB.Slice(AscendC::Te::MakeCoord(rankId * scaleKLen, nPos),
                AscendC::Te::MakeShape(scaleKLen, Get<MNK_N>(singleShape)));
            mmadOp_(gmBlockA_self, gmBlockB_self, gmBlockScaleA_self, gmBlockScaleB_self,
                    gmBlockBias, gmCWorkspace, singleShape, 0);

            // Phase 2: wait（已在外层处理 dependTileIdx）

            // Phase 3: 遍历其它 rank，在 L0C 上累加（最后一个 rank 触发 fixpipe → workspace）
            uint32_t remoteRankCnt = 1;
            for (uint64_t rank = 0; rank < rankSize; rank++) {
                if (rank == rankId) continue;
                auto actualMPos = rank * oriM + mPos;
                auto gmBlockA_remote =
                    gmA.Slice(AscendC::Te::MakeCoord(actualMPos, kPos),
                        AscendC::Te::MakeShape(Get<MNK_M>(singleShape), Get<MNK_K>(params.problemShape)));
                auto gmBlockScaleA_remote = gmScaleA.Slice(AscendC::Te::MakeCoord(actualMPos, kPos),
                        AscendC::Te::MakeShape(Get<MNK_M>(singleShape), scaleKLen));
                auto gmBlockB_r = gmB.Slice(AscendC::Te::MakeCoord(rank * Get<MNK_K>(params.problemShape), nPos),
                    AscendC::Te::MakeShape(Get<MNK_K>(params.problemShape), Get<MNK_N>(singleShape)));
                auto gmBlockScaleB_r = gmScaleB.Slice(AscendC::Te::MakeCoord(rank * scaleKLen, nPos),
                    AscendC::Te::MakeShape(scaleKLen, Get<MNK_N>(singleShape)));
                mmadOp_(gmBlockA_remote, gmBlockB_r, gmBlockScaleA_remote, gmBlockScaleB_r,
                        gmBlockBias, gmCWorkspace, singleShape, remoteRankCnt);
                remoteRankCnt++;
            }
        } else {
            // REMOTE 模式（v2 默认）：遍历所有 rank 的 A×B → L0C 跨 rank 累加 → fixpipe → workspace
            auto remoteRankCnt = 0UL;
            for (uint64_t rank = 0; rank < rankSize; rank++) {
                auto actualMPos = rank * oriM + mPos;
                // 从通信 buffer 上切片（本 rank 用 localAGmAddr_）
                auto gmBlockA =
                    gmA.Slice(AscendC::Te::MakeCoord(actualMPos, kPos),
                        AscendC::Te::MakeShape(Get<MNK_M>(singleShape), Get<MNK_K>(params.problemShape)));
                auto gmBlockScaleA = gmScaleA.Slice(AscendC::Te::MakeCoord(actualMPos, kPos),
                        AscendC::Te::MakeShape(Get<MNK_M>(singleShape), scaleKLen));

                if (rank == rankId) {
                    gmBlockA = gmALocal.Slice(AscendC::Te::MakeCoord(actualMPos, kPos),
                        AscendC::Te::MakeShape(Get<MNK_M>(singleShape), Get<MNK_K>(params.problemShape)));
                    gmBlockScaleA = gmScaleALocal.Slice(AscendC::Te::MakeCoord(actualMPos, kPos),
                        AscendC::Te::MakeShape(Get<MNK_M>(singleShape), scaleKLen));
                }

                auto gmBlockB = gmB.Slice(AscendC::Te::MakeCoord(rank * Get<MNK_K>(params.problemShape), nPos),
                    AscendC::Te::MakeShape(Get<MNK_K>(params.problemShape), Get<MNK_N>(singleShape)));
                auto gmBlockScaleB = gmScaleB.Slice(AscendC::Te::MakeCoord(rank * scaleKLen, nPos),
                    AscendC::Te::MakeShape(scaleKLen, Get<MNK_N>(singleShape)));

                // L0C 上累加（最后一个 rank 触发 fixpipe → workspace）
                mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmCWorkspace,
                    singleShape, remoteRankCnt);
                remoteRankCnt++;
            }
        }

        // v2 通道 2：通知 AIV mm 完成（flagId=5，AIV 侧阻塞 MTE2 流水）
        CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_MM_DONE_FLAG);
        cubeNum_++;
    }

    // 存在尾核没用满核，所以这里要等 flag 兜底
    while (readyTileIdx < totalTiles - 1) {
        readyTileIdx++;
        commPolicy_.WaitTile(readyTileIdx);
    }
}

// v2 PostCompute：drain 剩余回压 flag（等待 AIV reduceSum 消费完所有 workspace slot）
QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernel<QBMM_MX_KERNEL_FUNC_TEM_PARAMS>::PostCompute()
{
    if ASCEND_IS_AIV {
        return;
    }
    // drain Min(cubeNum_, pipelineNum_) 个回压 flag
    uint32_t drainCnt = (cubeNum_ < pipelineNum_) ? cubeNum_ : pipelineNum_;
    for (uint32_t idx = 0; idx < drainCnt; ++idx) {
        CrossCoreWaitFlag<SYNC_MODE2, PIPE_M>(SYNC_AIV_AIC_FLAG);
    }
}
} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
