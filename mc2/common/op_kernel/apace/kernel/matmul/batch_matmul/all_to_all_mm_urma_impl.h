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
 * \file all_to_all_mm_urma_impl.h
 * \brief AlltoAll MatMul（非量化，GM 累加版）— 通信+计算融合实现
 *
 * Run():
 *   AIV: RunAllToAll()  根据通信轮次下发通信任务和通知AIC通信已完成
 *   AIC: RunMatmul()    等待通信完成后开始cube计算
 *
 * 与量化版(all_to_all_mx_quant_matmul_urma_impl.h)的差异：
 * 1. 无 scale：删除 scaleA/scaleB 通信通道与相关参数。
 * 2. 非量化 kernel 无 splitKIdx 接口，统一采用 GM 累加（首次覆盖写 + 其余原子累加）。
 * 3. 增加 zeroBiasGmAddr（tiling 分配的清零 workspace），用于抑制基本 mmad 的重复 bias 累加。
 */

#pragma once

#include "basic_api/kernel_basic_intf.h"
#include "include/tensor_api/tensor.h"

#include "adv_api/hcomm/hcomm.h"
#include "all_to_all_mm_kernel.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "all_to_all_mm_tiling_data.h"

#include "apace/core/aiv_comm/collective_comm_api.h"
#include "apace/core/aiv_comm/collective_comm_context.h"
#include "apace/tiling/comm_tiling_data.h"
#include "apace/core/aiv_comm/barrier/barrier_ubmem.h"

namespace Apace {

using namespace AscendC;
using namespace Blaze::Gemm;
using AscendC::Te::Get;
using namespace Apace::AivComm;

struct UrmaCommWaitPolicy {
    __aicore__ inline void WaitTile(uint32_t tileIdx)
    {
        AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(tileIdx);
    }
};

// 定义问题形状：[M, N, K, Batch]
using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

/**
 * @brief All2All-Matmul 核心实现类 (Hcomm GET 版本，计算通信融合，非量化)
 * 实现思路：
 * 1. AIC 负责本地计算 和远程数据计算。
 * 2. AIV 负责背景通信，通过 Hcomm GET 接口从其他卡拉取 A 数据。
 * 3. 通过流水线掩盖通信开销。
 * 4. 计算与通信逻辑合并于此类，便于精细化流水线控制。
 * 5. 非量化 mmad 自包含（无 splitKIdx），GM 累加：首次写入覆盖写，其余写入原子累加。
 */
template <typename AType, typename BType, typename CType, bool TransA, bool TransB>
class AllToAllMmUrmaImpl {
public:
    __aicore__ inline AllToAllMmUrmaImpl(){};
    __aicore__ inline ~AllToAllMmUrmaImpl() {}

    /**
     * @brief 初始化算子状态和参数
     */
    __aicore__ inline void Init(__gm__ CommContext *hcommCtx, GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM,
                                GM_ADDR zeroBiasGM, const allToAllMmMatmulTilingData *tilingData);
    /**
     * @brief 执行算子逻辑（包含 AIC/AIV 分离逻辑）
     */
    __aicore__ inline void Run();

    using TypeA = AType;
    using TypeB = BType;
    using TypeC = CType;
    using BiasType = float;

    // Layout 定义
    using LayoutA =
        typename AscendC::Std::conditional_t<TransA, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
    using LayoutB =
        typename AscendC::Std::conditional_t<TransB, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;

    // 组件定义
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, NONE_FULL_LOAD_MODE,
                                                                                LayoutA, LayoutB, TypeA>;
    using DispatchPolicy =
        Blaze::Gemm::MatmulMultiBlockBasic<NONE_FULL_LOAD_MODE, 0, Blaze::Gemm::KernelMmadMultiBlockBasic, 0>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, TypeA, LayoutA, TypeB, LayoutB, TypeC, LayoutC,
                                                    BiasType, LayoutBias>;
    using MatmulKernelImpl =
        Kernel::AllToAllMmKernel<ProblemShape, BlockMmad, BlockScheduler, UrmaCommWaitPolicy>;

    // 参数类型
    using Params = typename MatmulKernelImpl::Params;
    using BlockMmadParams = typename MatmulKernelImpl::BlockMmadParams;
    using LocalParams = typename MatmulKernelImpl::LocalParams;
    using BlockSchedulerParams = typename MatmulKernelImpl::BlockSchedulerParams;
    using MMTiling = typename MatmulKernelImpl::MMTiling;
    using MatmulMode = typename MatmulKernelImpl::MatmulMode;

    MatmulKernelImpl matmulKernelImpl_;

private:
    // ---------------- 通信相关组件与参数 ----------------
    __gm__ CommUbmemContext *syncBuffer_{nullptr};
    __gm__ CommUdmaContext *udmaCtx_{nullptr};
    CollectiveComm<CommCollectiveOp::AllToAll, CommMode::PUT, AType, TeamBarrier> allToAllA_;
    TeamBarrier teamBarrier_;

    struct BaseParams {
        GM_ADDR selfWinAddr{nullptr}; // 通信窗口地址
        GM_ADDR aGm{nullptr};
        GM_ADDR bGm{nullptr};
        GM_ADDR cGm{nullptr};
        GM_ADDR biasGm{nullptr};
        GM_ADDR zeroBiasGm{nullptr}; // zero-bias workspace（tiling 分配并清零，kernel 只读）

        uint32_t rankId{0};
        int32_t commTurn{0};  // 总流水步数
        uint32_t rankSize{0}; // 卡数
        uint64_t axisM{0};    // M 轴总大小
        uint64_t axisKa{0};   // K 轴大小
        uint64_t headMSize{0}; // 长块 M 大小
        uint64_t rankDataBytes{0};
    } baseParams_;

    const allToAllMmMatmulTilingData *tilingData_{nullptr};

    // ---------------- 私有方法 ----------------
    __aicore__ inline void InitBaseParams(const allToAllMmMatmulTilingData *tilingData);
    __aicore__ inline void SetupParams(const QuantMatmulTilingData *mmTile, Params &out, MatmulMode matmulMode);

    __aicore__ inline void RunAllToAll(); // AIV 通信任务
    __aicore__ inline void RunLocalMatmul(); // AIC 本地计算任务
    __aicore__ inline void RunMatmul(); // AIC 远程计算任务
};

// =================================================================================
// 公有方法实现
// =================================================================================

template <typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllMmUrmaImpl<AType, BType, CType, TransA, TransB>::Init(
    __gm__ CommContext *hcommCtx, GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR zeroBiasGM,
    const allToAllMmMatmulTilingData *tilingData)
{
    tilingData_ = tilingData;
    baseParams_.aGm = aGM;
    baseParams_.bGm = bGM;
    baseParams_.cGm = cGM;
    baseParams_.biasGm = biasGM;
    baseParams_.zeroBiasGm = zeroBiasGM;

    syncBuffer_ = &(hcommCtx->ubmemCtx);
    udmaCtx_ = &(hcommCtx->udmaCtx);
    baseParams_.rankId = udmaCtx_->rankId;
    baseParams_.rankSize = udmaCtx_->rankSize;

    InitBaseParams(tilingData);

    baseParams_.selfWinAddr = reinterpret_cast<GM_ADDR>(udmaCtx_->commBufferAddrs[baseParams_.rankId]);
    uint32_t ubOffset = 0;
    auto commBuf = AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, uint8_t>(ubOffset); // 用512B
    ubOffset += COMM_WORKSPACE_SIZE;
    auto barrierBuf = AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, uint8_t>(ubOffset);
    teamBarrier_.Init(barrierBuf.Get(), syncBuffer_, baseParams_.rankSize, static_cast<uint32_t>(GetBlockIdx()));

    allToAllA_.template Init<BARRIER_NONE>(udmaCtx_, teamBarrier_, tilingData->commTilingData, baseParams_.aGm,
                                           commBuf.Get(), baseParams_.rankSize, static_cast<uint32_t>(GetBlockIdx()));
}

template <typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllMmUrmaImpl<AType, BType, CType, TransA, TransB>::Run()
{
    if ASCEND_IS_AIV {
        RunAllToAll(); // AIV 通信
    }

    if ASCEND_IS_AIC {
        if (tilingData_->localMatmul == 2) {
            RunLocalMatmul(); // AIC local计算前置（GM累加模式）
        }
        RunMatmul(); // AIC 通信块计算（带同步等待）
    }
}

// =================================================================================
// 通信与流水线控制实现 (AIV 侧)
// =================================================================================

template <typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllMmUrmaImpl<AType, BType, CType, TransA, TransB>::RunAllToAll()
{
    for (uint32_t tid = 0; tid < baseParams_.commTurn; ++tid) {
        // 必选保证baseParams_.rankSize <= BlockNum
        if (AscendC::GetBlockIdx() < baseParams_.rankSize) {
            allToAllA_.Commit();
            allToAllA_.template Wait<BARRIER_DEVICE>();
        }

        AscendC::SyncAll<true>();
        CrossCoreSetFlag<0x2, PIPE_MTE3>(tid);
    }

    allToAllA_.Finalize();

    // 通信接口使用了 channel 内的字段，需要刷新 GM 地址保证跨 kernel launch 的 GM 状态与 aiCore cache 一致
    // 后续通信接口会内置刷 cache 机制，届时可移除此处调用
    dcci(reinterpret_cast<__gm__ void *>(udmaCtx_->commBufferAddrs),
         static_cast<uint64_t>(CacheLine::ENTIRE_DATA_CACHE), static_cast<uint64_t>(DcciDst::CACHELINE_OUT));
}

// =================================================================================
// 计算逻辑实现 (AIC 侧)
// =================================================================================

template <typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllMmUrmaImpl<AType, BType, CType, TransA, TransB>::InitBaseParams(
    const allToAllMmMatmulTilingData *tilingData)
{
    baseParams_.headMSize = tilingData->commTilingData.splitAxisTileSize;
    baseParams_.commTurn = tilingData->commTilingData.splitAxisTileCnt + tilingData->commTilingData.splitAxisTailCnt;

    baseParams_.axisM = tilingData->tileMmTilingData.m;
    baseParams_.axisKa = tilingData->tileMmTilingData.k;

    // 计算地址偏移所需的参数（static_cast<uint64_t> 提升首操作数，避免中间溢出）
    baseParams_.rankDataBytes =
        static_cast<uint64_t>(baseParams_.axisM) * static_cast<uint64_t>(baseParams_.axisKa) * sizeof(AType);
}

template <typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllMmUrmaImpl<AType, BType, CType, TransA, TransB>::SetupParams(
    const QuantMatmulTilingData *mmTile, Params &out, MatmulMode matmulMode)
{
    ProblemShape problemShape{mmTile->m, mmTile->n, mmTile->k, 1UL};
    BlockMmadParams mmadParams;
    LocalParams localParams{baseParams_.rankId,
                            baseParams_.rankSize,
                            baseParams_.axisM,
                            baseParams_.aGm,
                            baseParams_.zeroBiasGm,
                            tilingData_->localMatmul,
                            matmulMode,
                            static_cast<uint32_t>(baseParams_.headMSize)};

    if (matmulMode == MatmulMode::LOCAL) {
        mmadParams.cGmAddr = baseParams_.cGm; // 本地模式直接写入输出
    }

    // 基本 mmad 的 L1/L0 分块与缓冲参数
    mmadParams.bGmAddr = baseParams_.bGm;
    mmadParams.biasGmAddr = baseParams_.biasGm;
    mmadParams.mL1 = mmTile->baseM;
    mmadParams.nL1 = mmTile->baseN;
    mmadParams.kL1 = static_cast<uint64_t>(mmTile->stepK) * mmTile->baseK;
    mmadParams.mL0 = mmTile->baseM;
    mmadParams.nL0 = mmTile->baseN;
    mmadParams.kL0 = mmTile->baseK;
    mmadParams.l1Stages = mmTile->nBufferNum;
    mmadParams.l0cStages = (mmTile->dbL0c > 1) ? 2 : 1;

    // 调度器参数
    BlockSchedulerParams schedulerParams{mmTile->baseM,
                                         mmTile->baseN,
                                         mmTile->mTailTile,
                                         mmTile->nTailTile,
                                         mmTile->mBaseTailSplitCnt,
                                         mmTile->nBaseTailSplitCnt,
                                         mmTile->mTailMain,
                                         mmTile->nTailMain};
    // 基础 Tiling 参数
    MMTiling mmTiling{mmTile->baseM, mmTile->baseN, mmTile->baseK, mmTile->dbL0c};

    out = {problemShape, mmadParams, schedulerParams, mmTiling, localParams};
}

template <typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllMmUrmaImpl<AType, BType, CType, TransA, TransB>::RunLocalMatmul()
{
    Params localParams;
    SetupParams(&tilingData_->tileMmTilingData, localParams, MatmulMode::LOCAL);
    matmulKernelImpl_(localParams);
}

template <typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllMmUrmaImpl<AType, BType, CType, TransA, TransB>::RunMatmul()
{
    Params params;
    MatmulMode mode = (tilingData_->localMatmul == 1) ? MatmulMode::DEFERRED_SYNC : MatmulMode::REMOTE;
    SetupParams(&tilingData_->tileMmTilingData, params, mode);
    params.mmadParams.aGmAddr = baseParams_.selfWinAddr; // 远程数据基址（通信缓冲区）
    params.mmadParams.cGmAddr = baseParams_.cGm;
    params.localParams.localAGmAddr = baseParams_.aGm;
    matmulKernelImpl_(params);
}

} // namespace Apace
