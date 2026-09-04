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
  * \file mx_quant_matmul_all_to_all_urma_impl.h
  * \brief MatmulAllToAll MX 算子 kernel 实现

  * 核心设计：
  * - AIC 负责计算：委托给 QuantBatchMatmulMxKernel，遍历所有 rank 的 B 段做 QBMM，结果写 Win 区
  * - AIV 负责通信：通过 AllToAll GET 从其他卡的 Win 区拉回本 rank 的 C 段到 cGM
  */

#pragma once

#include "basic_api/kernel_basic_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "matmul_all_to_all_tiling_data.h"
#include "adv_api/hcomm/hcomm.h"

#include "../../matmul/quant_batch_matmul/all_to_all_qbmm_mx_kernel_tpn.h"
#include "../../../core/aiv_comm/collective_comm_api.h"
#include "../../../core/aiv_comm/collective_comm_context.h"
#include "../../../tiling/comm_tiling_data.h"
#include "../../../core/aiv_comm/barrier/barrier_ubmem.h"

namespace Apace {

using namespace AscendC;
using namespace Blaze::Gemm;
using namespace Apace::AivComm;

template <typename AType, typename BType, typename CType, bool LocalDelay>
class MatmulAllToAllMxImpl {
public:
    __aicore__ inline MatmulAllToAllMxImpl(){};
    __aicore__ inline ~MatmulAllToAllMxImpl() {}

    /**
     * @brief 初始化算子状态和参数
     */
    __aicore__ inline void Init(__gm__ CommContext *hcommCtx, GM_ADDR aGM, GM_ADDR scaleAGM, GM_ADDR bGM,
                                GM_ADDR scaleBGM, GM_ADDR biasGM, GM_ADDR cGM,
                                const MatmulAllToAllTilingData *tilingData);

    /**
     * @brief 执行算子逻辑（包含 AIC/AIV 分离逻辑）
     */
    __aicore__ inline void Run();

    // Layout 定义
    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutB = AscendC::Te::DNExtLayoutPtn;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;
    using BiasType = float;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    // 组件定义
    using BlockScheduler =
        Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, 0, LayoutA, LayoutB, AType>;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<0, false>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC,
                                                    BiasType, LayoutBias>;
    using QuantMatmulKernelImpl =
        Blaze::Gemm::Kernel::QuantBatchMatmulMxKernel<ProblemShape, BlockMmad, BlockScheduler>;

    // 参数类型
    using Params = typename QuantMatmulKernelImpl::Params;
    using BlockMmadParams = typename QuantMatmulKernelImpl::BlockMmadParams;
    using L1Params = typename QuantMatmulKernelImpl::L1Params;
    using LocalParams = typename QuantMatmulKernelImpl::LocalParams;
    using BlockSchedulerParams = typename QuantMatmulKernelImpl::BlockSchedulerParams;
    using QBMMTiling = typename QuantMatmulKernelImpl::QBMMTiling;

    QuantMatmulKernelImpl quantMatmulKernelImpl_;

private:
    // ---------------- 通信相关组件与参数 ----------------
    __gm__ CommUbmemContext *barrierCtx_{nullptr};
    __gm__ CommUdmaContext *udmaCtx_{nullptr};
    CollectiveComm<CommCollectiveOp::AllToAll, CommMode::GET, CType, TeamBarrier> allToAll_;
    TeamBarrier teamBarrier_;

    struct BaseParams {
        GM_ADDR selfWinAddr{nullptr};
        GM_ADDR aGm{nullptr};
        GM_ADDR scaleAGm{nullptr};
        GM_ADDR bGm{nullptr};
        GM_ADDR scaleBGm{nullptr};
        GM_ADDR cGm{nullptr};
        GM_ADDR selfCBase{nullptr};
        GM_ADDR biasGm{nullptr};

        uint32_t rankId{0};
        uint32_t rankSize{0};
        uint32_t tileCnt{0};
        uint32_t tileM{0};
        uint32_t tailM{0};
        uint32_t tailCnt{0};
        uint32_t totalTileCnt{0};

        uint64_t axisM{0};
        uint64_t axisK{0};
        uint64_t axisNPerRank{0};
        uint64_t maxTileM{0};

        uint64_t aByteStrideK{0};
        uint64_t cByteStrideN{0};
        uint64_t scaleAByteStride{0};
        uint64_t scaleAElemCnt{0};

        uint64_t tileChunkBytes{0};
        uint64_t bufferSlotBytes{0};
    } baseParams_;

    static constexpr uint16_t MTE1_MTE2_EVENT_FLAG = 6;
    const MatmulAllToAllTilingData *tilingData_{nullptr};

    // ---------------- 私有方法 ----------------
    __aicore__ inline void InitBaseParams(const MatmulAllToAllTilingData *td);
    __aicore__ inline void SetupParams(const QuantMatmulTilingData &mmTile, uint32_t batchSize,
                                       uint32_t batchRankOffset, bool isLocal, Params &params);
    __aicore__ inline void RunRemoteMatmul();
    __aicore__ inline void RunLocalMatmul();
    __aicore__ inline void RunAllToAll();
};

// =================================================================================
// 公有方法实现
// =================================================================================

template <typename AType, typename BType, typename CType, bool LocalDelay>
__aicore__ inline void MatmulAllToAllMxImpl<AType, BType, CType, LocalDelay>::Init(
    __gm__ CommContext *hcommCtx, GM_ADDR aGM, GM_ADDR scaleAGM, GM_ADDR bGM, GM_ADDR scaleBGM, GM_ADDR biasGM,
    GM_ADDR cGM, const MatmulAllToAllTilingData *tilingData)
{
    tilingData_ = tilingData;

    barrierCtx_ = &(hcommCtx->ubmemCtx);
    udmaCtx_ = &(hcommCtx->udmaCtx);

    baseParams_.rankId = udmaCtx_->rankId;
    baseParams_.rankSize = udmaCtx_->rankSize;
    baseParams_.aGm = aGM;
    baseParams_.scaleAGm = scaleAGM;
    baseParams_.bGm = bGM;
    baseParams_.scaleBGm = scaleBGM;
    baseParams_.cGm = cGM;
    baseParams_.biasGm = biasGM;

    InitBaseParams(tilingData);

    baseParams_.selfCBase =
        baseParams_.cGm + static_cast<uint64_t>(baseParams_.rankId) * baseParams_.axisM * baseParams_.cByteStrideN;
    baseParams_.selfWinAddr = reinterpret_cast<GM_ADDR>(udmaCtx_->commBufferAddrs[baseParams_.rankId]);

    uint32_t ubOffset = 0;
    auto commPtr = AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, uint8_t>(ubOffset);
    ubOffset += COMM_WORKSPACE_SIZE;
    auto barrierPtr = AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, uint8_t>(ubOffset);
    teamBarrier_.Init(barrierPtr.Get(), barrierCtx_, udmaCtx_->rankSize, static_cast<uint32_t>(GetBlockIdx()));

    CommTilingData commTilingData = tilingData->commTilingData;
    allToAll_.Init(udmaCtx_, teamBarrier_, commTilingData, cGM, commPtr.Get(),
                   static_cast<uint32_t>(udmaCtx_->rankSize), static_cast<uint32_t>(GetBlockIdx()));
}

template <typename AType, typename BType, typename CType, bool LocalDelay>
__aicore__ inline void MatmulAllToAllMxImpl<AType, BType, CType, LocalDelay>::Run()
{
    if ASCEND_IS_AIC {
        RunRemoteMatmul();
        if constexpr (LocalDelay) {
            RunLocalMatmul();
        }
    }
    if ASCEND_IS_AIV {
        RunAllToAll();
    }
}

// =================================================================================
// 参数推导与构造
// =================================================================================

template <typename AType, typename BType, typename CType, bool LocalDelay>
__aicore__ inline void MatmulAllToAllMxImpl<AType, BType, CType, LocalDelay>::InitBaseParams(
    const MatmulAllToAllTilingData *tilingData)
{
    const auto &commTd = tilingData->commTilingData;
    baseParams_.axisK = tilingData->tileQbmmTilingData.k;
    baseParams_.axisNPerRank = tilingData->tileQbmmTilingData.n;
    baseParams_.tileCnt = static_cast<uint32_t>(commTd.splitAxisTileCnt);
    baseParams_.tileM = static_cast<uint32_t>(commTd.splitAxisTileSize);
    baseParams_.tailM = static_cast<uint32_t>(commTd.splitAxisTailSize);
    baseParams_.tailCnt = static_cast<uint32_t>(commTd.splitAxisTailCnt);
    baseParams_.totalTileCnt = baseParams_.tileCnt + baseParams_.tailCnt;
    baseParams_.axisM = static_cast<uint64_t>(baseParams_.tileCnt) * baseParams_.tileM +
                        static_cast<uint64_t>(baseParams_.tailCnt) * baseParams_.tailM;
    baseParams_.maxTileM = (baseParams_.tileM > baseParams_.tailM) ? baseParams_.tileM : baseParams_.tailM;
    baseParams_.cByteStrideN = baseParams_.axisNPerRank * sizeof(CType);
    uint32_t scaleKGroups = (baseParams_.axisK + Blaze::Gemm::MXFP_DIVISOR_SIZE - 1) / Blaze::Gemm::MXFP_DIVISOR_SIZE;
    baseParams_.scaleAElemCnt = scaleKGroups * Blaze::Gemm::MXFP_MULTI_BASE_SIZE;
    constexpr uint64_t sizeShift = Blaze::Gemm::IsFp4<AType>() ? 1 : 0;
    baseParams_.aByteStrideK = (baseParams_.axisK * sizeof(AType)) >> sizeShift;
    baseParams_.scaleAByteStride = baseParams_.scaleAElemCnt * sizeof(AscendC::fp8_e8m0_t);
    baseParams_.tileChunkBytes = baseParams_.maxTileM * baseParams_.cByteStrideN;
    baseParams_.bufferSlotBytes = baseParams_.rankSize * baseParams_.tileChunkBytes;
}

template <typename AType, typename BType, typename CType, bool LocalDelay>
__aicore__ inline void MatmulAllToAllMxImpl<AType, BType, CType, LocalDelay>::SetupParams(
    const QuantMatmulTilingData &mmTile, uint32_t batchSize, uint32_t batchRankOffset, bool isLocal, Params &params)
{
    params.problemShape = ProblemShape{static_cast<int64_t>(mmTile.m), static_cast<int64_t>(baseParams_.axisNPerRank),
                                       static_cast<int64_t>(baseParams_.axisK), static_cast<int64_t>(batchSize)};

    constexpr uint64_t sizeShift = Blaze::Gemm::IsFp4<AType>() ? 1 : 0;
    GM_ADDR biasAddr = nullptr;
    if (baseParams_.biasGm != nullptr) {
        biasAddr =
            baseParams_.biasGm + static_cast<uint64_t>(batchRankOffset) * baseParams_.axisNPerRank * sizeof(BiasType);
    }
    params.mmadParams = BlockMmadParams{
        baseParams_.aGm,
        baseParams_.bGm +
            ((static_cast<uint64_t>(batchRankOffset) * baseParams_.axisNPerRank * baseParams_.axisK) >> sizeShift),
        isLocal ? baseParams_.selfCBase : baseParams_.selfWinAddr,
        biasAddr,
        baseParams_.scaleAGm,
        baseParams_.scaleBGm +
            static_cast<uint64_t>(batchRankOffset) * baseParams_.axisNPerRank * baseParams_.scaleAElemCnt};
    params.l1Params = L1Params{static_cast<uint64_t>(mmTile.stepK) * mmTile.baseK, mmTile.scaleKL1, mmTile.nBufferNum};
    params.schParams = BlockSchedulerParams{mmTile.baseM,
                                            mmTile.baseN,
                                            mmTile.mTailTile,
                                            mmTile.nTailTile,
                                            mmTile.mBaseTailSplitCnt,
                                            mmTile.nBaseTailSplitCnt,
                                            mmTile.mTailMain,
                                            mmTile.nTailMain};
    params.qbmmParams = QBMMTiling{batchSize,
                                   mmTile.baseM,
                                   mmTile.baseN,
                                   mmTile.baseK,
                                   baseParams_.biasGm != nullptr ? QBMMTiling::BIAS_ENABLED : QBMMTiling::BIAS_DISABLED,
                                   mmTile.dbL0c,
                                   LocalDelay && !isLocal};
    params.localParams = LocalParams{baseParams_.rankId, baseParams_.rankSize,  baseParams_.maxTileM,
                                     baseParams_.rankId, baseParams_.selfCBase, LocalDelay && !isLocal};
}

// =================================================================================
// 计算逻辑实现 (AIC 侧)
// =================================================================================

template <typename AType, typename BType, typename CType, bool LocalDelay>
__aicore__ inline void MatmulAllToAllMxImpl<AType, BType, CType, LocalDelay>::RunRemoteMatmul()
{
    uint32_t remoteBatchSize = LocalDelay ? baseParams_.rankSize - 1 : baseParams_.rankSize;
    uint32_t remoteBatchRankOffset = 0;
    Params param;
    SetupParams(tilingData_->tileQbmmTilingData, remoteBatchSize, remoteBatchRankOffset, false, param);

    GM_ADDR aCur = baseParams_.aGm;
    GM_ADDR aScaleCur = baseParams_.scaleAGm;
    GM_ADDR cSelfCur = baseParams_.selfCBase;
    uint32_t lastTiling = 0; // 修改命名tile块
    for (uint32_t tid = 0; tid < baseParams_.totalTileCnt; ++tid) {
        bool isTail = (tid >= baseParams_.tileCnt);
        uint32_t curTileM = isTail ? baseParams_.tailM : baseParams_.tileM;
        uint32_t curTiling = isTail ? 1 : 0;
        if (curTiling != lastTiling) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_FLAG);
            SetupParams(isTail ? tilingData_->tailQbmmTilingData : tilingData_->tileQbmmTilingData, remoteBatchSize,
                        remoteBatchRankOffset, false, param);
            lastTiling = curTiling;
        }
        GM_ADDR cCur = baseParams_.selfWinAddr + tid * baseParams_.bufferSlotBytes;
        param.mmadParams.aGmAddr = aCur;
        param.mmadParams.scaleAGmAddr = aScaleCur;
        param.mmadParams.cGmAddr = cCur;
        if (!LocalDelay) {
            param.localParams.cGmSelfAddr = cSelfCur;
        }
        quantMatmulKernelImpl_(param);

        CrossCoreSetFlag<0x2, PIPE_FIX>(tid);
        aCur += curTileM * baseParams_.aByteStrideK;
        aScaleCur += curTileM * baseParams_.scaleAByteStride;
        cSelfCur += curTileM * baseParams_.cByteStrideN;
    }
}

template <typename AType, typename BType, typename CType, bool LocalDelay>
__aicore__ inline void MatmulAllToAllMxImpl<AType, BType, CType, LocalDelay>::RunLocalMatmul()
{
    // 注释
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_FLAG);
    Params localParam;
    SetupParams(tilingData_->localQbmmTilingData, 1, baseParams_.rankId, true, localParam);
    quantMatmulKernelImpl_(localParam);
}

// =================================================================================
// 通信实现 (AIV 侧)
// =================================================================================

template <typename AType, typename BType, typename CType, bool LocalDelay>
__aicore__ inline void MatmulAllToAllMxImpl<AType, BType, CType, LocalDelay>::RunAllToAll()
{
    for (uint32_t tid = 0; tid < baseParams_.totalTileCnt; ++tid) {
        CrossCoreWaitFlag<0x2, PIPE_S>(tid);
        if (GetBlockIdx() < baseParams_.rankSize) {
            allToAll_.Commit();
            allToAll_.Wait(true);
        }
    }
    allToAll_.Finalize();
}

} // namespace Apace
