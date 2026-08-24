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
 * \file mx_quant_matmul_all_to_all_hcomm_impl.h
 * \brief MatmulAllToAll MX 算子 hcomm 实现（HCCL PUT 模式）
 *
 * 基于PR#10054 urma方案，将UDMA GET改为HCCL PUT:
 * - AIC: for循环逐tile matmul(QuantBatchMatmulMxKernel) + CrossCoreSetFlag通知AIV
 * - AIV: for循环 CrossCoreWaitFlag + HCCL AllToAll(PUT发送mmResult到y)
 *
 * 数据流: MatMul→AllToAll (计算在前,通信在后)
 * matmul结果写mmResult(GM), AIV从mmResult PUT发送到各卡y_
 */

#pragma once

#include "lib/hccl/hccl.h"
#include "basic_api/kernel_basic_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "../../matmul/quant_batch_matmul/quant_batch_matmul_mx_kernel.h"
#include "include/tensor_api/tensor.h"
#include "quant_matmul_all_to_all_tiling_data.h"

namespace Apace {

template <typename T>
constexpr uint64_t GetYHcclDataType()
{
    if constexpr (AscendC::IsSameType<T, half>::value) {
        return AscendC::HCCL_DATA_TYPE_FP16;
    } else if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        return AscendC::HCCL_DATA_TYPE_BFP16;
    } else {
        return AscendC::HCCL_DATA_TYPE_FP32;
    }
}

template <typename X1Type, typename X2Type, typename YType,
          typename QuantMatmulAllToAllTilingDataType,
          AscendC::HcclServerType ServerType, bool IsMxFp4>
class QuantMatmulAllToAllHcommImpl {
public:
    explicit __aicore__ inline QuantMatmulAllToAllHcommImpl(QuantMatmulAllToAllTilingDataType *tilingData)
        : tilingData_(tilingData)
    {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y,
                                GM_ADDR x1_scale, GM_ADDR x2_scale, GM_ADDR workspaceGM);
    __aicore__ inline void Run();

    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutB = AscendC::Te::DNExtLayoutPtn;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;
    using BiasType = float;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
        ProblemShape, 0, LayoutA, LayoutB, X1Type>;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<0, false>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, X1Type, LayoutA, X2Type, LayoutB, YType, LayoutC, BiasType, LayoutBias>;
    using QuantMatmulKernelImpl = Blaze::Gemm::Kernel::QuantBatchMatmulMxKernel<
        ProblemShape, BlockMmad, BlockScheduler>;

    using Params = typename QuantMatmulKernelImpl::Params;
    using BlockMmadParams = typename QuantMatmulKernelImpl::BlockMmadParams;
    using L1Params = typename QuantMatmulKernelImpl::L1Params;
    using LocalParams = typename QuantMatmulKernelImpl::LocalParams;
    using BlockSchedulerParams = typename QuantMatmulKernelImpl::BlockSchedulerParams;
    using QBMMTiling = typename QuantMatmulKernelImpl::QBMMTiling;

    QuantMatmulKernelImpl quantMatmulKernelImpl_;

private:
    QuantMatmulAllToAllTilingDataType *tilingData_;

    GM_ADDR x1_;
    GM_ADDR x2_;
    GM_ADDR y_;
    GM_ADDR bias_;
    GM_ADDR x1Scale_;
    GM_ADDR x2Scale_;
    GM_ADDR workspaceGM_;
    GM_ADDR mmResultGM_;

    AscendC::Hccl<ServerType> hccl_;

    uint32_t rankId_{0};
    uint32_t rankDim_{0};
    uint32_t tileM_{0};
    uint32_t tailM_{0};
    uint32_t tileCnt_{0};
    uint32_t tailCnt_{0};
    uint32_t totalTileCnt_{0};
    uint64_t axisM_{0};
    uint64_t axisK_{0};
    uint64_t axisNPerRank_{0};
    uint64_t maxTileM_{0};
    uint64_t cByteStrideN_{0};
    uint64_t aByteStrideK_{0};
    uint64_t scaleAByteStride_{0};
    uint64_t scaleAElemCnt_{0};

    bool isBias_{false};

    static constexpr uint64_t yHcclDataType_ = GetYHcclDataType<YType>();
    static constexpr uint64_t MXFP_GROUP_SIZE = 64UL;
    static constexpr uint64_t MXFP_MULTI_BASE_SIZE = 2UL;
    static constexpr uint64_t MXFP_DATA_NUM_PER_BYTE = 2UL;
    static constexpr uint16_t MTE1_MTE2_EVENT_FLAG = 6;

    __aicore__ inline void InitBaseParams();
    __aicore__ inline void SetupParams(const QuantMatmulTilingData &mmTile, uint32_t batchSize,
                                       uint32_t batchRankOffset, uint64_t origM, bool isLocal, Params &params);
    __aicore__ inline void RunRemoteMatmul();
};

template <typename X1Type, typename X2Type, typename YType,
          typename QuantMatmulAllToAllTilingDataType,
          AscendC::HcclServerType ServerType, bool IsMxFp4>
__aicore__ inline void
QuantMatmulAllToAllHcommImpl<X1Type, X2Type, YType,
                              QuantMatmulAllToAllTilingDataType, ServerType, IsMxFp4>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y, GM_ADDR x1_scale, GM_ADDR x2_scale,
    GM_ADDR workspaceGM)
{
    tilingData_ = tilingData_;
    x1_ = x1;
    x2_ = x2;
    y_ = y;
    bias_ = bias;
    isBias_ = (bias != nullptr);
    x1Scale_ = x1_scale;
    x2Scale_ = x2_scale;
    workspaceGM_ = workspaceGM;
    mmResultGM_ = workspaceGM;

    InitBaseParams();

    hccl_.InitV2(AscendC::GetHcclContext<0>(), &(tilingData_->mc2InitTiling));
    hccl_.SetCcTilingV2(static_cast<uint64_t>(offsetof(QuantMatmulAllToAllTilingDataType, mc2CcTiling)));
    rankId_ = hccl_.GetRankId();
    rankDim_ = hccl_.GetRankDim();
}

template <typename X1Type, typename X2Type, typename YType,
          typename QuantMatmulAllToAllTilingDataType,
          AscendC::HcclServerType ServerType, bool IsMxFp4>
__aicore__ inline void
QuantMatmulAllToAllHcommImpl<X1Type, X2Type, YType,
                              QuantMatmulAllToAllTilingDataType, ServerType, IsMxFp4>::InitBaseParams()
{
    auto &&commTD = tilingData_->commTilingData;
    axisK_ = tilingData_->tileQbmmTilingData.k;
    axisNPerRank_ = tilingData_->tileQbmmTilingData.n;
    tileM_ = static_cast<uint32_t>(commTD.splitAxisTileSize);
    tailM_ = static_cast<uint32_t>(commTD.splitAxisTailSize);
    tileCnt_ = static_cast<uint32_t>(commTD.splitAxisTileCnt);
    tailCnt_ = static_cast<uint32_t>(commTD.splitAxisTailCnt);
    totalTileCnt_ = tileCnt_ + tailCnt_;
    axisM_ = static_cast<uint64_t>(tileCnt_) * tileM_ + static_cast<uint64_t>(tailCnt_) * tailM_;
    maxTileM_ = (tileM_ > tailM_) ? tileM_ : tailM_;
    cByteStrideN_ = axisNPerRank_ * sizeof(YType);
    uint32_t scaleKGroups = (axisK_ + Blaze::Gemm::MXFP_DIVISOR_SIZE - 1) / Blaze::Gemm::MXFP_DIVISOR_SIZE;
    scaleAElemCnt_ = scaleKGroups * Blaze::Gemm::MXFP_MULTI_BASE_SIZE;
    aByteStrideK_ = axisK_ * sizeof(X1Type);
    scaleAByteStride_ = scaleAElemCnt_ * sizeof(AscendC::fp8_e8m0_t);
}

template <typename X1Type, typename X2Type, typename YType,
          typename QuantMatmulAllToAllTilingDataType,
          AscendC::HcclServerType ServerType, bool IsMxFp4>
__aicore__ inline void
QuantMatmulAllToAllHcommImpl<X1Type, X2Type, YType,
                              QuantMatmulAllToAllTilingDataType, ServerType, IsMxFp4>::SetupParams(
    const QuantMatmulTilingData &mmTile, uint32_t batchSize, uint32_t batchRankOffset,
    uint64_t origM, bool isLocal, Params &params)
{
    params.problemShape = ProblemShape{
        static_cast<int64_t>(mmTile.m),
        static_cast<int64_t>(axisNPerRank_),
        static_cast<int64_t>(axisK_),
        static_cast<int64_t>(batchSize)};

    constexpr uint64_t sizeShift = IsMxFp4 ? 1 : 0;
    params.mmadParams = BlockMmadParams{
        x1_,
        x2_ + ((static_cast<uint64_t>(batchRankOffset) * axisNPerRank_ * axisK_) >> sizeShift),
        isLocal ? y_ : mmResultGM_, nullptr,
        x1Scale_, x2Scale_ + static_cast<uint64_t>(batchRankOffset) * axisNPerRank_ * scaleAElemCnt_};

    params.l1Params = L1Params{static_cast<uint64_t>(mmTile.stepK) * mmTile.baseK, mmTile.scaleKL1, mmTile.nBufferNum};
    params.schParams = BlockSchedulerParams{mmTile.baseM, mmTile.baseN, mmTile.mTailTile, mmTile.nTailTile,
        mmTile.mBaseTailSplitCnt, mmTile.nBaseTailSplitCnt, mmTile.mTailMain, mmTile.nTailMain};
    params.qbmmParams = QBMMTiling{batchSize, mmTile.baseM, mmTile.baseN, mmTile.baseK,
        isBias_ ? QBMMTiling::BIAS_ENABLED : QBMMTiling::BIAS_DISABLED, mmTile.dbL0c, false};
    params.localParams = LocalParams{rankId_, rankDim_, origM, rankId_, y_, false};
}

template <typename X1Type, typename X2Type, typename YType,
          typename QuantMatmulAllToAllTilingDataType,
          AscendC::HcclServerType ServerType, bool IsMxFp4>
__aicore__ inline void
QuantMatmulAllToAllHcommImpl<X1Type, X2Type, YType,
                              QuantMatmulAllToAllTilingDataType, ServerType, IsMxFp4>::RunRemoteMatmul()
{
    Params param;
    SetupParams(tilingData_->tileQbmmTilingData, rankDim_, 0, maxTileM_, false, param);
    quantMatmulKernelImpl_.InitMmad(param);

    GM_ADDR aCur = x1_;
    GM_ADDR aScaleCur = x1Scale_;
    GM_ADDR cSelfCur = y_;
    uint32_t lastTiling = 0;
    for (uint32_t tid = 0; tid < totalTileCnt_; ++tid) {
        bool isTail = (tid >= tileCnt_);
        uint32_t curTileM = isTail ? tailM_ : tileM_;
        uint32_t curTiling = isTail ? 1 : 0;
        if (curTiling != lastTiling) {
            SetupParams(isTail ? tilingData_->tailQbmmTilingData : tilingData_->tileQbmmTilingData,
                rankDim_, 0, maxTileM_, false, param);
            quantMatmulKernelImpl_.InitMmad(param);
            lastTiling = curTiling;
        }
        // C输出: 远端rank写mmResult, 本rank写y_(cGmSelfAddr)
        param.mmadParams.aGmAddr = aCur;
        param.mmadParams.scaleAGmAddr = aScaleCur;
        param.mmadParams.cGmAddr = mmResultGM_ + static_cast<uint64_t>(tid) * maxTileM_ * axisNPerRank_ * sizeof(YType);
        param.localParams.cGmSelfAddr = cSelfCur;
        quantMatmulKernelImpl_(param);

        aCur += static_cast<uint64_t>(curTileM) * aByteStrideK_;
        aScaleCur += static_cast<uint64_t>(curTileM) * scaleAByteStride_;
        cSelfCur += static_cast<uint64_t>(curTileM) * cByteStrideN_;

        if (tailCnt_ > 0 && tid < totalTileCnt_ - 1) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
        }
    }
}

template <typename X1Type, typename X2Type, typename YType,
          typename QuantMatmulAllToAllTilingDataType,
          AscendC::HcclServerType ServerType, bool IsMxFp4>
__aicore__ inline void
QuantMatmulAllToAllHcommImpl<X1Type, X2Type, YType,
                              QuantMatmulAllToAllTilingDataType, ServerType, IsMxFp4>::Run()
{
    // AIC_ONLY模式: matmul+HCCL通信都在AIC上
    // for循环逐tile matmul + CrossCoreSetFlag不需要(AIC_ONLY无AIV)
    // 改为: for循环逐tile matmul, 每个tile完成后SyncAll+异步AllToAll
    // 但for循环中调用AlltoAll<true>会编译卡死(HCCL重模板)
    // 解决: for循环只做matmul, SyncAll后统一下发AllToAll(头块+尾块固定2次)

    if ASCEND_IS_AIC {
        RunRemoteMatmul();

        // matmul完成后统一下发AllToAll
        AscendC::SyncAll();
        uint64_t perRankN = axisNPerRank_;
        uint32_t headM = tileM_ * tileCnt_;
        uint32_t tailM = tailM_ * tailCnt_;

        AscendC::HcclHandle h0{0}, h1{0};
        if (headM > 0) {
            uint64_t sendCount = static_cast<uint64_t>(headM) * perRankN;
            h0 = hccl_.template AlltoAll<true>(
                mmResultGM_, y_,
                sendCount, static_cast<AscendC::HcclDataType>(yHcclDataType_),
                sendCount, 1);
        }
        if (tailM > 0) {
            uint64_t headOffset = static_cast<uint64_t>(headM) * perRankN * sizeof(YType);
            uint64_t sendCount = static_cast<uint64_t>(tailM) * perRankN;
            h1 = hccl_.template AlltoAll<true>(
                mmResultGM_ + headOffset, y_ + headOffset,
                sendCount, static_cast<AscendC::HcclDataType>(yHcclDataType_),
                sendCount, 1);
        }

        AscendC::SyncAll();
        if (headM > 0) { hccl_.Wait(h0); }
        if (tailM > 0) { hccl_.Wait(h1); }

        AscendC::SyncAll();
        hccl_.Finalize();
    }
}

} // namespace Apace
