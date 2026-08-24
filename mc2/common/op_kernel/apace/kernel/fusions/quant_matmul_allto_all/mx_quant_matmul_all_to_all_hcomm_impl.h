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
 * \brief QuantMatmul AllToAll — 计算→通信融合实现（HCCL纯C通信）
 *
 * 基于PR#10054 QuantBatchMatmulMxKernel(纯计算kernel):
 * - ProblemShape.N=perRankN, x2按rank切分N, matmul输出[curTileM, perRankN]连续
 * - AllToAll整块发送(sendCount=curTileM*perRankN, batchCnt=1)
 * - AIC_ONLY: for循环逐tile matmul, SyncAll后统一下发AllToAll
 */

#pragma once

#include "lib/hccl/hccl.h"
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
    __aicore__ inline void SetupParams(const QuantMatmulTilingData &mmTile, uint32_t curTileM, Params &params);
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
    const QuantMatmulTilingData &mmTile, uint32_t curTileM, Params &params)
{
    // batchSize=1: 每卡只算自己的x1×x2
    // x2按K切分: [rankDim, K, totalN], 每卡通过rankId*K偏移读取自己的[K, totalN]
    // ProblemShape.N=totalN(完整N), matmul输出[curTileM, totalN]写mmResult
    // AllToAll逐行发送perRankN列(sendCount=perRankN, strideCount=perRankN, batchCnt=curTileM)
    uint64_t totalN = axisNPerRank_ * rankDim_;
    params.problemShape = ProblemShape{
        static_cast<int64_t>(curTileM),
        static_cast<int64_t>(totalN),
        static_cast<int64_t>(axisK_),
        1};

    constexpr uint64_t sizeShift = IsMxFp4 ? 1 : 0;
    uint64_t scaleKLen = Blaze::Gemm::CeilDiv(static_cast<uint64_t>(axisK_), MXFP_GROUP_SIZE) * MXFP_MULTI_BASE_SIZE;
    // x2按K切分: bGmAddr = x2_ + rankId_ * K
    params.mmadParams = BlockMmadParams{
        x1_,
        x2_ + (rankId_ * axisK_ * sizeof(X2Type) / (IsMxFp4 ? MXFP_DATA_NUM_PER_BYTE : 1)),
        mmResultGM_, nullptr,
        x1Scale_,
        x2Scale_ + rankId_ * scaleKLen * sizeof(AscendC::fp8_e8m0_t)};

    params.l1Params = L1Params{static_cast<uint64_t>(mmTile.stepK) * mmTile.baseK, mmTile.scaleKL1, mmTile.nBufferNum};
    params.schParams = BlockSchedulerParams{mmTile.baseM, mmTile.baseN, mmTile.mTailTile, mmTile.nTailTile,
        mmTile.mBaseTailSplitCnt, mmTile.nBaseTailSplitCnt, mmTile.mTailMain, mmTile.nTailMain};
    params.qbmmParams = QBMMTiling{1, mmTile.baseM, mmTile.baseN, mmTile.baseK,
        isBias_ ? QBMMTiling::BIAS_ENABLED : QBMMTiling::BIAS_DISABLED, mmTile.dbL0c, false};
    params.localParams = LocalParams{0, 1, curTileM, 0, 0, false};
}

template <typename X1Type, typename X2Type, typename YType,
          typename QuantMatmulAllToAllTilingDataType,
          AscendC::HcclServerType ServerType, bool IsMxFp4>
__aicore__ inline void
QuantMatmulAllToAllHcommImpl<X1Type, X2Type, YType,
                              QuantMatmulAllToAllTilingDataType, ServerType, IsMxFp4>::RunRemoteMatmul()
{
    Params param;
    SetupParams(tilingData_->tileQbmmTilingData, tileM_, param);
    quantMatmulKernelImpl_.InitMmad(param);

    GM_ADDR aCur = x1_;
    GM_ADDR aScaleCur = x1Scale_;
    uint32_t lastTiling = 0;
    for (uint32_t tid = 0; tid < totalTileCnt_; ++tid) {
        bool isTail = (tid >= tileCnt_);
        uint32_t curTileM = isTail ? tailM_ : tileM_;
        uint32_t curTiling = isTail ? 1 : 0;
        if (curTiling != lastTiling) {
            SetupParams(isTail ? tilingData_->tailQbmmTilingData : tilingData_->tileQbmmTilingData,
                curTileM, param);
            quantMatmulKernelImpl_.InitMmad(param);
            lastTiling = curTiling;
        }
        param.mmadParams.aGmAddr = aCur;
        param.mmadParams.scaleAGmAddr = aScaleCur;
        // mmResult偏移按totalN步进(每tile输出[curTileM, totalN])
        uint64_t totalN = axisNPerRank_ * rankDim_;
        uint64_t tileOffset = (tid < tileCnt_)
            ? static_cast<uint64_t>(tid) * tileM_
            : static_cast<uint64_t>(tileCnt_) * tileM_ + static_cast<uint64_t>(tid - tileCnt_) * tailM_;
        param.mmadParams.cGmAddr = mmResultGM_ + tileOffset * totalN * sizeof(YType);
        quantMatmulKernelImpl_(param);

        aCur += static_cast<uint64_t>(curTileM) * aByteStrideK_;
        aScaleCur += static_cast<uint64_t>(curTileM) * scaleAByteStride_;

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
    // AIC: for循环逐tile matmul(QuantBatchMatmulMxKernel)
    // SyncAll后统一下发AllToAll(头块+尾块固定2次,整块发送)
    if ASCEND_IS_AIC {
        RunRemoteMatmul();

        AscendC::SyncAll();
        uint64_t totalN = axisNPerRank_ * rankDim_;
        uint32_t headM = tileM_ * tileCnt_;
        uint32_t tailM = tailM_ * tailCnt_;

        AscendC::HcclHandle h0{0}, h1{0};
        if (headM > 0) {
            // 逐行发送: sendCount=perRankN, strideCount=perRankN(行内相邻rank间距), batchCnt=headM(行数)
            h0 = hccl_.template AlltoAll<true>(
                mmResultGM_, y_,
                axisNPerRank_, static_cast<AscendC::HcclDataType>(yHcclDataType_),
                axisNPerRank_, static_cast<uint8_t>(headM > 255 ? 255 : headM));
        }
        if (tailM > 0) {
            uint64_t headOffset = static_cast<uint64_t>(headM) * totalN * sizeof(YType);
            uint64_t tailRecvOffset = static_cast<uint64_t>(headM) * axisNPerRank_ * sizeof(YType);
            h1 = hccl_.template AlltoAll<true>(
                mmResultGM_ + headOffset, y_ + tailRecvOffset,
                axisNPerRank_, static_cast<AscendC::HcclDataType>(yHcclDataType_),
                axisNPerRank_, static_cast<uint8_t>(tailM > 255 ? 255 : tailM));
        }

        AscendC::SyncAll();
        if (headM > 0) { hccl_.Wait(h0); }
        if (tailM > 0) { hccl_.Wait(h1); }

        AscendC::SyncAll();
        hccl_.Finalize();
    }
}

} // namespace Apace
