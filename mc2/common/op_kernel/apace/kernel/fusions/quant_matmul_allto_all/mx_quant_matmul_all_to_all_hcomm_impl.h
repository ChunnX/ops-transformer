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
 * \brief MatmulAllToAll MX hcomm实现(PR#10391方案)
 *
 * AIC_ONLY + 逐tile compute+SyncAll+comm:
 * - for循环逐tile: kernel(tileM, perRankN) → SyncAll → AllToAll(tileM*perRankN)
 * - kernel逐rank遍历x2的N列, 输出rank-major [rankDim, tileM, perRankN]
 * - AllToAll整块发送(batchCnt=1)
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

    __aicore__ inline void InitBaseParams();
    __aicore__ inline void SetupParams(Params &params, uint64_t rowOffset, const QuantMatmulTilingData &mmTile);
    __aicore__ inline void MatmulProcess(uint64_t rowOffset, const QuantMatmulTilingData &mmTile);
    __aicore__ inline void ProcessBatch(uint64_t startRow, uint32_t taskM, uint32_t taskCnt,
                                        const QuantMatmulTilingData &mmTile);
    // 类型级PackFactor: X1Type(FP4时=2), YType/scale(=1), 与PR#10391一致
    template <typename T>
    static constexpr uint64_t TypePackFactor() {
        if constexpr (AscendC::IsSameType<T, fp4x2_e2m1_t>::value) { return 2UL; }
        else { return 1UL; }
    }
    template <typename T>
    __aicore__ inline GM_ADDR GmAddr(GM_ADDR base, uint64_t elemCount) {
        return base + elemCount * sizeof(T) / TypePackFactor<T>();
    }
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
    auto &&ti = tilingData_->matmulAllToAllTilingInfo;
    axisK_ = ti.rankK;
    uint64_t rankN = ti.rankN;  // 完整N
    axisNPerRank_ = rankN / ti.rankDim;  // perRankN
    tileM_ = ti.tileM;
    tailM_ = ti.tailM;
    tileCnt_ = ti.tileCnt;
    tailCnt_ = ti.tailCnt;
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
    Params &params, uint64_t rowOffset, const QuantMatmulTilingData &mmTile)
{
    auto &&ti = tilingData_->matmulAllToAllTilingInfo;
    uint64_t rankNPerPeer = static_cast<uint64_t>(ti.rankN) / ti.rankDim;
    uint64_t scaleK = Blaze::Gemm::CeilDiv(static_cast<uint64_t>(ti.rankK), MXFP_GROUP_SIZE) * MXFP_MULTI_BASE_SIZE;

    // ProblemShape: M=mmTile.m, N=mmTile.n(perRankN), K=mmTile.k
    params.problemShape = ProblemShape{
        static_cast<int64_t>(mmTile.m),
        static_cast<int64_t>(mmTile.n),
        static_cast<int64_t>(mmTile.k),
        1};

    // A偏移: rowOffset * rankK (用类型级PackFactor处理FP4打包)
    params.mmParams = BlockMmadParams{
        GmAddr<X1Type>(x1_, rowOffset * ti.rankK),
        x2_,  // B不偏移
        GmAddr<YType>(workspaceGM_, rowOffset * rankNPerPeer),
        bias_,
        GmAddr<AscendC::fp8_e8m0_t>(x1Scale_, rowOffset * scaleK),
        x2Scale_};

    params.l1Params = L1Params{static_cast<uint64_t>(mmTile.stepK) * mmTile.baseK, mmTile.scaleKL1, mmTile.nBufferNum};
    params.schedulerParams = BlockSchedulerParams{mmTile.baseM, mmTile.baseN, mmTile.mTailTile, mmTile.nTailTile,
        mmTile.mBaseTailSplitCnt, mmTile.nBaseTailSplitCnt, mmTile.mTailMain, mmTile.nTailMain};
    params.qbmmParams = QBMMTiling{mmTile.baseM, mmTile.baseN, mmTile.baseK, mmTile.dbL0c,
        isBias_ ? 1u : 0u};
    // rankStride = rankM * rankNPerPeer (完整M, 与PR#10391一致)
    params.rankStride = static_cast<uint64_t>(ti.rankM) * rankNPerPeer;
    params.rankDim = ti.rankDim;
}

template <typename X1Type, typename X2Type, typename YType,
          typename QuantMatmulAllToAllTilingDataType,
          AscendC::HcclServerType ServerType, bool IsMxFp4>
__aicore__ inline void
QuantMatmulAllToAllHcommImpl<X1Type, X2Type, YType,
                              QuantMatmulAllToAllTilingDataType, ServerType, IsMxFp4>::MatmulProcess(
    uint64_t rowOffset, const QuantMatmulTilingData &mmTile)
{
    Params params;
    SetupParams(params, rowOffset, mmTile);
    quantMatmulKernelImpl_(params);
}

template <typename X1Type, typename X2Type, typename YType,
          typename QuantMatmulAllToAllTilingDataType,
          AscendC::HcclServerType ServerType, bool IsMxFp4>
__aicore__ inline void
QuantMatmulAllToAllHcommImpl<X1Type, X2Type, YType,
                              QuantMatmulAllToAllTilingDataType, ServerType, IsMxFp4>::ProcessBatch(
    uint64_t startRow, uint32_t taskM, uint32_t taskCnt, const QuantMatmulTilingData &mmTile)
{
    auto &&ti = tilingData_->matmulAllToAllTilingInfo;
    uint64_t rankNPerPeer = static_cast<uint64_t>(ti.rankN) / ti.rankDim;

    // AllToAll参数(与PR#10391一致)
    // sendBuffer = workspace + startRow * rankNPerPeer (绝对偏移)
    // recvBuffer = y_ + startRow * rankNPerPeer
    // sendOffset = taskM * rankNPerPeer * sizeof(YType) (tile间步长)
    // sendCount = taskM * rankNPerPeer (每tile发送量)
    // strideCount = rankM * rankNPerPeer (相邻rank间距, 完整M)
    GM_ADDR sendBuf = GmAddr<YType>(workspaceGM_, startRow * rankNPerPeer);
    GM_ADDR recvBuf = GmAddr<YType>(y_, startRow * rankNPerPeer);
    uint64_t sendOffset = static_cast<uint64_t>(taskM) * rankNPerPeer * sizeof(YType);
    uint64_t sendCount = static_cast<uint64_t>(taskM) * rankNPerPeer;
    uint64_t strideCount = static_cast<uint64_t>(ti.rankM) * rankNPerPeer;

    for (uint32_t index = 0; index < taskCnt; ++index) {
        // matmul: rowOffset = startRow + index * taskM
        MatmulProcess(startRow + static_cast<uint64_t>(index) * taskM, mmTile);
        AscendC::SyncAll();

        // AllToAll整块发送
        AscendC::HcclHandle h = hccl_.template AlltoAll<true>(
            sendBuf + static_cast<uint64_t>(index) * sendOffset,
            recvBuf + static_cast<uint64_t>(index) * sendOffset,
            sendCount, static_cast<AscendC::HcclDataType>(ti.hcclDataType),
            strideCount, 1);
        hccl_.Wait(h);
    }
}

template <typename X1Type, typename X2Type, typename YType,
          typename QuantMatmulAllToAllTilingDataType,
          AscendC::HcclServerType ServerType, bool IsMxFp4>
__aicore__ inline void
QuantMatmulAllToAllHcommImpl<X1Type, X2Type, YType,
                              QuantMatmulAllToAllTilingDataType, ServerType, IsMxFp4>::Run()
{
    if ASCEND_IS_AIC {
        auto &&ti = tilingData_->matmulAllToAllTilingInfo;
        // 头块: ProcessBatch(0, tileM, tileCnt, tileQbmmTilingData)
        if (ti.tileCnt > 0) {
            ProcessBatch(0, ti.tileM, ti.tileCnt, tilingData_->tileQbmmTilingData);
        }
        // 尾块: ProcessBatch(mainRows, tailM, tailCnt, tailQbmmTilingData)
        if (ti.tailCnt > 0) {
            uint64_t mainRows = static_cast<uint64_t>(ti.tileM) * ti.tileCnt;
            ProcessBatch(mainRows, ti.tailM, ti.tailCnt, tilingData_->tailQbmmTilingData);
        }
        AscendC::SyncAll();
        hccl_.Finalize();
    }
}

} // namespace Apace
