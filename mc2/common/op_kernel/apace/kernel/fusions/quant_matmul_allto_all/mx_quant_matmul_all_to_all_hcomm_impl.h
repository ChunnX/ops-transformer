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
 * \brief MatmulAllToAll MX hcomm实现
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

template <typename X1Type, typename X2Type, typename YType,
          typename QuantMatmulAllToAllTilingDataType,
          AscendC::HcclServerType ServerType, bool IsMxFp4>
class QuantMatmulAllToAllHcommImpl {
public:
    explicit __aicore__ inline QuantMatmulAllToAllHcommImpl(QuantMatmulAllToAllTilingDataType *tilingData)
        : tilingData_(tilingData) {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y,
                                GM_ADDR x1_scale, GM_ADDR x2_scale, GM_ADDR workspaceGM);
    __aicore__ inline void Run();

private:
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

    QuantMatmulAllToAllTilingDataType *tilingData_;
    QuantMatmulKernelImpl quantMatmulKernelImpl_;
    AscendC::Hccl<ServerType> hccl_;
    AscendC::HcclHandle prevHandle_{0};

    GM_ADDR x1_{};
    GM_ADDR x2_{};
    GM_ADDR y_{};
    GM_ADDR bias_{};
    GM_ADDR x1Scale_{};
    GM_ADDR x2Scale_{};
    GM_ADDR workspaceGM_{};

    bool isBias_{false};
    static constexpr uint64_t MXFP_GROUP_SIZE = 64UL;
    static constexpr uint64_t MXFP_MULTI_BASE_SIZE = 2UL;

    __aicore__ inline void SetupParams(Params &params, uint64_t rowOffset, const QuantMatmulTilingData &mmTile);
    __aicore__ inline void MatmulProcess(uint64_t rowOffset, const QuantMatmulTilingData &mmTile);
    __aicore__ inline void ProcessBatch(uint64_t startRow, uint32_t taskM, uint32_t taskCnt,
                                        const QuantMatmulTilingData &mmTile);

    template <typename T>
    __aicore__ inline GM_ADDR GmAddr(GM_ADDR base, uint64_t elemCount) {
        if constexpr (AscendC::IsSameType<T, fp4x2_e2m1_t>::value) {
            return base + elemCount * sizeof(T) / 2UL;
        } else {
            return base + elemCount * sizeof(T);
        }
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
    tilingData_ = tilingData_;
    x1_ = x1;
    x2_ = x2;
    y_ = y;
    bias_ = bias;
    isBias_ = (bias != nullptr);
    x1Scale_ = x1_scale;
    x2Scale_ = x2_scale;
    workspaceGM_ = workspaceGM;

    hccl_.InitV2(AscendC::GetHcclContext<0>(), &(tilingData_->mc2InitTiling));
    hccl_.SetCcTilingV2(static_cast<uint64_t>(offsetof(QuantMatmulAllToAllTilingDataType, mc2CcTiling)));
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

    params.problemShape = ProblemShape{
        static_cast<int64_t>(mmTile.m),
        static_cast<int64_t>(mmTile.n),
        static_cast<int64_t>(mmTile.k),
        1};

    params.mmParams = BlockMmadParams{
        GmAddr<X1Type>(x1_, rowOffset * ti.rankK),
        x2_,
        GmAddr<YType>(workspaceGM_, rowOffset * rankNPerPeer),
        bias_,
        GmAddr<AscendC::fp8_e8m0_t>(x1Scale_, rowOffset * scaleK),
        x2Scale_};

    params.l1Params = L1Params{static_cast<uint64_t>(mmTile.stepK) * mmTile.baseK, mmTile.scaleKL1, mmTile.nBufferNum};
    params.schedulerParams = BlockSchedulerParams{mmTile.baseM, mmTile.baseN, mmTile.mTailTile, mmTile.nTailTile,
        mmTile.mBaseTailSplitCnt, mmTile.nBaseTailSplitCnt, mmTile.mTailMain, mmTile.nTailMain};
    params.qbmmParams = QBMMTiling{mmTile.baseM, mmTile.baseN, mmTile.baseK, mmTile.dbL0c,
        isBias_ ? 1u : 0u};
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

    GM_ADDR sendBuf = GmAddr<YType>(workspaceGM_, startRow * rankNPerPeer);
    GM_ADDR recvBuf = GmAddr<YType>(y_, startRow * rankNPerPeer);
    uint64_t sendOffset = static_cast<uint64_t>(taskM) * rankNPerPeer * sizeof(YType);
    uint64_t sendCount = static_cast<uint64_t>(taskM) * rankNPerPeer;
    uint64_t strideCount = static_cast<uint64_t>(ti.rankM) * rankNPerPeer;

    bool hasPrev = false;

    for (uint32_t index = 0; index < taskCnt; ++index) {
        MatmulProcess(startRow + static_cast<uint64_t>(index) * taskM, mmTile);
        AscendC::SyncAll();

        if (hasPrev) {
            hccl_.Wait(prevHandle_);
        }

        prevHandle_ = hccl_.template AlltoAll<true>(
            sendBuf + static_cast<uint64_t>(index) * sendOffset,
            recvBuf + static_cast<uint64_t>(index) * sendOffset,
            sendCount, static_cast<AscendC::HcclDataType>(ti.hcclDataType),
            strideCount, 1);
        hasPrev = true;
    }

    if (hasPrev) {
        hccl_.Wait(prevHandle_);
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
        if (ti.tileCnt > 0) {
            ProcessBatch(0, ti.tileM, ti.tileCnt, tilingData_->tileQbmmTilingData);
        }
        if (ti.tailCnt > 0) {
            uint64_t mainRows = static_cast<uint64_t>(ti.tileM) * ti.tileCnt;
            ProcessBatch(mainRows, ti.tailM, ti.tailCnt, tilingData_->tailQbmmTilingData);
        }
        AscendC::SyncAll();
        hccl_.Finalize();
    }
}

} // namespace Apace
