/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms of the
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR
 * PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file all_gather_mx_matmul_ccu_impl.h
 * \brief AllGather + QuantMatmul fusion kernel implementation (CCU variant)
 *
 * Communication state is held by HcommCommState; wait policy is handled by
 * HcommCommWaitPolicy, which references HcommCommState via a state_ pointer.
 * Inside the kernel, per-tile WaitTile() waits for the corresponding HCCL handle.
 *
 * Init():
 *   AIC: hccl_.InitV2() / SetCcTilingV2(), then batch-submit AllGather<true>
 *        tasks (scale + dataHead + dataTail), and bind CommPolicy to commState_.
 *
 * Run():
 *   AIC: execute FragmentTensor kernel, SyncAll(), then hccl_.Finalize().
 *        Inside the kernel, per-tile commPolicy_.WaitTile() -> state_->hccl_.Wait(handle).
 *
 * AIC-only execution: communication dispatch, compute, and per-tile wait all happen
 * in the AIC core, with no AIV participation and no cross-core flag synchronization.
 */

#pragma once

#include "lib/hccl/hccl.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#define WINDOW_LEN 1L
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#undef WINDOW_LEN
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "include/tensor_api/tensor.h"
#include "apace/block/blaze_ext/gemm/block/qmm_mx_block_mmad_fragment.h"
#include "apace/basic/fragment_tensor/fragment_tensor.h"
#include "apace/basic/fragment_tensor/fragment_tensor_api.h"
#include "apace/tiling/quant_matmul_tiling_data.h"
#include "apace/tiling/comm_tiling_data.h"
#include "apace/kernel/all_gather_quant_matmul/all_gather_mx_matmul_ccu_tiling_data.h"
#include "apace/kernel/all_gather_quant_matmul/qmm_mx_kernel_ag_udma.h"

namespace AllGatherQuantMatmulImpl {
using namespace AscendC;

template <typename T>
constexpr uint64_t GetHcclDataType()
{
    if constexpr (AscendC::IsSameType<T, float8_e5m2_t>::value) {
        return AscendC::HCCL_DATA_TYPE_FP8E5M2;
    } else {
        return AscendC::HCCL_DATA_TYPE_FP8E4M3;
    }
}

struct HcommCommState {
    static constexpr AscendC::HcclServerType ServerType = AscendC::HcclServerType::HCCL_SERVER_TYPE_CCU;
    AscendC::Hccl<ServerType> hccl_;
    AscendC::HcclHandle scaleHandle_{0};
    AscendC::HcclHandle dataHeadHandle_{0};
    AscendC::HcclHandle dataTailHandle_{0};
    uint32_t tileCnt_{0};
};

struct HcommCommWaitPolicy {
    HcommCommState *state_{nullptr};

    __aicore__ inline void WaitTile(uint32_t tileIdx)
    {
        if (tileIdx == 0) {
            return;
        }
        if (tileIdx == 1) {
            state_->hccl_.Wait(state_->scaleHandle_);
        }
        if (tileIdx <= state_->tileCnt_) {
            state_->hccl_.Wait(state_->dataHeadHandle_);
        } else {
            state_->hccl_.Wait(state_->dataTailHandle_);
        }
    }
};

template <typename AType, typename BType, typename CType>
class AllGatherMxMatmulCcuImpl {
public:
    __aicore__ inline AllGatherMxMatmulCcuImpl(ccuAllGatherMxMatmulTilingData *tilingData)
        : tilingData_(tilingData) {}

    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR aScaleGM, GM_ADDR bGM, GM_ADDR bScaleGM, GM_ADDR cGM,
                                GM_ADDR workspaceGM);
    __aicore__ inline void Run();

    using QuantMatmulKernelImpl = QmmMxKernelAgUdma<AType, BType, CType, HcommCommWaitPolicy>;
    using KernelParams = typename QuantMatmulKernelImpl::Params;

    QuantMatmulKernelImpl quantMatmulKernelImpl_;
    HcommCommState commState_;

private:
    __aicore__ inline void InitBaseParams();
    __aicore__ inline void CommitAllGather();
    __aicore__ inline void SetupKernelParams(KernelParams &params);

    ccuAllGatherMxMatmulTilingData *tilingData_;

    GM_ADDR aGM_{};
    GM_ADDR aScaleGM_{};
    GM_ADDR bGM_{};
    GM_ADDR bScaleGM_{};
    GM_ADDR cGM_{};
    GM_ADDR workspaceGM_{};
    GM_ADDR gatherDataAddr_{};
    GM_ADDR gatherScaleAddr_{};

    uint32_t rankId_{};
    uint32_t rankSize_{};
    uint32_t m_{};
    uint32_t k_{};
    uint32_t n_{};
    uint32_t tileCnt_{};
    uint32_t tileM_{};
    uint32_t tailCnt_{};
    uint32_t tailM_{};
    uint32_t paddedTailM_{};
    uint32_t commTurn_{};
    uint64_t headRows_{};
    uint64_t scaleKLen_{};
    uint64_t scaleKGroups_{};
    uint64_t dataBytesPerMRow_{};
    uint64_t scaleBytesPerMRow_{};
    uint64_t cBytesPerM_{};
    uint64_t dataRegionBytes_{};
    uint64_t scaleRegionBytes_{};

    static constexpr uint32_t kPaddingLength = 16;
    static constexpr uint64_t scaleDataType_ = HCCL_DATA_TYPE_FP8E8M0;
    static constexpr uint64_t dataDataType_ = GetHcclDataType<AType>();
};

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherMxMatmulCcuImpl<AType, BType, CType>::InitBaseParams()
{
    const auto &ct = tilingData_->commTile;
    tileCnt_ = static_cast<uint32_t>(ct.splitAxisTileCnt);
    tileM_ = static_cast<uint32_t>(ct.splitAxisTileSize);
    tailCnt_ = static_cast<uint32_t>(ct.splitAxisTailCnt);
    tailM_ = static_cast<uint32_t>(ct.splitAxisTailSize);
    k_ = tilingData_->mmTile.k;
    n_ = tilingData_->mmTile.n;
    m_ = static_cast<uint32_t>(ct.splitAxisTileSize * ct.splitAxisTileCnt +
                               ct.splitAxisTailSize * ct.splitAxisTailCnt);
    commTurn_ = tileCnt_ + tailCnt_;
    paddedTailM_ = (tailM_ > 0) ? ((tailM_ + kPaddingLength - 1) / kPaddingLength * kPaddingLength) : 0U;
    headRows_ = static_cast<uint64_t>(tileCnt_) * tileM_;

    scaleKGroups_ = Blaze::Gemm::CeilDiv(static_cast<uint64_t>(k_), Blaze::Gemm::MXFP_DIVISOR_SIZE);
    scaleKLen_ = scaleKGroups_ * static_cast<uint64_t>(Blaze::Gemm::MXFP_MULTI_BASE_SIZE);
    dataBytesPerMRow_ = static_cast<uint64_t>(k_) * sizeof(AType);
    scaleBytesPerMRow_ = scaleKGroups_ * static_cast<uint64_t>(
        Blaze::Gemm::MXFP_MULTI_BASE_SIZE) * sizeof(AscendC::fp8_e8m0_t);
    cBytesPerM_ = static_cast<uint64_t>(n_) * sizeof(CType);
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherMxMatmulCcuImpl<AType, BType, CType>::Init(
    GM_ADDR aGM, GM_ADDR aScaleGM, GM_ADDR bGM, GM_ADDR bScaleGM, GM_ADDR cGM, GM_ADDR workspaceGM)
{
    aGM_ = aGM;
    aScaleGM_ = aScaleGM;
    bGM_ = bGM;
    bScaleGM_ = bScaleGM;
    cGM_ = cGM;
    workspaceGM_ = workspaceGM;

    InitBaseParams();

    // HCCL initialization
    commState_.hccl_.InitV2(AscendC::GetHcclContext<0>(), &(tilingData_->mc2InitTiling));
    commState_.hccl_.SetCcTilingV2(
        static_cast<uint64_t>(offsetof(ccuAllGatherMxMatmulTilingData, mc2CcTiling)));
    rankId_ = commState_.hccl_.GetRankId();
    rankSize_ = commState_.hccl_.GetRankDim();

    // Compute gather buffer addresses
    dataRegionBytes_ = static_cast<uint64_t>(rankSize_) * m_ * dataBytesPerMRow_;
    scaleRegionBytes_ = static_cast<uint64_t>(rankSize_) * m_ * scaleBytesPerMRow_;
    gatherDataAddr_ = workspaceGM_;
    gatherScaleAddr_ = workspaceGM_ + dataRegionBytes_;

    // Submit AllGather (non-blocking)
    CommitAllGather();

    // Bind CommPolicy to CommState
    quantMatmulKernelImpl_.GetCommPolicy().state_ = &commState_;
    commState_.tileCnt_ = tileCnt_;
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherMxMatmulCcuImpl<AType, BType, CType>::CommitAllGather()
{
    // Scale: full amount, single submission
    commState_.scaleHandle_ = commState_.hccl_.template AllGather<true>(
        aScaleGM_, gatherScaleAddr_,
        static_cast<uint64_t>(m_) * scaleKLen_,
        static_cast<AscendC::HcclDataType>(scaleDataType_),
        static_cast<uint64_t>(m_) * scaleKLen_,
        static_cast<uint8_t>(1));

    // Data Head: batch submit all head tiles via repeat
    commState_.dataHeadHandle_ = commState_.hccl_.template AllGather<true>(
        aGM_, gatherDataAddr_,
        static_cast<uint64_t>(tileM_) * k_,
        static_cast<AscendC::HcclDataType>(dataDataType_),
        static_cast<uint64_t>(m_) * k_,
        static_cast<uint8_t>(tileCnt_));

    // Data Tail: batch submit all tail tiles via repeat
    if (tailCnt_ > 0) {
        commState_.dataTailHandle_ = commState_.hccl_.template AllGather<true>(
            aGM_ + headRows_ * k_ * sizeof(AType),
            gatherDataAddr_ + headRows_ * k_ * sizeof(AType),
            static_cast<uint64_t>(tailM_) * k_,
            static_cast<AscendC::HcclDataType>(dataDataType_),
            static_cast<uint64_t>(m_) * k_,
            static_cast<uint8_t>(tailCnt_));
    }
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherMxMatmulCcuImpl<AType, BType, CType>::SetupKernelParams(KernelParams &params)
{
    params.mmTile = &tilingData_->mmTile;
    params.qbmmParams = {tilingData_->mmTile.baseM, tilingData_->mmTile.baseN,
                         tilingData_->mmTile.baseK, tilingData_->mmTile.dbL0c};
    params.fragParams = {tileCnt_, tileM_, tailCnt_, tailM_, paddedTailM_, commTurn_,
                         headRows_, rankId_, rankSize_, m_,
                         static_cast<uint64_t>(k_), static_cast<uint64_t>(n_), scaleKLen_};
    params.aGM = aGM_;
    params.aScaleGM = aScaleGM_;
    params.bGM = bGM_;
    params.bScaleGM = bScaleGM_;
    params.cGM = cGM_;
    params.winDataBase = gatherDataAddr_;
    params.winScaleBase = gatherScaleAddr_;
    params.dataBytesPerMRow = dataBytesPerMRow_;
    params.scaleBytesPerMRow = scaleBytesPerMRow_;
    params.cBytesPerM = cBytesPerM_;
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherMxMatmulCcuImpl<AType, BType, CType>::Run()
{
    KernelParams params;
    SetupKernelParams(params);
    quantMatmulKernelImpl_(params);
    AscendC::SyncAll();
    commState_.hccl_.Finalize();
}

} // namespace AllGatherQuantMatmulImpl

__global__ __aicore__ void AllGatherQuantMatmulCcuKernel(
    GM_ADDR aGM, GM_ADDR aScaleGM,
    GM_ADDR bGM, GM_ADDR bScaleGM, GM_ADDR cGM,
    GM_ADDR workspaceGM,
    ccuAllGatherMxMatmulTilingData tilingData)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_0);
    AllGatherQuantMatmulImpl::AllGatherMxMatmulCcuImpl<
        fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t> impl(&tilingData);
    impl.Init(aGM, aScaleGM, bGM, bScaleGM, cGM, workspaceGM);
    impl.Run();
}
