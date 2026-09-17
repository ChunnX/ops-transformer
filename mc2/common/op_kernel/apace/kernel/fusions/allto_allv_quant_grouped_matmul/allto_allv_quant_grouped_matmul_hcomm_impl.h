/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file allto_allv_quant_grouped_matmul_hcomm_impl.h
 * \brief AIC-only HCCL AllToAllV + Blaze MX GroupedMatMul implementation.
 */

#pragma once

#include "lib/hccl/hccl.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad_qgmm_mx.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "../../matmul/grouped_matmul/grouped_matmul_mx_kernel.h"
#include "allto_allv_quant_grouped_matmul_tiling_data.h"

namespace Apace {

template <typename T>
constexpr AscendC::HcclDataType GetA2avGmmHcclDataType()
{
    if constexpr (AscendC::IsSameType<T, float8_e5m2_t>::value) {
        return AscendC::HCCL_DATA_TYPE_FP8E5M2;
    } else if constexpr (AscendC::IsSameType<T, fp4x2_e2m1_t>::value) {
        return AscendC::HCCL_DATA_TYPE_UINT8;
    } else {
        return AscendC::HCCL_DATA_TYPE_FP8E4M3;
    }
}

template <typename AType, typename BType, typename CType, bool TransB, bool TransMmB,
          AscendC::HcclServerType ServerType>
class AllToAllvQuantGmmHcommImpl {
public:
    explicit __aicore__ inline AllToAllvQuantGmmHcommImpl(hcommA2avQuantGmmTilingData *tilingData)
        : tilingData_(tilingData)
    {}

    __aicore__ inline void Init(GM_ADDR gmmX, GM_ADDR gmmXScale, GM_ADDR gmmWeight, GM_ADDR gmmWeightScale,
                                GM_ADDR mmX, GM_ADDR mmXScale, GM_ADDR mmWeight, GM_ADDR mmWeightScale,
                                GM_ADDR gmmY, GM_ADDR mmY, GM_ADDR permuteOut, GM_ADDR workspace, GM_ADDR tilingGM)
    {
        xGM_ = gmmX;
        xScaleGM_ = gmmXScale;
        weightGM_ = gmmWeight;
        weightScaleGM_ = gmmWeightScale;
        yGM_ = gmmY;
        mmXGM_ = mmX;
        mmXScaleGM_ = mmXScale;
        mmWeightGM_ = mmWeight;
        mmWeightScaleGM_ = mmWeightScale;
        mmYGM_ = mmY;
        dataRecvGM_ = tilingData_->hasPermuteOut != 0U ? permuteOut : workspace + tilingData_->dataRecvOffset;
        scaleRecvGM_ = workspace + tilingData_->scaleRecvOffset;
        sparseGroupListGM_ = tilingGM + offsetof(hcommA2avQuantGmmTilingData, sparseGroupList);
        mmGroupListGM_ = tilingGM + offsetof(hcommA2avQuantGmmTilingData, mmGroupList);

        hccl_.InitV2(AscendC::GetHcclContext<0>(), &tilingData_->mc2InitTiling);
        hccl_.SetCcTilingV2(static_cast<uint64_t>(offsetof(hcommA2avQuantGmmTilingData, mc2CcTiling)));
    }

    __aicore__ inline void Run()
    {
        SubmitExpert(0U);
        if (tilingData_->localMatmul != 0U) {
            RunLocalMatmul();
        }

        for (uint32_t expertIdx = 0U; expertIdx < tilingData_->expertNum; ++expertIdx) {
            WaitExpert();
            if (expertIdx + 1U < tilingData_->expertNum) {
                SubmitExpert(expertIdx + 1U);
            }
            RunExpertMatmul(expertIdx);
        }

        AscendC::SyncAll();
        hccl_.Finalize();
    }

private:
    static constexpr uint8_t GROUP_LIST_TYPE_OFFSET = 0U;
    static constexpr uint8_t GROUP_LIST_TYPE_SPARSE = 2U;
    static constexpr AscendC::HcclDataType X_HCCL_DATA_TYPE = GetA2avGmmHcclDataType<AType>();
    static constexpr AscendC::HcclDataType SCALE_HCCL_DATA_TYPE = AscendC::HCCL_DATA_TYPE_FP8E8M0;

    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutB = AscendC::Std::conditional_t<TransB, asc::te::dn_ext_layout_ptn,
                                                asc::te::nd_ext_layout_ptn>;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using LayoutBias = asc::te::nd_ext_layout_ptn;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
    using DispatchPolicy =
        Blaze::Gemm::GroupedMatmulWithScaleMx<0, false, Blaze::Gemm::KernelGroupedMmadWithScaleMx>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC,
                                                    float, LayoutBias>;
    using GmmKernel = Blaze::Gemm::Kernel::GroupedMatmulMxKernel<ProblemShape, BlockMmad, BlockScheduler>;

    using MmLayoutB = AscendC::Std::conditional_t<TransMmB, asc::te::dn_ext_layout_ptn,
                                                  asc::te::nd_ext_layout_ptn>;
    using MmBlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, MmLayoutB, CType,
                                                      LayoutC, float, LayoutBias>;
    using MmKernel = Blaze::Gemm::Kernel::GroupedMatmulMxKernel<ProblemShape, MmBlockMmad, BlockScheduler>;

    __aicore__ inline uint32_t GetRecvRows(uint32_t expertIdx, uint32_t peer) const
    {
        const uint32_t segment = expertIdx * tilingData_->rankSize + peer;
        return static_cast<uint32_t>(tilingData_->sparseGroupList[segment * 2U + 1U]);
    }

    __aicore__ inline uint64_t GetExpertStartRow(uint32_t expertIdx) const
    {
        uint64_t rows = 0U;
        for (uint32_t previous = 0U; previous < expertIdx; ++previous) {
            for (uint32_t peer = 0U; peer < tilingData_->rankSize; ++peer) {
                rows += GetRecvRows(previous, peer);
            }
        }
        return rows;
    }

    __aicore__ inline void PrepareExpertCommunication(uint32_t expertIdx)
    {
        const uint64_t dataAxis = tilingData_->isMxFp4 != 0U ?
                                      (static_cast<uint64_t>(tilingData_->k) + 1U) / 2U :
                                      static_cast<uint64_t>(tilingData_->k);
        const uint64_t scaleAxis = tilingData_->scaleKLen;
        uint64_t sendPeerBaseRow = 0U;
        uint64_t recvRowOffset = GetExpertStartRow(expertIdx);

        for (uint32_t peer = 0U; peer < tilingData_->rankSize; ++peer) {
            const uint32_t segment = expertIdx * tilingData_->rankSize + peer;
            uint64_t sendExpertOffset = 0U;
            uint64_t sendPeerRows = 0U;
            for (uint32_t e = 0U; e < tilingData_->expertNum; ++e) {
                const uint64_t rows = tilingData_->segmentSendRows[e * tilingData_->rankSize + peer];
                if (e < expertIdx) {
                    sendExpertOffset += rows;
                }
                sendPeerRows += rows;
            }

            const uint64_t sendRows = tilingData_->segmentSendRows[segment];
            const uint64_t recvRows = GetRecvRows(expertIdx, peer);
            dataSendCount_[peer] = sendRows * dataAxis;
            dataSendOffset_[peer] = (sendPeerBaseRow + sendExpertOffset) * dataAxis;
            dataRecvCount_[peer] = recvRows * dataAxis;
            dataRecvOffset_[peer] = recvRowOffset * dataAxis;
            scaleSendCount_[peer] = sendRows * scaleAxis;
            scaleSendOffset_[peer] = (sendPeerBaseRow + sendExpertOffset) * scaleAxis;
            scaleRecvCount_[peer] = recvRows * scaleAxis;
            scaleRecvOffset_[peer] = recvRowOffset * scaleAxis;

            sendPeerBaseRow += sendPeerRows;
            recvRowOffset += recvRows;
        }
    }

    __aicore__ inline void SubmitExpert(uint32_t expertIdx)
    {
        PrepareExpertCommunication(expertIdx);
        scaleHandle_ = hccl_.template AlltoAllV<true>(
            xScaleGM_, scaleSendCount_, scaleSendOffset_, SCALE_HCCL_DATA_TYPE, scaleRecvGM_, scaleRecvCount_,
            scaleRecvOffset_, SCALE_HCCL_DATA_TYPE);
        dataHandle_ = hccl_.template AlltoAllV<true>(xGM_, dataSendCount_, dataSendOffset_, X_HCCL_DATA_TYPE,
                                                     dataRecvGM_, dataRecvCount_, dataRecvOffset_, X_HCCL_DATA_TYPE);
    }

    __aicore__ inline void WaitExpert()
    {
        hccl_.Wait(scaleHandle_);
        hccl_.Wait(dataHandle_);
        // Wait records the dependency on the data-move pipe.  Drain it before
        // the scalar side reuses the parameter arrays for the next collective.
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void RunExpertMatmul(uint32_t expertIdx)
    {
        int64_t expertRows = 0;
        for (uint32_t peer = 0U; peer < tilingData_->rankSize; ++peer) {
            expertRows += static_cast<int64_t>(GetRecvRows(expertIdx, peer));
        }
        if (expertRows == 0) {
            return;
        }

        const uint64_t expertStartRow = GetExpertStartRow(expertIdx);
        const uint64_t dataAxis = tilingData_->isMxFp4 != 0U ?
                                      (static_cast<uint64_t>(tilingData_->k) + 1U) / 2U :
                                      static_cast<uint64_t>(tilingData_->k);
        typename GmmKernel::GMMTiling gmmTiling{};
        FillGmmTiling(gmmTiling, tilingData_->gmmTiling, tilingData_->rankSize, expertRows, tilingData_->n,
                      tilingData_->k, GROUP_LIST_TYPE_SPARSE);

        typename GmmKernel::Params params{};
        params.problemShape = ProblemShape{expertRows, static_cast<int64_t>(tilingData_->n),
                                           static_cast<int64_t>(tilingData_->k), 0};
        params.mmadParams.aGmAddr = dataRecvGM_ + expertStartRow * dataAxis;
        params.mmadParams.scaleAGmAddr = scaleRecvGM_ + expertStartRow * tilingData_->scaleKLen;
        params.mmadParams.bGmAddr = weightGM_;
        params.mmadParams.scaleBGmAddr = weightScaleGM_;
        params.mmadParams.cGmAddr = yGM_ + expertStartRow * tilingData_->n * sizeof(CType);
        params.mmadParams.biasGmAddr = nullptr;
        params.groupListGmAddr = sparseGroupListGM_ + expertIdx * tilingData_->rankSize * 2U * sizeof(int64_t);
        params.gmmParams = gmmTiling;
        params.groupOffsetTables = {};

        {
            GmmKernel kernel;
            kernel.Init(params);
            kernel.Run();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void RunLocalMatmul()
    {
        typename MmKernel::GMMTiling mmTiling{};
        FillGmmTiling(mmTiling, tilingData_->mmTiling, 1U, tilingData_->mmM, tilingData_->mmN, tilingData_->mmK,
                      GROUP_LIST_TYPE_OFFSET);

        typename MmKernel::Params params{};
        params.problemShape = ProblemShape{static_cast<int64_t>(tilingData_->mmM),
                                           static_cast<int64_t>(tilingData_->mmN),
                                           static_cast<int64_t>(tilingData_->mmK), 0};
        params.mmadParams.aGmAddr = mmXGM_;
        params.mmadParams.scaleAGmAddr = mmXScaleGM_;
        params.mmadParams.bGmAddr = mmWeightGM_;
        params.mmadParams.scaleBGmAddr = mmWeightScaleGM_;
        params.mmadParams.cGmAddr = mmYGM_;
        params.mmadParams.biasGmAddr = nullptr;
        params.groupListGmAddr = mmGroupListGM_;
        params.gmmParams = mmTiling;
        params.groupOffsetTables = {};

        {
            MmKernel kernel;
            kernel.Init(params);
            kernel.Run();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <typename GmmTiling>
    __aicore__ inline void FillGmmTiling(GmmTiling &out, const A2avGmmMxTiling &in, uint32_t groupNum, int64_t m,
                                         int64_t n, int64_t k, uint8_t groupListType)
    {
        out.groupNum = groupNum;
        out.m = m;
        out.n = n;
        out.k = k;
        out.baseM = in.baseM;
        out.baseN = in.baseN;
        out.baseK = in.baseK;
        out.kAL1 = in.kAL1;
        out.kBL1 = in.kBL1;
        out.scaleKAL1 = in.scaleKAL1;
        out.isBias = 0U;
        out.dbL0C = in.dbL0C;
        out.l1BufferStage = in.l1BufferStage;
        out.groupListType = groupListType;
    }

private:
    hcommA2avQuantGmmTilingData *tilingData_{nullptr};
    AscendC::Hccl<ServerType> hccl_;
    AscendC::HcclHandle scaleHandle_{0};
    AscendC::HcclHandle dataHandle_{0};

    GM_ADDR xGM_{nullptr};
    GM_ADDR xScaleGM_{nullptr};
    GM_ADDR weightGM_{nullptr};
    GM_ADDR weightScaleGM_{nullptr};
    GM_ADDR yGM_{nullptr};
    GM_ADDR mmXGM_{nullptr};
    GM_ADDR mmXScaleGM_{nullptr};
    GM_ADDR mmWeightGM_{nullptr};
    GM_ADDR mmWeightScaleGM_{nullptr};
    GM_ADDR mmYGM_{nullptr};
    GM_ADDR dataRecvGM_{nullptr};
    GM_ADDR scaleRecvGM_{nullptr};
    GM_ADDR sparseGroupListGM_{nullptr};
    GM_ADDR mmGroupListGM_{nullptr};

    uint64_t dataSendCount_[A2AV_GMM_MAX_RANK_NUM] = {0};
    uint64_t dataSendOffset_[A2AV_GMM_MAX_RANK_NUM] = {0};
    uint64_t dataRecvCount_[A2AV_GMM_MAX_RANK_NUM] = {0};
    uint64_t dataRecvOffset_[A2AV_GMM_MAX_RANK_NUM] = {0};
    uint64_t scaleSendCount_[A2AV_GMM_MAX_RANK_NUM] = {0};
    uint64_t scaleSendOffset_[A2AV_GMM_MAX_RANK_NUM] = {0};
    uint64_t scaleRecvCount_[A2AV_GMM_MAX_RANK_NUM] = {0};
    uint64_t scaleRecvOffset_[A2AV_GMM_MAX_RANK_NUM] = {0};
};

} // namespace Apace
