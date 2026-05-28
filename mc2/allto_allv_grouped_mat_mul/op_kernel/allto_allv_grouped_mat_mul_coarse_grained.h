/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/* !
* \file allto_allv_grouped_mat_mul_coarse_grained.h
* \brief
*/
#ifndef ALL_TO_ALL_V_GROUPED_MAT_MUL_H
#define ALL_TO_ALL_V_GROUPED_MAT_MUL_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/hccl/hccl.h"
#include "allto_allv_gmm.h"
#include "lib/matmul_intf.h"
#include "allto_allv_grouped_mat_mul_tiling.h"

constexpr int CCU_COMM_MODE = 0;
constexpr int AICPU_COMM_MODE = 1;

namespace AscendC {
using namespace ALLTO_ALLV_GMM;

template<int commMode = CCU_COMM_MODE>
struct HcclTypeSelector {
    using type = Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU>;
};

template<>
struct HcclTypeSelector<AICPU_COMM_MODE> {
    using type = Hccl<HcclServerType::HCCL_SERVER_TYPE_AICPU>;
};

template <typename DataType, bool IsTranGmmW, bool IsTranMmW, int TILINGKEY_COMM_MODE = CCU_COMM_MODE>
class AlltoAllvGmmCoarseGrained
{
public:
    __aicore__ inline AlltoAllvGmmCoarseGrained()
    {}
    __aicore__ inline void Init(
        GM_ADDR gmmxGM, GM_ADDR gmmweightGM, GM_ADDR sendCountsTensorOptionalGM, GM_ADDR recvCountsTensorOptionalGM,
        GM_ADDR mmxOptionalGM, GM_ADDR mmweightOptionalGM, GM_ADDR gmmyGM, GM_ADDR mmyOptionalGM,
        GM_ADDR permuteOutOptionalGM, GM_ADDR workspaceGM, GM_ADDR contextGM, const AlltoAllvGmmTilingData* tilingData,
        __gm__ void* hcclInitTiling, __gm__ void* alltoAllvCcTiling, TPipe* tPipe);
    __aicore__ inline void Process();

private:
    using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DataType, false>;
    using gmmBType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DataType, IsTranGmmW>;
    using mmBType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DataType, IsTranMmW>;
    using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DataType, false>;
    using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DataType>;
    using gmmType = MMImplType<aType, gmmBType, cType, biasType, CFG_MDL>;
    using mmType = MMImplType<aType, mmBType, cType, biasType, CFG_MDL>;

    __aicore__ inline void CalcMatmul();
    __aicore__ inline void HcclAlltoAllvPrepare();
    __aicore__ inline void HcclAlltoAllvExec();
    __aicore__ inline void HcclFinalize();
    __aicore__ inline void Rearrange(uint32_t startExpert, uint32_t expertNum);
    __aicore__ inline uint32_t GetRoundExpertNum(uint32_t roundIdx);

    GM_ADDR gmmxGM_ = nullptr;
    GM_ADDR gmmwGM_ = nullptr;
    GM_ADDR sendCntsGM_ = nullptr;
    GM_ADDR recvCntsGM_ = nullptr;
    GM_ADDR mmxGM_ = nullptr;
    GM_ADDR mmwGM_ = nullptr;
    GM_ADDR gmmyGM_ = nullptr;
    GM_ADDR mmyGM_ = nullptr;
    GM_ADDR permuteOutGM_ = nullptr;
    GM_ADDR rearrangeGM_ = nullptr;
    const AlltoAllvGmmTilingData* tilingData_ = nullptr;
    uint32_t rankId_ = 0U;
    uint32_t rankDim_ = 8U;
    uint32_t expertNumInOneRank_ = 0U;
    uint32_t expertNumAll_ = 0U;
    uint32_t expertMergeNum_ = 1U;
    uint32_t commRoundCount_ = 0U;
    bool rearrangeNeeded_ = false;

    // alltoall flow data structures
    static constexpr uint64_t MAX_HANDLE_ID_NUM = 64U;
    typename HcclTypeSelector<TILINGKEY_COMM_MODE>::type hccl_;
    HcclDataType hcclDataType_ = HCCL_DATA_TYPE_FP16;
    HcclHandle alltoAllvHandleId_[MAX_HANDLE_ID_NUM] = {INVALID_HANDLE_ID};

    GlobalTensor<DataType> gmmxGMTensor_;
    GlobalTensor<DataType> permutedGMTensor_;
    GlobalTensor<DataType> rearrangeGMTensor_;

    // Per-round recv offsets for rearrange
    uint64_t roundRecvOffsetBase_[MAX_EP_RANK_SIZE] = {0UL};

    uint64_t axisH1_;
    uint64_t axisN1_;

    typename gmmType::MT gmm_;
    typename mmType::MT mm_;
    TPipe* tPipe_ = nullptr;
};

template <typename DataType, bool IsTranGmmW, bool IsTranMmW, int TILINGKEY_COMM_MODE>
__aicore__ inline void AlltoAllvGmmCoarseGrained<DataType, IsTranGmmW, IsTranMmW, TILINGKEY_COMM_MODE>::Init(
    GM_ADDR gmmxGM, GM_ADDR gmmweightGM, GM_ADDR sendCountsTensorOptionalGM, GM_ADDR recvCountsTensorOptionalGM,
    GM_ADDR mmxOptionalGM, GM_ADDR mmweightOptionalGM, GM_ADDR gmmyGM, GM_ADDR mmyOptionalGM,
    GM_ADDR permuteOutOptionalGM, GM_ADDR workspaceGM, GM_ADDR contextGM, const AlltoAllvGmmTilingData* tilingData,
    __gm__ void* hcclInitTiling, __gm__ void* alltoAllvCcTiling, TPipe* tPipe)
{
    gmmxGM_ = gmmxGM;
    gmmwGM_ = gmmweightGM;
    sendCntsGM_ = sendCountsTensorOptionalGM;
    recvCntsGM_ = recvCountsTensorOptionalGM;
    mmxGM_ = mmxOptionalGM;
    mmwGM_ = mmweightOptionalGM;
    gmmyGM_ = gmmyGM;
    mmyGM_ = mmyOptionalGM;
    tilingData_ = tilingData;
    tPipe_ = tPipe;
    permuteOutGM_ = tilingData_->commonTilingInfo.isPermuteOut ? permuteOutOptionalGM : workspaceGM;
    expertMergeNum_ = tilingData_->commonTilingInfo.expertMergeNum;
    commRoundCount_ = tilingData_->commonTilingInfo.commRoundCount;
    rearrangeNeeded_ = (expertMergeNum_ > 1U);

    // Compute rearrange buffer location in workspace
    if (rearrangeNeeded_) {
        uint64_t commOutSize = tilingData_->commonTilingInfo.A * tilingData_->commonTilingInfo.H1 * sizeof(DataType);
        uint64_t permOutSize = tilingData_->commonTilingInfo.isPermuteOut ? 0 : commOutSize;
        rearrangeGM_ = workspaceGM + commOutSize + permOutSize;
        rearrangeGMTensor_.SetGlobalBuffer((__gm__ DataType*)rearrangeGM_);
    }

    const void *hcclInitTilingV2 = &(tilingData_->hcclInitTiling);
    uint64_t hcclCcTilingOffset = offsetof(AlltoAllvGmmTilingData, alltoAllvCcTiling);
    hccl_.InitV2(contextGM, hcclInitTilingV2);
    hccl_.SetCcTilingV2(hcclCcTilingOffset);
    rankId_ = hccl_.GetRankId();
    rankDim_ = hccl_.GetRankDim();

    expertNumInOneRank_ = tilingData_->commonTilingInfo.E_ep;
    expertNumAll_ = expertNumInOneRank_ * rankDim_;

    axisH1_ = tilingData_->commonTilingInfo.H1;
    axisN1_ = tilingData_->commonTilingInfo.N1;

    if constexpr (AscendC::IsSameType<DataType, bfloat16_t>::value) {
        hcclDataType_ = HCCL_DATA_TYPE_BFP16;
    }

    gmmxGMTensor_.SetGlobalBuffer((__gm__ DataType*)this->gmmxGM_);
    permutedGMTensor_.SetGlobalBuffer((__gm__ DataType*)this->permuteOutGM_);
}

template <typename DataType, bool IsTranGmmW, bool IsTranMmW, int TILINGKEY_COMM_MODE>
__aicore__ inline uint32_t AlltoAllvGmmCoarseGrained<DataType, IsTranGmmW, IsTranMmW, TILINGKEY_COMM_MODE>::GetRoundExpertNum(uint32_t roundIdx)
{
    uint32_t tailRounds = expertNumInOneRank_ % commRoundCount_;
    return expertMergeNum_ + (roundIdx < tailRounds ? 1U : 0U);
}

template <typename DataType, bool IsTranGmmW, bool IsTranMmW, int TILINGKEY_COMM_MODE>
__aicore__ inline void AlltoAllvGmmCoarseGrained<DataType, IsTranGmmW, IsTranMmW, TILINGKEY_COMM_MODE>::Process()
{
    HcclAlltoAllvPrepare();
    if (tilingData_->commonTilingInfo.isNeedMM) {
        CalcMatmul();
        SyncAll<false>();
    }
    HcclAlltoAllvExec();
    SyncAll<false>();
    HcclFinalize();
}

template <typename DataType, bool IsTranGmmW, bool IsTranMmW, int TILINGKEY_COMM_MODE>
__aicore__ inline void AlltoAllvGmmCoarseGrained<DataType, IsTranGmmW, IsTranMmW, TILINGKEY_COMM_MODE>::CalcMatmul()
{
    mm_.Init(&(tilingData_->mmTilingData));
    GMMCompute<mmType> computeOp(mm_);
    computeOp.Init(mmxGM_, mmwGM_, mmyGM_);
    GMMProcess<decltype(computeOp)> mmOp(computeOp);
    mmOp.Init(
        tilingData_->mmTilingData.baseM, tilingData_->mmTilingData.baseN, tilingData_->commonTilingInfo.aicCoreNum);

    uint64_t mmInOffset[1] = {0};
    uint64_t mmOutOffset[1] = {0};
    uint64_t mmWeightOffset[1] = {0};
    uint32_t tokenNum[1] = {(uint32_t)(tilingData_->commonTilingInfo.BS)};
    if ASCEND_IS_AIC {
        mmOp.Process(
            this->rankId_, tilingData_->commonTilingInfo.H2, tilingData_->commonTilingInfo.N2, mmInOffset, mmOutOffset,
            tokenNum, 1, 0);
    }
}

template <typename DataType, bool IsTranGmmW, bool IsTranMmW, int TILINGKEY_COMM_MODE>
__aicore__ inline void AlltoAllvGmmCoarseGrained<DataType, IsTranGmmW, IsTranMmW,
                                                  TILINGKEY_COMM_MODE>::HcclAlltoAllvPrepare()
{
    if ASCEND_IS_AIV {
        if (GetBlockIdx() != 0) {
            return;
        }
        const auto* sendCnt = &tilingData_->aicpuTiling.sendCnt[0];
        const auto* recvCnt = &tilingData_->aicpuTiling.recvCnt[0];
        uint32_t expertSoFar = 0U;
        uint32_t tailRounds = expertNumInOneRank_ % commRoundCount_;
        for (uint32_t roundIdx = 0U; roundIdx < commRoundCount_; roundIdx++) {
            uint32_t expertNum = GetRoundExpertNum(roundIdx);
            uint32_t startExpert = expertSoFar;

            uint64_t alltoAllvRecvOffsetLastSum = 0UL;
            uint64_t alltoAllvSendCnt[MAX_EP_RANK_SIZE] = {0UL};
            uint64_t alltoAllvSendOffset[MAX_EP_RANK_SIZE] = {0UL};
            uint64_t alltoAllvRecvCnt[MAX_EP_RANK_SIZE] = {0UL};
            uint64_t alltoAllvRecvOffset[MAX_EP_RANK_SIZE] = {0UL};

            for (uint32_t i = 0U; i < rankDim_; i++) {
                alltoAllvSendCnt[i] = 0UL;
                alltoAllvRecvCnt[i] = 0UL;
                for (uint32_t e = 0U; e < expertNum; e++) {
                    alltoAllvSendCnt[i] += static_cast<uint64_t>(sendCnt[startExpert + e + i * expertNumInOneRank_]) * axisH1_;
                    alltoAllvRecvCnt[i] += static_cast<uint64_t>(recvCnt[startExpert + e + i * expertNumInOneRank_]) * axisH1_;
                }
            }

            // Send offset computation
            alltoAllvSendOffset[0] = 0UL;
            for (uint32_t j = 0U; j < startExpert; j++) {
                alltoAllvSendOffset[0U] += static_cast<uint64_t>(sendCnt[j]) * axisH1_;
            }
            for (uint32_t i = 1U; i < rankDim_; i++) {
                alltoAllvSendOffset[i] = alltoAllvSendOffset[i - 1U];
                for (uint32_t j = 0U; j < expertNumInOneRank_; j++) {
                    alltoAllvSendOffset[i] +=
                        static_cast<uint64_t>(sendCnt[startExpert + (i - 1U) * expertNumInOneRank_ + j]) * axisH1_;
                }
            }

            // Recv offset computation
            for (uint32_t i = 0U; i < rankDim_; i++) {
                if ((roundIdx == 0U) && (i == 0U)) {
                    alltoAllvRecvOffset[i] = 0UL;
                    alltoAllvRecvOffsetLastSum += alltoAllvRecvCnt[0];
                } else {
                    alltoAllvRecvOffset[i] = alltoAllvRecvOffsetLastSum;
                    alltoAllvRecvOffsetLastSum += alltoAllvRecvCnt[i];
                }
            }

            // Store recv offsets for rearrange
            if (rearrangeNeeded_) {
                for (uint32_t i = 0U; i < rankDim_; i++) {
                    roundRecvOffsetBase_[i] = alltoAllvRecvOffset[i];
                }
            }

            alltoAllvHandleId_[roundIdx] = hccl_.template AlltoAllV<true>(
                (__gm__ uint8_t*)this->gmmxGMTensor_.GetPhyAddr(), alltoAllvSendCnt, alltoAllvSendOffset, hcclDataType_,
                (__gm__ uint8_t*)this->permutedGMTensor_.GetPhyAddr(), alltoAllvRecvCnt, alltoAllvRecvOffset,
                hcclDataType_);

            expertSoFar += expertNum;
        }
    }
}

template <typename DataType, bool IsTranGmmW, bool IsTranMmW, int TILINGKEY_COMM_MODE>
__aicore__ inline void AlltoAllvGmmCoarseGrained<DataType, IsTranGmmW, IsTranMmW,
                                                  TILINGKEY_COMM_MODE>::HcclAlltoAllvExec()
{
    gmm_.Init(&(tilingData_->gmmTilingData));
    GMMCompute<gmmType> computeOp(gmm_);
    // When rearrangeNeeded, compute reads from rearranged buffer; otherwise from permuteOut
    GM_ADDR computeInputGM = rearrangeNeeded_ ? rearrangeGM_ : permuteOutGM_;
    computeOp.Init(computeInputGM, gmmwGM_, gmmyGM_);
    GMMProcess<decltype(computeOp)> gmmOp(computeOp);
    gmmOp.Init(
        tilingData_->gmmTilingData.baseM, tilingData_->gmmTilingData.baseN, tilingData_->commonTilingInfo.aicCoreNum);

    auto* recvCnt = &tilingData_->aicpuTiling.recvCnt[0];

    // Pre-compute expert token counts and offsets
    uint64_t expertTokenNum[MAX_HANDLE_ID_NUM] = {0UL};
    uint64_t expertoffset[MAX_HANDLE_ID_NUM] = {0UL};
    for (uint32_t e = 0U; e < expertNumInOneRank_; e++) {
        uint64_t curTokenNum = 0;
        for (uint32_t i = 0U; i < rankDim_; i++) {
            curTokenNum += static_cast<uint64_t>(recvCnt[e + i * expertNumInOneRank_]);
        }
        expertTokenNum[e] = curTokenNum;
    }
    for (uint64_t i = 0; i < expertNumInOneRank_; i++) {
        if (i >= 1) {
            expertoffset[i] = expertoffset[i - 1] + expertTokenNum[i - 1] * axisH1_;
        }
    }

    uint32_t expertSoFar = 0U;
    uint32_t tailRounds = expertNumInOneRank_ % commRoundCount_;
    for (uint32_t roundIdx = 0U; roundIdx < commRoundCount_; roundIdx++) {
        uint32_t expertNum = GetRoundExpertNum(roundIdx);
        uint32_t startExpert = expertSoFar;

        if ASCEND_IS_AIV {
            if (GetBlockIdx() == 0) {
                hccl_.Wait(alltoAllvHandleId_[roundIdx]);
            }
        }
        SyncAll<false>();

        if (rearrangeNeeded_) {
            Rearrange(startExpert, expertNum);
            SyncAll<false>();
        }

        uint64_t mmInOffset[2] = {0};
        uint64_t mmOutOffset[2] = {0};
        uint64_t mmWeightOffset[2] = {0};
        uint32_t tokenNum[2] = {0};
        uint32_t processNum = 1;

        for (uint32_t e = startExpert; e < startExpert + expertNum; e++) {
            mmInOffset[0] = expertoffset[e];
            mmOutOffset[0] = expertoffset[e] / axisH1_ * axisN1_;
            mmWeightOffset[0] = e * axisH1_ * axisN1_;
            tokenNum[0] = expertTokenNum[e];

            if ASCEND_IS_AIC {
                if (tokenNum[0] != 0) {
                    gmmOp.Process(this->rankId_, axisH1_, axisN1_, mmInOffset, mmOutOffset, tokenNum, processNum, e);
                }
            }
        }
        expertSoFar += expertNum;
    }
}

template <typename DataType, bool IsTranGmmW, bool IsTranMmW, int TILINGKEY_COMM_MODE>
__aicore__ inline void AlltoAllvGmmCoarseGrained<DataType, IsTranGmmW, IsTranMmW, TILINGKEY_COMM_MODE>::Rearrange(uint32_t startExpert, uint32_t expertNum)
{
    if ASCEND_IS_AIV {
        if (GetBlockIdx() != 0) { return; }
    }
    const auto* recvCnt = &tilingData_->aicpuTiling.recvCnt[0];
    // Compute global expert offsets: expertTokenOffset[e] = sum(recvCnt[0..e-1] from all ranks) * axisH1
    uint64_t expertTokenOffset[64] = {0UL};
    for (uint32_t e = 1U; e < expertNumInOneRank_; e++) {
        expertTokenOffset[e] = expertTokenOffset[e - 1];
        for (uint32_t r = 0U; r < rankDim_; r++) {
            expertTokenOffset[e] += static_cast<uint64_t>(recvCnt[e - 1 + r * expertNumInOneRank_]);
        }
        expertTokenOffset[e] *= axisH1_;
    }
    // GM-to-UB-to-GM copy for each (expert, rank) segment
    constexpr uint64_t REARRANGE_UB_BLOCK = 8192UL;
    TQue<QuePosition::VECIN, 1> rearrangeQue;
    tPipe_->InitBuffer(rearrangeQue, 1, REARRANGE_UB_BLOCK * sizeof(DataType));

    for (uint32_t localIdx = 0U; localIdx < expertNum; localIdx++) {
        uint32_t expertIdx = startExpert + localIdx;
        uint64_t dstExpertBase = expertTokenOffset[expertIdx];
        uint64_t dstRankOffset = 0UL;

        for (uint32_t rankIdx = 0U; rankIdx < rankDim_; rankIdx++) {
            uint64_t segTokenCnt = static_cast<uint64_t>(recvCnt[expertIdx + rankIdx * expertNumInOneRank_]);
            uint64_t segSize = segTokenCnt * axisH1_;
            if (segSize == 0UL) {
                dstRankOffset += segSize;
                continue;
            }
            // Source offset: within rank_i's merged block, expert j's data
            uint64_t srcExpertOffset = 0UL;
            for (uint32_t le = 0U; le < localIdx; le++) {
                srcExpertOffset += static_cast<uint64_t>(recvCnt[startExpert + le + rankIdx * expertNumInOneRank_]) * axisH1_;
            }
            uint64_t srcOffset = roundRecvOffsetBase_[rankIdx] + srcExpertOffset;
            uint64_t dstOffset = dstExpertBase + dstRankOffset;

            // Copy segment through UB
            for (uint64_t copyOffset = 0UL; copyOffset < segSize; copyOffset += REARRANGE_UB_BLOCK) {
                uint64_t curCopySize = (segSize - copyOffset > REARRANGE_UB_BLOCK) ?
                    REARRANGE_UB_BLOCK : (segSize - copyOffset);
                LocalTensor<DataType> ubBuf = rearrangeQue.AllocTensor<DataType>();
                DataCopy(ubBuf, permutedGMTensor_[srcOffset + copyOffset], curCopySize);
                rearrangeQue.EnQue(ubBuf);
                ubBuf = rearrangeQue.DeQue<DataType>();
                DataCopy(rearrangeGMTensor_[dstOffset + copyOffset], ubBuf, curCopySize);
                rearrangeQue.FreeTensor<DataType>(ubBuf);
            }
            dstRankOffset += segSize;
        }
    }
    rearrangeQue.FreeAllEvent();
}

template <typename DataType, bool IsTranGmmW, bool IsTranMmW, int TILINGKEY_COMM_MODE>
__aicore__ inline void AlltoAllvGmmCoarseGrained<DataType, IsTranGmmW, IsTranMmW, TILINGKEY_COMM_MODE>::HcclFinalize()
{
    if ASCEND_IS_AIV {
        hccl_.Finalize();
    }
}

} // namespace AscendC
#endif