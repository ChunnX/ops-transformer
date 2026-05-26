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
 * \file grouped_mat_mul_allto_allv.h
 * \brief
 */
#ifndef GROUPED_MAT_MUL_ALLTO_ALLV_H
#define GROUPED_MAT_MUL_ALLTO_ALLV_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/hccl/hccl.h"
#include "lib/matmul_intf.h"
#if __has_include("../../allto_allv_grouped_mat_mul/op_kernel/allto_allv_gmm.h")
#include "../../allto_allv_grouped_mat_mul/op_kernel/allto_allv_gmm.h"
#else
#include "../allto_allv_grouped_mat_mul/allto_allv_gmm.h"
#endif
#include "grouped_mat_mul_allto_allv_tiling.h"

namespace AscendC {
using namespace ALLTO_ALLV_GMM;
template <typename GMMATAV>
class GroupedMatmulAlltoAllv
{
public:
    __aicore__ inline GroupedMatmulAlltoAllv()
    {}
    __aicore__ inline void Init(
        GM_ADDR gmmxGM, GM_ADDR gmmweightGM, GM_ADDR sendCountsTensorGM, GM_ADDR recvCountsTensorGM, GM_ADDR mmGM,
        GM_ADDR mmweightGM, GM_ADDR yGM, GM_ADDR mmyGM, GM_ADDR workspaceGM, GM_ADDR contextGM,
        const GroupedMatMulAlltoAllvTilingData* tilingData, __gm__ void* hcclInitTiling, __gm__ void* alltoAllvCcTiling,
        TPipe* tPipe);
    __aicore__ inline void Process();
    using X_T = typename GMMATAV::xType;
    static constexpr bool NEED_MM = GMMATAV::isOptionalMm;
    static constexpr bool NEED_GMMW_TRANS = GMMATAV::isGmmWeightTrans;
    static constexpr bool NEED_MMW_TRANS = GMMATAV::isOptWeightTrans;

    using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, X_T, false>;
    using gmmBType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, X_T, NEED_GMMW_TRANS>;
    using mmBType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, X_T, NEED_MMW_TRANS>;
    using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, X_T, false>;
    using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, X_T>;
    using gmmType = ALLTO_ALLV_GMM::MMImplType<aType, gmmBType, cType, biasType, CFG_MDL>;
    using sharedmmType = ALLTO_ALLV_GMM::MMImplType<aType, mmBType, cType, biasType, CFG_MDL>;

private:
    __aicore__ inline void HcclFinalize();
    __aicore__ inline void GmmProcessAlltoallv();
    __aicore__ inline void PermuteGroup(uint64_t groupId, uint64_t groupOffset);
    __aicore__ inline void ShareMatMulCompute();

    GM_ADDR gmmxGM_;
    GM_ADDR gmmweightGM_;
    GM_ADDR mmxGM_;
    GM_ADDR mmweightGM_;
    GM_ADDR yGM_;
    GM_ADDR mmyGM_;
    GM_ADDR allGatherOutGM_;
    GM_ADDR gmmOutGM_;
    GM_ADDR workspaceGM_;
    const GroupedMatMulAlltoAllvTilingData* tilingData_ = nullptr;
    TCubeTiling matmulTiling_;

    TPipe* tPipe_;
    uint64_t rankId_{0};
    uint64_t rankDim_{8};

    uint64_t expertNumInOneRank_ = 0U; // 单卡专家数
    uint64_t expertNumAll_ = 0U;       // 通信域内专家数

    uint64_t axisBsK_ = 0U;
    uint64_t axisH_ = 0U;
    uint64_t axisA_ = 0U;
    uint64_t axisN1_ = 0U;
    uint64_t workSpaceOffset = 0U;
    uint64_t expertsPerComm_ = 1U;
    uint64_t commGroupNum_ = 1U;
    GM_ADDR permuteBufferGM_ = nullptr;
    TBuf<TPosition::VECCALC> permuteUbBuf_;

    // alltoallv 流程数据结构
    static constexpr uint64_t MAX_EP_RANK_SIZE = 128U;
    static constexpr uint64_t TOTAL_UBSIZE = static_cast<uint64_t>(190U * 1024U / 2U);
    static constexpr uint64_t MAX_AIV_NUM = 48U;
    static constexpr uint64_t MAX_HANDLE_ID_NUM = 64U;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl_;
#else
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
#endif
    HcclHandle allGatherHandleId_{INVALID_HANDLE_ID};
    HcclHandle alltoAllvHandleId_[MAX_HANDLE_ID_NUM] = {INVALID_HANDLE_ID};
    HcclDataType hcclDataType_;

    GlobalTensor<int64_t> allGatherOutGMTensor_;
    GlobalTensor<X_T> gmmXGMTensor_;
    GlobalTensor<X_T> gmmWeightGMTensor_;
    GlobalTensor<X_T> gmmOutGMTensor_;
    GlobalTensor<X_T> yOutGMTensor_;
    GlobalTensor<X_T> mmXGMTensor_;
    GlobalTensor<X_T> mmWeightGMTensor_;
    GlobalTensor<X_T> mmOutGMTensor_;
    typename gmmType::MT gmm;
    typename sharedmmType::MT sharedmm;
};
template <typename GMMATAV>
__aicore__ inline void GroupedMatmulAlltoAllv<GMMATAV>::Init(
    GM_ADDR gmmxGM, GM_ADDR gmmweightGM, GM_ADDR sendCountsTensorGM, GM_ADDR recvCountsTensorGM, GM_ADDR mmGM,
    GM_ADDR mmweightGM, GM_ADDR yGM, GM_ADDR mmyGM, GM_ADDR workspaceGM, GM_ADDR contextGM,
    const GroupedMatMulAlltoAllvTilingData* tilingData, __gm__ void* hcclInitTiling, __gm__ void* alltoAllvCcTiling,
    TPipe* tPipe)
{
    gmmxGM_ = gmmxGM;
    gmmweightGM_ = gmmweightGM;
    mmxGM_ = mmGM;
    mmweightGM_ = mmweightGM;
    yGM_ = yGM;
    mmyGM_ = mmyGM;
    gmmOutGM_ = workspaceGM;
    workspaceGM_ = workspaceGM;
    tilingData_ = tilingData;
    tPipe_ = tPipe;

    axisBsK_ = tilingData_->commonTilingInfo.BsK;
    axisH_ = tilingData_->commonTilingInfo.H;
    axisA_ = tilingData_->commonTilingInfo.A;
    axisN1_ = tilingData_->commonTilingInfo.N1;

    const void *hcclInitTilingV2 = &(tilingData_->hcclInitTiling);
    uint64_t hcclCcTilingOffset = offsetof(GroupedMatMulAlltoAllvTilingData, alltoAllvCcTiling);
    hccl_.InitV2(contextGM, hcclInitTilingV2);
    hccl_.SetCcTilingV2(hcclCcTilingOffset);
    rankId_ = hccl_.GetRankId();
    rankDim_ = hccl_.GetRankDim();

    expertNumInOneRank_ = tilingData_->commonTilingInfo.E_ep;
    expertNumAll_ = expertNumInOneRank_ * rankDim_;
    expertsPerComm_ = tilingData_->commonTilingInfo.expertsPerComm;
    commGroupNum_ = tilingData_->commonTilingInfo.commGroupNum;
    matmulTiling_ = tilingData_->matmulTiling;

    uint64_t gmmOutSize = axisA_ * axisN1_ * sizeof(X_T);
    permuteBufferGM_ = reinterpret_cast<GM_ADDR>(
        reinterpret_cast<__gm__ uint8_t*>(workspaceGM_) + gmmOutSize);
    tPipe_->InitBuffer(permuteUbBuf_, TOTAL_UBSIZE * sizeof(X_T));

    if (AscendC::IsSameType<X_T, bfloat16_t>::value) {
        hcclDataType_ = HcclDataType::HCCL_DATA_TYPE_BFP16;
    } else {
        hcclDataType_ = HcclDataType::HCCL_DATA_TYPE_FP16;
    }

    gmmXGMTensor_.SetGlobalBuffer((__gm__ X_T*)this->gmmxGM_);
    gmmWeightGMTensor_.SetGlobalBuffer((__gm__ X_T*)this->gmmweightGM_);
    gmmOutGMTensor_.SetGlobalBuffer((__gm__ X_T*)workspaceGM_);

    mmXGMTensor_.SetGlobalBuffer((__gm__ X_T*)this->mmxGM_);
    mmWeightGMTensor_.SetGlobalBuffer((__gm__ X_T*)this->mmweightGM_);
    mmOutGMTensor_.SetGlobalBuffer((__gm__ X_T*)this->mmyGM_);
    yOutGMTensor_.SetGlobalBuffer((__gm__ X_T*)this->yGM_);
}

template <typename GMMATAV>
__aicore__ inline void GroupedMatmulAlltoAllv<GMMATAV>::Process()
{
    // gmm + alltoall，做完一块gmm下发一块alltoall
    GmmProcessAlltoallv();
    if ASCEND_IS_AIV {
        if (GetBlockIdx() == 0) {
            for (uint64_t g = 0; g < commGroupNum_; g++) {
                hccl_.Wait(alltoAllvHandleId_[g]);
            }
        }
    }
    SyncAll<false>();
    HcclFinalize();
}

template <typename GMMATAV>
__aicore__ inline void GroupedMatmulAlltoAllv<GMMATAV>::PermuteGroup(uint64_t groupId, uint64_t groupOffset)
{
    if ASCEND_IS_AIV {
        auto* sendCnt = &tilingData_->aicpuTilingInfo.sendCnt[0];
        GlobalTensor<X_T> srcGm;
        GlobalTensor<X_T> dstGm;
        srcGm.SetGlobalBuffer((__gm__ X_T*)gmmOutGM_);
        dstGm.SetGlobalBuffer((__gm__ X_T*)permuteBufferGM_);

        LocalTensor<X_T> ubTensor = permuteUbBuf_.Get<X_T>();
        const uint64_t maxBatchSize = TOTAL_UBSIZE;

        uint64_t blockIdxLocal = GetBlockIdx();
        uint64_t blockNumLocal = GetBlockNum();

        uint64_t gStart = groupId * expertsPerComm_;
        uint64_t gEnd = (groupId + 1) * expertsPerComm_;
        if (gEnd > expertNumInOneRank_) {
            gEnd = expertNumInOneRank_;
        }

        // Each AIV block handles a subset of ranks via round-robin
        for (uint64_t r = blockIdxLocal; r < rankDim_; r += blockNumLocal) {
            uint64_t dstRankOffset = groupOffset;
            for (uint64_t prevR = 0; prevR < r; ++prevR) {
                for (uint64_t e = gStart; e < gEnd; ++e) {
                    dstRankOffset += static_cast<uint64_t>(sendCnt[e + prevR * expertNumInOneRank_]) * axisN1_;
                }
            }

            for (uint64_t e = gStart; e < gEnd; ++e) {
                uint64_t srcExpertOffset = 0;
                for (uint64_t prevE = 0; prevE < e; ++prevE) {
                    for (uint64_t j = 0; j < rankDim_; ++j) {
                        srcExpertOffset += static_cast<uint64_t>(sendCnt[prevE + j * expertNumInOneRank_]) * axisN1_;
                    }
                }
                for (uint64_t prevR = 0; prevR < r; ++prevR) {
                    srcExpertOffset += static_cast<uint64_t>(sendCnt[e + prevR * expertNumInOneRank_]) * axisN1_;
                }

                uint64_t tokenCnt = static_cast<uint64_t>(sendCnt[e + r * expertNumInOneRank_]);
                uint64_t dataSize = tokenCnt * axisN1_;
                if (dataSize == 0) {
                    continue;
                }

                uint64_t remaining = dataSize;
                uint64_t srcOffset = srcExpertOffset;
                uint64_t dstOffset = dstRankOffset;
                while (remaining > 0) {
                    uint64_t batchSize = (remaining > maxBatchSize) ? maxBatchSize : remaining;
                    DataCopy(ubTensor, srcGm[srcOffset], static_cast<uint32_t>(batchSize));
                    PipeBarrier<PIPE_ALL>();
                    DataCopy(dstGm[dstOffset], ubTensor, static_cast<uint32_t>(batchSize));
                    PipeBarrier<PIPE_ALL>();
                    remaining -= batchSize;
                    srcOffset += batchSize;
                    dstOffset += batchSize;
                }
                dstRankOffset += dataSize;
            }
        }
    }
}

template <typename GMMATAV>
__aicore__ inline void GroupedMatmulAlltoAllv<GMMATAV>::ShareMatMulCompute()
{
    if (NEED_MM) {
        if ASCEND_IS_AIC {
            uint64_t aSharedOffset = 0U;
            uint64_t bSharedOffset = 0U;
            uint64_t cSharedOffset = 0U;

            sharedmm.Init(&(tilingData_->sharedExpMatmulTiling));
            GMMCompute<sharedmmType> computeOp2(sharedmm);
            computeOp2.Init(mmxGM_, mmweightGM_, mmyGM_);
            GMMProcess<decltype(computeOp2)> mmOp(computeOp2);
            mmOp.Init(tilingData_->sharedExpMatmulTiling.baseM, tilingData_->sharedExpMatmulTiling.baseN, tilingData_->commonTilingInfo.aicCoreNum);

            uint64_t mmInOffset[1] = {0};
            uint64_t mmOutOffset[1] = {0};
            uint64_t mmWeightOffset[1] = {0};
            uint32_t tokenNum[1] = {static_cast<uint32_t>(tilingData_->commonTilingInfo.Bs)};
            mmOp.Process(
                this->rankId_, tilingData_->commonTilingInfo.sharedMatmulH, tilingData_->commonTilingInfo.N2,
                mmInOffset, mmOutOffset, tokenNum, 1, 0);
        }
    }
}

template <typename GMMATAV>
__aicore__ inline void GroupedMatmulAlltoAllv<GMMATAV>::GmmProcessAlltoallv()
{
    gmm.Init(&(tilingData_->matmulTiling));
    GMMCompute<gmmType> computeOp(gmm);
    computeOp.Init(gmmxGM_, gmmweightGM_, gmmOutGM_);
    GMMProcess<decltype(computeOp)> gmmOp(computeOp);
    gmmOp.Init(tilingData_->matmulTiling.baseM, tilingData_->matmulTiling.baseN, tilingData_->commonTilingInfo.aicCoreNum);
    auto* sendCnt = &tilingData_->aicpuTilingInfo.sendCnt[0];

    uint64_t mmInOffset[2] = {0};
    uint64_t mmOutOffset[2] = {0};
    uint64_t mmWeightOffset[2] = {0};
    uint32_t tokenNum[2] = {0};
    uint32_t processNum = 1;

    uint64_t expertoffset[MAX_HANDLE_ID_NUM] = {0UL};
    uint64_t gmmTokennum[MAX_HANDLE_ID_NUM] = {0UL};
    for (uint64_t i = 0UL; i < expertNumInOneRank_; i++) {
        uint64_t cur_gmmTokenNum = 0;
        for (uint64_t j = 0; j < rankDim_; j++) {
            cur_gmmTokenNum += static_cast<uint64_t>(sendCnt[i + j * expertNumInOneRank_]);
        }
        gmmTokennum[i] = cur_gmmTokenNum; // 本专家的tokenNum
    }
    for (uint64_t i = 0UL; i < expertNumInOneRank_; i++) {
        if (i >= 1UL) {
            expertoffset[i] = expertoffset[i - 1] + gmmTokennum[i - 1] * axisH_;
        }
    }
    for (uint64_t g = 0UL; g < commGroupNum_; g++) {
        uint64_t gStart = g * expertsPerComm_;
        uint64_t gEnd = (g + 1) * expertsPerComm_;
        if (gEnd > expertNumInOneRank_) {
            gEnd = expertNumInOneRank_;
        }

        // AIC: compute GMM for all experts in the group
        if ASCEND_IS_AIC {
            for (uint64_t i = gStart; i < gEnd; i++) {
                mmInOffset[0] = expertoffset[i];
                mmOutOffset[0] = expertoffset[i] / axisH_ * axisN1_;
                mmWeightOffset[0] = i * axisH_ * axisN1_;
                tokenNum[0] = gmmTokennum[i];
                if (tokenNum[0] != 0) {
                    gmmOp.Process(this->rankId_, axisH_, axisN1_, mmInOffset, mmOutOffset, tokenNum, processNum, i);
                }
            }
            PipeBarrier<PIPE_ALL>();
        }
        SyncAll<false>();

        // Compute group offset in permuteBuffer for this group
        uint64_t groupOffset = 0UL;
        for (uint64_t prevG = 0UL; prevG < g; ++prevG) {
            uint64_t prevGStart = prevG * expertsPerComm_;
            uint64_t prevGEnd = (prevG + 1) * expertsPerComm_;
            if (prevGEnd > expertNumInOneRank_) {
                prevGEnd = expertNumInOneRank_;
            }
            for (uint64_t r = 0UL; r < rankDim_; ++r) {
                for (uint64_t e = prevGStart; e < prevGEnd; ++e) {
                    groupOffset += static_cast<uint64_t>(sendCnt[e + r * expertNumInOneRank_]) * axisN1_;
                }
            }
        }

        // AIV: permute (always needed: GMM output is expert-major, alltoallv expects rank-major)
        PermuteGroup(g, groupOffset);
        SyncAll<false>();

        // AIV block0: launch alltoallv for the group (using <true> to avoid stale stack array issue)
        if ASCEND_IS_AIV {
            if (GetBlockIdx() == 0) {
                const auto* a2aSendCnt = &tilingData_->aicpuTilingInfo.sendCnt[0];
                const auto* a2aRecvCnt = &tilingData_->aicpuTilingInfo.recvCnt[0];

                uint64_t alltoAllvSendCnt[MAX_EP_RANK_SIZE] = {0UL};
                uint64_t alltoAllvSendOffset[MAX_EP_RANK_SIZE] = {0UL};
                uint64_t alltoAllvRecvCnt[MAX_EP_RANK_SIZE] = {0UL};
                uint64_t alltoAllvRecvOffset[MAX_EP_RANK_SIZE] = {0UL};

                // sendCnt per rank: sum across all experts in the group
                for (uint64_t r = 0UL; r < rankDim_; r++) {
                    for (uint64_t e = gStart; e < gEnd; e++) {
                        alltoAllvSendCnt[r] += static_cast<uint64_t>(a2aSendCnt[e + r * expertNumInOneRank_]);
                    }
                    alltoAllvSendCnt[r] *= axisN1_;
                }

                // sendOffset: permuteBufferGM_ is always rank-major after PermuteGroup
                alltoAllvSendOffset[0] = groupOffset;
                for (uint64_t r = 1UL; r < rankDim_; r++) {
                    alltoAllvSendOffset[r] = alltoAllvSendOffset[r - 1] + alltoAllvSendCnt[r - 1];
                }

                // recvCnt per rank: sum across all experts in the group
                for (uint64_t r = 0UL; r < rankDim_; r++) {
                    for (uint64_t e = gStart; e < gEnd; e++) {
                        alltoAllvRecvCnt[r] += static_cast<uint64_t>(a2aRecvCnt[e + r * expertNumInOneRank_]);
                    }
                    alltoAllvRecvCnt[r] *= axisN1_;
                }

                // recvOffset: y is rank-major (all experts from rank 0, then rank 1, etc.)
                for (uint64_t r = 0UL; r < rankDim_; r++) {
                    uint64_t offset = 0UL;
                    for (uint64_t prevR = 0; prevR < r; ++prevR) {
                        for (uint64_t e = 0; e < expertNumInOneRank_; ++e) {
                            offset += static_cast<uint64_t>(a2aRecvCnt[e + prevR * expertNumInOneRank_]) * axisN1_;
                        }
                    }
                    for (uint64_t e = 0; e < gStart; ++e) {
                        offset += static_cast<uint64_t>(a2aRecvCnt[e + r * expertNumInOneRank_]) * axisN1_;
                    }
                    alltoAllvRecvOffset[r] = offset;
                }

                GM_ADDR sendBuffer = permuteBufferGM_;
                alltoAllvHandleId_[g] = hccl_.AlltoAllV<true>(
                    (__gm__ uint8_t*)sendBuffer, alltoAllvSendCnt, alltoAllvSendOffset, hcclDataType_,
                    (__gm__ uint8_t*)yOutGMTensor_.GetPhyAddr(), alltoAllvRecvCnt, alltoAllvRecvOffset, hcclDataType_);
            }
        }
    }

    ShareMatMulCompute();
    SyncAll<false>();
}

template <typename GMMATAV>
__aicore__ inline void GroupedMatmulAlltoAllv<GMMATAV>::HcclFinalize()
{
    if ASCEND_IS_AIV {
        hccl_.Finalize();
    }
}
} // namespace AscendC
#endif
