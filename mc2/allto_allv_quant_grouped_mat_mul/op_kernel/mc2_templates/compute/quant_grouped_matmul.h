/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file quant_grouped_matmul.h
 * \brief
 */

#ifndef MC2_QUANT_GROUPED_MATMUL_H
#define MC2_QUANT_GROUPED_MATMUL_H

#include "kernel_operator.h"

#if __has_include("../../../3rd/grouped_matmul/op_kernel/gqmm_cube_on_the_fly.h")
#include "../../../3rd/grouped_matmul/op_kernel/gqmm_cube_on_the_fly.h"
#else
#include "../../../../3rd/grouped_matmul/op_kernel/gqmm_cube_on_the_fly.h"
#endif

using namespace AscendC;

namespace MC2KernelTemplate {

template <typename TilingDataType, typename GmmTilingDataType, class xType, class wType, class scaleType, class yType,
          CubeFormat wFormat, bool aTrans, bool bTrans, bool isLocal, bool isA2avGmm>
class QuantGroupedMatmul {
public:
    __aicore__ inline void Init(GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR xScaleGM, GM_ADDR weightScaleGM, GM_ADDR yGM,
                                GM_ADDR workspaceGM, const TilingDataType *tilingData,
                                const GmmTilingDataType *gmmTilingData, TILING_TYPE *gmmArrayAddrIn, TPipe *tPipe,
                                bool isA2avGmmFlag)
    {
        if ASCEND_IS_AIV {
            return;
        }
        xGM_ = xGM;
        wGM_ = weightGM;
        xScaleGM_ = xScaleGM;
        weightScaleGM_ = weightScaleGM;
        yGM_ = yGM;
        tilingData_ = tilingData;
        tPipe_ = tPipe;
        workspaceGM_ = workspaceGM;
        gmmTilingData_ = gmmTilingData;
        gmmArrayAddrIn_ = gmmArrayAddrIn;

        expertNumInOneRank_ = tilingData_->taskTilingInfo.e;
        epWorldSize_ = tilingData_->taskTilingInfo.epWorldSize;
        h1_ = tilingData_->taskTilingInfo.H1;
        n1_ = tilingData_->taskTilingInfo.N1;
        bs_ = tilingData_->taskTilingInfo.BS;
        a_ = tilingData_->taskTilingInfo.A;
        gmmParamsCopy_ = gmmTilingData_->gmmQuantParams;

        uint64_t groupListSize = sizeof(int64_t) * expertNumInOneRank_;
        groupListGm_ = workspaceGM_;

        ptrTableBase_ = groupListGm_ + groupListSize;
        cOffsetListGm_ = ptrTableBase_ + 64;  // 指针表后面存放 cOffsetList
        xGlobalBuffer_.SetGlobalBuffer((__gm__ xType *)this->xGM_);
        wGlobalBuffer_.SetGlobalBuffer((__gm__ wType *)this->wGM_);
        yGlobalBuffer_.SetGlobalBuffer((__gm__ yType *)this->yGM_);
        groupListGlobalBuffer_.SetGlobalBuffer((__gm__ int64_t *)groupListGm_);
        xScaleGlobalBuffer_.SetGlobalBuffer((__gm__ scaleType *)xScaleGM);
        wScaleGlobalBuffer_.SetGlobalBuffer((__gm__ scaleType *)weightScaleGM);

        const auto *opCnt =
            isA2avGmmFlag ? &tilingData_->taskTilingInfo.recvCnt[0] : &tilingData_->taskTilingInfo.sendCnt[0];
        for (uint32_t e = 0U; e < expertNumInOneRank_; e++) {
            for (uint32_t i = 0U; i < epWorldSize_; i++) {
                expertTokenNum_[e] += static_cast<uint64_t>(opCnt[e + i * expertNumInOneRank_]);
            }
        }
    }

    __aicore__ inline void Process(uint32_t expertIdx, uint32_t roundIdx = 0)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if (!isLocal && expertTokenNum_[expertIdx] == 0) {
            return;
        }
        this->UpdateAddr(expertIdx, roundIdx);
        __gm__ uint8_t *xAddr = reinterpret_cast<__gm__ uint8_t *>(xGM_);
        __gm__ uint8_t *yAddr = reinterpret_cast<__gm__ uint8_t *>(yGM_);
        __gm__ uint8_t *wAddr = reinterpret_cast<__gm__ uint8_t *>(wGM_);
        GM_ADDR xPtr = BuildPtrTable(reinterpret_cast<GM_ADDR>(xAddr), 0);
        GM_ADDR wPtr = BuildPtrTable(reinterpret_cast<GM_ADDR>(wAddr), 1);
        GM_ADDR scaleBPtr = BuildPtrTable(weightScaleGM_, 2);
        GM_ADDR yPtr = BuildPtrTable(reinterpret_cast<GM_ADDR>(yAddr), 3);

        uint64_t totalTokens = isLocal ? bs_ : expertTokenNum_[expertIdx];
        uint32_t startM = roundIdx * baseA_;
        uint32_t endM = (roundIdx + 1) * baseA_ < static_cast<uint32_t>(totalTokens) ? (roundIdx + 1) * baseA_ : static_cast<uint32_t>(totalTokens);
        uint64_t curRoundTokens = (endM > startM) ? (endM - startM) : 0;

        groupListGlobalBuffer_.SetValue(0, curRoundTokens);
        gmmParamsCopy_.groupNum = 1;
        AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(groupListGlobalBuffer_);
        Mc2GroupedMatmul::Mc2GmmASWKernel<xType, wType, biasType, scaleType, yType, wFormat, aTrans, bTrans>
            gmmASWKernel;
        tPipe_->Reset();
        gmmASWKernel.Init(xPtr, wPtr, nullptr, scaleBPtr, groupListGm_, xScaleGM_, yPtr, workspaceGM_,
                          &gmmParamsCopy_, &gmmTilingData_->mmTilingData, gmmArrayAddrIn_, tPipe_);
        gmmASWKernel.Process();
    }

    __aicore__ inline void ProcessAll(uint32_t roundIdx, uint32_t baseA)
    {
        if ASCEND_IS_AIV {
            return;
        }
        AscendC::printf("\n=== ProcessAll round=%u baseA=%u ===\n", roundIdx, baseA);

        uint32_t activeGroupNum = 0;
        uint64_t cumulativeM = 0;
        uint64_t cOffset = 0;

        AscendC::printf("--- Expert tokens per round ---\n");
        for (uint32_t e = 0U; e < expertNumInOneRank_; e++) {
            if (!isLocal && expertTokenNum_[e] == 0) {
                AscendC::printf("expert%u: skipped (tokenNum=0)\n", e);
                continue;
            }
            uint64_t totalTokens = isLocal ? bs_ : expertTokenNum_[e];
            uint32_t startM = roundIdx * baseA;
            uint32_t endM = (roundIdx + 1) * baseA < static_cast<uint32_t>(totalTokens) ? (roundIdx + 1) * baseA : static_cast<uint32_t>(totalTokens);
            uint64_t curRoundTokens = (endM > startM) ? (endM - startM) : 0;
            cumulativeM += curRoundTokens;
            groupListGlobalBuffer_.SetValue(activeGroupNum, cumulativeM);

            AscendC::printf("expert%u: totalTokens=%lu, startM=%u, endM=%u, curRoundTokens=%lu, cumulativeM=%lu\n",
                   e, totalTokens, startM, endM, curRoundTokens, cumulativeM);

            cOffsetListGlobalBuffer_.SetValue(activeGroupNum, cOffset);
            AscendC::printf("  cOffset[%u] = %lu (output offset in y)\n", activeGroupNum, cOffset);
            cOffset += curRoundTokens * n1_;
            activeGroupNum++;
        }

        if (activeGroupNum == 0) {
            AscendC::printf("ProcessAll: activeGroupNum=0, return\n");
            return;
        }

        gmmParamsCopy_.groupNum = activeGroupNum;
        AscendC::printf("activeGroupNum=%u, groupNum=%u\n", activeGroupNum, gmmParamsCopy_.groupNum);

        xGM_ = (GM_ADDR)xGlobalBuffer_.GetPhyAddr(0);
        yGM_ = (GM_ADDR)yGlobalBuffer_.GetPhyAddr(0);
        wGM_ = (GM_ADDR)wGlobalBuffer_.GetPhyAddr(0);

        AscendC::printf("\n--- GM Buffer Addresses ---\n");
        AscendC::printf("xGM (A矩阵输入)=%p\n", xGM_);
        AscendC::printf("wGM (B矩阵weight)=%p\n", wGM_);
        AscendC::printf("yGM (C矩阵输出)=%p\n", yGM_);
        AscendC::printf("xScaleGM=%p\n", xScaleGM_);
        AscendC::printf("weightScaleGM=%p\n", weightScaleGM_);
        AscendC::printf("workspaceGM=%p\n", workspaceGM_);
        AscendC::printf("groupListGm=%p\n", groupListGm_);

        AscendC::printf("\n--- Workspace Layout ---\n");
        AscendC::printf("workspace[0]: commOut (recvBuffer)\n");
        if (MX_QUANT_MODE) {
            AscendC::printf("workspace[offset1]: scaleCommOut\n");
        }
        AscendC::printf("workspace[offset2]: preCommBuffer\n");
        AscendC::printf("workspace[offset3]: groupList\n");
        AscendC::printf("workspace[offset4]: ptrTable (16 bytes per slot)\n");

        AscendC::printf("\n--- Matrix Dimensions ---\n");
        AscendC::printf("h1_=%lu (K axis), n1_=%lu (N axis), baseA=%u\n", h1_, n1_, baseA);

        __gm__ uint8_t *xAddr = reinterpret_cast<__gm__ uint8_t *>(xGM_);
        __gm__ uint8_t *yAddr = reinterpret_cast<__gm__ uint8_t *>(yGM_);
        __gm__ uint8_t *wAddr = reinterpret_cast<__gm__ uint8_t *>(wGM_);
        GM_ADDR xPtr = BuildPtrTable(reinterpret_cast<GM_ADDR>(xAddr), 0);
        GM_ADDR wPtr = BuildPtrTable(reinterpret_cast<GM_ADDR>(wAddr), 1);
        GM_ADDR scaleBPtr = BuildPtrTable(weightScaleGM_, 2);
        GM_ADDR yPtr = BuildPtrTable(reinterpret_cast<GM_ADDR>(yAddr), 3);

        AscendC::printf("\n--- PtrTable ---\n");
        AscendC::printf("xPtr (slot0)=%p -> points to xAddr=%p\n", xPtr, xAddr);
        AscendC::printf("wPtr (slot1)=%p -> points to wAddr=%p\n", wPtr, wAddr);
        AscendC::printf("scaleBPtr (slot2)=%p -> points to weightScaleGM=%p\n", scaleBPtr, weightScaleGM_);
        AscendC::printf("yPtr (slot3)=%p -> points to yAddr=%p\n", yPtr, yAddr);

        AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(groupListGlobalBuffer_);
        AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(cOffsetListGlobalBuffer_);

        AscendC::printf("\n--- GMM Kernel Params ---\n");
        AscendC::printf("groupNum=%u, h1=%lu, n1=%lu\n", gmmParamsCopy_.groupNum, h1_, n1_);

        AscendC::printf("\nProcessAll: GMM Process simulated (not actually called)\n");
        // Mc2GroupedMatmul::Mc2GmmASWKernel<...> gmmASWKernel;
        // tPipe_->Reset();
        // gmmASWKernel.Init(...);
        // gmmASWKernel.Process();
    }

    __aicore__ inline void End()
    {
        if ASCEND_IS_AIV {
            return;
        }
    }

protected:
    __aicore__ inline void UpdateAddr(uint32_t expertIdx, uint32_t roundIdx = 0)
    {
        uint64_t roundOffset = roundIdx * baseA_;

        xGM_ = (GM_ADDR)xGlobalBuffer_.GetPhyAddr((expertTokenOffset_ + roundOffset) * h1_);
        wGM_ = (GM_ADDR)wGlobalBuffer_.GetPhyAddr(expertIdx * h1_ * n1_);
        yGM_ = (GM_ADDR)yGlobalBuffer_.GetPhyAddr((expertTokenOffset_ + roundOffset) * n1_);

        if constexpr (Mc2QuantUtils::IsMxType<scaleType>()) {
            uint64_t scaleK = Mc2QuantUtils::MXFP_MULTI_BASE_SIZE *
                              Mc2QuantUtils::CeilDiv(h1_, static_cast<uint64_t>(Mc2QuantUtils::MXFP_DIVISOR_SIZE));
            xScaleGM_ = (GM_ADDR)xScaleGlobalBuffer_.GetPhyAddr((expertTokenOffset_ + roundOffset) * scaleK);
            weightScaleGM_ = (GM_ADDR)wScaleGlobalBuffer_.GetPhyAddr(expertIdx * n1_ * scaleK);
        }

        expertTokenOffset_ += expertTokenNum_[expertIdx];
    }

    __aicore__ inline GM_ADDR BuildPtrTable(GM_ADDR dataAddr, uint32_t slotIdx)
    {
        __gm__ uint64_t *slot =
            reinterpret_cast<__gm__ uint64_t *>(reinterpret_cast<__gm__ uint8_t *>(ptrTableBase_) + slotIdx * 16);
        slot[0] = sizeof(uint64_t);
        slot[1] = reinterpret_cast<uint64_t>(dataAddr);
        return reinterpret_cast<GM_ADDR>(slot);
    }

    __aicore__ inline uint64_t CalcGroupListOffset(bool isPermuteOut, uint64_t permuteOutSize,
                                                   uint64_t permuteXScaleSize)
    {
        uint64_t offset = 0;
        if (!isPermuteOut) {
            offset += permuteOutSize;
        }
        if constexpr (Mc2QuantUtils::IsMxType<scaleType>()) {
            offset += permuteXScaleSize;
        }
        return offset;
    }

private:
    using biasType = float;

    GM_ADDR xGM_;
    GM_ADDR wGM_;
    GM_ADDR xScaleGM_;
    GM_ADDR weightScaleGM_;
    GM_ADDR yGM_;
    GM_ADDR groupListGm_;
    GM_ADDR workspaceGM_;
    GM_ADDR ptrTableBase_ = nullptr;
    GM_ADDR cOffsetListGm_ = nullptr;
    GlobalTensor<xType> xGlobalBuffer_;
    GlobalTensor<wType> wGlobalBuffer_;
    GlobalTensor<scaleType> xScaleGlobalBuffer_;
    GlobalTensor<scaleType> wScaleGlobalBuffer_;
    GlobalTensor<yType> yGlobalBuffer_;
    GlobalTensor<int64_t> groupListGlobalBuffer_;
    GlobalTensor<uint64_t> cOffsetListGlobalBuffer_;
    const TilingDataType *tilingData_;
    TPipe *tPipe_;
    uint64_t expertTokenNum_[32] = {0};
    uint64_t expertTokenOffset_ = 0;
    uint64_t expertNumInOneRank_ = 0;
    uint64_t epWorldSize_ = 0;
    uint64_t h1_;
    uint64_t n1_;
    uint64_t bs_;
    uint64_t a_;
    uint32_t totalRound_ = 0;
    const GmmTilingDataType *gmmTilingData_;
    GMMQuantParams gmmParamsCopy_;
    TILING_TYPE *gmmArrayAddrIn_;
};
} // namespace MC2KernelTemplate
#endif