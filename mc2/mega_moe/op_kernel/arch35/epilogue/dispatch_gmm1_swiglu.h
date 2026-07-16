/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_DISPATCH_GMM1_SWIGLU_H
#define MEGA_MOE_DISPATCH_GMM1_SWIGLU_H

#include "../aiv_comm/dispatch.h"

namespace MegaMoeImpl {

__aicore__ inline Gmm1ExpertLoopState CreateGmm1ExpertLoopState(
    const DispatchGmm1SwigluContext &context)
{
    Gmm1ProblemShape shape;
    Get<N_VALUE>(shape) = context.gmm1OutputDim;
    Get<K_VALUE>(shape) = context.tokenHiddenDim;
    Gmm1BlockOffset offset{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    return Gmm1ExpertLoopState{shape, offset, 0};
}

template <typename ActivationType, typename WeightType, typename SwigluOutType, typename BlockEpilogue>
__aicore__ inline bool UpdateGmm1GroupParams(
    const DispatchGmm1SwigluContext &context,
    const DispatchGmm1SwigluArgs<BlockEpilogue> &args,
    DispatchGmm1SwigluScratch<ActivationType> &scratch,
    Gmm1ExpertLoopState &state, uint32_t expertIdx, uint64_t sendCnt)
{
    constexpr uint32_t aElemsPerByte = Std::IsSame<ActivationType, uint8_t>::value ? 2U : 1U;
    constexpr uint32_t bElemsPerByte = Std::IsSame<WeightType, fp4x2_e2m1_t>::value ? 2U : 1U;
    constexpr uint32_t cElemsPerByte = Std::IsSame<SwigluOutType, fp4x2_e2m1_t>::value ? 2U : 1U;

    if (expertIdx != 0) {
        uint64_t m = Get<M_VALUE>(state.problemShape);
        uint64_t n = Get<N_VALUE>(state.problemShape);
        uint64_t k = Get<K_VALUE>(state.problemShape);
        state.expertBeforeCnt += m;
        Get<IDX_A_OFFSET>(state.baseOffset) += m * k / aElemsPerByte;
        Get<IDX_B_OFFSET>(state.baseOffset) += n * k / bElemsPerByte;
        auto scaleK = Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_A_SCALE_OFFSET>(state.baseOffset) += m * scaleK;
        Get<IDX_B_SCALE_OFFSET>(state.baseOffset) += n * scaleK;
        Get<IDX_C_OFFSET>(state.baseOffset) += m * n / SWIGLU_N_HALF / cElemsPerByte;
        Get<IDX_C_SCALE_OFFSET>(state.baseOffset) +=
            m * Ops::Base::CeilDiv(n / SWIGLU_N_HALF,
                static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_FLAG_OFFSET>(state.baseOffset) += 1;
        Get<IDX_B2_OFFSET>(state.baseOffset) += k * n / SWIGLU_N_HALF / bElemsPerByte;
        Get<IDX_B2_SCALE_OFFSET>(state.baseOffset) +=
            k * Ops::Base::CeilDiv(n / SWIGLU_N_HALF,
                static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_Y2_OFFSET>(state.baseOffset) += m * k;
        Get<IDX_M_OFFSET>(state.baseOffset) += m;
        Get<IDX_GMM1_OFFSET>(state.baseOffset) += m * n;
        Get<IDX_GMM2_OFFSET>(state.baseOffset) += m * k;
    }

    if (GetSubBlockIdx() == 0) {
        __gm__ int32_t *sendCntFlag = reinterpret_cast<__gm__ int32_t *>(
            args.sendCntFlagPtr) +
            static_cast<uint64_t>(expertIdx) * context.countWorkspace.blockNum * INT_CACHELINE +
            static_cast<uint64_t>(context.countWorkspace.blockIdx) * INT_CACHELINE;
        while (AscendC::ReadGmByPassDCache(sendCntFlag) == 0) {
            int64_t startCycle = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - startCycle < 100) {
            }
        }
        uint64_t offset = expertIdx * INT32_PER_256B * context.countWorkspace.blockNum +
            INT32_PER_256B * context.countWorkspace.blockIdx;
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
            scratch.expertRevNumsGlobalTensor[offset]);
        Get<M_VALUE>(state.problemShape) = scratch.expertRevNumsGlobalTensor.GetValue(offset);
    } else {
        Get<M_VALUE>(state.problemShape) = sendCnt;
    }
    return Get<M_VALUE>(state.problemShape) != 0;
}

template <typename ActivationType, typename WeightType, typename SwigluOutType,
    typename QuantScaleType, bool EnableA8W4, typename BlockEpilogue>
__aicore__ inline void UpdateGmm1GlobalBuffer(
    const DispatchGmm1SwigluContext &context,
    const DispatchGmm1SwigluArgs<BlockEpilogue> &args,
    GMMAddrInfo &gmmAddrInfo, const Gmm1ExpertLoopState &state)
{
    if constexpr (EnableA8W4) {
        gmmAddrInfo.gmm1OutGlobal = args.gmm1MmadResPtr +
            Get<IDX_GMM1_OFFSET>(state.baseOffset) * sizeof(bfloat16_t);
    }
    gmmAddrInfo.aGlobal = args.dispatchRevDataPtr +
        Get<IDX_A_OFFSET>(state.baseOffset) * sizeof(ActivationType);
    gmmAddrInfo.aScaleGlobal = args.dispatchRevScalePtr +
        Get<IDX_A_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleType);
    gmmAddrInfo.bGlobal = args.weight1Ptr +
        Get<IDX_B_OFFSET>(state.baseOffset) * sizeof(ActivationType);
    gmmAddrInfo.bScaleGlobal = args.weight1ScalePtr +
        Get<IDX_B_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleType);
    if constexpr (g_coreType == AIV) {
        AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
            Get<IDX_C_OFFSET>(state.baseOffset), Get<IDX_C_SCALE_OFFSET>(state.baseOffset),
            Get<IDX_FLAG_OFFSET>(state.baseOffset), 0L, 0L, 0L};
        args.epilogueOp->UpdateGlobalAddr(vecBaseOffset);
    }
    gmmAddrInfo.swigluToGmm2Flag = reinterpret_cast<__gm__ int32_t *>(
        args.swigluToGmm2FlagPtr) +
        Get<IDX_FLAG_OFFSET>(state.baseOffset) * INT_CACHELINE;
    gmmAddrInfo.dispatchToGmm1Flag = reinterpret_cast<__gm__ int32_t *>(
        args.dispatchToGmm1FlagPtr) +
        Get<IDX_FLAG_OFFSET>(state.baseOffset) * context.dispatchFlagSlotsPerExpert;
}

template <typename QuantOutType, typename ActivationType, typename WeightType,
    typename SwigluOutType, typename QuantScaleType, bool EnableA8W4, typename BlockEpilogue>
__aicore__ inline void RunGmm1SwigluStageOneCore(
    const DispatchGmm1SwigluContext &context,
    const DispatchGmm1SwigluArgs<BlockEpilogue> &args,
    const GMMAddrInfo &gmmAddrInfo, const Gmm1ExpertLoopState &state)
{
    if constexpr (EnableA8W4) {
        MegaMoeImpl::GroupMatmulSwigluQuantA8W4<
            QuantOutType, WeightType, bfloat16_t, QuantScaleType, QuantScaleType>(
            *args.epilogueOp, *args.gmmParams, state.problemShape, gmmAddrInfo,
            *args.startBlockIdx, *args.vecSetSyncCom, context.threadBlock);
    } else if (context.groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ ||
        context.groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ) {
        MegaMoeImpl::GroupMatmulSwigluQuant<
            QuantOutType, SwigluOutType, QuantOutType, bfloat16_t, QuantScaleType, QuantScaleType, true>(
            *args.epilogueOp, *args.gmmParams, state.problemShape, gmmAddrInfo,
            *args.startBlockIdx, *args.vecSetSyncCom, context.threadBlock);
    } else {
        MegaMoeImpl::GroupMatmulSwigluQuant<
            QuantOutType, SwigluOutType, QuantOutType, bfloat16_t, QuantScaleType, QuantScaleType>(
            *args.epilogueOp, *args.gmmParams, state.problemShape, gmmAddrInfo,
            *args.startBlockIdx, *args.vecSetSyncCom, context.threadBlock);
    }
}

// Every physical core in one MIX_AIC_1_2 block calls this function with the
// same logical threadBlock and countWorkspace contexts. The function preserves
// dispatch(e + 1) prefetch and GMM1/Swiglu(e) consumption order.
template <typename QuantOutType, typename ActivationType, typename WeightType,
    typename SwigluOutType, typename QuantScaleType, bool EnableA8W4, typename BlockEpilogue>
__aicore__ inline void DispatchGmm1SwigluOneCore(
    const DispatchGmm1SwigluContext &context,
    const DispatchGmm1SwigluArgs<BlockEpilogue> &args,
    DispatchGmm1SwigluScratch<ActivationType> &scratch)
{
    if (context.threadBlock.threadBlockNum == 0U ||
        context.threadBlock.threadBlockIdx >= context.threadBlock.threadBlockNum ||
        context.countWorkspace.blockNum == 0U ||
        context.countWorkspace.blockIdx >= context.countWorkspace.blockNum ||
        context.expertPerRank == 0U || context.blockNumPerRank == 0U) {
        return;
    }
    InitDispatchGmm1SwigluBuffers<ActivationType, EnableA8W4>(context, args, scratch);
    Gmm1ExpertLoopState dispatchState = CreateGmm1ExpertLoopState(context);
    Gmm1ExpertLoopState gmm1State = CreateGmm1ExpertLoopState(context);
    GMMAddrInfo dispatchAddrInfo;
    GMMAddrInfo gmm1AddrInfo;
    uint64_t currentSendCnt = 0;
    uint64_t nextSendCnt = 0;

    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 1) {
            SendCntCalOneCore<ActivationType, EnableA8W4>(context, args, scratch, 0, nextSendCnt);
            if (UpdateGmm1GroupParams<ActivationType, WeightType, SwigluOutType>(
                context, args, scratch, dispatchState, 0, nextSendCnt)) {
                UpdateGmm1GlobalBuffer<ActivationType, WeightType, SwigluOutType,
                    QuantScaleType, EnableA8W4>(context, args, dispatchAddrInfo, dispatchState);
                DispatchTokensOneCore<ActivationType, QuantScaleType, EnableA8W4>(
                    context, args, scratch, dispatchAddrInfo, 0);
            }
        }
    }

    for (uint32_t expertIdx = 0; expertIdx < context.expertPerRank; ++expertIdx) {
        currentSendCnt = nextSendCnt;
        if constexpr (g_coreType == AIV) {
            if (GetSubBlockIdx() == 1 && expertIdx + 1 < context.expertPerRank) {
                SendCntCalOneCore<ActivationType, EnableA8W4>(
                    context, args, scratch, expertIdx + 1, nextSendCnt);
                if (UpdateGmm1GroupParams<ActivationType, WeightType, SwigluOutType>(
                    context, args, scratch, dispatchState, expertIdx + 1, nextSendCnt)) {
                    UpdateGmm1GlobalBuffer<ActivationType, WeightType, SwigluOutType,
                        QuantScaleType, EnableA8W4>(context, args, dispatchAddrInfo, dispatchState);
                    DispatchTokensOneCore<ActivationType, QuantScaleType, EnableA8W4>(
                        context, args, scratch, dispatchAddrInfo, expertIdx + 1);
                }
            }
        }

        if (!UpdateGmm1GroupParams<ActivationType, WeightType, SwigluOutType>(
            context, args, scratch, gmm1State, expertIdx, currentSendCnt)) {
            continue;
        }
        UpdateGmm1GlobalBuffer<ActivationType, WeightType, SwigluOutType,
            QuantScaleType, EnableA8W4>(context, args, gmm1AddrInfo, gmm1State);
        RunGmm1SwigluStageOneCore<QuantOutType, ActivationType, WeightType,
            SwigluOutType, QuantScaleType, EnableA8W4>(
                context, args, gmm1AddrInfo, gmm1State);
    }
    EndSync(*args.vecSetSyncCom);
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_DISPATCH_GMM1_SWIGLU_H
