/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */
#ifndef DISPATCH_FFN_COMBINE_H
#define DISPATCH_FFN_COMBINE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "dispatch_ffn_combine_tiling.h"

#include "gmmsq_base.h"
#include "workspace_info.h"
#include "hccl_shmem.h"

#include "dispatch_ffn_combine_catlass_impl.h"

using namespace AscendC;

namespace DispatchFFNCombineImpl {

constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024;

struct Mc2MoeContext {
    uint32_t epRankId;
    uint32_t epRankSize;
    uint64_t winSize;
    uint64_t epHcclBuffer_[HCCL_MAX_RANK_SIZE];
};

template <typename BlockMmad1, typename BlockMmad2, typename BlockEpilogue, typename BlockScheduler>
class KernelGmmSwiGluMix;

template <typename T>
class DispatchFFNCombine {
public:
    __aicore__ inline DispatchFFNCombine(){};
    
    __aicore__ inline void Init(
        GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights,
        GM_ADDR weight1, GM_ADDR weight2,
        GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2,
        GM_ADDR scales, GM_ADDR yOut, GM_ADDR expertTokenNumsOut,
        GM_ADDR workspaceGM, GM_ADDR tilingGM);
    
    __aicore__ inline void Process();

private:
    __aicore__ inline void NotifyCube(uint16_t value = 0);
    __aicore__ inline void WaitForVector(uint16_t value = 0);
    __aicore__ inline void NotifyVector(uint16_t value = 0);
    __aicore__ inline void WaitForCube(uint16_t value = 0);
    __aicore__ inline void EndSync(bool isVecSetSync);
    __aicore__ inline void EndGMM2Sync(uint32_t syncNum);
    __aicore__ inline void AivSyncAllForAic();

    __aicore__ inline void InitParamsAndTensor();
    __aicore__ inline void UpdateGroupParams(uint32_t groupIdx);
    __aicore__ inline void UpdateGlobalBuffer();
    __aicore__ inline void InitRouting();
    __aicore__ inline void CrossRankSyncTokenPerExpert(GM_ADDR tokenPerExpertCurRank);
    __aicore__ inline void DispatchTokens(uint32_t groupIdx);
    __aicore__ inline void GroupMatmulSwigluQuant();
    __aicore__ inline void GroupMatmul2(uint32_t groupIdx);
    __aicore__ inline void FinalizeRouting();

    __aicore__ inline bool IsFirstSubBlock();
    __aicore__ inline uint32_t GetMValue();
    __aicore__ inline uint32_t GetNValue();
    __aicore__ inline uint32_t GetKValue();

    template<typename GmmRunner>
    __aicore__ inline void RunGmm(GM_ADDR aGm, GM_ADDR bGm, GM_ADDR cUb, uint32_t m, uint32_t n, uint32_t k);

    template<typename GmmSwigluRunner>
    __aicore__ inline void RunGmmSwiglu(
        const typename GmmSwigluRunner::GmmSwigluParams& params,
        LocalTensor<bfloat16_t> l0cOutUbFirst,
        LocalTensor<bfloat16_t> l0cOutUbSecond,
        GM_ADDR notifyFn, GM_ADDR waitFn);

    template<typename Gmm2Runner>
    __aicore__ inline void RunGmm2(
        GM_ADDR aGm, GM_ADDR bGm, GM_ADDR cUb, GM_ADDR outputGm,
        uint32_t m, uint32_t n, uint32_t k, GM_ADDR notifyFn, GM_ADDR waitFn);

private:
    __gm__ Mc2MoeContext* mc2Context_;
    GM_ADDR commBuffer_[HCCL_MAX_RANK_SIZE];
    const DispatchFFNCombineTilingData* tilingData_;

    using TupleShape = Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockOffset = Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    TupleShape problemShape_;
    BlockOffset baseOffset_;

    uint32_t blockNum_;
    uint32_t blockIdx_;
    uint32_t subBlockIdx_;
    uint32_t aivCoreIdx_;
    int32_t prevSum_;

    bool isVecSetSyncCom_;
    uint32_t vecSetSyncCom2_;
    uint32_t startBlockIdx_;
    uint32_t dispatchBlockNumPerEP_;
    uint16_t gmm2PingpongIdx_;

    GlobalTensor<int32_t> groupListGm_;
    GlobalTensor<int32_t> groupFlagListGm_;
    GlobalTensor<int32_t> groupFlagListGm2_;
    GlobalTensor<int32_t> tokenPerExpertGm_;
    GlobalTensor<int32_t> preSumBeforeRankGm_;
    GlobalTensor<int32_t> cumsumMMGm_;
    Layout3D tokenPerExpertLayout_;

    LocalTensor<bfloat16_t> l0cOutUbFirst_;
    LocalTensor<bfloat16_t> l0cOutUbSecond_;
    LocalTensor<bfloat16_t> l0cOutUbGMM2First_;
    LocalTensor<bfloat16_t> l0cOutUbGMM2Second_;
};

template <typename T>
__aicore__ inline void DispatchFFNCombine<T>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights,
    GM_ADDR weight1, GM_ADDR weight2,
    GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2,
    GM_ADDR scales, GM_ADDR yOut, GM_ADDR expertTokenNumsOut,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    tilingData_ = reinterpret_cast<const DispatchFFNCombineTilingData*>(tilingGM);
    mc2Context_ = reinterpret_cast<__gm__ Mc2MoeContext*>(context);
    
    for (uint32_t i = 0; i < tilingData_->megaMoeInfo.epWorldSize; i++) {
        commBuffer_[i] = reinterpret_cast<GM_ADDR>(mc2Context_->epHcclBuffer_[i]);
    }

    blockNum_ = GetBlockNum();
    blockIdx_ = GetBlockIdx() / GetTaskRation();
    subBlockIdx_ = GetSubBlockIdx();
    aivCoreIdx_ = GetBlockIdx();

    isVecSetSyncCom_ = false;
    vecSetSyncCom2_ = 0;
    startBlockIdx_ = 0;
    gmm2PingpongIdx_ = 0;
    prevSum_ = 0;
    dispatchBlockNumPerEP_ = 2;
}

template <typename T>
__aicore__ inline void DispatchFFNCombine<T>::NotifyCube(uint16_t value)
{
    AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIV_SYNC_AIC_FLAG + value);
}

template <typename T>
__aicore__ inline void DispatchFFNCombine<T>::WaitForVector(uint16_t value)
{
    AscendC::CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIV_FLAG + value);
}

template <typename T>
__aicore__ inline void DispatchFFNCombine<T>::NotifyVector(uint16_t value)
{
    AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG + value);
}

template <typename T>
__aicore__ inline void DispatchFFNCombine<T>::WaitForCube(uint16_t value)
{
    AscendC::CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIC_SYNC_AIV_FLAG + value);
}

template <typename T>
__aicore__ inline void DispatchFFNCombine<T>::EndSync(bool isVecSetSync)
{
    if (!isVecSetSync) return;
    if constexpr(g_coreType == AscendC::AIC) {
        WaitForVector();
    }
}

template <typename T>
__aicore__ inline void DispatchFFNCombine<T>::EndGMM2Sync(uint32_t syncNum)
{
    if constexpr(g_coreType == AscendC::AIV) return;
    if (syncNum <= 0) return;
    else if (syncNum == 1) WaitForVector();
    else {
        WaitForVector(gmm2PingpongIdx_);
        WaitForVector(1 - gmm2PingpongIdx_);
    }
}

template <typename T>
__aicore__ inline void DispatchFFNCombine<T>::AivSyncAllForAic()
{
    if constexpr(g_coreType == AscendC::AIC) {
        AscendC::CrossCoreWaitFlag<0x2>(0);
    }
    if constexpr(g_coreType == AscendC::AIV) {
        AscendC::SyncAll<true>();
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0);
    }
}

template <typename T>
__aicore__ inline void DispatchFFNCombine<T>::InitParamsAndTensor()
{
    baseOffset_ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    Get<1>(problemShape_) = tilingData_->megaMoeInfo.n;
    Get<2>(problemShape_) = tilingData_->megaMoeInfo.k;
    tokenPerExpertLayout_ = Layout3D(
        CeilAlign(tilingData_->megaMoeInfo.epWorldSize * (tilingData_->megaMoeInfo.moeExpertNum / tilingData_->megaMoeInfo.epWorldSize), 128),
        tilingData_->megaMoeInfo.moeExpertNum / tilingData_->megaMoeInfo.epWorldSize);
    dispatchBlockNumPerEP_ = tilingData_->mc2InitTiling.blockNumPerEP;
}

template <typename T>
__aicore__ inline bool DispatchFFNCombine<T>::UpdateGroupParams(uint32_t groupIdx)
{
    return true;
}

template <typename T>
__aicore__ inline void DispatchFFNCombine<T>::UpdateGlobalBuffer()
{
}

template <typename T>
__aicore__ inline void DispatchFFNCombine<T>::InitRouting()
{
}

template <typename T>
__aicore__ inline void DispatchFFNCombine<T>::CrossRankSyncTokenPerExpert(GM_ADDR tokenPerExpertCurRank)
{
}

template <typename T>
__aicore__ inline void DispatchFFNCombine<T>::DispatchTokens(uint32_t groupIdx)
{
}

template <typename T>
__aicore__ inline void DispatchFFNCombine<T>::GroupMatmulSwigluQuant()
{
    typename Gmm1SwigluRunner::GmmSwigluParams params = {
        .aGm = 0,
        .bGm = 0,
        .bScaleGm = 0,
        .aScaleGm = 0,
        .cUb = 0,
        .cUb2 = 0,
        .outputUb = 0,
        .outputScaleUb = 0,
        .m = GetMValue(),
        .n = GetNValue(),
        .k = GetKValue(),
        .blockNum = blockNum_,
        .blockIdx = blockIdx_,
        .subBlockIdx = subBlockIdx_,
        .startBlockIdx = startBlockIdx_,
        .isVecSetSyncCom = isVecSetSyncCom_
    };

    RunGmmSwiglu<Gmm1SwigluRunner>(params, l0cOutUbFirst_, l0cOutUbSecond_,
                        reinterpret_cast<GM_ADDR>(&NotifyVector), reinterpret_cast<GM_ADDR>(&WaitForCube));
}

template <typename T>
__aicore__ inline void DispatchFFNCombine<T>::GroupMatmul2(uint32_t groupIdx)
{
    RunGmm2<Gmm2CombineRunner>(0, 0, 0, 0, GetMValue(), GetKValue(), GetNValue() / 2, 
                        reinterpret_cast<GM_ADDR>(&NotifyVector), reinterpret_cast<GM_ADDR>(&WaitForCube));
    vecSetSyncCom2_++;
    gmm2PingpongIdx_ = 1 - gmm2PingpongIdx_;
}

template <typename T>
template<typename GmmRunner>
__aicore__ inline void DispatchFFNCombine<T>::RunGmm(GM_ADDR aGm, GM_ADDR bGm, GM_ADDR cUb, uint32_t m, uint32_t n, uint32_t k)
{
    GmmRunner::Run(aGm, bGm, cUb, m, n, k, blockNum_, blockIdx_, startBlockIdx_);
}

template <typename T>
template<typename GmmSwigluRunner>
__aicore__ inline void DispatchFFNCombine<T>::RunGmmSwiglu(
    const typename GmmSwigluRunner::GmmSwigluParams& params,
    LocalTensor<bfloat16_t> l0cOutUbFirst,
    LocalTensor<bfloat16_t> l0cOutUbSecond,
    GM_ADDR notifyFn, GM_ADDR waitFn)
{
    GmmSwigluRunner::RunTile(params, l0cOutUbFirst, l0cOutUbSecond, 
                             gmm2PingpongIdx_, isVecSetSyncCom_, startBlockIdx_, 
                             blockNum_, blockIdx_, notifyFn, waitFn);
    startBlockIdx_ = (startBlockIdx_ + 1) % blockNum_;
}

template <typename T>
template<typename GmmRunner>
__aicore__ inline void DispatchFFNCombine<T>::RunGmm2(
    GM_ADDR aGm, GM_ADDR bGm, GM_ADDR cUb, GM_ADDR outputGm,
    uint32_t m, uint32_t n, uint32_t k, GM_ADDR notifyFn, GM_ADDR waitFn)
{
    GmmRunner::RunTileGmm2(aGm, bGm, cUb, outputGm, m, n, k, 
                           blockNum_, blockIdx_, startBlockIdx_,
                           vecSetSyncCom2_, gmm2PingpongIdx_,
                           notifyFn, waitFn);
    startBlockIdx_ = (startBlockIdx_ + 1) % blockNum_;
}

template <typename T>
__aicore__ inline void DispatchFFNCombine<T>::FinalizeRouting()
{
}

template <typename T>
__aicore__ inline bool DispatchFFNCombine<T>::IsFirstSubBlock()
{
    return subBlockIdx_ == 0;
}

template <typename T>
__aicore__ inline uint32_t DispatchFFNCombine<T>::GetMValue()
{
    return static_cast<uint32_t>(Get<0>(problemShape_));
}

template <typename T>
__aicore__ inline uint32_t DispatchFFNCombine<T>::GetNValue()
{
    return static_cast<uint32_t>(Get<1>(problemShape_));
}

template <typename T>
__aicore__ inline uint32_t DispatchFFNCombine<T>::GetKValue()
{
    return static_cast<uint32_t>(Get<2>(problemShape_));
}

template <typename T>
template<typename GmmRunner>
__aicore__ inline void DispatchFFNCombine<T>::RunGmm(GM_ADDR aGm, GM_ADDR bGm, GM_ADDR cUb, uint32_t m, uint32_t n, uint32_t k)
{
    GmmRunner::Run(aGm, bGm, cUb, m, n, k, blockNum_, blockIdx_, startBlockIdx_);
}

}

template <typename T>
__aicore__ inline void DispatchFFNCombine<T>::Process()
{
    InitParamsAndTensor();
    
    GM_ADDR tokenPerExpertCurRank = 0;
    InitRouting();
    
    SyncAll<true>();
    CrossRankSyncTokenPerExpert(tokenPerExpertCurRank);
    AivSyncAllForAic();

    uint32_t expertPerRank = tilingData_->megaMoeInfo.moeExpertNum / tilingData_->megaMoeInfo.epWorldSize;
    for (uint32_t groupIdx = 0; groupIdx < expertPerRank; groupIdx++) {
        if (!UpdateGroupParams(groupIdx)) continue;
        UpdateGlobalBuffer();

        if (subBlockIdx_ == 1) {
            DispatchTokens(groupIdx);
        }
        if (subBlockIdx_ == 0) {
            GroupMatmulSwigluQuant();
        }
    }

    baseOffset_ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (uint32_t groupIdx = 0; groupIdx < expertPerRank; groupIdx++) {
        if (!UpdateGroupParams(groupIdx)) continue;
        UpdateGlobalBuffer();
        if (subBlockIdx_ == 0) {
            GroupMatmul2(groupIdx);
        }
    }

    EndGMM2Sync(vecSetSyncCom2_);
    SyncAll<true>();
    FinalizeRouting();
}

#endif