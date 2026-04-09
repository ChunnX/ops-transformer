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
#include "block_epilogue_swiglu_mx_quant.h"

#include "dispatch_ffn_combine_catlass_impl.h"

using namespace AscendC;

namespace DispatchFFNCombineImpl {

constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024;

// 模板参数定义
// XType: 输入X的数据类型 (fp8_e4m3fn_t, bf16等)
// Weight1Type: 权重的数据类型 (fp8_e4m3fn_t等)
// WeightScale1Type: 权重量化的scale类型 (fp8_e8m0_t等)
// Weight2Type: GMM2的权重数据类型
// WeightScale2Type: GMM2权重量化的scale类型
// OutputType: 最终输出的数据类型
// QuantMode: 量化模式 (0: 无量化, 1: per-token量化, 2: per-block量化)
// QuantOutType: 量化输出类型 (0: 输出不量化, 1: 输出量化为fp8等)
#define TemplateDispatchFFNCombineTypeClass typename XType, typename Weight1Type, typename WeightScale1Type, typename Weight2Type, typename WeightScale2Type, typename OutputType, int32_t QuantMode, int32_t QuantOutType
#define TemplateDispatchFFNCombineTypeFunc XType, Weight1Type, WeightScale1Type, Weight2Type, WeightScale2Type, OutputType, QuantMode, QuantOutType
template <TemplateDispatchFFNCombineTypeClass>
class DispatchFFNCombine {
public:
    // 类型别名 - 使用模板参数
    using Gmm1TypesAlias = Gmm1Types<XType, Weight1Type, WeightScale1Type, GMM1OutType, OutputType>;
    using Gmm2TypesAlias = Gmm2Types<GMM1OutType, Weight2Type, WeightScale2Type, OutputType>;
    using Gmm1SwigluRunner = GmmSwigluRunner<Gmm1TypesAlias>;
    using Gmm2CombineRunner = Gmm2WithCombineRunner<Gmm2TypesAlias>;
    
    __aicore__ inline DispatchFFNCombine(){};
    
    __aicore__ inline void Init(
        GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights,
        GM_ADDR weight1, GM_ADDR weight2,
        GM_ADDR weightScales1, GM_ADDR weightScales2,
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

    // Callback context for static function pointer conversion (lambda with capture cannot convert to function pointer)
    struct CallbackContext {
        static DispatchFFNCombine* instance;
        static __aicore__ void NotifyVectorCallback(uint16_t value) {
            instance->NotifyVector(value);
        }
        static __aicore__ void WaitForCubeCallback(uint16_t value) {
            instance->WaitForCube(value);
        }
    };

    __aicore__ inline void InitParamsAndTensor();
    __aicore__ inline void UpdateOffset(uint32_t groupIdx);
    __aicore__ inline int32_t GetSplitValueFromGroupList(uint32_t groupIdx, uint8_t groupListType);
    __aicore__ inline bool UpdateGroupParams(uint32_t groupIdx);
    __aicore__ inline void UpdateGlobalBuffer(const Params& params);
    __aicore__ inline void InitRouting();
    __aicore__ inline void CrossRankSyncTokenPerExpert(GM_ADDR tokenPerExpertCurRank);
    __aicore__ inline void DispatchTokens(uint32_t groupIdx);
    __aicore__ inline void GroupMatmulSwigluQuant(const Params& params);
    __aicore__ inline void GroupMatmul2(const Params& params, uint32_t groupIdx);
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

    // GMM1 L0C输出 (使用GMM1OutType)
    LocalTensor<GMM1OutType> l0cOutUbFirst_;
    LocalTensor<GMM1OutType> l0cOutUbSecond_;
    // GMM2 L0C输出
    LocalTensor<OutputType> l0cOutUbGMM2First_;
    LocalTensor<OutputType> l0cOutUbGMM2Second_;
    
    GM_ADDR weight1_;
    GM_ADDR weight2_;
    GM_ADDR weightScales1_;
    GM_ADDR weightScales2_;
    GM_ADDR x_;
    GM_ADDR yOut_;
    GM_ADDR topkIds_;
    GM_ADDR topkWeights_;
    GM_ADDR workspaceGM_;
    
    // GMM1 tensors - 使用模板类型
    GlobalTensor<XType> aGlobal_;           // 输入X
    GlobalTensor<WeightType> bGlobal_;      // 权重W1
    GlobalTensor<WeightScaleType> aScaleGlobal_;  // X的scale
    GlobalTensor<WeightScaleType> bScaleGlobal_;  // W1的scale
    
    // GMM2 tensors - GMM2的A输入是GMM1的输出，B是W2
    GlobalTensor<GMM1OutType> a2Global_;    // GMM1的输出作为GMM2的输入
    GlobalTensor<WeightType> b2Global_;     // 权重W2
    GlobalTensor<WeightScaleType> a2ScaleGlobal_;  // GMM2 A的scale
    GlobalTensor<WeightScaleType> b2ScaleGlobal_;  // W2的scale
    GlobalTensor<OutputType> outputGm_;     // 最终输出
    
    // BlockEpilogue for SwiGLU+Quant
    using BlockEpilogue = BlockEpilogueSwiGluMxQuant<GMM1OutType, OutputType>;
    BlockEpilogue epilogueOp;
    
    // Params for function calls
    Params params_;
};

template <TemplateDispatchFFNCombineTypeClass>
__aicore__ inline void DispatchFFNCombine<TemplateDispatchFFNCombineTypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights,
    GM_ADDR weight1, GM_ADDR weight2,
    GM_ADDR weightScales1, GM_ADDR weightScales2,
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
    
    x_ = x;
    weight1_ = weight1;
    weight2_ = weight2;
    weightScales1_ = weightScales1;
    weightScales2_ = weightScales2;
    yOut_ = yOut;
    topkIds_ = topkIds;
    topkWeights_ = topkWeights;
    workspaceGM_ = workspaceGM;
    
    // Initialize params_ from workspaceGM and tilingData_
    params_.aGmAddr = workspaceGM_;
    params_.expertIdxGmAddr = topkIds_;
    params_.bGmAddr = weight1_;
    params_.bScaleGmAddr = weightScales1_;
    params_.b2GmAddr = weight2_;
    params_.b2ScaleGmAddr = weightScales2_;
    params_.y2GmAddr = yOut_;
    params_.probsGmAddr = topkWeights_;
    params_.tilingData = tilingData_;
    
    // Initialize peermemInfo
    for (uint32_t i = 0; i < tilingData_->megaMoeInfo.epWorldSize; i++) {
        params_.peermemInfo.peermemBaseAddr[i] = commBuffer_[i];
    }
    
    InitParamsAndTensor();
}

template <TemplateDispatchFFNCombineTypeClass>
__aicore__ inline void DispatchFFNCombine<TemplateDispatchFFNCombineTypeFunc>::NotifyCube(uint16_t value)
{
    AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIV_SYNC_AIC_FLAG + value);
}

template <TemplateDispatchFFNCombineTypeClass>
__aicore__ inline void DispatchFFNCombine<TemplateDispatchFFNCombineTypeFunc>::WaitForVector(uint16_t value)
{
    AscendC::CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIV_FLAG + value);
}

template <TemplateDispatchFFNCombineTypeClass>
__aicore__ inline void DispatchFFNCombine<TemplateDispatchFFNCombineTypeFunc>::NotifyVector(uint16_t value)
{
    AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG + value);
}

template <TemplateDispatchFFNCombineTypeClass>
__aicore__ inline void DispatchFFNCombine<TemplateDispatchFFNCombineTypeFunc>::WaitForCube(uint16_t value)
{
    AscendC::CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIC_SYNC_AIV_FLAG + value);
}

template <TemplateDispatchFFNCombineTypeClass>
__aicore__ inline void DispatchFFNCombine<TemplateDispatchFFNCombineTypeFunc>::EndSync(bool isVecSetSync)
{
    if (!isVecSetSync) return;
    if constexpr(g_coreType == AscendC::AIC) {
        WaitForVector();
    }
}

template <TemplateDispatchFFNCombineTypeClass>
__aicore__ inline void DispatchFFNCombine<TemplateDispatchFFNCombineTypeFunc>::EndGMM2Sync(uint32_t syncNum)
{
    if constexpr(g_coreType == AscendC::AIV) return;
    if (syncNum <= 0) return;
    else if (syncNum == 1) WaitForVector();
    else {
        WaitForVector(gmm2PingpongIdx_);
        WaitForVector(1 - gmm2PingpongIdx_);
    }
}

template <TemplateDispatchFFNCombineTypeClass>
__aicore__ inline void DispatchFFNCombine<TemplateDispatchFFNCombineTypeFunc>::AivSyncAllForAic()
{
    if constexpr(g_coreType == AscendC::AIC) {
        AscendC::CrossCoreWaitFlag<0x2>(0);
    }
    if constexpr(g_coreType == AscendC::AIV) {
        AscendC::SyncAll<true>();
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0);
    }
}

template <TemplateDispatchFFNCombineTypeClass>
__aicore__ inline void DispatchFFNCombine<TemplateDispatchFFNCombineTypeFunc>::InitParamsAndTensor()
{
    baseOffset_ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    Get<1>(problemShape_) = tilingData_->megaMoeInfo.n;
    Get<2>(problemShape_) = tilingData_->megaMoeInfo.k;
    tokenPerExpertLayout_ = Layout3D(
        CeilAlign(tilingData_->megaMoeInfo.epWorldSize * (tilingData_->megaMoeInfo.moeExpertNum / tilingData_->megaMoeInfo.epWorldSize), 128),
        tilingData_->megaMoeInfo.moeExpertNum / tilingData_->megaMoeInfo.epWorldSize);
    dispatchBlockNumPerEP_ = tilingData_->megaMoeInfo.blockNumPerEP;
}

template <TemplateDispatchFFNCombineTypeClass>
__aicore__ inline int32_t DispatchFFNCombine<TemplateDispatchFFNCombineTypeFunc>::GetSplitValueFromGroupList(uint32_t groupIdx, uint8_t groupListType)
{
    int32_t splitValue = 0;
    if (groupListType == 0) {
        int32_t offset = static_cast<int32_t>(groupListGm_.GetValue(groupIdx));
        splitValue = offset - prevSum_;
        prevSum_ = offset;
    } else {
        splitValue = static_cast<int32_t>(groupListGm_.GetValue(groupIdx));
    }
    return splitValue;
}

template <TemplateDispatchFFNCombineTypeClass>
__aicore__ inline void DispatchFFNCombine<TemplateDispatchFFNCombineTypeFunc>::UpdateOffset(uint32_t groupIdx)
{
    if (groupIdx == 0) {
        return;
    }
    uint64_t m = Get<0>(problemShape_);
    uint64_t n = Get<1>(problemShape_);
    uint64_t k = Get<2>(problemShape_);
    auto scaleK = CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
    
    Get<0>(baseOffset_) += m * k;
    Get<1>(baseOffset_) += n * k;
    Get<2>(baseOffset_) += m * scaleK;
    Get<3>(baseOffset_) += n * scaleK;
    Get<4>(baseOffset_) += m * n / 2;
    Get<5>(baseOffset_) += m * CeilDiv(n / 2, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
    Get<6>(baseOffset_) += 1;
    Get<7>(baseOffset_) += k * n / 2;
    Get<8>(baseOffset_) += k * CeilDiv(n / 2, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
    Get<9>(baseOffset_) += m * k;
    Get<10>(baseOffset_) += m;
}

template <TemplateDispatchFFNCombineTypeClass>
__aicore__ inline bool DispatchFFNCombine<TemplateDispatchFFNCombineTypeFunc>::UpdateGroupParams(uint32_t groupIdx)
{
    UpdateOffset(groupIdx);
    int32_t splitValue = GetSplitValueFromGroupList(groupIdx, tilingData_->megaMoeInfo.groupListType);
    Get<0>(problemShape_) = splitValue;
    if (Get<0>(problemShape_) == 0) {
        return false;
    }
    return true;
}

template <TemplateDispatchFFNCombineTypeClass>
__aicore__ inline void DispatchFFNCombine<TemplateDispatchFFNCombineTypeFunc>::UpdateGlobalBuffer(const Params& params)
{
    // 使用params中的地址，参考老代码
    aGlobal_.SetGlobalBuffer((__gm__ XType*)(params.workspaceInfo.ptrA + Get<0>(baseOffset_)));
    aScaleGlobal_.SetGlobalBuffer((__gm__ WeightScaleType*)(params.workspaceInfo.ptrAScale + Get<2>(baseOffset_)));
    if constexpr(g_coreType == AscendC::AIC) {
        bGlobal_.SetGlobalBuffer((__gm__ WeightType*)(params.bGmAddr + Get<1>(baseOffset_)));
        bScaleGlobal_.SetGlobalBuffer((__gm__ WeightScaleType*)(params.bScaleGmAddr + Get<3>(baseOffset_)));
        // GMM2的A输入是GMM1的输出
        a2Global_.SetGlobalBuffer((__gm__ GMM1OutType*)(params.workspaceInfo.ptrA2 + Get<4>(baseOffset_)));
        a2ScaleGlobal_.SetGlobalBuffer((__gm__ WeightScaleType*)(params.workspaceInfo.ptrA2Scale + Get<5>(baseOffset_)));
        b2Global_.SetGlobalBuffer((__gm__ WeightType*)(params.b2GmAddr + Get<7>(baseOffset_)));
        b2ScaleGlobal_.SetGlobalBuffer((__gm__ WeightScaleType*)(params.b2ScaleGmAddr + Get<8>(baseOffset_)));
    }
    if constexpr(g_coreType == AscendC::AIV) {
        epilogueOp.UpdateGlobalAddr({Get<4>(baseOffset_), Get<5>(baseOffset_), Get<6>(baseOffset_), 0, 0, 0});
        outputGm_.SetGlobalBuffer((__gm__ OutputType*)(params.y2GmAddr + Get<9>(baseOffset_)));
    }
    groupFlagListGm_.SetGlobalBuffer((__gm__ int32_t*)(params.workspaceInfo.ptrFlagList + Get<6>(baseOffset_) * INT_CACHELINE * MAX_AICORE_NUM));
    groupFlagListGm2_.SetGlobalBuffer((__gm__ int32_t*)(params.workspaceInfo.ptrFlagList2 + Get<6>(baseOffset_) * INT_CACHELINE * MAX_AICORE_NUM + 128));
}

template <TemplateDispatchFFNCombineTypeClass>
__aicore__ inline void DispatchFFNCombine<TemplateDispatchFFNCombineTypeFunc>::InitRouting()
{
    // Call interface - implementation in dispatch_ffn_combine_ops_impl.h
    DispatchFFNCombineImpl::InitRouting(params_, tokenPerExpertGm_, 0);
}

template <TemplateDispatchFFNCombineTypeClass>
__aicore__ inline void DispatchFFNCombine<TemplateDispatchFFNCombineTypeFunc>::CrossRankSyncTokenPerExpert(GM_ADDR tokenPerExpertCurRank)
{
    // Call interface - implementation in dispatch_ffn_combine_ops_impl.h
    DispatchFFNCombineImpl::CrossRankSyncTokenPerExpert(params_, tokenPerExpertGm_, preSumBeforeRankGm_, cumsumMMGm_);
}

template <TemplateDispatchFFNCombineTypeClass>
__aicore__ inline void DispatchFFNCombine<TemplateDispatchFFNCombineTypeFunc>::DispatchTokens(uint32_t groupIdx)
{
    GlobalTensor<bfloat16_t> aGlobal;
    GlobalTensor<fp8_e8m0_t> aScaleGlobal;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x_));
    aScaleGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ fp8_e8m0_t*>(weightScales1_));
    
    // Call interface - implementation in dispatch_ffn_combine_ops_impl.h
    DispatchFFNCombineImpl::DispatchTokens(params_, groupIdx, tokenPerExpertGm_, 
                                          preSumBeforeRankGm_, cumsumMMGm_,
                                          aGlobal, aScaleGlobal, 0);
}

template <TemplateDispatchFFNCombineTypeClass>
__aicore__ inline void DispatchFFNCombine<TemplateDispatchFFNCombineTypeFunc>::GroupMatmulSwigluQuant(const Params& params)
{
    uint32_t m = GetMValue();
    uint32_t n = GetNValue();
    uint32_t k = GetKValue();
    uint32_t outputN = n / SWIGLU_N_HALF;
    
    epilogueOp.UpdateNextProblem({m, outputN, k, 0});

    typename Gmm1SwigluRunner::GmmSwigluParams gmmParams = {
        .aGm = x_,
        .bGm = weight1_,
        .bScaleGm = weightScales1_,
        .aScaleGm = 0,  // x scale, 需要根据实际情况设置
        .cUb = 0,
        .cUb2 = 0,
        .outputUb = 0,
        .outputScaleUb = 0,
        .m = m,
        .n = n,
        .k = k,
        .blockNum = blockNum_,
        .blockIdx = blockIdx_,
        .subBlockIdx = subBlockIdx_,
        .startBlockIdx = startBlockIdx_,
        .isVecSetSyncCom = isVecSetSyncCom_
    };

    // Set callback context for static function pointer conversion
    CallbackContext::instance = this;
    Gmm1SwigluRunner::RunTile(gmmParams, l0cOutUbFirst_, l0cOutUbSecond_,
                              gmm2PingpongIdx_, isVecSetSyncCom_, startBlockIdx_,
                              blockNum_, blockIdx_,
                              reinterpret_cast<GM_ADDR>(CallbackContext::NotifyVectorCallback),
                              reinterpret_cast<GM_ADDR>(CallbackContext::WaitForCubeCallback));
    startBlockIdx_ = (startBlockIdx_ + 1) % blockNum_;
}

template <TemplateDispatchFFNCombineTypeClass>
__aicore__ inline void DispatchFFNCombine<TemplateDispatchFFNCombineTypeFunc>::GroupMatmul2(const Params& params, uint32_t groupIdx)
{
    // Set callback context for static function pointer conversion
    CallbackContext::instance = this;
    
    // Get correct GM_ADDR for GMM2 data tensors
    GM_ADDR a2Gm = reinterpret_cast<GM_ADDR>(a2Global_.GetPhyAddr());
    GM_ADDR b2Gm = reinterpret_cast<GM_ADDR>(b2Global_.GetPhyAddr());
    GM_ADDR a2ScaleGm = reinterpret_cast<GM_ADDR>(a2ScaleGlobal_.GetPhyAddr());
    GM_ADDR b2ScaleGm = reinterpret_cast<GM_ADDR>(b2ScaleGlobal_.GetPhyAddr());
    GM_ADDR outputGm = reinterpret_cast<GM_ADDR>(outputGm_.GetPhyAddr());
    
    // L0C buffer address - use first buffer
    GM_ADDR cUb = reinterpret_cast<GM_ADDR>(l0cOutUbGMM2First_.GetPhyAddr());
    
    // Fix parameter order: m, n, k (not m, k, n/2)
    Gmm2CombineRunner::RunTileGmm2(a2Gm, b2Gm, a2ScaleGm, b2ScaleGm, cUb, outputGm,
                                   GetMValue(), GetNValue(), GetKValue(),
                                   blockNum_, blockIdx_, startBlockIdx_,
                                   vecSetSyncCom2_, gmm2PingpongIdx_,
                                   reinterpret_cast<GM_ADDR>(CallbackContext::NotifyVectorCallback),
                                   reinterpret_cast<GM_ADDR>(CallbackContext::WaitForCubeCallback),
                                   reinterpret_cast<GM_ADDR>(tokenPerExpertGm_.GetPhyAddr()),
                                   reinterpret_cast<GM_ADDR>(preSumBeforeRankGm_.GetPhyAddr()),
                                   reinterpret_cast<GM_ADDR>(cumsumMMGm_.GetPhyAddr()),
                                   groupIdx, params_);
    vecSetSyncCom2_++;
    gmm2PingpongIdx_ = 1 - gmm2PingpongIdx_;
}

template <TemplateDispatchFFNCombineTypeClass>
__aicore__ inline void DispatchFFNCombine<TemplateDispatchFFNCombineTypeFunc>::FinalizeRouting()
{
    GlobalTensor<bfloat16_t> combinedOutput;
    GlobalTensor<int32_t> expandedRowIdx;
    GlobalTensor<bfloat16_t> topkWeights;
    GlobalTensor<int32_t> topkIds;
    
    combinedOutput.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(yOut_));
    expandedRowIdx.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspaceGM_));
    topkWeights.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(topkWeights_));
    topkIds.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(topkIds_));
    
    // Call interface - implementation in dispatch_ffn_combine_ops_impl.h
    DispatchFFNCombineImpl::FinalizeRouting(params_, combinedOutput, expandedRowIdx, 
                                           0, topkWeights, topkIds);
}

template <TemplateDispatchFFNCombineTypeClass>
__aicore__ inline bool DispatchFFNCombine<TemplateDispatchFFNCombineTypeFunc>::IsFirstSubBlock()
{
    return subBlockIdx_ == 0;
}

template <TemplateDispatchFFNCombineTypeClass>
__aicore__ inline uint32_t DispatchFFNCombine<TemplateDispatchFFNCombineTypeFunc>::GetMValue()
{
    return static_cast<uint32_t>(Get<0>(problemShape_));
}

template <TemplateDispatchFFNCombineTypeClass>
__aicore__ inline uint32_t DispatchFFNCombine<TemplateDispatchFFNCombineTypeFunc>::GetNValue()
{
    return static_cast<uint32_t>(Get<1>(problemShape_));
}

template <TemplateDispatchFFNCombineTypeClass>
__aicore__ inline uint32_t DispatchFFNCombine<TemplateDispatchFFNCombineTypeFunc>::GetKValue()
{
    return static_cast<uint32_t>(Get<2>(problemShape_));
}

template <TemplateDispatchFFNCombineTypeClass>
__aicore__ inline void DispatchFFNCombine<TemplateDispatchFFNCombineTypeFunc>::Process()
{
    
    GM_ADDR tokenPerExpertCurRank = 0;
    InitRouting();
    
    SyncAll<true>();
    CrossRankSyncTokenPerExpert(tokenPerExpertCurRank);
    AivSyncAllForAic();

    uint32_t expertPerRank = tilingData_->megaMoeInfo.expertPerRank;
    for (uint32_t groupIdx = 0; groupIdx < expertPerRank; groupIdx++) {
        if (!UpdateGroupParams(groupIdx)) continue;
        UpdateGlobalBuffer(params_);

        if (subBlockIdx_ == 1) {
            DispatchTokens(groupIdx);
        }
        if (subBlockIdx_ == 0) {
            GroupMatmulSwigluQuant(params_);
        }
    }

    baseOffset_ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (uint32_t groupIdx = 0; groupIdx < expertPerRank; groupIdx++) {
        if (!UpdateGroupParams(groupIdx)) continue;
        UpdateGlobalBuffer(params_);
        if (subBlockIdx_ == 0) {
            GroupMatmul2(params_, groupIdx);
        }
    }

    EndGMM2Sync(vecSetSyncCom2_);
    SyncAll<true>();
    FinalizeRouting();
}

}  // namespace DispatchFFNCombineImpl

#endif