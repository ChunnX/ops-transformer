#ifndef MOE_MOE_DISPATCH_FFN_COMBINE_H
#define MOE_MOE_DISPATCH_FFN_COMBINE_H
 
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_dispatch_ffn_combine_tiling.h"
 
#include "gmmsq_base.h"
#include "workspace_info.h"
#include "hccl_shmem.h"
#include "block_epilogue_swiglu_mx_quant.h"
 
#include "moe_dispatch_ffn_combine_catlass_impl.h"
 
using namespace AscendC;
using namespace tla;
 
namespace MoeDispatchFFNCombineImpl {
 
constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024;
struct Mc2MoeContext {
    uint32_t epRankId;
    uint32_t epRankSize;
    uint64_t winSize;
    uint64_t epHcclBuffer_[HCCL_MAX_RANK_SIZE];
};
 
// 模板参数定义
// XType: 输入X的数据类型 (fp8_e4m3fn_t, bf16等)
// Weight1Type: 权重的数据类型 (fp8_e4m3fn_t等)
// Weight1ScaleType: 权重量化的scale类型 (fp8_e8m0_t等)
// GMM1OutType: GMM1(SwiGLU)输出的数据类型
// OutputType: 最终输出的数据类型
// QuantMode: 量化模式 (0: 无量化, 1: per-token量化, 2: per-block量化)
#define TemplateMoeDispatchFFNCombineTypeClass typename XType, typename Weight1Type, typename WeightScale1Type, typename Weight2Type, typename WeightScale2Type, typename OutputType, int32_t QuantMode, int32_t QuantOutType
#define TemplateMoeDispatchFFNCombineTypeFunc XType, Weight1Type, WeightScale1Type, Weight2Type, WeightScale2Type, OutputType, QuantMode, QuantOutType

struct GMMAddrInfo {
    GM_ADDR aGlobal;
    GM_ADDR aScaleGlobal;
    GM_ADDR bGlobal;
    GM_ADDR bScaleGlobal;
};

template <TemplateMoeDispatchFFNCombineTypeClass>
class MoeDispatchFFNCombine {
public:
    // 类型别名 - 使用模板参数
    using GMM1OutType = bfloat16_t;
    // TupleShape 用于更新shape BlockOffset 用于更新偏移
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockOffset = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t,
                                       int64_t, int64_t, int64_t, int64_t, int64_t,
                                       int64_t, int64_t>;
    __aicore__ inline MoeDispatchFFNCombine(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights,
        GM_ADDR weight1, GM_ADDR weight2, GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2,
        GM_ADDR scales, GM_ADDR yOut, GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, GM_ADDR tilingGM);
    __aicore__ inline void Process();
 
private:
    __aicore__ inline void ResetWorkSpaceFlagList();
    __aicore__ inline void GetCumsumForMMAIV();
    __aicore__ inline void CrossRankSyncCntPerExpert();
    __aicore__ inline void CopyGMToGMPerTokenV2(GlobalTensor<quantOutType_> dst, GlobalTensor<quantOutScaleType_> dstScale,
        GlobalTensor<quantOutType_> src, int32_t rows, int32_t hiddenSize, int32_t& pingpongId);
    __aicore__ inline void DispatchTokens(uint32_t expertIdx);
    __aicore__ inline bool UpdateGroupParams(uint32_t expertIdx);
    __aicore__ inline void UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo, uint32_t gmmIndex);
    __aicore__ inline void ArgSortExpandedRowIdx();
    __aicore__ inline void ResetTokenPerExpert();
    __aicore__ inline void Unpermute();
 
    __aicore__ inline void NotifyCube(uint16_t value = 0)
    {
        AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIV_SYNC_AIC_FLAG + value);
    }
 
    __aicore__ inline void WaitForVector(uint16_t value = 0)
    {
        AscendC::CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG + value);
    }
 
    __aicore__ inline void NotifyVector(uint16_t value = 0)
    {
        AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG + value);
    }
 
    __aicore__ inline void WaitForCube(uint16_t value = 0)
    {
        AscendC::CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIC_SYNC_AIV_FLAG + value);
    }
 
    __aicore__ inline void EndSync(const int32_t syncNum)
    {
        if (syncNum == 0) {
            return;
        }
        if constexpr(g_coreType == AscendC::AIC) {
            WaitForVector();
        }
    }
 
    __aicore__ inline void EndGMM2Sync(const int32_t syncNum)
    {
        if constexpr(g_coreType == AscendC::AIV) {
            return;
        }
        if (syncNum <= 0) {
            return;
        } else if (syncNum == 1) {
            WaitForVector();
        } else {
            WaitForVector(0);
            WaitForVector(1);
        }
    }
 
    __aicore__ inline void TilingByCore(int32_t totalLen, int32_t &coreLen, int32_t& coreOffset)
    {
        int32_t coreIdx = GetBlockIdx();
        int32_t coreNum = GetBlockNum();
        int32_t lenPerCore = CeilDiv((uint32_t)totalLen, coreNum) / 32 * 32;
        int32_t lenLastCore = totalLen - lenPerCore * (coreNum - 1);
        coreLen = (coreIdx == coreNum - 1) ? lenLastCore : lenPerCore;
        coreOffset = coreIdx * lenPerCore;
    }
 
    __aicore__ inline void  TokenPerExpertLayout(int64_t dim0, int64_t dim1, int64_t dim2)
    {
        return dim0 * strides_ + dim1 * expertPerRank_ + dim2;
    }
 
    __aicore__ inline void  CeilAlign(int64_t a, int64_t b)
    {
        if (b == 0) {
            return a;
        }
        return (a + b - 1) / b * b;
    }
 
    __aicore__ inline void gm_signal_wait_until_ne(__gm__ int32_t *sig_addr, int32_t cmp_val) {
        do {
            if (AscendC::ReadGmByPassDCache(sig_addr) != cmp_val) {
                return;
            }
        } while (true);
    }

    __simt_vf__ __aicore__ LAUNCH_BOUND(MoeInitRoutingV3::SIMT_THREAD_NUM) inline void ArgsortSimt(int64_t elements, int64_t indexBase,
                                            __gm__ int32_t *sortedExpertIndicesGmAddr, __gm__ int32_t *expandedRowIdxGmAddr) {
        for (int32_t index = static_cast<int32_t>(Simt::GetThreadIdx()); index < static_cast<int32_t>(elements);
            index += static_cast<int32_t>(Simt::GetThreadNum())) {
            int64_t outIndices = sortedExpertIndicesGmAddr[indexBase + index];
            expandedRowIdxGmAddr[outIndices] = indexBase + index;
        }
    }
 
    __gm__ Mc2MoeContext* mc2Context_{nullptr};
    Params params_;
    TupleShape problemShape_{};
    BlockOffset baseOffset_{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
 
    GM_ADDR tokenPerExpertCurRank_;
 
    GlobalTensor<int32_t> preSumBeforeRankWorkSpaceGM_;
    GlobalTensor<int32_t> tokenPerExpertWinGM_;
    GlobalTensor<int32_t> cumsumInWorkSpaceGM_;
    GlobalTensor<int32_t> groupFlagListWorkSpaceGM_;
    GlobalTensor<int32_t> groupFlagListGm_;
    GlobalTensor<int32_t> groupFlagListGm2_;
    GlobalTensor<int32_t> groupListGm_;
 
    uint32_t blockNum_ = GetBlockNum();
    uint32_t blockIdx_ = GetBlockIdx() / AscendC::GetTaskRation();
    uint32_t subBlockIdx_ = GetSubBlockIdx();
    uint32_t aivCoreIdx_ = GetBlockIdx();

    uint32_t m_;
    uint32_t k_;
    uint32_t topK_;
    uint32_t rankId_;
    uint32_t worldSize_;
    uint32_t expertPerRank_;
    uint32_t groupListType_;
    int32_t preOffset_ = 0;
    int64_t strides_;
 
    int32_t prevSum_ = 0;
    int32_t dispatchRows_ = 0;
    int32_t vecSetSyncCom_ = 0;
    uint32_t dispatchBlockNumPerEP_ = 2;
    uint32_t startBlockIdx_ = 0;
    uint16_t gmm2PingpongIdx_ = 0;

    BlockEpilogue epilogueOp_;
    LocalTensor<bfloat16_t> l0cOutUbFirst_;
    LocalTensor<bfloat16_t> l0cOutUbSecond_;

    using quantOutType_ = fp8_e4m3fn_t;
    using quantOutScaleType_ = fp8_e8m0_t;

    static __aicore__ void NotifyVectorCallback(uint16_t value)
    {
        AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG + value);
    }

    static __aicore__ void WaitForCubeCallback(uint16_t value)
    {
        AscendC::CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIC_SYNC_AIV_FLAG + value);
    }
};
 
template <TemplateMoeDispatchFFNCombineTypeClass>
__aicore__ inline void MoeDispatchFFNCombine<TemplateMoeDispatchFFNCombineTypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1, GM_ADDR weight2,
    GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2,GM_ADDR scales, GM_ADDR yOut, 
    GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, TPipe *pipe, const MoeDispatchFFNCombineTilingData *tilingData)
{
    m_ = tilingData->m;
    k_ = tilingData->k;
    topK_ = tilingData->topK;
    rankId_ = tilingData->rank;
    worldSize_ = tilingData->worldSize_;
    expertPerRank_ = tilingData->expertPerRank_;
    dispatchBlockNumPerEP_ = tilingData->blockNumPerEP;
    dispatchRows_ = tilingData->dispatchRows;
    groupListType_ = tilingData->groupListType;
    Get<N_VALUE>(problemShape_) = tilingData->n;
    Get<K_VALUE>(problemShape_) = k_;
    GM_ADDR buff[HCCL_MAX_RANK_SIZE];
    mc2Context_ = reinterpret_cast<__gm__ Mc2MoeContext*>(context);
    for (int i = 0; i < worldSize_; i++) {
        buff[i] = (GM_ADDR)mc2Context_->epHcclBuffer_[i];
    }
    params_ = {
        .aGmAddr = x,
        .expertIdxGmAddr = topkIds,
        .bGmAddr = weight1,
        .bScaleGmAddr = weightScales1,
        .b2GmAddr = weight2,
        .b2ScaleGmAddr = weightScales2,
        .y2GmAddr = yOut,
        .expertTokenNumsOutGmAddr = expertTokenNumsOut,
        .probsGmAddr = topkWeights,
        .workspaceInfo = WorkspaceInfo(workspace, tilingData_),
        .peermemInfo = PeermemInfo(buff[rankId_], tilingData_),
        .shmem = HcclShmem(buff, rankId_, worldSize_),
        .tilingData_ = tilingData
    };
    preSumBeforeRankWorkSpaceGM_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrSumBeforeRank);
    tokenPerExpertWinGM_.SetGlobalBuffer((__gm__ int32_t*)params_.peermemInfo.ptrTokenPerExpert);
    cumsumInWorkSpaceGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.ptrcumsumMM));
    groupListGm_ = cumsumMM[(worldSize_ - 1) * expertPerRank_]; // cumsumMM的最后一行
    groupFlagListWorkSpaceGM_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrFlagList);
 
    ResetWorkSpaceFlagList();
    using L1TileShape = tla::Shape<Int<256>, Int<256>, Int<448>>;
    epilogueOp_ = BlockEpilogueSwigluQuant<L1TileShape, fp8_e4m3fn_t, bfloat16_t, fp8_e8m0_t, fp8_e8m0_t, true>; // swiglu
    epilogueOp_.Init({params_.workspaceInfo.ptrA2, params_.workspaceInfo.ptrA2Scale, params_.workspaceInfo.ptrFlagList,
                    nullptr, nullptr, nullptr, L1_TILE_M, L1_TILE_N});
    strides_ = CeilAlign(worldSize_ * expertPerRank_, ALIGN_128);
    tokenPerExpertCurRank_ = params_.peermemInfo.ptrTokenPerExpert + TokenPerExpertLayout(params_.rankId_, 0, 0) * sizeof(int32_t);
    if (constexpr (QuantMode == 2)){
        if constexpr (QuantOutType == 2) { // 暂定1是fp8_e4m3 2是fp8_e5m2
            quantOutType_ = fp8_e5m2fn_t;
        } else  if constexpr (QuantOutType == 2) {
            quantOutType_ = fp8_e4m3fn_t ;
        }
        quantOutScaleType_ = fp8_e8m0_t;
    }
}
 
template <TemplateMoeDispatchFFNCombineTypeClass>
__aicore__ inline void MoeDispatchFFNCombine<TemplateMoeDispatchFFNCombineTypeFunc>::ResetWorkSpaceFlagList()
{
    // workSpace groupFlagList清零
    groupFlagListGm_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrFlagList);
    if constexpr(g_coreType == AscendC::AIV) {
        int32_t flagNum = expertPerRank_ * INT_CACHELINE * MAX_AICORE_NUM * 2;
        int32_t coreLen, coreOffset;
        TilingByCore(flagNum, coreLen, coreOffset);
        AscendC::LocalTensor<int32_t> initTensor{AscendC::TPosition::VECIN, 0, 0};
        AscendC::Duplicate(initTensor, 0, coreLen);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::DataCopy(groupFlagListGm_[coreOffset], initTensor, coreLen);
    }
}
 
template <TemplateMoeDispatchFFNCombineTypeClass>
__aicore__ inline void MoeDispatchFFNCombine<TemplateMoeDispatchFFNCombineTypeFunc>::GetCumsumForMMAIV()
{
    int32_t expertPerRankAligned = (expertPerRank_ + 8 - 1) / 8 * 8;
    AscendC::LocalTensor<int32_t> tmpBuffer1 = {AscendC::TPosition::VECIN, 0, 0};
    #define U16(x) static_cast<uint16_t>(x)
 
    AscendC::DataCopyPad(
        tmpBuffer1,
        tokenPerExpertWinGM_[rankId_ * expertPerRank_], // win区存放ep域内所有卡的专家cnt情况的起始地址，偏移到本卡的位置上，按照offset取每张卡上要发给本卡专家的数据
        {U16(worldSize_), U16(expertPerRank_ * sizeof(int32_t)), U16((AlignUp(worldSize_ * expertPerRank_, 128) - expertPerRank_) * sizeof(int32_t)), 0},
        {}
    );
 
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
 
    for (uint32_t i = 1; i < worldSize_; ++i) {
        AscendC::Add(tmpBuffer1[i * expertPerRankAligned], tmpBuffer1[i * expertPerRankAligned], tmpBuffer1[(i - 1) * expertPerRankAligned], expertPerRank_);
        AscendC::PipeBarrier<PIPE_V>();
    }
 
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
 
    AscendC::DataCopyPad(cumsumInWorkSpaceGM_, tmpBuffer1, {U16(worldSize_), U16((expertPerRank_) * sizeof(int32_t)), 0, 0}); // workSpace上
}
 
 
template <TemplateMoeDispatchFFNCombineTypeClass>
__aicore__ inline void MoeDispatchFFNCombine<TemplateMoeDispatchFFNCombineTypeFunc>::CrossRankSyncCntPerExpert()
{
    if constexpr(g_coreType == AscendC::AIC) {
        return;
    }
    uint32_t numPerCore = CeilAlign(worldSize_ * expertPerRank_, ALIGN_128);
    AscendC::LocalTensor<int32_t> tmpBuffer{AscendC::TPosition::VECIN, 0, 0};
    AscendC::LocalTensor<int32_t> prevSumBuf = tmpBuffer[numPerCore];
    // 给对端发送
    for(int32_t dstEpIdx = aivCoreIdx; dstEpIdx < worldSize_; dstEpIdx += blockNum) {
        if (dstEpIdx == rankId_) {
            continue;
        }
        GlobalTensor<int32_t> srcAddress;
        srcAddress.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tokenPerExpertCurRank_));
        GlobalTensor<int32_t> dstAddress;
        __gm__ void* dstPeermemPtr = params_.shmem(tokenPerExpertCurRank_, dstEpIdx);
        dstAddress.SetGlobalBuffer((__gm__ int32_t * )dstPeermemPtr);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::DataCopy(tmpBuffer, srcAddress, numPerCore);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Adds(tmpBuffer, tmpBuffer, 0x800000, numPerCore);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopy(dstAddress, tmpBuffer, numPerCore);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    // 本端接收
    int32_t intPer512 = ALIGN_512 / sizeof(int);
    for(int32_t dstEpIdx = aivCoreIdx_; dstEpIdx < worldSize_; dstEpIdx += blockNum_) {
        if (dstEpIdx != rankId_) {
            for(int32_t checkIdx = 0; checkIdx < numPerCore; checkIdx += intPer512) {
                __gm__ int32_t* sync_check = reinterpret_cast<__gm__ int32_t*>(params_.peermemInfo.ptrTokenPerExpert) + TokenPerExpertLayout(dstEpIdx, 0, checkIdx);
                gm_signal_wait_until_ne(sync_check, 0);
            }
            AscendC::DataCopy(tmpBuffer, tokenPerExpertWinGM_[TokenPerExpertLayout(dstEpIdx, 0, 0)], numPerCore);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Adds(tmpBuffer, tmpBuffer, -0x800000, numPerCore);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopy(tokenPerExpertWinGM_[TokenPerExpertLayout(dstEpIdx, 0, 0)], tmpBuffer, numPerCore);
        } else {
            AscendC::DataCopy(tmpBuffer, tokenPerExpertWinGM_[TokenPerExpertLayout(dstEpIdx, 0, 0)], numPerCore);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        int32_t tmpSum = 0;
        int32_t j = 0;
        for (int32_t i = 0; i < (rankId_ + 1) * expertPerRank_; i++) {
            if (i >= rankId_ * expertPerRank_) {
                prevSumBuf(j) = tmpSum;
                j++;
            }
            tmpSum += tmpBuffer(i);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyPad(preSumBeforeRankWorkSpaceGM_[dstEpIdx * expertPerRank_], prevSumBuf,
        DataCopyParams{1, static_cast<uint16_t>(expertPerRank_ * sizeof(int32_t)), 0, 0});
    }
    AscendC::SyncAll<true>();
    if (aivCoreIdx == 0) {
        GetCumsumForMMAIV();
    }
    if (blockIdx_ < worldSize_ * dispatchBlockNumPerEP_) {
        prevSum_ = preSumBeforeRankWorkSpaceGM_(blockIdx_ / dispatchBlockNumPerEP_ * expertPerRank_);
    }
}
 
template <TemplateMoeDispatchFFNCombineTypeClass>
__aicore__ inline void MoeDispatchFFNCombine<TemplateMoeDispatchFFNCombineTypeFunc>::CopyGMToGMPerTokenV2(GlobalTensor<quantOutType_> dst, GlobalTensor<quantOutScaleType_> dstScale,
    GlobalTensor<quantOutType_> src, int32_t rows, int32_t hiddenSize, int32_t& pingpongId)
{
    constexpr int32_t BufferNum = 2;
    AscendC::LocalTensor<quantOutType_> tmpBuffer1{AscendC::TPosition::VECIN, 0, 0};
    AscendC::LocalTensor<quantOutType_> tmpBuffer2{AscendC::TPosition::VECIN, 96 * 1024, 0};
    uint32_t scaleLen = CeilDiv(hiddenSize, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
    uint32_t copyInNum = hiddenSize + scaleLen;
    auto processCount = CeilDiv(rows, dispatchRows_);
    for (uint32_t processIndex = 0; processIndex < processCount; ++processIndex) {
        pingpongId = (pingpongId + 1) % BufferNum;
        AscendC::TEventID EVENT_ID = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
        AscendC::LocalTensor<quantOutType_> buf = pingpongId == 0 ? tmpBuffer1 : tmpBuffer2;
        AscendC::LocalTensor<Weight2ScaleType> bufScale = buf[hiddenSize].template ReinterpretCast<Weight2ScaleType>();
        
        auto inputOffset = processIndex * dispatchRows_ * copyInNum;
        
        int32_t rowNum = dispatchRows_;
        if (processIndex == processCount - 1) {
            rowNum = rows - processIndex * dispatchRows_;
        }
 
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        int64_t dataLen = rowNum * copyInNum;
        AscendC::DataCopy(buf, src[inputOffset], dataLen);
 
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        auto outputOffset = processIndex * dispatchRows_ * hiddenSize;
        #define U16(x) static_cast<uint16_t>(x)
        AscendC::DataCopyPad(dst[outputOffset], 
            buf, {U16(rowNum), U16(hiddenSize), U16(scaleLen / 32), 0, 0});
 
        AscendC::DataCopyPad(dstScale[processIndex * dispatchRows_ * scaleLen], 
            bufScale, {U16(rowNum), U16(scaleLen), U16(hiddenSize / 32), 0, 0});
 
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
    }
}
 
 
template <TemplateMoeDispatchFFNCombineTypeClass>
__aicore__ inline void MoeDispatchFFNCombine<TemplateMoeDispatchFFNCombineTypeFunc>::DispatchTokens(uint32_t expertIdx)
{
    int64_t widthA = Get<K_VALUE>(problemShape_); // H长度
    int64_t widthAScale = CeilDiv(widthA, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
    auto &tilingData = params_.tilingData;
    int32_t pingpongId = 0;
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for(int32_t idx = blockIdx; idx < tilingData->worldSize_ * dispatchBlockNumPerEP; idx += blockNum) {
        uint32_t dstEpIdx = idx / dispatchBlockNumPerEP;
        uint32_t idxInEp = idx % dispatchBlockNumPerEP;
        uint32_t rowStart = (dstEpIdx == 0 ? 0 : cumsumMM((dstEpIdx - 1) * tilingData->expertPerRank_ + expertIdx));
        if (rowStart < tilingData->maxOutputSize) {
            uint32_t rows = tokenPerExpertWinGM_(TokenPerExpertLayout(dstEpIdx, tilingData->rank, expertIdx));
            if (rowStart + rows > tilingData->maxOutputSize) {
                rows = tilingData->maxOutputSize - rowStart;
            }
            uint32_t rowSrc = prevSum_;
            prevSum_ += rows;
            uint32_t rowsPerCore = CeilDiv(rows, dispatchBlockNumPerEP);
            uint32_t rowStartPerCore = rowStart + idxInEp * rowsPerCore;
            uint32_t rowSrcPerCore = rowSrc + idxInEp * rowsPerCore;
            if (rowStartPerCore + rowsPerCore > rowStart + rows) {
                rowsPerCore = rowStart + rows - rowStartPerCore;
            }
            if (rowsPerCore > 0) {
                GlobalTensor <quantOutType_> gmRemoteA;
                gmRemoteA.SetGlobalBuffer(reinterpret_cast<__gm__ quantOutType_ * > (params_.shmem(params_.peermemInfo.ptrA0, dstEpIdx)));
                int64_t gmOffsetA = rowStartPerCore * widthA;
                int64_t gmOffsetAScale = rowStartPerCore * widthAScale;
                int64_t gmOffsetPeer = rowSrcPerCore * (widthA + widthAScale);
                GM_ADDR aScalePtr = (GM_ADDR) aScaleGlobal[gmOffsetAScale].GetPhyAddr();
                CopyGMToGMPerTokenV2(aGlobal[gmOffsetA], aScaleGlobal[gmOffsetAScale], gmRemoteA[gmOffsetPeer], rowsPerCore, widthA, pingpongId);
            }
        }
        AtomicAddFlag((__gm__ int32_t*)groupFlagListGm2_.GetPhyAddr(), MAX_AICORE_NUM * INT_CACHELINE); // TODO 不需要这么大
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}
 
 
template <TemplateMoeDispatchFFNCombineTypeClass>
__aicore__ inline bool MoeDispatchFFNCombine<TemplateMoeDispatchFFNCombineTypeFunc>::UpdateGroupParams(uint32_t expertIdx)
{
    // baseOffset_ is 0 when expertIdx = 0
    if (expertIdx == 0) {
        return;
    }
    uint64_t m = Get<M_VALUE>(problemShape_); // 待确认 这个m没有初始值
    uint64_t n = Get<N_VALUE>(problemShape_);
    uint64_t k = Get<K_VALUE>(problemShape_);
    Get<IDX_A_OFFSET>(baseOffset_) += m * k;
    Get<IDX_B_OFFSET>(baseOffset_) += n * k;
    // only splitM
    auto scaleK = CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
    // scaleAAxisBaseOffset (m, ceil(k,64), 2)
    Get<IDX_A_SCALE_OFFSET>(baseOffset_) += m * scaleK;
    // scaleBAxisBaseOffset (g, n, ceil(k,64), 2) or (g, ceil(k,64), n, 2)
    Get<IDX_B_SCALE_OFFSET>(baseOffset_) += n * scaleK;
    Get<IDX_C_OFFSET>(baseOffset_) += m * n / SWIGLU_N_HALF;
    Get<IDX_C_SCALE_OFFSET>(baseOffset_) +=
        m * CeilDiv(n / SWIGLU_N_HALF, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
    Get<IDX_FLAG_OFFSET>(baseOffset_) += 1;
    Get<IDX_B2_OFFSET>(baseOffset_) += k * n / SWIGLU_N_HALF;
    Get<IDX_B2_SCALE_OFFSET>(baseOffset_) += k * CeilDiv(n / SWIGLU_N_HALF, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
    Get<IDX_Y2_OFFSET>(baseOffset_) += m * k;
    Get<IDX_M_OFFSET>(baseOffset_) += m;
 
    int32_t splitValue = 0;
    if (groupListType_ == 0) {
        int32_t offset = static_cast<int32_t>(groupListGm.GetValue(expertIdx));
        splitValue = offset - preOffset;
        preOffset = offset;
    } else {
        splitValue = static_cast<uint64_t>(groupListGm.GetValue(expertIdx));
    }
    Get<M_VALUE>(problemShape_) = splitValue;
 
    // split_m，when m=0, skip 这里groupListType写死1, 就是没收到值就跳过
    if (Get<M_VALUE>(problemShape_) == 0) {
        return false;
    }
    return true;
}
 
template <TemplateMoeDispatchFFNCombineTypeClass>
__aicore__ inline void MoeDispatchFFNCombine<TemplateMoeDispatchFFNCombineTypeFunc>::UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo, uint32_t gmmIndex)
{
    if (gmmIndex == 0) {
        gmmAddrInfo.aGlobal = params_.workspaceInfo.ptrA + Get<IDX_A_OFFSET>(baseOffset_) * sizeof(quantOutType_);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.ptrAScale + Get<IDX_A_SCALE_OFFSET>(baseOffset_) * sizeof(Weight2ScaleType);
        if constexpr(g_coreType == AscendC::AIC) {
            gmmAddrInfo.bGlobal = params_.bGmAddr + Get<IDX_B_OFFSET>(baseOffset_) * sizeof(Weight1Type);
            gmmAddrInfo.bScaleGlobal = params_.bScaleGmAddr + Get<IDX_B_SCALE_OFFSET>(baseOffset_) * sizeof(WeightScale1Type); 
        } else {
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                Get<IDX_C_OFFSET>(baseOffset_), Get<IDX_C_SCALE_OFFSET>(baseOffset_),
                Get<IDX_FLAG_OFFSET>(baseOffset_), 0L, 0L, 0L};
            epilogueOp_.UpdateGlobalAddr(vecBaseOffset);
        }
    } else if constexpr(g_coreType == AscendC::AIC) {
        gmmAddrInfo.aGlobal = params_.workspaceInfo.ptrA2 + Get<IDX_C_OFFSET>(baseOffset_) * sizeof(quantOutType_);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.ptrA2Scale + Get<IDX_C_SCALE_OFFSET>(baseOffset_) * sizeof(quantOutScaleType_);
        gmmAddrInfo.bGlobal = params_.b2GmAddr + Get<IDX_B2_OFFSET>(baseOffset_) * sizeof(Weight2Type2);
        gmmAddrInfo.bScaleGlobal = params_.b2ScaleGmAddr + Get<IDX_B2_SCALE_OFFSET>(baseOffset_) * sizeof(WeightScale2Type2);
    }
 
    groupFlagListGm_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrFlagList + Get<IDX_FLAG_OFFSET>(baseOffset_) * INT_CACHELINE * MAX_AICORE_NUM);
    groupFlagListGm2_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrFlagList2 + Get<IDX_FLAG_OFFSET>(baseOffset_) * INT_CACHELINE * MAX_AICORE_NUM);
}

template <TemplateMoeDispatchFFNCombineTypeClass>
__aicore__ inline void MoeDispatchFFNCombine<TemplateMoeDispatchFFNCombineTypeFunc>::ArgSortExpandedRowIdx()
{
    int32_t coreLen, coreOffset;
    TilingByCore(m_ * topK_, coreLen, coreOffset, 1);
    __gm__ int32_t *src = (__gm__ int32_t *)params_.workspaceInfo.expandedRowIdx;
    __gm__ int32_t *dst = (__gm__ int32_t *)params_.workspaceInfo.expandedRowIdxGather;
    Simt::VF_CALL<ArgsortSimt>(Simt::Dim3{MoeInitRoutingV3::SIMT_THREAD_NUM, 1, 1}, coreLen, coreOffset,
                                src, dst);
}
 
template <TemplateMoeDispatchFFNCombineTypeClass>
__aicore__ inline void MoeDispatchFFNCombine<TemplateMoeDispatchFFNCombineTypeFunc>::ResetTokenPerExpert()
{
    if (aivCoreIdx != blockNum - 1) {
        return;
    }
    int32_t num = worldSize_ * CeilAlign(worldSize_ * expertPerRank_, ALIGN_128);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    AscendC::LocalTensor<int32_t> tmp = resource.ubBuf.template GetBufferByByte<int32_t>(0); // 这啥玩意？
    AscendC::Duplicate(tmp, 0, num);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::DataCopy(tokenPerExpert, tmp, num);
}
 
template <TemplateMoeDispatchFFNCombineTypeClass>
__aicore__ inline void MoeDispatchFFNCombine<TemplateMoeDispatchFFNCombineTypeFunc>::Unpermute()
{
    if constexpr(g_coreType == AscendC::AIC) {
        return;
    }
    int32_t coreLen, coreOffset;
    TilingByCore(m_, coreLen, coreOffset, 1);
    AscendC::GlobalTensor<int32_t> expandedRowIdx;
    expandedRowIdx.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.expandedRowIdxGather);
    AscendC::GlobalTensor<bfloat16_t> probs;
    probs.SetGlobalBuffer((__gm__ bfloat16_t*)params_.probsGmAddr);
    AscendC::GlobalTensor<bfloat16_t> expandedX;
    expandedX.SetGlobalBuffer((__gm__ bfloat16_t*)params_.peermemInfo.ptrD);
    AscendC::GlobalTensor<bfloat16_t> output;
    output.SetGlobalBuffer((__gm__ bfloat16_t*)params_.y2GmAddr);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    for(int32_t tokenIdx = coreOffset; tokenIdx < coreLen + coreOffset; tokenIdx++) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::LocalTensor<bfloat16_t> dataRes{AscendC::TPosition::VECIN, 0, 0};
        AscendC::LocalTensor<bfloat16_t> dataIn0 = dataRes[k_];
        AscendC::LocalTensor<bfloat16_t> dataIn1 = dataRes[k_ * 2];
        for(int32_t expId = 0; expId < topK_; ++expId) {
            int32_t rowIdx = expandedRowIdx(tokenIdx * topK_+ expId);
            bfloat16_t expScale = probs(tokenIdx * topK_ + expId);
            auto event = (expId % 2 == 0) ? EVENT_ID0 : EVENT_ID1;
            auto dataIn = (expId % 2 == 0) ? dataIn0 : dataIn1;
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::DataCopy(dataIn, expandedX[rowIdx * k_], k_);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event);
            AscendC::SetFlag<AscendC::HardEvent::S_V>(event);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(event);
            if (expId == 0) {
                AscendC::Muls(dataRes, dataIn, expScale, k_);
            } else {
                AscendC::Muls(dataIn, dataIn, expScale, k_);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(dataRes, dataRes, dataIn, k_);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopy(output[tokenIdx * k_], dataRes, k_);
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
}
 
template <TemplateMoeDispatchFFNCombineTypeClass>
__aicore__ inline void MoeDispatchFFNCombine<TemplateMoeDispatchFFNCombineTypeFunc>::Process()
{
    // 1. moe_init_routing_v3调用
    moe_init_routing_v3<bfloat16_t, fp8_e4m3fn_t>(params_.aGmAddr, params_.expertIdxGmAddr, nullptr, nullptr,
            params_.peermemInfo.ptrA0, params_.workspaceInfo.expandedRowIdx, tokenPerExpertCurRank_, nullptr,
            params_.workspaceInfo.ptrA, &params_.tilingData_->moeInitRoutingTilingData,
            params_.tilingData_->moeInitRoutingTilingData.tilingKey);
    SyncAll<true>();
 
    // 2. cnt值全ep域内的发送以及本端接收整理
    CrossRankSyncCntPerExpert();
    SyncAll<false>();
 
    // 3. for循环按照本卡专家id进行Disptch，内部M1SwigluQuant按照基本块进行流水
    GMMAddrInfo gmmAddrInfo;
    for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; expertIdx++) {
        if (!UpdateGroupParams(expertIdx)) {
            continue;
        }
        UpdateGlobalBuffer(gmmAddrInfo, 0);
        if (subBlockIdx_ == 1) {
            DispatchTokens(expertIdx);
        }
        if (subBlockIdx_ == 0) {
            MoeDispatchFFNCombineImpl::GroupMatmulSwigluQuant<XType, Weight1Type, GMM1OutType, WeightScale1Type, WeightScale1Type>(
                epilogueOp_, params_, problemShape_, gmmAddrInfo, startBlockIdx_, vecSetSyncCom_);
        }
    }
    EndSync(vecSetSyncCom_);
 
    // 4. for循环按照本卡专家id进行gmm2 & combine，内部gmm2与combine之间按照基本块进行流水
    preOffset_ = 0;
    vecSetSyncCom_ = 0;
    baseOffset_ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; expertIdx++) {
        if (!UpdateGroupParams(expertIdx)) {
            continue;
        }
        UpdateGlobalBuffer(gmmAddrInfo, 1);
        if (subBlockIdx_ == 0) {
            MoeDispatchFFNCombineImpl::GroupMatmul2<GMM1OutType, Weight2Type, OutputType, WeightScale2Type, WeightScale2Type>(
                params_, problemShape_, gmmAddrInfo, startBlockIdx_, vecSetSyncCom_, expertIdx);
        }
    }
    EndGMM2Sync(vecSetSyncCom_);
    AscendC::SyncAll<true>();
 
    // 5. unpermute & 清理win区上放置token的位置
    ArgSortExpandedRowIdx();
    if constexpr(g_coreType == AscendC::AIV) {
        ResetTokenPerExpert();
        params_.shmem.CrossRankSync();
    }
    Unpermute(params_);
}
 
}   // namespace MoeDispatchFFNCombineImpl
#endif