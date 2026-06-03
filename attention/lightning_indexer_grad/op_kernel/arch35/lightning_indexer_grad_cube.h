#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "simt_api/math_functions.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/device_functions.h"
#include "lightning_indexer_grad_simd.h"
#include "lightning_indexer_grad_base.h"
#include "lightning_indexer_grad_cube.h"

using namespace AscendC;
using namespace MicroAPI;

namespace {
constexpr int32_t DOUBLE_BUF = 2;
constexpr int32_t L1_ALLOC_SIZE = 512 * 1024;
constexpr int32_t BASE_N_ALIGNED = 32;
}

namespace LIG {
template<typename T>
class LightningIndexerGradKernel {
public:
    __aicore__ inline LightningIndexerGradKernel() = default;

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR dy,
        GM_ADDR sparseIndices, GM_ADDR weights, GM_ADDR actualSeqLenQuery,
        GM_ADDR actualSeqLenKey, GM_ADDR dq, GM_ADDR dk, GM_ADDR dweights,
        GM_ADDR workspace, LightningIndexerGradArch35TilingData *tilingData,  TPipe* pipe)
    {
        pipe_ = pipe;
        if ASCEND_IS_AIV {
            blkIdx_ =  GetBlockIdx() / 2;
            subBlkIdx_ = GetSubBlockIdx();
        } else {
            blkIdx_ =  GetBlockIdx();
            subBlkIdx_ = 0;
        }

        InitTiling(tilingData);
        InitGM(query, key, dy, sparseIndices, weights, actualSeqLenQuery,
            actualSeqLenKey, dq, dk, dweights, workspace);
        InitBuffer();
    }
    __aicore__ inline void Process();

private:
    TPipe* pipe_;
    int64_t blkIdx_, subBlkIdx_;
    int64_t aivNum_, aicNum_;
    int64_t batchSize_;
    int64_t headNumQ_;
    int64_t headDim_;
    int64_t querySeqLen_;
    int64_t keySeqLen_;
    int32_t tileTopK_, tileTopKAligned_;
    int64_t topK_;
    int64_t inputLayout_;
    int64_t totalQuerySeqLength_;
    int64_t totalKeySeqLength_;
    int64_t tileSeqQ_;
    int64_t topKAligned_;
    int64_t sparseMode_;
    
    GlobalTensor<int32_t> sparseIndicesGlobal_;
    GlobalTensor<int32_t> actualSeqLenQueryGlobal_;
    GlobalTensor<int32_t> actualSeqLenKeyGlobal_;

    GlobalTensor<T> queryGlobal_;
    GlobalTensor<T> keyGlobal_;
    GlobalTensor<T> gradValueGlobal_;
    GlobalTensor<T> weightsGlobal_;
    GlobalTensor<T> gradQueryGlobal_;
    GlobalTensor<T> gradKeyGlobal_;
    GlobalTensor<T> gradWeightsGlobal_;

    LocalTensor<T> queryL1Local_;
    LocalTensor<T> gradSelectProbL1Local_;
    LocalTensor<T> selectKeyLocal_;
    LocalTensor<T> selectKeyL1Local_;
    LocalTensor<float> selectProbLocal_;
    LocalTensor<int32_t> sparseIndicesLocal_;
    LocalTensor<T> gradValueLocal_;
    LocalTensor<float> gradValueFloatLocal_;
    LocalTensor<float> gradWeightsFloatLocal_;
    LocalTensor<T> weightsLocal_;
    LocalTensor<float> weightsFloatLocal_;
    LocalTensor<float> gradSelectProbFloatLocal_;
    LocalTensor<T> gradSelectProbNZLocal_;
    LocalTensor<float> gradKeyLocal_;
    
    TBuf<TPosition::VECOUT> selectKeyBuf_;
    TBuf<TPosition::VECCALC> sparseIndicesBuf_;
    TBuf<TPosition::VECOUT> selectProbBuf_;
    TBuf<TPosition::VECCALC> gradValueBuf_;
    TBuf<TPosition::VECCALC> gradValueFloatBuf_;
    TBuf<TPosition::VECCALC> gradWeightsFloatBuf_;
    TBuf<TPosition::VECCALC> mmTmpWorkspaceBuf_;
    TBuf<TPosition::VECCALC> weightsBuf_;
    TBuf<TPosition::VECCALC> weightsFloatBuf_;
    TBuf<TPosition::VECCALC> gradSelectProbFloatBuf_;
    TBuf<TPosition::VECCALC> gradSelectProbNZBuf_;
    TBuf<TPosition::VECCALC> gradKeyBuf_;

private:
    __aicore__ inline void InitTiling(
        const LightningIndexerGradArch35TilingData* tilingData)
    {
        aivNum_ = tilingData->aivNum;
        aicNum_ = aivNum_ / 2;
        inputLayout_ = tilingData->inputLayout;
        totalQuerySeqLength_ = tilingData->totalQuerySeqLength;
        totalKeySeqLength_ = tilingData->totalKeySeqLength;
        tileSeqQ_ = tilingData->tileSeqQ;
        tileTopK_ = tilingData->tileTopK;
        topK_ = tilingData->topK;
        querySeqLen_ = tilingData->querySeqLen;
        keySeqLen_ = tilingData->keySeqLen;
        headDim_ = tilingData->headDim;
        headNumQ_ = tilingData->headNumQ;
        batchSize_ = tilingData->batchSize;
        topKAligned_ = AlignUp(topK_, UB_BLOCK_BYTE_SIZE / sizeof(int32_t));
        sparseMode_ = tilingData->sparseMode;
    }

    __aicore__ inline void InitGM(
        GM_ADDR query, GM_ADDR key, GM_ADDR dy, GM_ADDR sparseIndices,
        GM_ADDR weights, GM_ADDR actualSeqLenQuery, GM_ADDR actualSeqLenKey,
        GM_ADDR dq, GM_ADDR dk, GM_ADDR dweights, GM_ADDR workspace)
    {
        queryGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(query));
        keyGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(key));
        gradValueGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dy));
        sparseIndicesGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sparseIndices));
        weightsGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(weights));
        gradQueryGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dq));
        gradKeyGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dk));
        gradWeightsGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dweights));

        if (inputLayout_ == 1) {
            // TND need
            actualSeqLenQueryGlobal_.SetGlobalBuffer(
                reinterpret_cast<__gm__ int32_t*>(actualSeqLenQuery));
            actualSeqLenKeyGlobal_.SetGlobalBuffer(
                reinterpret_cast<__gm__ int32_t*>(actualSeqLenKey));
        }
        
        if ASCEND_IS_AIV {
            int32_t totalTask = totalQuerySeqLength_;
            int32_t aivIdx = blkIdx_ * 2 + subBlkIdx_;
            int32_t bigCoreCount = totalTask % aivNum_;
            int32_t avgTaskCount = totalTask / aivNum_;
            int32_t taskOffset = aivIdx < bigCoreCount?
                aivIdx * (avgTaskCount + 1) : bigCoreCount * (avgTaskCount + 1) + (aivIdx - bigCoreCount) * avgTaskCount;
            int32_t curAivCoreTaskCount = aivIdx < bigCoreCount? avgTaskCount + 1: avgTaskCount;

            GlobalTensor<T> tmpGlobal1 = gradWeightsGlobal_[taskOffset * headNumQ_];
            Fill(tmpGlobal1, curAivCoreTaskCount * headNumQ_, static_cast<T>(0));

            totalTask = totalKeySeqLength_;
            bigCoreCount = totalTask % aivNum_;
            avgTaskCount = totalTask / aivNum_;
            taskOffset = aivIdx < bigCoreCount?
                aivIdx * (avgTaskCount + 1) : bigCoreCount * (avgTaskCount + 1) + (aivIdx - bigCoreCount) * avgTaskCount;
            curAivCoreTaskCount = aivIdx < bigCoreCount? avgTaskCount + 1: avgTaskCount;
            GlobalTensor<T> tmpGlobal2 = gradKeyGlobal_[taskOffset * headDim_];
            Fill(tmpGlobal2, curAivCoreTaskCount * headDim_, static_cast<T>(0));
            SyncAll();
        }
    }

    __aicore__ inline void InitBuffer()
    {
        int32_t tileTopKAlignedB32 = AlignUp(tileTopK_, UB_BLOCK_BYTE_SIZE / sizeof(float));
        int32_t tileTopKAlignedB16 = AlignUp(tileTopK_, UB_BLOCK_BYTE_SIZE / sizeof(T));

        pipe_->InitBuffer(selectProbBuf_,
            tileTopKAlignedB32 * headNumQ_ * sizeof(float));
        pipe_->InitBuffer(gradSelectProbFloatBuf_,
            tileTopKAlignedB32 * headNumQ_ * sizeof(float));
        pipe_->InitBuffer(gradSelectProbNZBuf_,
            tileTopKAlignedB32 * headNumQ_ * sizeof(T));
        
        pipe_->InitBuffer(sparseIndicesBuf_, topKAligned_ * sizeof(int32_t));
        pipe_->InitBuffer(gradValueBuf_, tileTopKAlignedB16 * sizeof(T));
        pipe_->InitBuffer(weightsBuf_, headNumQ_ * sizeof(T));
        pipe_->InitBuffer(weightsFloatBuf_, headNumQ_ * sizeof(float));
        pipe_->InitBuffer(gradValueFloatBuf_, tileTopKAlignedB32 * sizeof(float));
        pipe_->InitBuffer(gradWeightsFloatBuf_, headDim_ * sizeof(float));
        pipe_->InitBuffer(selectKeyBuf_,
            tileTopKAlignedB16 * headDim_ * sizeof(T));
        pipe_->InitBuffer(gradKeyBuf_,
            tileTopKAlignedB16 * headDim_ * sizeof(float));

        sparseIndicesLocal_ = sparseIndicesBuf_.Get<int32_t>();
        selectProbLocal_ = selectProbBuf_.Get<float>();
        gradSelectProbFloatLocal_ = gradSelectProbFloatBuf_.Get<float>();
        gradSelectProbNZLocal_ = gradSelectProbNZBuf_.Get<T>();
        gradValueLocal_ = gradValueBuf_.Get<T>();
        gradValueFloatLocal_ = gradValueFloatBuf_.Get<float>();
        gradWeightsFloatLocal_ = gradWeightsFloatBuf_.Get<float>();
        selectKeyLocal_ = selectKeyBuf_.Get<T>();
        weightsLocal_ = weightsBuf_.Get<T>();
        weightsFloatLocal_ = weightsFloatBuf_.Get<float>();
        gradKeyLocal_ = gradKeyBuf_.Get<float>();

        
        TBuf<TPosition::A1> l1Buffer;
        pipe_->InitBuffer(l1Buffer, L1_ALLOC_SIZE);
        
        selectKeyL1Local_ = l1Buffer.Get<T>();
        queryL1Local_ = selectKeyL1Local_[tileTopKAlignedB16 * headDim_ * 2];
        gradSelectProbL1Local_ = queryL1Local_[headNumQ_ * headDim_];
    }

    __aicore__ inline void GatherKey(
        const int32_t tileTopKTask, const int32_t sparseIndicesOffset,
        const int32_t innerBatchQueryOffset, const int32_t curBatchKeyOffset);
    __aicore__ inline void ScatterKey(const int32_t tileTopKTask,
        const int32_t sparseIndicesOffset, const int32_t curBatchKeyOffset);
    __aicore__ inline void CopyInKey(const int32_t taskOffset, const int32_t queryTaskCount);
    __aicore__ inline void CopyInSparseIndices(
        const int32_t queryOffset, const int32_t queryTaskCount);
    __aicore__ inline void ProcessTileVector(
        const int32_t curBatchQueryOffset, const int32_t innerBatchQueryOffset,
        const int32_t curBatchKeyOffset, const int32_t queryTaskCount);
    __aicore__ inline int32_t GetRealTopK(const int32_t queryOffset);
    __aicore__ inline void QueryMatmulSelectKeyVector(
        const int32_t queryOffset, const int32_t tileTopKTask, const int32_t aiCoreTopKTask);
    __aicore__ inline void ComputeGradWeight(
        const int32_t queryOffset, const int32_t topKOffset,
        const int32_t tileTopKTask, const bool initGradWeights);
    __aicore__ inline void ComputeGradRelu(
        const int32_t queryOffset, const int32_t tileTopKTask);
    __aicore__ inline void ComputeGradQuery(
        const int32_t queryOffset, const int32_t tileTopKTask,
        const int32_t aiCoreTopKTask, const bool firstTileFlag,
        const int32_t lastTileFlag);
};

template<typename T>
__aicore__ inline void LightningIndexerGradKernel<T>::Process()
{
    int32_t execCoreIdx = 0;
    for (int32_t bIdx = 0; bIdx < batchSize_; bIdx++) {

        int32_t curBatchQueryOffset = bIdx * querySeqLen_;
        int32_t curBatchKeyOffset = bIdx * keySeqLen_;

        if (inputLayout_ == 1) {
            // TND
            curBatchQueryOffset = bIdx == 0? 0 : actualSeqLenQueryGlobal_.GetValue(bIdx - 1);
            curBatchKeyOffset = bIdx == 0? 0 : actualSeqLenKeyGlobal_.GetValue(bIdx - 1);
            querySeqLen_ = actualSeqLenQueryGlobal_.GetValue(bIdx) - curBatchQueryOffset;
            keySeqLen_ = actualSeqLenKeyGlobal_.GetValue(bIdx) - curBatchKeyOffset;
        }
        
        for (int32_t innerBatchQueryOffset = 0;
            innerBatchQueryOffset < querySeqLen_; innerBatchQueryOffset += tileSeqQ_) {
            int32_t queryTaskCount = min(querySeqLen_ - innerBatchQueryOffset, tileSeqQ_);

            if (blkIdx_ == execCoreIdx) {
                ProcessTileVector(curBatchQueryOffset, innerBatchQueryOffset, curBatchKeyOffset,
                    queryTaskCount);
            }

            execCoreIdx = (execCoreIdx + 1) % aicNum_;
        }
    }
}

template<typename T>
__aicore__ inline void LightningIndexerGradKernel<T>::ProcessTileVector(
    const int32_t curBatchQueryOffset, const int32_t innerBatchQueryOffset,
    const int32_t curBatchKeyOffset, const int32_t queryTaskCount)
{
    
    for (int32_t i = 0; i < queryTaskCount; i++) {
        bool initGradWeights = true;
        int32_t realTopK = GetRealTopK(innerBatchQueryOffset + i);
        int32_t realTopKAligned = AlignUp(realTopK, tileTopK_);
        int32_t queryOffset = curBatchQueryOffset + innerBatchQueryOffset + i;

        CopyInSparseIndices(queryOffset, 1);

        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);

        for (int32_t topKOffset = 0;
                topKOffset < realTopK; topKOffset += tileTopK_) {
            int32_t curCoreTaskCount = min(realTopK - topKOffset, tileTopK_);
            int32_t tileTopKTask = subBlkIdx_ == 0? CeilDiv(curCoreTaskCount, 2) : curCoreTaskCount / 2;
            int32_t realTopKOffset = subBlkIdx_ == 0? topKOffset : topKOffset + CeilDiv(curCoreTaskCount, 2);

            GatherKey(tileTopKTask, realTopKOffset,
                innerBatchQueryOffset, curBatchKeyOffset);
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);

            QueryMatmulSelectKeyVector(queryOffset, tileTopKTask, curCoreTaskCount);
            ComputeGradWeight(queryOffset, realTopKOffset, tileTopKTask, initGradWeights);
            ComputeGradRelu(queryOffset, tileTopKTask);
            ComputeGradQuery(queryOffset, tileTopKTask,
                curCoreTaskCount, topKOffset == 0, topKOffset + tileTopK_ >= realTopK);
            
            ScatterKey(tileTopKTask, realTopKOffset, curBatchKeyOffset);
            initGradWeights = false;
        }

        // CopyOutGradWeight
        Cast(gradWeightsFloatLocal_.template ReinterpretCast<T>(),
            gradWeightsFloatLocal_, RoundMode::CAST_ROUND, headNumQ_);

        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);

        SetAtomicAdd<T>();
        DataCopyExtParams copyParams{static_cast<uint16_t>(1),
            static_cast<uint32_t>(headNumQ_ * sizeof(T)), 0, 0, 0};
        DataCopyPad(gradWeightsGlobal_[queryOffset * headNumQ_],
            gradWeightsFloatLocal_.template ReinterpretCast<T>(), copyParams);
        SetAtomicNone();
    }
}

template<typename T>
__aicore__ inline void LightningIndexerGradKernel<T>::CopyInSparseIndices(
    const int32_t queryOffset, const int32_t queryTaskCount)
{
    if (queryTaskCount <= 0)
        return;

    if ASCEND_IS_AIV {
        DataCopyExtParams copyParams{static_cast<uint16_t>(queryTaskCount),
            static_cast<uint32_t>(topK_ * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(sparseIndicesLocal_, sparseIndicesGlobal_[topK_ * queryOffset],
            copyParams, padParams);
    }
}

template<typename T>
__aicore__ inline int32_t LightningIndexerGradKernel<T>::GetRealTopK(const int32_t queryOffset)
{
    int64_t realTopK =  0;
    if (sparseMode_ == 0) {
        realTopK = min(keySeqLen_, topK_);
    } else {
        // sparseMode_ = 3
        realTopK = (keySeqLen_ - querySeqLen_ + queryOffset + 1);
        realTopK = realTopK < 0? 0 : min(topK_, realTopK);
    }

    return realTopK;
}

template<typename T>
__aicore__ inline void LightningIndexerGradKernel<T>::GatherKey(
    const int32_t tileTopKTask, const int32_t sparseIndicesOffset,
    const int32_t innerBatchQueryOffset, const int32_t curBatchKeyOffset)
{
    if (tileTopKTask <= 0)
        return;

    if ASCEND_IS_AIV {
        uint32_t tileTopKTaskAligned = AlignUp(tileTopKTask, UB_BLOCK_BYTE_SIZE / sizeof(T));
        uint32_t xThreadCount = headDim_ / (sizeof(float4) / sizeof(T));
        uint32_t thread_num = min(static_cast<uint32_t>(THREAD_NUM),
            static_cast<uint32_t>(tileTopKTaskAligned * xThreadCount));
        uint32_t yThreadCount = thread_num / xThreadCount;
        
        Simt::VF_CALL<GatherKeyNZSimt<T>>(Simt::Dim3{xThreadCount, yThreadCount},
            (__ubuf__ float4*) selectKeyLocal_.GetPhyAddr(),
            (__ubuf__ int32_t*) sparseIndicesLocal_[sparseIndicesOffset].GetPhyAddr(),
            (__gm__ float4*) keyGlobal_[curBatchKeyOffset * headDim_].GetPhyAddr(),
            tileTopKTask);
    }
}

template<typename T>
__aicore__ inline void LightningIndexerGradKernel<T>::QueryMatmulSelectKeyVector(
    const int32_t queryOffset, const int32_t tileTopKTask, const int32_t aiCoreTopKTask)
{
    if (aiCoreTopKTask == 0)
        return;
    
    if ASCEND_IS_AIV {
        int32_t tileTopKTaskAligned = AlignUp(tileTopKTask == 0? aiCoreTopKTask : tileTopKTask, 16);
        DataCopyParams copyParams;
        copyParams.blockCount = headDim_ / 16;
        copyParams.blockLen = tileTopKTaskAligned;
        copyParams.srcGap = 0;
        copyParams.dstGap =
            AlignUp(aiCoreTopKTask, BASE_N_ALIGNED) - tileTopKTaskAligned;

        if (tileTopKTask == 0) {
            InitConstValue(selectKeyLocal_,
                {1, static_cast<uint16_t>(tileTopKTaskAligned * headDim_ * sizeof(T) / 32), 0, 0.0f});
        }

        if (GetSubBlockIdx() == 0) {
            // AIV 0
            DataCopy(selectKeyL1Local_, selectKeyLocal_, copyParams);
        } else {
            // AIV 1
            int32_t selectKeyOffset = AlignUp(aiCoreTopKTask, BASE_N_ALIGNED) * 16 / 2;
            DataCopy(selectKeyL1Local_[selectKeyOffset], selectKeyLocal_, copyParams);
        }
        CrossCoreSetFlag<0x2, PIPE_MTE3>(0);
    }
    
    if ASCEND_IS_AIC {
        LightningIndexerGradCube1Service<T, float> mmService1;
        
        mmService1.Init(headNumQ_, AlignUp(aiCoreTopKTask, BASE_N_ALIGNED), headDim_,
            headNumQ_, AlignUp(aiCoreTopKTask, BASE_N_ALIGNED), headDim_);
        mmService1.CopyInA1Nd2Nz(queryL1Local_,
            queryGlobal_[queryOffset * headDim_ * headNumQ_]);
        SetFlag<HardEvent::MTE2_MTE1>(0);
        WaitFlag<HardEvent::MTE2_MTE1>(0);
        CrossCoreWaitFlag<0x2>(0);
        mmService1.CopyInA2(queryL1Local_);
        mmService1.CopyInB2(selectKeyL1Local_);
        SetFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::MTE1_M>(0);
        mmService1.Process();
        mmService1.CopyOut(selectProbLocal_, AlignUp((aiCoreTopKTask + 1) / 2, 8));
        CrossCoreSetFlag<0x2, PIPE_FIX>(0);
    }
}

template<typename T>
__aicore__ inline void LightningIndexerGradKernel<T>::ComputeGradWeight(
    const int32_t queryOffset, const int32_t topKOffset,
    const int32_t tileTopKTask, const bool initGradWeights)
{
    // if (tileTopKTask <= 0)
    //     return;

    if ASCEND_IS_AIV {
        const int32_t tileTopKTaskAligned = AlignUp(tileTopKTask, 8);

        // Copy in GradValue
        DataCopyExtParams copyParams{static_cast<uint16_t>(1),
            static_cast<uint32_t>(tileTopKTask * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0,
            static_cast<uint8_t>(tileTopKTaskAligned - tileTopKTask), 0};
        DataCopyPad(gradValueLocal_, gradValueGlobal_[topK_ * queryOffset + topKOffset],
            copyParams, padParams);
        
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);

        Cast(gradValueFloatLocal_, gradValueLocal_, RoundMode::CAST_NONE, tileTopKTaskAligned);
        const int32_t repeatTimesRow = CeilDiv(headNumQ_, (UB_BLOCK_BYTE_SIZE / sizeof(float)));
        const int32_t repeatTimesCol = CeilDiv(tileTopKTask, (UB_BLOCK_BYTE_SIZE / sizeof(float)));

        CrossCoreWaitFlag<0x2>(0);
        ComputegradWeightsSimd<float>(gradWeightsFloatLocal_, selectProbLocal_,
            gradValueFloatLocal_, repeatTimesRow, repeatTimesCol, initGradWeights);
    }
}

template<typename T>
__aicore__ inline void LightningIndexerGradKernel<T>::ComputeGradRelu(
    const int32_t queryOffset, const int32_t tileTopKTask)
{
    if (tileTopKTask <= 0)
        return;

    if ASCEND_IS_AIV {
        DataCopy(weightsLocal_, weightsGlobal_[queryOffset * headNumQ_], headNumQ_);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        Cast(weightsFloatLocal_, weightsLocal_, RoundMode::CAST_NONE, headNumQ_);
        
        const int32_t repeatTimesRow = CeilDiv(headNumQ_, (UB_BLOCK_BYTE_SIZE / sizeof(float)));
        const int32_t repeatTimesCol = CeilDiv(tileTopKTask, (UB_BLOCK_BYTE_SIZE / sizeof(float)));
        ComputeGradReluSimd<float>(gradSelectProbFloatLocal_, weightsFloatLocal_,
            selectProbLocal_, gradValueFloatLocal_, repeatTimesRow, repeatTimesCol);
    }
}

template<typename T>
__aicore__ inline void LightningIndexerGradKernel<T>::ComputeGradQuery(
    const int32_t queryOffset, const int32_t tileTopKTask,
    const int32_t aiCoreTopKTask, const bool firstTileFlag,
    const int32_t lastTileFlag)
{
    if ASCEND_IS_AIV {
        int32_t tileTopKB32Aligned = AlignUp(tileTopKTask == 0? aiCoreTopKTask : tileTopKTask,
            (UB_BLOCK_BYTE_SIZE / sizeof(float)));
        int32_t tileTopKB16Aligned = AlignUp(tileTopKTask == 0? aiCoreTopKTask : tileTopKTask,
            (UB_BLOCK_BYTE_SIZE / sizeof(T)));
        if (tileTopKTask == 0) {
            InitConstValue(gradSelectProbNZLocal_,
                {1, static_cast<uint16_t>(headNumQ_ * tileTopKB16Aligned * sizeof(T) / 32), 0, 0.0f});
        } else {
            ND2NZSimd(gradSelectProbNZLocal_, gradSelectProbFloatLocal_,
                headNumQ_, tileTopKB32Aligned);
        }
        
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);

        if (GetSubBlockIdx() == 0) {
            DataCopy(gradSelectProbL1Local_,
                gradSelectProbNZLocal_, headNumQ_ * tileTopKB16Aligned);
        } else {
            DataCopy(gradSelectProbL1Local_[headNumQ_ * tileTopKB16Aligned],
                gradSelectProbNZLocal_, headNumQ_ * tileTopKB16Aligned);
        }
        CrossCoreSetFlag<0x2, PIPE_MTE3>(0);
        CrossCoreWaitFlag<0x2>(0);
    }

    if ASCEND_IS_AIC {
        int32_t topKAligned = AlignUp(aiCoreTopKTask, 32);
        LightningIndexerGradCube2Service<T, float> mmService2;
        mmService2.Init(headNumQ_, headDim_, topKAligned);
        CrossCoreWaitFlag<0x2>(0);
        mmService2.CopyInA2(gradSelectProbL1Local_);
        mmService2.CopyInB2(selectKeyL1Local_, queryL1Local_);
        mmService2.Process(firstTileFlag);
        if (lastTileFlag)
            mmService2.CopyOutMM1Reslut(gradQueryGlobal_[queryOffset * headNumQ_ * headDim_]);
        mmService2.CopyOutMM2Reslut(gradKeyLocal_);
        CrossCoreSetFlag<0x2, PIPE_FIX>(0);
    }
}

template<typename T>
__aicore__ inline void LightningIndexerGradKernel<T>::ScatterKey(
    const int32_t tileTopKTask, const int32_t sparseIndicesOffset,
    const int32_t curBatchKeyOffset)
{
    if (tileTopKTask == 0)
        return;

    if ASCEND_IS_AIV {
        uint32_t tileTopKTaskAligned = AlignUp(tileTopKTask, UB_BLOCK_BYTE_SIZE / sizeof(T));
        uint32_t xThreadCount = headDim_ / (sizeof(float2) / sizeof(float));
        uint32_t thread_num = min(static_cast<uint32_t>(THREAD_NUM),
            static_cast<uint32_t>(tileTopKTask * xThreadCount));
        uint32_t yThreadCount = thread_num / xThreadCount;

        if (std::is_same<T, half>::value) {
            using VECTOR_DTYPE = half2;
            Simt::VF_CALL<ScatterKeySimt<half, VECTOR_DTYPE>>(Simt::Dim3{xThreadCount, yThreadCount},
            (__gm__ VECTOR_DTYPE*) gradKeyGlobal_[curBatchKeyOffset * headDim_].GetPhyAddr(),
            (__ubuf__ float2*) gradKeyLocal_.GetPhyAddr(),
            (__ubuf__ int32_t*) sparseIndicesLocal_[sparseIndicesOffset].GetPhyAddr(),
            tileTopKTask);
        } else {
            using VECTOR_DTYPE = bfloat16x2;
            Simt::VF_CALL<ScatterKeySimt<bfloat16_t, VECTOR_DTYPE>>(Simt::Dim3{xThreadCount, yThreadCount},
            (__gm__ VECTOR_DTYPE*) gradKeyGlobal_[curBatchKeyOffset * headDim_].GetPhyAddr(),
            (__ubuf__ float2*) gradKeyLocal_.GetPhyAddr(),
            (__ubuf__ int32_t*) sparseIndicesLocal_[sparseIndicesOffset].GetPhyAddr(),
            tileTopKTask);
        }
    }
}
}